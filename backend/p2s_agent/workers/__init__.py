"""Background pipeline worker layer for the PNG-to-Shader pipeline.

Framework-free home for the daemon-thread worker that runs the pipeline after a
submit request has returned, plus the global/variant worker-pool backpressure
that bounds how many of those threads may run concurrently. Extracted from
``app.routers.png_shader`` so the web layer holds no thread-spawning logic of
its own.

Dependency direction:
  * workers → store (attribute access) — run-state reads/writes.
  * workers → core.pipeline.graph — ``run_png_shader_pipeline`` (the heavy call;
    also the load-bearing monkeypatch target in tests).
  * workers → orchestration.sessions — ``_finalize_fusion_for_run`` is imported
    LAZILY (function-body import) at its call site. A future module-level
    ``import p2s_agent.workers`` from ``orchestration.sessions`` therefore does
    not close an import cycle at load time. Do NOT import ``sessions`` at module
    level here.

This module imports no ``app.*`` / ``fastapi`` symbols (enforced by
test_agent_web_boundary), so the worker caps read the environment directly via a
local ``_env_int`` rather than the web-layer ``app.api.guards._env_int``.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from pathlib import Path
from time import time
from typing import Optional

from p2s_agent import store
from p2s_agent.config import ModelConfig, use_active_model
from p2s_agent.core.logging_config import attach_run_log, log_event
from p2s_agent.core.pipeline.graph import run_png_shader_pipeline
from p2s_agent.core.tracing import trace_context
from p2s_agent.orchestration.checkpoints import save_timeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker-pool concurrency caps (configurable via env).
# ---------------------------------------------------------------------------
# ``_env_int`` is inlined here (rather than imported from app.api.guards) to keep
# this module free of any web-layer import — see module docstring.

def _env_int(name: str, default: int) -> int:
    """Read a positive int from the environment, falling back to *default*."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


# Variant exploration concurrency cap (bounds how many variant CHILD workers run
# at once). The per-request variant-COUNT cap (_MAX_VARIANT_COUNT) is a route
# input-validation limit and stays in the web layer.
_MAX_VARIANT_CONCURRENCY = 2
_variant_worker_semaphore = threading.Semaphore(_MAX_VARIANT_CONCURRENCY)

# Global top-level worker backpressure. Every top-level pipeline worker
# (/run, /branch-refine, fusion) must acquire one of these slots at submission
# time; if none is free the endpoint returns 429 instead of spawning an
# unbounded daemon thread. Variant CHILD runs are bounded separately by
# ``_variant_worker_semaphore`` and do NOT consume a global slot (their parent
# already held one), so the two pools never double-count or deadlock.
# Configurable via env MAX_ACTIVE_RUNS (default 6). Bounded so an over-release
# bug surfaces loudly instead of silently inflating capacity.
_MAX_ACTIVE_RUNS = _env_int("MAX_ACTIVE_RUNS", 6)
_global_worker_semaphore = threading.BoundedSemaphore(_MAX_ACTIVE_RUNS)


class WorkerCapacityError(Exception):
    """Raised when the global top-level worker pool is saturated at submission.

    Endpoints translate this into ``HTTPException(429, ...)``.
    """


def _variant_preserved(stored: dict) -> dict:
    """Extract the fields that must survive a terminal store overwrite for a variant run.

    Used in both the cancelled-before-acquire and cancelled-after-acquire paths so
    that a stopped variant child retains its group identity in /status.
    """
    out = {
        "strategy": stored.get("strategy"),
        "stop_requested": stored.get("stop_requested", False),
        "strategy_revision": stored.get("strategy_revision", 1),
    }
    for k in ("variant_group_id", "variant_index", "variant_label", "lineage"):
        if k in stored:
            out[k] = stored[k]
    return out


def _run_png_shader_background(
    *,
    run_id: str,
    image_path: Path,
    upload_dir: Optional[Path],
    pipeline_input_spec: Optional[dict],
    seed_glsl: Optional[str],
    model_config: Optional[ModelConfig],
    trace_input: dict,
    trace_metadata: dict,
    pipeline_extra: Optional[dict] = None,
    variant_semaphore: Optional[threading.Semaphore] = None,
    global_slot: Optional[threading.BoundedSemaphore] = None,
) -> None:
    """Run the PNG shader pipeline after the submit request has returned.

    ``upload_dir`` is the temp dir of an uploaded image and is removed on
    completion. Branch runs reuse the parent's reference image and pass
    ``upload_dir=None`` so the parent's ``run_dir`` is never deleted.
    ``pipeline_extra`` carries human-in-loop kwargs (human_feedback_notes,
    directed_acceptance, force_first_refinement_iteration, lineage,
    extra_artifacts) forwarded to ``run_png_shader_pipeline``.
    ``variant_semaphore`` when provided puts the run through a queued→acquired
    lifecycle: the run waits in "queued" until the semaphore is free, then
    flips to "running"/"acquired".
    ``global_slot`` (set only for top-level /run, /branch-refine and fusion
    runs) is the global worker-pool slot acquired at submission; it is released
    in the ``finally`` so the next top-level submission can be admitted.
    """
    # Lazy (function-body) import to avoid a workers↔orchestration.sessions
    # module-load cycle: sessions.py imports the worker layer at module level in
    # a later task, so we resolve this symbol at call time instead.
    from p2s_agent.orchestration.sessions import _finalize_fusion_for_run

    # --- Variant queued lifecycle ---
    if variant_semaphore is not None:
        # Mark as queued and check for pre-acquire cancellation.
        with store._run_store_lock:
            stored = store._run_store.get(run_id, {})
            stored["current_phase"] = "queued"
            store._store_run_locked(run_id, stored)
            stop_early = bool(stored.get("stop_requested"))

        if stop_early:
            store._index_updated(run_id, {"status": "cancelled", "completed_at": time()})
            with store._run_store_lock:
                stored = store._run_store.get(run_id, {})
                store._store_run_locked(run_id, {
                    "run_id": run_id,
                    "status": "cancelled",
                    "completed_at": time(),
                    **_variant_preserved(stored),
                })
            if upload_dir is not None:
                shutil.rmtree(upload_dir, ignore_errors=True)
            return

        variant_semaphore.acquire()
        with store._run_store_lock:
            stored = store._run_store.get(run_id)
            stop_after_acquire = bool((stored or {}).get("stop_requested"))
            if stored is not None:
                stored["status"] = "running"
                stored["current_phase"] = "acquired"
                store._store_run_locked(run_id, stored)

        if stop_after_acquire:
            store._index_updated(run_id, {"status": "cancelled", "completed_at": time()})
            with store._run_store_lock:
                stored = store._run_store.get(run_id, {})
                store._store_run_locked(run_id, {
                    "run_id": run_id,
                    "status": "cancelled",
                    "completed_at": time(),
                    **_variant_preserved(stored),
                })
            variant_semaphore.release()
            if upload_dir is not None:
                shutil.rmtree(upload_dir, ignore_errors=True)
            return

    def _progress(phase: str) -> None:
        with store._run_store_lock:
            stored = store._run_store.get(run_id)
            if stored is not None:
                stored["current_phase"] = phase

    def _strategy_reader() -> dict:
        with store._run_store_lock:
            stored = store._run_store.get(run_id) or {}
            return {
                "strategy": dict(stored.get("strategy") or {}),
                "stop_requested": bool(stored.get("stop_requested")),
            }

    # Mutable cell: a closure can't rebind a bare outer name in Python 3.9; this
    # remembers the first run_dir a partial reveals so terminal updates can reuse it.
    seen = {"run_dir": None}

    def _publish_partial(partial: dict) -> None:
        rd = partial.get("run_dir")
        if rd and seen["run_dir"] is None:
            seen["run_dir"] = str(rd)
            store._index_updated(run_id, {"run_dir": str(rd), "status": "running"})
        store._publish_partial_to_store(run_id, partial)

    run_log_path = Path("artifacts") / run_id / "run.log"
    with attach_run_log(run_id=run_id, log_file=run_log_path):
        log_event(
            logger,
            "pipeline_worker_start",
            run_id=run_id,
            image=image_path.name,
            run_log=str(run_log_path),
            model=model_config.model if model_config else None,
        )
        try:
            with trace_context(
                "PNG Shader Pipeline",
                inputs=trace_input,
                metadata=trace_metadata,
                tags=["png-shader", run_id],
            ) as run_tree, use_active_model(model_config):
                pipeline_result = run_png_shader_pipeline(
                    image_path,
                    pipeline_input_spec,
                    run_id=run_id,
                    seed_glsl=seed_glsl,
                    progress_callback=_progress,
                    strategy_reader=_strategy_reader,
                    publish_partial=_publish_partial,
                    **(pipeline_extra or {}),
                )
                result_with_status = {**pipeline_result, "status": "completed", "current_phase": "done"}
                final_run_dir = seen["run_dir"] or (
                    str(pipeline_result.get("run_dir"))
                    if pipeline_result.get("run_dir") else None
                )
                if final_run_dir:
                    try:
                        save_timeline(final_run_dir, pipeline_result, run_id=run_id)
                    except Exception:
                        logger.warning("save_timeline failed", exc_info=True)
                store._index_updated(run_id, {
                    "status": "completed",
                    "final_score": pipeline_result.get("quality_router", {}).get("final_score"),
                    "completed_at": time(),
                    **({"run_dir": final_run_dir} if final_run_dir else {}),
                })
                with store._run_store_lock:
                    stored = store._run_store.get(run_id, {})
                    preserved = {
                        "strategy": stored.get("strategy"),
                        "stop_requested": stored.get("stop_requested", False),
                        "strategy_revision": stored.get("strategy_revision", 1),
                        # Preserve variant identity fields so they survive the
                        # result overwrite and remain queryable from /status.
                        **({"variant_group_id": stored["variant_group_id"]}
                           if "variant_group_id" in stored else {}),
                        **({"variant_index": stored["variant_index"]}
                           if "variant_index" in stored else {}),
                        **({"variant_label": stored["variant_label"]}
                           if "variant_label" in stored else {}),
                        **({"lineage": stored["lineage"]}
                           if "lineage" in stored else {}),
                    }
                    store._store_run_locked(run_id, {**result_with_status, **preserved})
                # Best-effort: close out a fusion plan record so its /fusions
                # poll terminates instead of reporting 'running' forever (Bug 5).
                _finalize_fusion_for_run(run_id, "completed")
                if run_tree is not None:
                    run_tree.end(
                        outputs={
                            "run_id": pipeline_result.get("run_id"),
                            "selected_id": pipeline_result.get("scoreboard", {}).get("selected_id"),
                            "score": pipeline_result.get("quality_router", {}).get("final_score"),
                            "refinement": pipeline_result.get("refinement_summary", {}),
                        }
                    )
                log_event(
                    logger,
                    "pipeline_worker_done",
                    run_id=run_id,
                    selected_id=pipeline_result.get("scoreboard", {}).get("selected_id"),
                    score=pipeline_result.get("quality_router", {}).get("final_score"),
                )
        except Exception as exc:
            logger.exception("worker failed: run_id=%s", run_id)
            log_event(
                logger,
                "pipeline_worker_failed",
                level=logging.ERROR,
                run_id=run_id,
                error=f"{exc.__class__.__name__}: {exc}",
            )
            store._index_updated(run_id, {
                "status": "failed",
                "completed_at": time(),
                **({"run_dir": seen["run_dir"]} if seen["run_dir"] else {}),
            })
            with store._run_store_lock:
                stored = store._run_store.get(run_id, {})
                preserved = {
                    "strategy": stored.get("strategy"),
                    "stop_requested": stored.get("stop_requested", False),
                    "strategy_revision": stored.get("strategy_revision", 1),
                    # Preserve variant identity fields on failure path too.
                    **({"variant_group_id": stored["variant_group_id"]}
                       if "variant_group_id" in stored else {}),
                    **({"variant_index": stored["variant_index"]}
                       if "variant_index" in stored else {}),
                    **({"variant_label": stored["variant_label"]}
                       if "variant_label" in stored else {}),
                    **({"lineage": stored["lineage"]}
                       if "lineage" in stored else {}),
                }
                store._store_run_locked(run_id, {
                    "run_id": run_id,
                    "status": "failed",
                    "error": f"Pipeline error: {exc}",
                    "completed_at": time(),
                    **preserved,
                })
            # Best-effort fusion-plan finalization on failure (Bug 5).
            _finalize_fusion_for_run(run_id, "failed")
        finally:
            if upload_dir is not None:
                shutil.rmtree(upload_dir, ignore_errors=True)
            if variant_semaphore is not None:
                variant_semaphore.release()
            # Release the global top-level worker slot (only set for /run,
            # /branch-refine and fusion runs) so the next submission is admitted.
            if global_slot is not None:
                global_slot.release()


def _start_pipeline_worker(
    *,
    run_id: str,
    image_path: Path,
    upload_dir: Optional[Path],
    pipeline_input_spec: Optional[dict],
    seed_glsl: Optional[str],
    model_config: Optional[ModelConfig],
    trace_input: dict,
    trace_metadata: dict,
    pipeline_extra: Optional[dict] = None,
    variant_semaphore: Optional[threading.Semaphore] = None,
) -> None:
    """Register the run's model and launch the background pipeline thread.

    Shared by ``/run`` (uploaded image, ``upload_dir`` set) and
    ``/branch-refine`` (parent reference image, ``upload_dir=None``).
    ``variant_semaphore`` is forwarded to the worker for the queued→acquired
    lifecycle used by variant child runs.

    Backpressure: TOP-LEVEL runs (``variant_semaphore is None`` — i.e. /run,
    /branch-refine, fusion) acquire one global worker slot here, before the
    thread is spawned. If the pool is saturated this raises
    ``WorkerCapacityError`` and NO thread is started; callers translate that
    into HTTP 429. Variant child runs (``variant_semaphore`` set) are bounded by
    their own semaphore and skip the global slot. The acquired global slot is
    forwarded to the worker, which releases it in its ``finally``.
    """
    global_slot: Optional[threading.BoundedSemaphore] = None
    if variant_semaphore is None:
        # Resolve via the module attribute so tests can monkeypatch the cap.
        sem = _global_worker_semaphore
        if not sem.acquire(blocking=False):
            raise WorkerCapacityError(
                "worker pool at capacity; retry shortly"
            )
        global_slot = sem

    try:
        store._store_run_model(run_id, model_config)
        threading.Thread(
            target=_run_png_shader_background,
            kwargs={
                "run_id": run_id,
                "image_path": image_path,
                "upload_dir": upload_dir,
                "pipeline_input_spec": pipeline_input_spec,
                "seed_glsl": seed_glsl,
                "model_config": model_config,
                "trace_input": trace_input,
                "trace_metadata": trace_metadata,
                "pipeline_extra": pipeline_extra,
                "variant_semaphore": variant_semaphore,
                "global_slot": global_slot,
            },
            daemon=True,
        ).start()
    except BaseException:
        # Thread never launched → release the slot we just took so it is not
        # leaked (the worker's finally only runs if the thread actually starts).
        if global_slot is not None:
            global_slot.release()
        raise
