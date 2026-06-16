"""FastAPI router for PNG-to-Shader pipeline.

Endpoints:
  POST /png-shader/run             — submit image, get run_id + scoreboard
  GET  /png-shader/status/{run_id} — get cached result (in-memory store)
  POST /png-shader/refine/{run_id} — human-in-the-loop refinement
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import shutil
import tempfile
import threading
from pathlib import Path
from time import time
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.config import ModelConfig, use_active_model
from app.llm.model_resolver import ModelResolutionError, resolve_model_config

from app.pipeline.checkpoints import (
    CheckpointError,
    _selected_candidate,
    build_timeline,
    checkpoint_metadata,
    list_checkpoints,
    resolve_checkpoint,
    resolve_checkpoint_artifact,
    save_timeline,
)
from app.pipeline.variant_groups import (
    VariantGroupRecord,
    aggregate_group_status,
    append_group_event,
    build_variant_strategies,
    load_group,
    save_group,
)
from app.pipeline.draw_sessions import (
    DrawSessionRecord,
    aggregate_draw_status,
    append_session_event,
    load_session,
    load_session_events,
    plan_draw_batches,
    save_session,
)
from app.pipeline.preferences import (
    PreferenceEvent,
    append_preference_event,
    build_preference_notes,
    clear_preferences,
    default_profile,
    load_preference_events,
    load_profile,
    patch_profile,
    rebuild_profile,
    save_profile,
)
from app.pipeline.graph import run_png_shader_pipeline
from app.pipeline.human_feedback import (
    MODES,
    FeedbackValidationError,
    build_human_feedback_notes,
    validate_feedback,
)
from app.pipeline.human_constraints import (
    HumanConstraintSpec,
    RegionConstraint,
    build_constraint_notes,
    parse_constraint_spec,
    spec_to_dict,
    validate_constraint_spec,
)
from app.pipeline.region_metrics import compute_region_metrics
from app.pipeline.artifacts import save_json
from app.pipeline.input_spec import build_input_spec, validate_input_spec
from app.pipeline.run_index import (
    RunIndexError,
    RunLineageRecord,
    append_run_created,
    append_run_updated,
    build_branch_tree,
    load_run_index,
    update_run_metadata,
)
from app.services.langsmith_tracing import trace_context
from app.services.logging_config import attach_run_log, log_event, logging_context

logger = logging.getLogger(__name__)

_run_store: dict[str, dict] = {}
_run_store_lock = threading.Lock()
_MAX_STORE_SIZE = 100

# Tests override this to isolate from the real backend/test_results/run_index.jsonl.
_RUN_INDEX_PATH: Optional[str] = None

# Resolved per-run model configs (may hold api_keys). Kept SEPARATE from
# _run_store so the selected model — and any secret key — is never returned by
# the client-facing /status endpoint.
_run_models: dict[str, ModelConfig] = {}
_run_models_lock = threading.Lock()

# Metadata fields that are allowed to be patched and mirrored into the in-memory store.
_METADATA_MIRROR_KEYS: frozenset[str] = frozenset({"title", "favorite", "tags"})

# Variant exploration concurrency constants.
_MAX_VARIANT_COUNT = 6
_MAX_VARIANT_CONCURRENCY = 2
_variant_worker_semaphore = threading.Semaphore(_MAX_VARIANT_CONCURRENCY)
_VARIANT_GROUPS_ROOT: Optional[str] = None  # tests override to isolate
_DRAW_SESSIONS_ROOT: Optional[str] = None  # tests override to isolate
_PREFERENCES_ROOT: Optional[str] = None  # tests override to isolate

# Draw-session (V3.5 batch draw) card-event vocabulary.
_DRAW_CARD_EVENT_TYPES: frozenset[str] = frozenset({
    "favorite", "eliminate", "tag", "note", "use_as_fusion_base", "use_as_region_source",
})

# V4.2: allowlist regex for region_id — mirrors _CANDIDATE_ID_RE in checkpoints.py.
# Rejects path-traversal characters (/, ..) and leading dots.
_REGION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")


# ---------------------------------------------------------------------------
# Best-effort run-index helpers — I/O errors only log; they never 500 a request
# or kill a worker thread.
# ---------------------------------------------------------------------------

def _index_created(record: RunLineageRecord) -> None:
    try:
        append_run_created(record, path=_RUN_INDEX_PATH)
    except Exception:
        logger.warning("run index append_created failed", exc_info=True)


def _index_updated(run_id: str, fields: dict) -> None:
    try:
        append_run_updated(run_id, fields, path=_RUN_INDEX_PATH)
    except Exception:
        logger.warning("run index append_updated failed", exc_info=True)


def _store_run_model(run_id: str, model_config: ModelConfig) -> None:
    with _run_models_lock:
        if run_id not in _run_models and len(_run_models) >= _MAX_STORE_SIZE:
            del _run_models[next(iter(_run_models))]
        _run_models[run_id] = model_config


def _get_run_model(run_id: str) -> Optional[ModelConfig]:
    with _run_models_lock:
        return _run_models.get(run_id)

router = APIRouter(prefix="/png-shader", tags=["png-shader"])


def _store_run(run_id: str, payload: dict) -> None:
    """Store a PNG shader run state with bounded in-memory retention."""
    with _run_store_lock:
        if run_id not in _run_store and len(_run_store) >= _MAX_STORE_SIZE:
            oldest_key = next(iter(_run_store))
            del _run_store[oldest_key]
        _run_store[run_id] = payload


def _publish_partial_to_store(run_id: str, partial: dict) -> None:
    """Merge a partial pipeline result into a still-running run's store entry.

    Best-effort: silently no-ops when the run was evicted or already reached a
    terminal state, so a late partial can't resurrect a finished run. Only data
    fields are merged; control fields (strategy / stop_requested /
    strategy_revision / status / ...) are preserved.
    """
    with _run_store_lock:
        stored = _run_store.get(run_id)
        if stored is None or stored.get("status") != "running":
            return
        stored.update(partial)


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
    """
    # --- Variant queued lifecycle ---
    if variant_semaphore is not None:
        # Mark as queued and check for pre-acquire cancellation.
        with _run_store_lock:
            stored = _run_store.get(run_id, {})
            stored["current_phase"] = "queued"
            _run_store[run_id] = stored
            stop_early = bool(stored.get("stop_requested"))

        if stop_early:
            _index_updated(run_id, {"status": "cancelled", "completed_at": time()})
            with _run_store_lock:
                stored = _run_store.get(run_id, {})
                _run_store[run_id] = {
                    "run_id": run_id,
                    "status": "cancelled",
                    "completed_at": time(),
                    **_variant_preserved(stored),
                }
            if upload_dir is not None:
                shutil.rmtree(upload_dir, ignore_errors=True)
            return

        variant_semaphore.acquire()
        with _run_store_lock:
            stored = _run_store.get(run_id, {})
            stop_after_acquire = bool(stored.get("stop_requested"))
            if stored is not None and run_id in _run_store:
                _run_store[run_id]["status"] = "running"
                _run_store[run_id]["current_phase"] = "acquired"

        if stop_after_acquire:
            _index_updated(run_id, {"status": "cancelled", "completed_at": time()})
            with _run_store_lock:
                stored = _run_store.get(run_id, {})
                _run_store[run_id] = {
                    "run_id": run_id,
                    "status": "cancelled",
                    "completed_at": time(),
                    **_variant_preserved(stored),
                }
            variant_semaphore.release()
            if upload_dir is not None:
                shutil.rmtree(upload_dir, ignore_errors=True)
            return

    def _progress(phase: str) -> None:
        with _run_store_lock:
            stored = _run_store.get(run_id)
            if stored is not None:
                stored["current_phase"] = phase

    def _strategy_reader() -> dict:
        with _run_store_lock:
            stored = _run_store.get(run_id) or {}
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
            _index_updated(run_id, {"run_dir": str(rd), "status": "running"})
        _publish_partial_to_store(run_id, partial)

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
                _index_updated(run_id, {
                    "status": "completed",
                    "final_score": pipeline_result.get("quality_router", {}).get("final_score"),
                    "completed_at": time(),
                    **({"run_dir": final_run_dir} if final_run_dir else {}),
                })
                with _run_store_lock:
                    stored = _run_store.get(run_id, {})
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
                    _run_store[run_id] = {**result_with_status, **preserved}
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
            _index_updated(run_id, {
                "status": "failed",
                "completed_at": time(),
                **({"run_dir": seen["run_dir"]} if seen["run_dir"] else {}),
            })
            with _run_store_lock:
                stored = _run_store.get(run_id, {})
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
                _run_store[run_id] = {
                    "run_id": run_id,
                    "status": "failed",
                    "error": f"Pipeline error: {exc}",
                    "completed_at": time(),
                    **preserved,
                }
        finally:
            if upload_dir is not None:
                shutil.rmtree(upload_dir, ignore_errors=True)
            if variant_semaphore is not None:
                variant_semaphore.release()


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
    """
    _store_run_model(run_id, model_config)
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
        },
        daemon=True,
    ).start()


@router.post("/run")
async def run_png_shader(
    image: UploadFile = File(...),
    input_spec_json: Optional[str] = Form(default=None),
    seed_glsl: Optional[str] = Form(default=None),
) -> dict:
    """Submit an image and run the PNG-to-Shader pipeline in the background."""
    input_spec: Optional[dict] = None
    if input_spec_json:
        try:
            input_spec = json.loads(input_spec_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid input_spec_json: {exc}",
            ) from exc

    if input_spec is not None and not isinstance(input_spec, dict):
        raise HTTPException(
            status_code=422,
            detail="input_spec_json must decode to an object",
        )

    # Resolve the selected model (preset or custom). `model` is consumed here and
    # not part of the pipeline input-spec schema, so remove it before building.
    model_selection = input_spec.pop("model", None) if input_spec is not None else None
    try:
        model_config = resolve_model_config(model_selection)
    except ModelResolutionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if seed_glsl is not None:
        if not seed_glsl.strip():
            raise HTTPException(
                status_code=422,
                detail="seed_glsl must be a non-empty string when provided",
            )
        # Default the seed run to always-refine unless the caller set the mode.
        overrides = dict(input_spec or {})
        quality = dict(overrides.get("quality") or {})
        quality.setdefault("refinement_mode", "on")
        overrides["quality"] = quality
        input_spec = overrides

    run_id = "run_" + str(uuid4())[:8]
    with logging_context(run_id=run_id):
        log_event(
            logger,
            "pipeline_submit_received",
            method="POST",
            path="/png-shader/run",
            run_id=run_id,
            filename=image.filename,
            content_type=image.content_type,
            has_input_spec=input_spec is not None,
            has_seed_glsl=seed_glsl is not None,
        )
    upload_dir = Path(tempfile.mkdtemp(prefix=f"png_shader_{run_id}_"))
    try:
        suffix = Path(image.filename or "upload.png").suffix or ".png"
        image_path = upload_dir / f"input{suffix}"
        try:
            contents = await image.read()
            image_path.write_bytes(contents)
        except Exception as exc:
            shutil.rmtree(upload_dir, ignore_errors=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save uploaded image: {exc}",
            ) from exc

        pipeline_input_spec = None
        if input_spec is not None:
            pipeline_input_spec = build_input_spec(image_path, **input_spec)
            errors = validate_input_spec(pipeline_input_spec)
            if errors:
                shutil.rmtree(upload_dir, ignore_errors=True)
                raise HTTPException(
                    status_code=422,
                    detail={"input_spec_errors": errors},
                )

        if pipeline_input_spec is not None:
            quality_for_store = dict(pipeline_input_spec.get("quality") or {})
        else:
            from app.pipeline.input_spec import build_input_spec as _default_spec
            quality_for_store = dict(_default_spec(image_path)["quality"])

        trace_input = {
            "run_id": run_id,
            "filename": image.filename,
            "content_type": image.content_type,
            "input_spec": pipeline_input_spec,
        }
        trace_metadata = {
            "run_id": run_id,
            "pipeline": "png-shader",
            "filename": image.filename,
            "content_type": image.content_type,
        }
        initial_result = {
            "run_id": run_id,
            "status": "running",
            "filename": image.filename,
            "content_type": image.content_type,
            "submitted_at": time(),
            "strategy": quality_for_store,
            "stop_requested": False,
            "strategy_revision": 1,
        }
        _store_run(run_id, initial_result)
        _index_created(RunLineageRecord(
            run_id=run_id, root_run_id=run_id, parent_run_id=None,
            source_checkpoint_id=None, source_checkpoint_label=None,
            mode=None, feedback=None, title=None,
            status="pending", run_dir=None, created_at=time(),
        ))
        _start_pipeline_worker(
            run_id=run_id,
            image_path=image_path,
            upload_dir=upload_dir,
            pipeline_input_spec=pipeline_input_spec,
            seed_glsl=seed_glsl,
            model_config=model_config,
            trace_input=trace_input,
            trace_metadata=trace_metadata,
        )
        return initial_result
    except HTTPException:
        raise
    except Exception as exc:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline submit error: {exc}",
        ) from exc


@router.get("/status/{run_id}")
async def get_status(run_id: str) -> dict:
    """Retrieve a cached pipeline result by run_id."""
    with _run_store_lock:
        result = _run_store.get(run_id)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"run_id '{run_id}' not found",
        )
    return result


@router.get("/runs/{run_id}/checkpoints")
async def get_checkpoints(run_id: str) -> dict:
    """List the branchable checkpoints (candidates / iterations / final) of a run.

    Works for both running and completed runs; GLSL payloads are omitted and
    re-resolved on demand by ``/branch-refine``.
    """
    with _run_store_lock:
        stored = _run_store.get(run_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"run_id '{run_id}' not found")
    return {
        "run_id": run_id,
        "status": stored.get("status"),
        "checkpoints": list_checkpoints(stored),
    }


@router.post("/runs/{run_id}/branch-refine")
async def branch_refine(run_id: str, payload: dict) -> dict:
    """Create a directed-refinement child run seeded from a parent checkpoint.

    The child is an independent run (own run_id / run_dir / lifecycle) seeded
    with the checkpoint's GLSL and the user's feedback. The parent is never
    overwritten and its reference image is never deleted.
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="payload must be a JSON object")

    with _run_store_lock:
        parent = _run_store.get(run_id)
    if parent is None:
        raise HTTPException(status_code=404, detail=f"run_id '{run_id}' not found")

    checkpoint_id = str(payload.get("checkpoint_id") or "final:selected")
    mode = str(payload.get("mode") or "refine")
    feedback = str(payload.get("feedback") or "")
    locks = payload.get("locks") or {}
    stop_parent = bool(payload.get("stop_parent"))
    quality_overrides = payload.get("quality") or {}
    if not isinstance(locks, dict) or not isinstance(quality_overrides, dict):
        raise HTTPException(status_code=422, detail="locks and quality must be objects")
    if mode not in MODES:
        raise HTTPException(status_code=422, detail=f"mode must be one of {list(MODES)}")
    try:
        validate_feedback(feedback, mode)
    except FeedbackValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Parse and validate optional structured constraints (V4.1).
    _constraints_payload = payload.get("constraints")
    _has_constraints = isinstance(_constraints_payload, dict) and bool(_constraints_payload)
    _constraint_spec: HumanConstraintSpec | None = None
    if _has_constraints:
        _constraint_spec = parse_constraint_spec(_constraints_payload)
        _constraint_errors = validate_constraint_spec(_constraint_spec)
        if _constraint_errors:
            raise HTTPException(status_code=422, detail={"constraint_errors": _constraint_errors})

    # Parent must have produced at least one branchable checkpoint.
    if not list_checkpoints(parent):
        raise HTTPException(
            status_code=409, detail="parent run has no branchable checkpoint yet"
        )
    try:
        checkpoint = resolve_checkpoint(parent, checkpoint_id)
    except CheckpointError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    parent_run_dir = parent.get("run_dir")
    if not parent_run_dir:
        raise HTTPException(status_code=409, detail="parent run_dir is not available yet")
    reference_path = Path(parent_run_dir) / "reference_input.png"
    if not reference_path.exists():
        raise HTTPException(
            status_code=409, detail="parent reference image is not available"
        )

    child_run_id = "run_" + uuid4().hex[:8]
    notes = build_human_feedback_notes(
        feedback=feedback, mode=mode, locks=locks, checkpoint=checkpoint
    )
    # V4.1: append constraint notes when constraints were supplied.
    if _constraint_spec is not None:
        notes += build_constraint_notes(_constraint_spec)

    # V4.4: inject user preference notes (gated by use_preferences flag).
    _use_preferences: bool = _constraint_spec.use_preferences if _constraint_spec is not None else True
    _pref_profile = load_profile(root=_PREFERENCES_ROOT)
    _pref_notes = build_preference_notes(_pref_profile)

    parent_lineage = parent.get("lineage") or {}
    root_run_id = parent_lineage.get("root_run_id") or run_id
    lineage = {
        "parent_run_id": run_id,
        "root_run_id": root_run_id,
        "source_checkpoint_id": checkpoint_id,
        "source_checkpoint_label": checkpoint.label,
        "mode": mode,
        "feedback": feedback,
    }

    # Child quality: force the directed closed loop on, at least one iteration.
    quality = dict(quality_overrides)
    quality["refinement_mode"] = "on"
    quality["max_refinement_iterations"] = max(
        int(quality.get("max_refinement_iterations", 0) or 0), 1
    )

    # Directed acceptance (V1.2): goal-driven modes may accept a small score drop
    # when the VLM judge prefers the candidate for the human goal. JSON-only
    # config — the callable is built in the pipeline from run context.
    directed_enabled = mode in ("refine", "polish")
    directed_acceptance = {
        "enabled": directed_enabled,
        "feedback": feedback,
        "mode": mode,
        # polish keeps structure stable: no score regression tolerated.
        "score_drop_tolerance": 0.0 if mode == "polish" else 0.03,
        "require_vlm_for_score_drop": True,
    }
    if directed_enabled:
        # Enable the VLM judge so directed acceptance can arbitrate a small drop;
        # degrades to metric-only acceptance if the VLM is unavailable.
        quality["vlm_judge_enabled"] = 1

    child_input_spec = build_input_spec(str(reference_path), quality=quality)
    errors = validate_input_spec(child_input_spec)
    if errors:
        raise HTTPException(status_code=422, detail={"input_spec_errors": errors})

    branch_request = {
        "checkpoint_id": checkpoint_id,
        "feedback": feedback,
        "mode": mode,
        "locks": locks,
        "stop_parent": stop_parent,
        "quality": quality,
    }
    extra_artifacts = {
        "branch_request.json": branch_request,
        "lineage.json": lineage,
        "source_checkpoint.json": checkpoint_metadata(checkpoint),
        "source_checkpoint.glsl": checkpoint.glsl,
        "human_feedback.txt": feedback,
        "directed_acceptance.json": directed_acceptance,
    }
    # V4.1: add constraints artifacts when present.
    if _constraint_spec is not None:
        _constraint_dict = spec_to_dict(_constraint_spec)
        extra_artifacts["constraints.json"] = _constraint_dict
        directed_acceptance["constraints"] = _constraint_dict

    # V4.4: inject preference notes + snapshot when enabled and profile is non-empty.
    if _use_preferences and _pref_notes:
        notes += _pref_notes
        extra_artifacts["preference_profile_snapshot.json"] = _pref_profile
        directed_acceptance["preference_score_drop_tolerance_hint"] = _pref_profile.get(
            "score_drop_tolerance_hint"
        )

    log_event(
        logger,
        "branch_refine_request",
        run_id=run_id,
        child_run_id=child_run_id,
        checkpoint_id=checkpoint_id,
        mode=mode,
        feedback_len=len(feedback),
    )

    initial_result = {
        "run_id": child_run_id,
        "status": "running",
        "parent_run_id": run_id,
        "source_checkpoint_id": checkpoint_id,
        "lineage": lineage,
        "submitted_at": time(),
        "strategy": dict(child_input_spec.get("quality") or quality),
        "stop_requested": False,
        "strategy_revision": 1,
    }
    _store_run(child_run_id, initial_result)
    _index_created(RunLineageRecord(
        run_id=child_run_id, root_run_id=root_run_id, parent_run_id=run_id,
        source_checkpoint_id=checkpoint_id,
        source_checkpoint_label=checkpoint.label,
        mode=mode, feedback=feedback, title=None,
        status="pending", run_dir=None, created_at=time(),
    ))

    _start_pipeline_worker(
        run_id=child_run_id,
        image_path=reference_path,
        upload_dir=None,  # reuse parent reference; never delete the parent run_dir
        pipeline_input_spec=child_input_spec,
        seed_glsl=checkpoint.glsl,
        model_config=_get_run_model(run_id),
        trace_input={
            "run_id": child_run_id,
            "parent_run_id": run_id,
            "checkpoint_id": checkpoint_id,
        },
        trace_metadata={
            "run_id": child_run_id,
            "pipeline": "png-shader-branch",
            "parent_run_id": run_id,
        },
        pipeline_extra={
            "human_feedback_notes": notes,
            "directed_acceptance": directed_acceptance,
            "force_first_refinement_iteration": True,
            "lineage": lineage,
            "extra_artifacts": extra_artifacts,
        },
    )

    if stop_parent:
        with _run_store_lock:
            stored_parent = _run_store.get(run_id)
            if stored_parent is not None and stored_parent.get("status") == "running":
                stored_parent["stop_requested"] = True

    return initial_result


def _create_variant_group(
    *,
    parent_run_id: str,
    root_run_id: str,
    checkpoint,
    checkpoint_id: str,
    reference_path: Path,
    feedback: str,
    mode: str,
    diversity: str,
    strategies: list[dict],
    quality_overrides: dict,
    draw_session_id: "str | None" = None,
    constraint_spec: "HumanConstraintSpec | None" = None,
    use_preferences: bool = True,
) -> tuple[str, list[str]]:
    """Create one variant group: spawn N child runs (one per strategy), persist
    the VariantGroupRecord, append the 'created' event. Returns (group_id, child_run_ids).

    Behavior-identical to the inline loop previously in explore_variants. When
    draw_session_id is set it is recorded on the group record and on each child's
    run-index lineage (additive; None preserves prior explore-variants behavior).
    """
    group_id = "group_" + uuid4().hex[:8]
    child_run_ids: list[str] = []

    # V4.4: load preference profile once for the whole group (all children share it).
    _vg_pref_profile = load_profile(root=_PREFERENCES_ROOT)
    _vg_pref_notes = build_preference_notes(_vg_pref_profile)

    for idx, strategy in enumerate(strategies):
        child_run_id = "run_" + uuid4().hex[:8]

        variant_lineage = {
            "parent_run_id": parent_run_id,
            "root_run_id": root_run_id,
            "source_checkpoint_id": checkpoint_id,
            "source_checkpoint_label": checkpoint.label,
            "mode": mode,
            "feedback": feedback,
            "variant_group_id": group_id,
            "variant_index": idx,
            "variant_label": strategy["label"],
            "variant_strategy": strategy,
        }

        notes = build_human_feedback_notes(
            feedback=feedback,
            mode=mode,
            locks=strategy.get("locks") or {},
            checkpoint=checkpoint,
        ) + list(strategy.get("notes") or [])
        # V4.1: append constraint notes when constraints were supplied.
        if constraint_spec is not None:
            notes += build_constraint_notes(constraint_spec)

        quality = dict(quality_overrides)
        quality["refinement_mode"] = "on"
        quality["max_refinement_iterations"] = max(
            int(quality.get("max_refinement_iterations", 0) or 0), 1
        )
        quality["vlm_judge_enabled"] = 1

        directed_acceptance = {
            "enabled": True,
            "feedback": feedback,
            "mode": mode,
            "score_drop_tolerance": strategy["score_drop_tolerance"],
            "require_vlm_for_score_drop": True,
        }

        child_input_spec = build_input_spec(str(reference_path), quality=quality)
        errors = validate_input_spec(child_input_spec)
        if errors:
            raise HTTPException(status_code=422, detail={"input_spec_errors": errors})

        branch_request = {
            "checkpoint_id": checkpoint_id,
            "feedback": feedback,
            "mode": mode,
            "variant_index": idx,
            "variant_label": strategy["label"],
            "group_id": group_id,
        }
        extra_artifacts = {
            "branch_request.json": branch_request,
            "lineage.json": variant_lineage,
            "source_checkpoint.json": checkpoint_metadata(checkpoint),
            "source_checkpoint.glsl": checkpoint.glsl,
            "human_feedback.txt": feedback,
            "directed_acceptance.json": directed_acceptance,
            "variant_strategy.json": strategy,
        }
        # V4.1: add constraints artifacts when present.
        if constraint_spec is not None:
            _cspec_dict = spec_to_dict(constraint_spec)
            extra_artifacts["constraints.json"] = _cspec_dict
            directed_acceptance["constraints"] = _cspec_dict

        # V4.4: inject preference notes + snapshot when enabled and profile is non-empty.
        if use_preferences and _vg_pref_notes:
            notes += _vg_pref_notes
            extra_artifacts["preference_profile_snapshot.json"] = _vg_pref_profile
            directed_acceptance["preference_score_drop_tolerance_hint"] = _vg_pref_profile.get(
                "score_drop_tolerance_hint"
            )

        initial_result = {
            "run_id": child_run_id,
            "status": "queued",
            "current_phase": "queued",
            "parent_run_id": parent_run_id,
            "source_checkpoint_id": checkpoint_id,
            "lineage": variant_lineage,
            "variant_group_id": group_id,
            "variant_index": idx,
            "variant_label": strategy["label"],
            "submitted_at": time(),
            "strategy": dict(child_input_spec.get("quality") or quality),
            "stop_requested": False,
            "strategy_revision": 1,
        }
        _store_run(child_run_id, initial_result)
        _index_created(RunLineageRecord(
            run_id=child_run_id,
            root_run_id=root_run_id,
            parent_run_id=parent_run_id,
            source_checkpoint_id=checkpoint_id,
            source_checkpoint_label=checkpoint.label,
            mode=mode,
            feedback=feedback,
            title=None,
            status="queued",
            run_dir=None,
            created_at=time(),
            variant_group_id=group_id,
            variant_index=idx,
            variant_label=strategy["label"],
            draw_session_id=draw_session_id,
        ))
        _start_pipeline_worker(
            run_id=child_run_id,
            image_path=reference_path,
            upload_dir=None,
            pipeline_input_spec=child_input_spec,
            seed_glsl=checkpoint.glsl,
            model_config=_get_run_model(parent_run_id),
            trace_input={
                "run_id": child_run_id,
                "parent_run_id": parent_run_id,
                "checkpoint_id": checkpoint_id,
                "variant_group_id": group_id,
                "variant_index": idx,
            },
            trace_metadata={
                "run_id": child_run_id,
                "pipeline": "png-shader-variant",
                "parent_run_id": parent_run_id,
                "variant_group_id": group_id,
                "variant_index": idx,
            },
            pipeline_extra={
                "human_feedback_notes": notes,
                "directed_acceptance": directed_acceptance,
                "force_first_refinement_iteration": True,
                "lineage": variant_lineage,
                "extra_artifacts": extra_artifacts,
            },
            variant_semaphore=_variant_worker_semaphore,
        )
        child_run_ids.append(child_run_id)

    # Persist the group record best-effort (I/O errors must not fail the request).
    record = VariantGroupRecord(
        group_id=group_id,
        root_run_id=root_run_id,
        parent_run_id=parent_run_id,
        source_checkpoint_id=checkpoint_id,
        feedback=feedback,
        mode=mode,
        variant_count=len(strategies),
        diversity=diversity,
        status="running",
        child_run_ids=child_run_ids,
        created_at=time(),
        draw_session_id=draw_session_id,
    )
    try:
        save_group(record, root=_VARIANT_GROUPS_ROOT)
        append_group_event(
            group_id,
            {"event": "created", "child_run_ids": child_run_ids, "at": time()},
            root=_VARIANT_GROUPS_ROOT,
        )
    except Exception:
        logger.warning("variant group persist failed for group_id=%s", group_id, exc_info=True)

    return group_id, child_run_ids


@router.post("/runs/{run_id}/explore-variants")
async def explore_variants(run_id: str, payload: dict) -> dict:
    """Fan one checkpoint into N variant child runs, each guided by a different strategy.

    Body: { checkpoint_id?, feedback, variant_count?=4, diversity?="medium",
            mode?="explore", quality?={}, stop_parent?=false }

    Each child is queued behind a shared semaphore (max _MAX_VARIANT_CONCURRENCY
    concurrent workers). The endpoint returns immediately with the group_id and
    child_run_ids; callers poll /status/<child_run_id> for individual progress.
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="payload must be a JSON object")

    with _run_store_lock:
        parent = _run_store.get(run_id)
    if parent is None:
        raise HTTPException(status_code=404, detail=f"run_id '{run_id}' not found")

    variant_count = int(payload.get("variant_count") or 4)
    diversity = str(payload.get("diversity") or "medium")
    mode = str(payload.get("mode") or "explore")
    feedback = str(payload.get("feedback") or "")
    checkpoint_id_raw = payload.get("checkpoint_id")
    quality_overrides = payload.get("quality") or {}
    stop_parent = bool(payload.get("stop_parent"))

    if not isinstance(quality_overrides, dict):
        raise HTTPException(status_code=422, detail="quality must be an object")

    if variant_count < 2 or variant_count > _MAX_VARIANT_COUNT:
        raise HTTPException(
            status_code=422,
            detail=f"variant_count must be between 2 and {_MAX_VARIANT_COUNT}, got {variant_count}",
        )

    if not feedback or not feedback.strip():
        raise HTTPException(status_code=422, detail="feedback is required for explore-variants")

    # Parse and validate optional structured constraints (V4.1).
    _ev_constraints_payload = payload.get("constraints")
    _ev_has_constraints = isinstance(_ev_constraints_payload, dict) and bool(_ev_constraints_payload)
    _ev_constraint_spec: HumanConstraintSpec | None = None
    if _ev_has_constraints:
        _ev_constraint_spec = parse_constraint_spec(_ev_constraints_payload)
        _ev_constraint_errors = validate_constraint_spec(_ev_constraint_spec)
        if _ev_constraint_errors:
            raise HTTPException(status_code=422, detail={"constraint_errors": _ev_constraint_errors})

    if not list_checkpoints(parent):
        raise HTTPException(status_code=409, detail="no branchable checkpoint yet")

    checkpoint_id = str(checkpoint_id_raw) if checkpoint_id_raw is not None else "final:selected"
    try:
        checkpoint = resolve_checkpoint(parent, checkpoint_id)
    except CheckpointError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    parent_run_dir = parent.get("run_dir")
    if not parent_run_dir:
        raise HTTPException(status_code=409, detail="parent run_dir is not available yet")
    reference_path = Path(parent_run_dir) / "reference_input.png"
    if not reference_path.exists():
        raise HTTPException(status_code=409, detail="parent reference image is not available")

    parent_lineage = parent.get("lineage") or {}
    root_run_id = parent_lineage.get("root_run_id") or run_id

    strategies = build_variant_strategies(
        feedback=feedback,
        count=variant_count,
        diversity=diversity,
        mode=mode,
    )

    _ev_use_preferences: bool = _ev_constraint_spec.use_preferences if _ev_constraint_spec is not None else True
    group_id, child_run_ids = _create_variant_group(
        parent_run_id=run_id,
        root_run_id=root_run_id,
        checkpoint=checkpoint,
        checkpoint_id=checkpoint_id,
        reference_path=reference_path,
        feedback=feedback,
        mode=mode,
        diversity=diversity,
        strategies=strategies,
        quality_overrides=quality_overrides,
        constraint_spec=_ev_constraint_spec,
        use_preferences=_ev_use_preferences,
    )

    if stop_parent:
        with _run_store_lock:
            stored_parent = _run_store.get(run_id)
            if stored_parent is not None and stored_parent.get("status") == "running":
                stored_parent["stop_requested"] = True

    log_event(
        logger,
        "variant_group_created",
        run_id=run_id,
        group_id=group_id,
        variant_count=variant_count,
    )

    return {
        "group_id": group_id,
        "status": "running",
        "parent_run_id": run_id,
        "source_checkpoint_id": checkpoint_id,
        "child_run_ids": child_run_ids,
    }


# ---------------------------------------------------------------------------
# V3.5 draw-session (gacha-style batch draw) helpers + endpoints
# ---------------------------------------------------------------------------


def _resolve_draw_checkpoint(parent: dict, checkpoint_id: str):
    """Resolve a draw-session checkpoint + reference image from a parent run.

    Mirrors explore_variants' pre-checks. Returns ``(checkpoint, reference_path)``.

    Raises:
        HTTPException(409): no branchable checkpoint / no run_dir / no reference.
        HTTPException(422): checkpoint_id cannot be resolved.
    """
    if not list_checkpoints(parent):
        raise HTTPException(status_code=409, detail="no branchable checkpoint yet")
    try:
        checkpoint = resolve_checkpoint(parent, checkpoint_id)
    except CheckpointError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    parent_run_dir = parent.get("run_dir")
    if not parent_run_dir:
        raise HTTPException(status_code=409, detail="parent run_dir is not available yet")
    reference_path = Path(parent_run_dir) / "reference_input.png"
    if not reference_path.exists():
        raise HTTPException(status_code=409, detail="parent reference image is not available")
    return checkpoint, reference_path


def _prevalidate_draw_quality(reference_path: Path, quality: dict) -> None:
    """Validate ONCE up-front, before any child run is created, to avoid leaving
    orphaned/partial groups behind when an input-spec error would otherwise fire
    mid-loop inside ``_create_variant_group``.

    Builds the same probe quality dict the helper applies per child.

    Raises:
        HTTPException(422): when the probe input-spec has validation errors.
    """
    probe = {
        **quality,
        "refinement_mode": "on",
        "max_refinement_iterations": max(int(quality.get("max_refinement_iterations", 0) or 0), 1),
        "vlm_judge_enabled": 1,
    }
    probe_spec = build_input_spec(str(reference_path), quality=probe)
    errors = validate_input_spec(probe_spec)
    if errors:
        raise HTTPException(status_code=422, detail={"input_spec_errors": errors})


def _create_draw_groups(
    *,
    parent_run_id: str,
    root_run_id: str,
    checkpoint,
    checkpoint_id: str,
    reference_path: Path,
    feedback: str,
    mode: str,
    diversity: str,
    quality: dict,
    request_locks: dict,
    draw_id: str,
    card_count: int,
    on_group=None,
    constraint_spec: "HumanConstraintSpec | None" = None,
    use_preferences: bool = True,
) -> tuple[list[str], list[str]]:
    """Plan *card_count* into batches and create one variant group per batch.

    Each strategy gets request_locks merged in. ``on_group(gid, cids)`` (if given)
    is invoked after each successful group so callers can incrementally persist —
    a mid-loop failure then still leaves a consistent record of what succeeded.

    Returns ``(group_ids, card_run_ids)`` across all batches.
    """
    group_ids: list[str] = []
    card_run_ids: list[str] = []
    for batch in plan_draw_batches(card_count):
        strategies = build_variant_strategies(
            feedback=feedback, count=batch, diversity=diversity, mode=mode,
        )
        for s in strategies:
            s["locks"] = {**(s.get("locks") or {}), **request_locks}
        gid, cids = _create_variant_group(
            parent_run_id=parent_run_id,
            root_run_id=root_run_id,
            checkpoint=checkpoint,
            checkpoint_id=checkpoint_id,
            reference_path=reference_path,
            feedback=feedback,
            mode=mode,
            diversity=diversity,
            strategies=strategies,
            quality_overrides=quality,
            draw_session_id=draw_id,
            constraint_spec=constraint_spec,
            use_preferences=use_preferences,
        )
        group_ids.append(gid)
        card_run_ids.extend(cids)
        if on_group is not None:
            on_group(gid, cids)
    return group_ids, card_run_ids


@router.post("/runs/{run_id}/draw-session")
async def create_draw_session(run_id: str, payload: dict) -> dict:
    """Start a gacha-style batch draw: fan one checkpoint into N cards across
    one or more variant groups (each a different exploration strategy).

    Body: { checkpoint_id?, feedback (required), card_count?=8, diversity?="medium",
            quality?={}, constraints?={"locks":{...}}, mode?="batch_draw",
            stop_parent?=false }
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="payload must be a JSON object")

    with _run_store_lock:
        parent = _run_store.get(run_id)
    if parent is None:
        raise HTTPException(status_code=404, detail=f"run_id '{run_id}' not found")

    feedback = str(payload.get("feedback") or "")
    if not feedback.strip():
        raise HTTPException(status_code=422, detail="feedback is required for draw-session")

    card_count = int(payload.get("card_count") or 8)
    if card_count < 2 or card_count > 12:
        raise HTTPException(
            status_code=422, detail=f"card_count must be between 2 and 12, got {card_count}"
        )

    quality = payload.get("quality") or {}
    if not isinstance(quality, dict):
        raise HTTPException(status_code=422, detail="quality must be an object")

    diversity = str(payload.get("diversity") or "medium")
    mode = str(payload.get("mode") or "batch_draw")
    checkpoint_id_raw = payload.get("checkpoint_id")
    checkpoint_id = str(checkpoint_id_raw) if checkpoint_id_raw is not None else "final:selected"
    _ds_constraints_payload = payload.get("constraints")
    _ds_has_constraints = isinstance(_ds_constraints_payload, dict) and bool(_ds_constraints_payload)
    request_locks = (_ds_constraints_payload or {}).get("locks") or {}

    # Parse and validate optional structured constraints (V4.1).
    _ds_constraint_spec: HumanConstraintSpec | None = None
    if _ds_has_constraints:
        _ds_constraint_spec = parse_constraint_spec(_ds_constraints_payload)
        _ds_constraint_errors = validate_constraint_spec(_ds_constraint_spec)
        if _ds_constraint_errors:
            raise HTTPException(status_code=422, detail={"constraint_errors": _ds_constraint_errors})

    checkpoint, reference_path = _resolve_draw_checkpoint(parent, checkpoint_id)
    # Pre-validate ONCE up-front so a bad spec can't leave partial groups behind.
    _prevalidate_draw_quality(reference_path, quality)

    draw_id = "draw_" + uuid4().hex[:8]
    root_run_id = (parent.get("lineage") or {}).get("root_run_id") or run_id

    _ds_use_preferences: bool = _ds_constraint_spec.use_preferences if _ds_constraint_spec is not None else True
    group_ids, card_run_ids = _create_draw_groups(
        parent_run_id=run_id,
        root_run_id=root_run_id,
        checkpoint=checkpoint,
        checkpoint_id=checkpoint_id,
        reference_path=reference_path,
        feedback=feedback,
        mode=mode,
        diversity=diversity,
        quality=quality,
        request_locks=request_locks,
        draw_id=draw_id,
        card_count=card_count,
        constraint_spec=_ds_constraint_spec,
        use_preferences=_ds_use_preferences,
    )

    # Persist the session record + creation event best-effort (I/O must not 500).
    record = DrawSessionRecord(
        draw_id=draw_id,
        root_run_id=root_run_id,
        parent_run_id=run_id,
        source_checkpoint_id=checkpoint_id,
        feedback=feedback,
        status="running",
        requested_count=card_count,
        diversity=diversity,
        mode=mode,
        group_ids=group_ids,
        card_run_ids=card_run_ids,
        created_at=time(),
        metadata={"locks": request_locks, "quality": quality, "mode": mode},
    )
    try:
        save_session(record, root=_DRAW_SESSIONS_ROOT)
        append_session_event(
            draw_id,
            {"event": "draw_session_created", "card_run_ids": card_run_ids,
             "group_ids": group_ids, "at": time()},
            root=_DRAW_SESSIONS_ROOT,
        )
    except Exception:
        logger.warning("draw session persist failed for draw_id=%s", draw_id, exc_info=True)

    if stop_parent := bool(payload.get("stop_parent")):
        with _run_store_lock:
            stored_parent = _run_store.get(run_id)
            if stored_parent is not None and stored_parent.get("status") == "running":
                stored_parent["stop_requested"] = True

    log_event(
        logger, "draw_session_created",
        run_id=run_id, draw_id=draw_id, card_count=card_count, groups=len(group_ids),
    )

    return {
        "draw_id": draw_id,
        "status": "running",
        "parent_run_id": run_id,
        "source_checkpoint_id": checkpoint_id,
        "group_ids": group_ids,
        "card_run_ids": card_run_ids,
    }


def _fold_draw_overlay(draw_id: str) -> dict[str, dict]:
    """Fold a draw session's events into a per-run overlay map.

    Tracks (later events override earlier):
    - ``favorite`` (event "favorite" -> bool(value))
    - ``eliminated`` (event "eliminate"/"draw_card_eliminated" -> bool(value, default True);
      an "eliminate" with value False clears it)
    - ``tags`` (event "tag" -> union of any ``tags`` list on the event)
    """
    overlay: dict[str, dict] = {}
    for ev in load_session_events(draw_id, root=_DRAW_SESSIONS_ROOT):
        run_id = ev.get("run_id")
        if not run_id:
            continue
        cur = overlay.setdefault(run_id, {})
        etype = ev.get("event")
        if etype == "favorite":
            cur["favorite"] = bool(ev.get("value"))
        elif etype in ("eliminate", "draw_card_eliminated"):
            value = ev.get("value", True)
            cur["eliminated"] = bool(value)
        elif etype == "tag":
            tags = ev.get("tags") or []
            if tags:
                existing = cur.get("tags") or []
                merged = list(existing)
                # Tags are append-only in V3.5 (no untag event type); each tag event unions in new tags.
                for t in tags:
                    if t not in merged:
                        merged.append(t)
                cur["tags"] = merged
    return overlay


@router.get("/draw-sessions/{draw_id}")
async def get_draw_session(draw_id: str) -> dict:
    """Return a draw session's aggregated status + per-card details."""
    record = load_session(draw_id, root=_DRAW_SESSIONS_ROOT)
    if record is None:
        raise HTTPException(status_code=404, detail=f"draw_id '{draw_id}' not found")

    overlay = _fold_draw_overlay(draw_id)
    index_records = load_run_index(path=_RUN_INDEX_PATH)

    cards: list[dict] = []
    with _run_store_lock:
        for position, run_id in enumerate(record.card_run_ids):
            ov = overlay.get(run_id, {})
            idx_rec = index_records.get(run_id)
            stored = _run_store.get(run_id)
            if stored is not None:
                lineage = stored.get("lineage") or {}
                gid = stored.get("variant_group_id") or lineage.get("variant_group_id")
                vi = stored.get("variant_index")
                if vi is None:
                    vi = lineage.get("variant_index", position)
                status = stored.get("status") or "queued"
                label = stored.get("variant_label") or "card"
                qr = stored.get("quality_router") or {}
                final_score = qr.get("final_score") if status == "completed" else None
                current_score = qr.get("final_score") if status == "running" else None
                error = stored.get("error") or None
            elif idx_rec is not None:
                gid = idx_rec.variant_group_id
                vi = idx_rec.variant_index if idx_rec.variant_index is not None else position
                status = idx_rec.status or "queued"
                label = idx_rec.variant_label or "card"
                final_score = idx_rec.final_score if status == "completed" else None
                current_score = None
                error = None
            else:
                gid = None
                vi = position
                status = "queued"
                label = "card"
                final_score = None
                current_score = None
                error = None

            # Favorite: run-index OR overlay. Tags: overlay OR run-index OR [].
            idx_favorite = bool(idx_rec.favorite) if idx_rec is not None else False
            favorite = idx_favorite or bool(ov.get("favorite", False))
            if "tags" in ov:
                tags = ov["tags"]
            elif idx_rec is not None and idx_rec.tags:
                tags = list(idx_rec.tags)
            else:
                tags = []
            replacement_of = idx_rec.replacement_of_run_id if idx_rec is not None else None

            cards.append({
                "card_id": run_id,
                "run_id": run_id,
                "group_id": gid,
                "index": vi,
                "status": status,
                "label": label,
                "strategy_label": label,
                "final_score": final_score,
                "current_score": current_score,
                "thumbnail_url": f"/png-shader/runs/{run_id}/artifacts/selected_render",
                "feedback": record.feedback,
                "favorite": favorite,
                "eliminated": bool(ov.get("eliminated", False)),
                "tags": tags,
                "replacement_of_run_id": replacement_of,
                "can_use_for_fusion": status == "completed",
                "error": error,
            })

    statuses = [c["status"] for c in cards]
    completed_count = sum(1 for s in statuses if s == "completed")
    failed_count = sum(1 for s in statuses if s == "failed")
    running_count = sum(1 for s in statuses if s in ("running", "queued"))
    status = aggregate_draw_status(statuses)

    return {
        "draw_id": record.draw_id,
        "parent_run_id": record.parent_run_id,
        "source_checkpoint_id": record.source_checkpoint_id,
        "feedback": record.feedback,
        "status": status,
        "requested_count": record.requested_count,
        "completed_count": completed_count,
        "running_count": running_count,
        "failed_count": failed_count,
        "winner_run_id": record.winner_run_id,
        "group_ids": record.group_ids,
        "cards": cards,
    }


def _load_draw_session_or_404(draw_id: str) -> DrawSessionRecord:
    record = load_session(draw_id, root=_DRAW_SESSIONS_ROOT)
    if record is None:
        raise HTTPException(status_code=404, detail=f"draw_id '{draw_id}' not found")
    return record


def _draw_parent_or_409(record: DrawSessionRecord) -> dict:
    with _run_store_lock:
        parent = _run_store.get(record.parent_run_id)
    if parent is None:
        raise HTTPException(status_code=409, detail="parent run no longer available")
    return parent


@router.post("/draw-sessions/{draw_id}/draw-more")
async def draw_more(draw_id: str, payload: dict) -> dict:
    """Append more cards to an existing draw session, inheriting its feedback,
    locks, mode, and (by default) quality.

    Body: { card_count?=4, diversity?, quality?={} }
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="payload must be a JSON object")

    record = _load_draw_session_or_404(draw_id)
    parent = _draw_parent_or_409(record)

    card_count = int(payload.get("card_count") or 4)
    if card_count < 2 or card_count > 12:
        raise HTTPException(
            status_code=422, detail=f"card_count must be between 2 and 12, got {card_count}"
        )

    quality = payload.get("quality")
    if quality is None:
        quality = record.metadata.get("quality") or {}
    if not isinstance(quality, dict):
        raise HTTPException(status_code=422, detail="quality must be an object")

    feedback = record.feedback
    mode = record.mode
    request_locks = record.metadata.get("locks") or {}
    diversity = str(payload.get("diversity") or record.diversity)
    checkpoint_id = record.source_checkpoint_id

    # Parse and validate optional structured constraints from draw-more payload (V4.1).
    _dm_constraints_payload = payload.get("constraints")
    _dm_has_constraints = isinstance(_dm_constraints_payload, dict) and bool(_dm_constraints_payload)
    _dm_constraint_spec: HumanConstraintSpec | None = None
    if _dm_has_constraints:
        _dm_constraint_spec = parse_constraint_spec(_dm_constraints_payload)
        _dm_constraint_errors = validate_constraint_spec(_dm_constraint_spec)
        if _dm_constraint_errors:
            raise HTTPException(status_code=422, detail={"constraint_errors": _dm_constraint_errors})

    checkpoint, reference_path = _resolve_draw_checkpoint(parent, checkpoint_id)
    _prevalidate_draw_quality(reference_path, quality)

    new_group_ids: list[str] = []
    new_card_ids: list[str] = []

    def _commit(gid: str, cids: list[str]) -> None:
        # After EACH successful group, persist progress so a mid-way failure
        # still leaves a consistent record of what succeeded.
        new_group_ids.append(gid)
        new_card_ids.extend(cids)
        record.group_ids.append(gid)
        record.card_run_ids.extend(cids)
        try:
            save_session(record, root=_DRAW_SESSIONS_ROOT)
        except Exception:
            logger.warning("draw-more incremental save failed for draw_id=%s", draw_id, exc_info=True)

    _dm_use_preferences: bool = _dm_constraint_spec.use_preferences if _dm_constraint_spec is not None else True
    _create_draw_groups(
        parent_run_id=record.parent_run_id,
        root_run_id=record.root_run_id,
        checkpoint=checkpoint,
        checkpoint_id=checkpoint_id,
        reference_path=reference_path,
        feedback=feedback,
        mode=mode,
        diversity=diversity,
        quality=quality,
        request_locks=request_locks,
        draw_id=draw_id,
        card_count=card_count,
        on_group=_commit,
        constraint_spec=_dm_constraint_spec,
        use_preferences=_dm_use_preferences,
    )

    record.status = "running"
    record.updated_at = time()
    try:
        save_session(record, root=_DRAW_SESSIONS_ROOT)
        append_session_event(
            draw_id,
            {"event": "draw_more_requested", "card_count": card_count,
             "group_ids": new_group_ids, "card_run_ids": new_card_ids, "at": time()},
            root=_DRAW_SESSIONS_ROOT,
        )
    except Exception:
        logger.warning("draw-more persist failed for draw_id=%s", draw_id, exc_info=True)

    log_event(logger, "draw_more", draw_id=draw_id, card_count=card_count, groups=len(new_group_ids))

    return {
        "draw_id": draw_id,
        "status": "running",
        "group_ids": new_group_ids,
        "card_run_ids": new_card_ids,
    }


@router.post("/draw-sessions/{draw_id}/redraw")
async def redraw_card(draw_id: str, payload: dict) -> dict:
    """Replace one card with a single freshly-drawn card; auto-eliminate the
    original (it is NOT deleted) and link the replacement back to it.

    Body: { run_id (required, in record.card_run_ids), reason?, diversity?="medium" }
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="payload must be a JSON object")

    record = _load_draw_session_or_404(draw_id)

    target_run_id = payload.get("run_id")
    if not target_run_id or target_run_id not in record.card_run_ids:
        raise HTTPException(status_code=422, detail="run_id must be a card in this draw session")

    parent = _draw_parent_or_409(record)
    diversity = str(payload.get("diversity") or "medium")
    feedback = record.feedback
    mode = record.mode
    request_locks = record.metadata.get("locks") or {}
    quality = record.metadata.get("quality") or {}
    checkpoint_id = record.source_checkpoint_id

    checkpoint, reference_path = _resolve_draw_checkpoint(parent, checkpoint_id)
    _prevalidate_draw_quality(reference_path, quality)

    # Build a ONE-element strategies list.
    # build_variant_strategies requires count>=2, so we request 2 and take only the first.
    strategies = build_variant_strategies(
        feedback=feedback, count=2, diversity=diversity, mode=mode,
    )[:1]
    for s in strategies:
        s["locks"] = {**(s.get("locks") or {}), **request_locks}

    gid, cids = _create_variant_group(
        parent_run_id=record.parent_run_id,
        root_run_id=record.root_run_id,
        checkpoint=checkpoint,
        checkpoint_id=checkpoint_id,
        reference_path=reference_path,
        feedback=feedback,
        mode=mode,
        diversity=diversity,
        strategies=strategies,
        quality_overrides=quality,
        draw_session_id=draw_id,
    )
    new_run_id = cids[0]

    # Link replacement -> original (best-effort).
    _index_updated(new_run_id, {"replacement_of_run_id": target_run_id})
    with _run_store_lock:
        stored_new = _run_store.get(new_run_id)
        if stored_new is not None:
            stored_new["replacement_of_run_id"] = target_run_id

    record.group_ids.append(gid)
    record.card_run_ids.extend(cids)
    record.updated_at = time()
    try:
        save_session(record, root=_DRAW_SESSIONS_ROOT)
        append_session_event(
            draw_id,
            {"event": "draw_card_eliminated", "run_id": target_run_id,
             "value": True, "auto": True, "at": time()},
            root=_DRAW_SESSIONS_ROOT,
        )
        append_session_event(
            draw_id,
            {"event": "draw_card_redrawn", "run_id": target_run_id,
             "replacement_run_id": new_run_id, "reason": payload.get("reason"), "at": time()},
            root=_DRAW_SESSIONS_ROOT,
        )
    except Exception:
        logger.warning("redraw persist failed for draw_id=%s", draw_id, exc_info=True)

    log_event(logger, "draw_card_redrawn", draw_id=draw_id, run_id=target_run_id,
              replacement_run_id=new_run_id)

    return {
        "draw_id": draw_id,
        "group_id": gid,
        "replaced_run_id": target_run_id,
        "replacement_run_id": new_run_id,
    }


@router.post("/draw-sessions/{draw_id}/cards/{run_id}/event")
async def draw_card_event(draw_id: str, run_id: str, payload: dict) -> dict:
    """Record a per-card review event (favorite/eliminate/tag/note/use_as_*).

    Body: { event_type (required), value?, reason?, tags?=[] }

    Note: ``tag`` events are append-only in V3.5 (no untag); ``eliminate``/``favorite``
    accept value=false to clear.
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="payload must be a JSON object")

    record = _load_draw_session_or_404(draw_id)
    if run_id not in record.card_run_ids:
        raise HTTPException(status_code=422, detail="run_id must be a card in this draw session")

    event_type = payload.get("event_type")
    if event_type not in _DRAW_CARD_EVENT_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"event_type must be one of {sorted(_DRAW_CARD_EVENT_TYPES)}",
        )

    try:
        append_session_event(
            draw_id,
            {"event": event_type, "run_id": run_id, "value": payload.get("value"),
             "reason": payload.get("reason"), "tags": payload.get("tags") or [], "at": time()},
            root=_DRAW_SESSIONS_ROOT,
        )
    except Exception:
        logger.warning("append_session_event failed for draw_id=%s", draw_id, exc_info=True)

    if event_type == "favorite":
        favorite = bool(payload.get("value", True))
        try:
            update_run_metadata(run_id, {"favorite": favorite}, path=_RUN_INDEX_PATH)
        except RunIndexError:
            pass
        except Exception:
            logger.warning("draw favorite mirror failed for run_id=%s", run_id, exc_info=True)
        # Mirror to run-store for the /status/{run_id} consumer; draw-session GET
        # treats the events overlay as authoritative.
        with _run_store_lock:
            stored = _run_store.get(run_id)
            if stored is not None:
                stored["favorite"] = favorite

    log_event(logger, "draw_card_event", draw_id=draw_id, run_id=run_id, event_type=event_type)

    return {"draw_id": draw_id, "run_id": run_id, "event_type": event_type, "ok": True}


@router.post("/refine/{run_id}")
async def refine_png_shader(
    run_id: str,
    feedback: str = Form(...),
    modified_dsl_json: Optional[str] = Form(default=None),
) -> dict:
    """Run one round of human-in-the-loop LLM refinement on a previous result.

    Args:
        run_id: The run_id from a previous POST /png-shader/run call.
        feedback: Natural language feedback describing what to improve.
        modified_dsl_json: Optional JSON string with a manually edited DSL.
            When provided, the LLM uses this as the starting point instead of
            the pipeline's selected DSL.

    Returns:
        Updated result dict with refinement applied.
    """
    log_event(
        logger,
        "human_refine_request",
        run_id=run_id,
        feedback_len=len(feedback or ""),
        has_modified_dsl=bool(modified_dsl_json),
    )
    with _run_store_lock:
        stored = _run_store.get(run_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"run_id '{run_id}' not found")
    if stored.get("status") != "completed":
        raise HTTPException(status_code=409, detail="Pipeline has not completed yet")

    current_dsl = stored.get("selected_dsl")
    if modified_dsl_json:
        try:
            current_dsl = json.loads(modified_dsl_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid modified_dsl_json: {exc}") from exc
    if not current_dsl:
        raise HTTPException(status_code=409, detail="No selected DSL available for refinement")

    current_metrics = dict(stored.get("objective_metrics") or {})
    current_quality = dict(stored.get("quality_router") or {})
    preprocess = stored.get("preprocess") or {}
    canvas = (current_dsl.get("canvas") or {})
    canvas_width = int(canvas.get("width", 512))
    canvas_height = int(canvas.get("height", 512))

    try:
        from app.candidates.llm_scene import generate_llm_refinement

        with use_active_model(_get_run_model(run_id)):
            revised = generate_llm_refinement(
                preprocess=preprocess,
                current_dsl=current_dsl,
                metrics=current_metrics,
                quality_router=current_quality,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
                extra_feedback=[f"[HUMAN FEEDBACK] {feedback}"],
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Refinement failed: {exc}") from exc

    io_record = revised.pop("_io", None) if isinstance(revised, dict) else None
    parse_error = revised.pop("_parse_error", None) if isinstance(revised, dict) else None
    if revised is None or parse_error:
        log_event(
            logger,
            "human_refine_llm_parse_failed",
            level=logging.WARNING,
            run_id=run_id,
            error=parse_error or "LLM returned no usable DSL",
            attempts=len((io_record or {}).get("attempts") or []),
            raw_response_len=len(str((io_record or {}).get("raw_response") or "")),
        )
        raise HTTPException(
            status_code=502,
            detail={
                "message": parse_error or "LLM returned no usable DSL",
                "llm_io": io_record,
            },
        )

    from app.dsl.compiler import compile_dsl
    from app.dsl.validator import validate_dsl

    val = validate_dsl(revised)
    if not val.valid:
        raise HTTPException(status_code=422, detail={"validation_errors": val.errors})

    cr = compile_dsl(revised)

    result = dict(stored)
    result["selected_dsl"] = revised
    result["selected_glsl"] = cr.glsl if cr.success else stored.get("selected_glsl")
    result["human_refinement"] = {
        "feedback": feedback,
        "llm_io": io_record,
        "validation_valid": val.valid,
        "compile_success": cr.success,
        "compile_errors": cr.errors,
    }
    _store_run(run_id, result)
    log_event(
        logger,
        "human_refine_done",
        run_id=run_id,
        validation_valid=val.valid,
        compile_success=cr.success,
        compile_error_count=len(cr.errors or []),
    )
    return result


@router.post("/parameterize/{run_id}")
async def parameterize_png_shader(
    run_id: str,
    glsl: str = Form(...),
) -> dict:
    """Lift hardcoded constants in a candidate's GLSL into tunable ``#define``s.

    Stateless w.r.t. the run store: operates on the GLSL sent in the request
    body (the currently-previewed candidate) and returns the parameterized
    shader. Uses the run's selected model when known, else the default model.

    Args:
        run_id: The run_id the candidate came from (used only to pick the model).
        glsl: The shader source to parameterize.

    Returns:
        Dict with the parameterized GLSL, extracted tunables, and before/after
        parameter counts.
    """
    if not glsl or not glsl.strip():
        raise HTTPException(status_code=422, detail="glsl must be a non-empty string")

    log_event(
        logger,
        "glsl_parameterize_request",
        run_id=run_id,
        glsl_len=len(glsl),
    )

    try:
        from app.candidates.llm_scene import parameterize_glsl

        with use_active_model(_get_run_model(run_id)):
            result = parameterize_glsl(glsl)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Parameterization failed: {exc}") from exc

    io_record = result.pop("_io", None) if isinstance(result, dict) else None
    if not result or not result.get("glsl"):
        log_event(
            logger,
            "glsl_parameterize_llm_failed",
            level=logging.WARNING,
            run_id=run_id,
        )
        raise HTTPException(
            status_code=502,
            detail={
                "message": "LLM returned no usable GLSL",
                "llm_io": io_record,
            },
        )

    from app.services.shader_validator import validate_shader_static

    val = validate_shader_static(result["glsl"])
    if not val.get("valid", False):
        raise HTTPException(
            status_code=422,
            detail={
                "validation_errors": val.get("errors", []),
                "glsl": result["glsl"],
                "llm_io": io_record,
            },
        )

    log_event(
        logger,
        "glsl_parameterize_done",
        run_id=run_id,
        param_count_before=result.get("param_count_before"),
        param_count_after=result.get("param_count_after"),
        warnings=result.get("postprocess_warnings"),
    )
    return {
        "glsl": result["glsl"],
        "tunable_parameters": result.get("tunable_parameters", []),
        "param_count_before": result.get("param_count_before"),
        "param_count_after": result.get("param_count_after"),
        "warnings": result.get("postprocess_warnings", []),
        "validation_warnings": val.get("warnings", []),
        "llm_io": io_record,
    }


@router.patch("/runs/{run_id}/strategy")
async def patch_strategy(run_id: str, payload: dict) -> dict:
    """Update strategy fields for a running pipeline.

    Body shape: { "quality": { ...partial fields... } }
    Only listed fields are merged; others retain their current value.
    """
    logger.info(
        "patch strategy: run_id=%s fields=%s",
        run_id,
        list((payload or {}).get("quality", {}).keys()) if isinstance(payload, dict) else [],
    )
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="payload must be a JSON object")
    quality_patch = payload.get("quality")
    if not isinstance(quality_patch, dict) or not quality_patch:
        raise HTTPException(
            status_code=422,
            detail="payload must contain a non-empty 'quality' object",
        )

    with _run_store_lock:
        stored = _run_store.get(run_id)
        if stored is None:
            raise HTTPException(status_code=404, detail=f"run_id '{run_id}' not found")
        if stored.get("status") != "running":
            raise HTTPException(status_code=409, detail="run is not currently running")

        current = dict(stored.get("strategy") or {})
        merged = {**current, **quality_patch}

        probe_spec = build_input_spec("__probe__.png", quality=merged)
        errors = validate_input_spec(probe_spec)
        if errors:
            raise HTTPException(status_code=422, detail={"strategy_errors": errors})

        stored["strategy"] = merged
        stored["strategy_revision"] = int(stored.get("strategy_revision", 1)) + 1
        _run_store[run_id] = stored
        return {
            "strategy": merged,
            "strategy_revision": stored["strategy_revision"],
        }


@router.get("/runs/{run_id}/timeline")
async def get_timeline(run_id: str) -> dict:
    """Return the ordered timeline of checkpoints for a run.

    For runs still in the in-memory store the timeline is built live from
    the store entry. For evicted/historic runs it falls back to the run
    index + timeline.json on disk (if present). Never resolves a path
    when run_dir is None.
    """
    with _run_store_lock:
        stored = _run_store.get(run_id)
    if stored is not None:
        return {
            "run_id": run_id,
            "status": stored.get("status"),
            "timeline": build_timeline(stored, run_id=run_id),
        }

    # Not in store — look in the run index.
    records = load_run_index(path=_RUN_INDEX_PATH)
    rec = records.get(run_id)
    if rec is None:
        raise HTTPException(status_code=404, detail=f"run_id '{run_id}' not found")

    # Try to read a previously written timeline.json from disk.
    if rec.run_dir:
        timeline_path = Path(rec.run_dir) / "timeline.json"
        if timeline_path.exists():
            try:
                with timeline_path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                return {
                    "run_id": run_id,
                    "status": rec.status,
                    "timeline": data.get("timeline", []),
                }
            except Exception:
                logger.warning("timeline.json unreadable for run_id=%s", run_id, exc_info=True)

    # No run_dir, or timeline.json missing/unreadable — return empty.
    return {"run_id": run_id, "status": rec.status, "timeline": []}


@router.get("/runs/{run_id}/branches")
async def get_branches(run_id: str) -> dict:
    """Return the full branch tree rooted at the given run's root.

    Tries the run index first; falls back to a synthesised single-node
    tree for runs that exist only in the in-memory store.
    """
    records = load_run_index(path=_RUN_INDEX_PATH)
    if run_id in records:
        try:
            tree = build_branch_tree(records, run_id)
        except RunIndexError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        root_run_id = records[run_id].root_run_id
        return {"root_run_id": root_run_id, "active_run_id": run_id, "tree": tree}

    with _run_store_lock:
        stored = _run_store.get(run_id)
    if stored is not None:
        # Synthesise a single-node tree for store-only runs.
        tree = {
            "run_id": run_id,
            "root_run_id": run_id,
            "parent_run_id": None,
            "source_checkpoint_id": None,
            "source_checkpoint_label": None,
            "title": stored.get("title"),
            "mode": None,
            "feedback": None,
            "status": stored.get("status"),
            "final_score": (stored.get("quality_router") or {}).get("final_score"),
            "created_at": stored.get("submitted_at"),
            "completed_at": None,
            "favorite": False,
            "children": [],
        }
        return {"root_run_id": run_id, "active_run_id": run_id, "tree": tree}

    raise HTTPException(status_code=404, detail=f"run_id '{run_id}' not found")


@router.patch("/runs/{run_id}/metadata")
async def patch_metadata(run_id: str, payload: dict) -> dict:
    """Patch metadata-only fields (title, favorite, tags) for a run.

    Mirrors allowed fields into the in-memory store if the run is present,
    so /status and /checkpoints remain consistent with the index.
    """
    try:
        updated = update_run_metadata(run_id, payload, path=_RUN_INDEX_PATH)
    except RunIndexError as exc:
        # RunIndexError is a subclass of ValueError; catch it first for 404.
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Mirror allowed fields into the in-memory store (best-effort).
    mirror = {k: v for k, v in payload.items() if k in _METADATA_MIRROR_KEYS}
    if mirror:
        with _run_store_lock:
            stored = _run_store.get(run_id)
            if stored is not None:
                stored.update(mirror)

    return dataclasses.asdict(updated)


@router.get("/runs/{run_id}/artifacts/{artifact_id:path}")
async def get_artifact(run_id: str, artifact_id: str) -> FileResponse:
    """Serve a checkpoint artifact (shader GLSL, render PNG, llm_io JSON).

    artifact_id forms:
      - ``selected_shader``          → final:selected shader
      - ``selected_render``          → candidate:selected render PNG
      - ``checkpoint:<cp_id>:<kind>`` → arbitrary checkpoint + kind
    """
    # 1. Resolve run result + run_dir.
    with _run_store_lock:
        stored = _run_store.get(run_id)

    if stored is not None:
        result = stored
        run_dir = stored.get("run_dir")
    else:
        records = load_run_index(path=_RUN_INDEX_PATH)
        rec = records.get(run_id)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"run_id '{run_id}' not found")
        run_dir = rec.run_dir
        result = {"run_dir": run_dir}
        # Load scoreboard from disk so candidate ids can be validated.
        if run_dir:
            scoreboard_path = Path(run_dir) / "scoreboard.json"
            if scoreboard_path.exists():
                try:
                    with scoreboard_path.open("r", encoding="utf-8") as fh:
                        result["scoreboard"] = json.load(fh)
                except Exception:
                    pass

    if not run_dir:
        raise HTTPException(status_code=409, detail="run_dir not available")

    # 2. Parse artifact_id → (checkpoint_id, kind).
    if artifact_id == "selected_shader":
        checkpoint_id = "final:selected"
        kind = "shader"
    elif artifact_id == "selected_render":
        checkpoint_id = "candidate:selected"
        kind = "render"
    elif artifact_id.startswith("checkpoint:"):
        rest = artifact_id[len("checkpoint:"):]
        parts = rest.rsplit(":", 1)
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise HTTPException(
                status_code=422,
                detail=f"malformed checkpoint artifact_id: {artifact_id!r}",
            )
        checkpoint_id, kind = parts
    else:
        raise HTTPException(
            status_code=422,
            detail=f"unknown artifact id: {artifact_id!r}",
        )

    # 3. Resolve to a safe file path via the security-checked resolver.
    # Traversal note: Starlette's {artifact_id:path} converter decodes %2e%2e%2f to "../"
    # but does NOT collapse it — the decoded string reaches this handler. Traversal is blocked
    # at the APPLICATION layer: artifact_id must begin with "selected_shader"/"selected_render"/
    # "checkpoint:", and resolve_checkpoint_artifact enforces a candidate-id regex + path
    # containment + suffix allowlist (unit-tested in test_checkpoints.py).
    try:
        path = resolve_checkpoint_artifact(result, checkpoint_id, kind, run_dir=run_dir)
    except CheckpointError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"artifact not found: {path.name}",
        )

    return FileResponse(path)


@router.post("/runs/{run_id}/stop")
async def stop_run(run_id: str) -> dict:
    """Signal the running pipeline to stop after the current iteration."""
    logger.info("stop request: run_id=%s", run_id)
    with _run_store_lock:
        stored = _run_store.get(run_id)
        if stored is None:
            raise HTTPException(status_code=404, detail=f"run_id '{run_id}' not found")
        if stored.get("status") not in ("running", "queued"):
            raise HTTPException(status_code=409, detail="run is not currently running")
        stored["stop_requested"] = True
        _run_store[run_id] = stored
        return {"stopping": True}


# ---------------------------------------------------------------------------
# V3.1: Variant-group read/aggregate + action endpoints
# ---------------------------------------------------------------------------

_STATUS_RANK = {"completed": 0, "running": 1, "queued": 2, "cancelled": 3, "failed": 4}


def _get_group_or_404(group_id: str):
    record = load_group(group_id, root=_VARIANT_GROUPS_ROOT)
    if record is None:
        raise HTTPException(status_code=404, detail=f"group_id '{group_id}' not found")
    return record


@router.get("/variant-groups/{group_id}")
async def get_variant_group(group_id: str) -> dict:
    """Return aggregated status + sorted variants for a variant group."""
    record = _get_group_or_404(group_id)

    # Build per-child variant dicts.
    index_records = load_run_index(path=_RUN_INDEX_PATH)
    variants = []
    with _run_store_lock:
        for position, run_id in enumerate(record.child_run_ids):
            stored = _run_store.get(run_id)
            if stored is not None:
                # Prefer store entry.
                vi = stored.get("variant_index")
                if vi is None:
                    lineage = stored.get("lineage") or {}
                    vi = lineage.get("variant_index", position)
                status = stored.get("status") or "queued"
                qr = stored.get("quality_router") or {}
                final_score = qr.get("final_score") if status == "completed" else None
                current_score = qr.get("final_score") if status == "running" else None
                rh = stored.get("refinement_history") or []
                changes_summary = rh[-1].get("changes_summary") if rh else None
                # Favorite: run index has priority, then store, then default False.
                idx_rec = index_records.get(run_id)
                if idx_rec is not None:
                    favorite = bool(idx_rec.favorite)
                else:
                    favorite = bool(stored.get("favorite", False))
                variant = {
                    "run_id": run_id,
                    "variant_index": vi,
                    "label": stored.get("variant_label") or "variant",
                    "status": status,
                    "final_score": final_score,
                    "current_score": current_score,
                    "selected_glsl": stored.get("selected_glsl") or None,
                    "thumbnail_url": f"/png-shader/runs/{run_id}/artifacts/selected_render",
                    "changes_summary": changes_summary,
                    "error": stored.get("error") or None,
                    "favorite": favorite,
                }
            else:
                # Evicted — best-effort from run index.
                idx_rec = index_records.get(run_id)
                if idx_rec is not None:
                    status = idx_rec.status or "queued"
                    variant = {
                        "run_id": run_id,
                        "variant_index": idx_rec.variant_index if idx_rec.variant_index is not None else position,
                        "label": idx_rec.variant_label or "variant",
                        "status": status,
                        "final_score": idx_rec.final_score if status == "completed" else None,
                        "current_score": None,
                        "selected_glsl": None,
                        "thumbnail_url": f"/png-shader/runs/{run_id}/artifacts/selected_render",
                        "changes_summary": None,
                        "error": None,
                        "favorite": bool(idx_rec.favorite),
                    }
                else:
                    # Completely unknown — minimal stub.
                    variant = {
                        "run_id": run_id,
                        "variant_index": position,
                        "label": "variant",
                        "status": "queued",
                        "final_score": None,
                        "current_score": None,
                        "selected_glsl": None,
                        "thumbnail_url": f"/png-shader/runs/{run_id}/artifacts/selected_render",
                        "changes_summary": None,
                        "error": None,
                        "favorite": False,
                    }
            variants.append(variant)

    # Sort: winner first, then status rank, then final_score desc (None last), then variant_index.
    winner = record.winner_run_id

    def _sort_key(v: dict):
        is_winner = 0 if v["run_id"] == winner else 1
        rank = _STATUS_RANK.get(v["status"], 99)
        score = v["final_score"]
        score_key = (0, -(score)) if score is not None else (1, 0.0)
        return (is_winner, rank, score_key, v["variant_index"])

    variants.sort(key=_sort_key)

    status = aggregate_group_status([v["status"] for v in variants])

    return {
        "group_id": record.group_id,
        "parent_run_id": record.parent_run_id,
        "source_checkpoint_id": record.source_checkpoint_id,
        "feedback": record.feedback,
        "status": status,
        "winner_run_id": record.winner_run_id,
        "variants": variants,
    }


@router.post("/variant-groups/{group_id}/stop")
async def stop_variant_group(group_id: str) -> dict:
    """Signal all queued/running children of a variant group to stop."""
    record = _get_group_or_404(group_id)

    with _run_store_lock:
        for run_id in record.child_run_ids:
            stored = _run_store.get(run_id)
            if stored is not None and stored.get("status") in ("queued", "running"):
                stored["stop_requested"] = True

    try:
        append_group_event(
            group_id,
            {"event": "stopped", "at": time()},
            root=_VARIANT_GROUPS_ROOT,
        )
    except Exception:
        logger.warning("append_group_event failed for group_id=%s", group_id, exc_info=True)

    return {"stopping": True, "group_id": group_id}


@router.post("/variant-groups/{group_id}/winner")
async def set_variant_winner(group_id: str, payload: dict) -> dict:
    """Mark one variant child as the winner of its group."""
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="payload must be a JSON object")

    record = _get_group_or_404(group_id)

    winner_run_id = payload.get("winner_run_id")
    if not winner_run_id or winner_run_id not in record.child_run_ids:
        raise HTTPException(
            status_code=422,
            detail="winner_run_id is required and must be a member of this group's child_run_ids",
        )

    record.winner_run_id = winner_run_id
    try:
        save_group(record, root=_VARIANT_GROUPS_ROOT)
    except Exception as exc:
        logger.error("save_group failed for group_id=%s", group_id, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to persist winner") from exc

    # Mark the winner favorite in the run index (best-effort).
    try:
        update_run_metadata(winner_run_id, {"favorite": True}, path=_RUN_INDEX_PATH)
    except RunIndexError:
        pass  # Run not in the index yet — silently ignore.
    except Exception:
        logger.warning(
            "update_run_metadata(favorite) failed for run_id=%s", winner_run_id, exc_info=True
        )

    # Mirror into the in-memory store (best-effort).
    with _run_store_lock:
        stored = _run_store.get(winner_run_id)
        if stored is not None:
            stored["favorite"] = True

    try:
        append_group_event(
            group_id,
            {
                "event": "winner",
                "run_id": winner_run_id,
                "reason": payload.get("reason"),
                "at": time(),
            },
            root=_VARIANT_GROUPS_ROOT,
        )
    except Exception:
        logger.warning("append_group_event failed for group_id=%s", group_id, exc_info=True)

    log_event(logger, "variant_winner_selected", group_id=group_id, run_id=winner_run_id)

    # Mirror into preference events (best-effort, additive — never 500).
    try:
        append_preference_event(
            PreferenceEvent(
                event_id="pref_" + uuid4().hex[:8],
                event_type="winner_selected",
                timestamp=time(),
                group_id=group_id,
                winner_run_id=winner_run_id,
                reason=payload.get("reason"),
                context={"source": "variant_winner"},
            ),
            root=_PREFERENCES_ROOT,
        )
    except Exception:
        logger.warning(
            "append_preference_event(winner_selected) failed for group_id=%s", group_id, exc_info=True
        )

    return {"group_id": group_id, "winner_run_id": winner_run_id}


@router.post("/variant-groups/{group_id}/ratings")
async def rate_variant(group_id: str, payload: dict) -> dict:
    """Append a rating event to a variant group."""
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="payload must be a JSON object")

    record = _get_group_or_404(group_id)

    run_id = payload.get("run_id")
    if not run_id or run_id not in record.child_run_ids:
        raise HTTPException(
            status_code=422,
            detail="run_id is required and must be a member of this group's child_run_ids",
        )

    rating = payload.get("rating")
    if rating not in (-1, 0, 1):
        raise HTTPException(
            status_code=422,
            detail="rating must be an integer in {-1, 0, 1}",
        )

    try:
        append_group_event(
            group_id,
            {
                "event": "rating",
                "run_id": run_id,
                "rating": rating,
                "reason": payload.get("reason"),
                "tags": payload.get("tags") or [],
                "at": time(),
            },
            root=_VARIANT_GROUPS_ROOT,
        )
    except Exception:
        logger.warning("append_group_event failed for group_id=%s", group_id, exc_info=True)

    log_event(logger, "variant_rated", group_id=group_id, run_id=run_id, rating=rating)

    # Mirror into preference events (best-effort, additive — never 500).
    try:
        append_preference_event(
            PreferenceEvent(
                event_id="pref_" + uuid4().hex[:8],
                event_type="variant_rated",
                timestamp=time(),
                run_id=run_id,
                group_id=group_id,
                rating=rating,
                reason=payload.get("reason"),
                tags=payload.get("tags") or [],
                context={"source": "variant_rating"},
            ),
            root=_PREFERENCES_ROOT,
        )
    except Exception:
        logger.warning(
            "append_preference_event(variant_rated) failed for group_id=%s", group_id, exc_info=True
        )

    return {"group_id": group_id, "run_id": run_id, "rating": rating}


# ---------------------------------------------------------------------------
# V4.2 region-mask endpoint
# ---------------------------------------------------------------------------

@router.post("/runs/{run_id}/region-mask")
async def region_mask(run_id: str, payload: dict) -> dict:
    """Validate, persist, and compute metrics for a normalized region rect.

    Body: { region_id, geometry_type?, geometry, label?, mode?, instruction?, strength? }
    Returns: { region_id, mask_artifact_id, mask_url, geometry, metrics }
    """
    # 1. Payload must be a dict (FastAPI handles JSON body; guard against non-dict).
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="payload must be a JSON object")

    # 2. Look up run.
    with _run_store_lock:
        stored = _run_store.get(run_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"run_id '{run_id}' not found")

    # 3. Build RegionConstraint — region_id is required and must pass the allowlist.
    region_id = payload.get("region_id")
    if not region_id or not str(region_id).strip():
        raise HTTPException(status_code=422, detail="region_id is required and must be non-blank")
    region_id = str(region_id).strip()
    if not _REGION_ID_RE.match(region_id):
        raise HTTPException(
            status_code=422,
            detail=f"region_id {region_id!r} contains disallowed characters",
        )

    if "strength" in payload:
        try:
            strength = float(payload["strength"])
        except (ValueError, TypeError):
            raise HTTPException(status_code=422, detail="strength must be a number")
    else:
        strength = 0.5

    region = RegionConstraint(
        id=region_id,
        label=str(payload.get("label") or region_id),
        mode=str(payload.get("mode") or "modify"),
        instruction=str(payload.get("instruction") or ""),
        geometry_type=str(payload.get("geometry_type") or "rect"),
        geometry=payload.get("geometry") if isinstance(payload.get("geometry"), dict) else {},
        strength=strength,
    )

    # 4. Validate via HumanConstraintSpec wrapper.
    spec = HumanConstraintSpec(regions=[region])
    errors = validate_constraint_spec(spec)
    if errors:
        raise HTTPException(status_code=422, detail={"region_errors": errors})

    run_dir = stored.get("run_dir")

    # 5. Persist geometry best-effort.
    region_dict = {
        "id": region.id,
        "label": region.label,
        "mode": region.mode,
        "instruction": region.instruction,
        "geometry_type": region.geometry_type,
        "geometry": region.geometry,
        "strength": region.strength,
    }
    if run_dir:
        try:
            mask_path = Path(run_dir) / "region_masks" / f"{region_id}.json"
            save_json(mask_path, region_dict)
        except Exception:
            logger.warning(
                "region_mask: failed to persist geometry for region_id=%s run_id=%s",
                region_id, run_id, exc_info=True,
            )

    # 6. Compute region metrics if both reference and selected render are available.
    metrics: dict | None = None
    if run_dir:
        try:
            reference_path = Path(run_dir) / "reference_input.png"
            # Resolve selected render: reuse the same logic as GET /artifacts/selected_render.
            # That maps to candidate:selected + kind=render → <run_dir>/candidates/<selected_id>_render.png
            selected_cand = _selected_candidate(stored)
            render_path: Path | None = None
            if selected_cand is not None:
                sid = selected_cand.get("id")
                if sid:
                    render_path = Path(run_dir) / "candidates" / f"{sid}_render.png"

            if (
                reference_path.exists()
                and render_path is not None
                and render_path.exists()
            ):
                metrics = compute_region_metrics(reference_path, render_path, [region])
                # Persist region metrics best-effort.
                try:
                    metrics_path = Path(run_dir) / "region_metrics" / f"{region_id}.json"
                    save_json(metrics_path, metrics)
                except Exception:
                    logger.warning(
                        "region_mask: failed to persist metrics for region_id=%s run_id=%s",
                        region_id, run_id, exc_info=True,
                    )
        except Exception:
            logger.warning(
                "region_mask: metrics computation failed for region_id=%s run_id=%s",
                region_id, run_id, exc_info=True,
            )
            metrics = None

    return {
        "region_id": region_id,
        "mask_artifact_id": f"mask:{region_id}",
        "mask_url": f"/png-shader/runs/{run_id}/artifacts/mask:{region_id}",
        "geometry": region.geometry,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# V4.3 Preference CRUD endpoints
# ---------------------------------------------------------------------------


@router.get("/preferences/profile")
async def get_preference_profile() -> dict:
    """Return the current preference profile (default if none saved)."""
    try:
        return load_profile(root=_PREFERENCES_ROOT)
    except Exception:
        logger.warning("load_profile failed, returning default", exc_info=True)
        return default_profile()


@router.patch("/preferences/profile")
async def patch_preference_profile(payload: dict) -> dict:
    """Patch editable fields of the preference profile.

    Only editable keys (enabled, default_locks, positive_preferences,
    negative_preferences, score_drop_tolerance_hint) are accepted.
    Disallowed keys → 422. Payload must be a dict → 422.
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="payload must be a JSON object")
    try:
        return patch_profile(payload, updated_at=time(), root=_PREFERENCES_ROOT)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        logger.warning("patch_profile failed", exc_info=True)
        return load_profile(root=_PREFERENCES_ROOT)


@router.post("/preferences/events")
async def post_preference_event(payload: dict) -> dict:
    """Append a manual preference event.

    Accepts any event_type; unknown types are defaulted to "manual_note".
    Returns {"event_id": ..., "ok": True}.
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="payload must be a JSON object")

    _ALLOWED_EVENT_TYPES = frozenset({
        "winner_selected", "variant_rated", "branch_accepted", "manual_note"
    })
    event_type = payload.get("event_type") or "manual_note"
    if event_type not in _ALLOWED_EVENT_TYPES:
        event_type = "manual_note"

    # Coerce rating to int or None.
    raw_rating = payload.get("rating")
    rating: int | None = None
    if raw_rating is not None:
        try:
            rating = int(raw_rating)
        except (TypeError, ValueError):
            rating = None

    event_id = "pref_" + uuid4().hex[:8]
    event = PreferenceEvent(
        event_id=event_id,
        event_type=event_type,
        timestamp=time(),
        run_id=payload.get("run_id"),
        group_id=payload.get("group_id"),
        feedback=payload.get("feedback"),
        winner_run_id=payload.get("winner_run_id"),
        loser_run_ids=list(payload.get("loser_run_ids") or []),
        rating=rating,
        reason=payload.get("reason"),
        tags=list(payload.get("tags") or []),
        context=dict(payload.get("context") or {}),
    )
    try:
        append_preference_event(event, root=_PREFERENCES_ROOT)
    except Exception:
        logger.warning("append_preference_event failed", exc_info=True)

    return {"event_id": event_id, "ok": True}


@router.post("/preferences/rebuild")
async def rebuild_preference_profile() -> dict:
    """Rebuild the preference profile from all stored events and save it."""
    try:
        events = load_preference_events(root=_PREFERENCES_ROOT)
        base = load_profile(root=_PREFERENCES_ROOT)
        profile = rebuild_profile(events, updated_at=time(), base_profile=base)
        save_profile(profile, root=_PREFERENCES_ROOT)
        return profile
    except Exception:
        logger.warning("rebuild_preference_profile failed", exc_info=True)
        return load_profile(root=_PREFERENCES_ROOT)


@router.post("/preferences/clear")
async def clear_preference_data() -> dict:
    """Clear all preference events and reset the profile to defaults."""
    try:
        clear_preferences(root=_PREFERENCES_ROOT)
    except Exception:
        logger.warning("clear_preferences failed", exc_info=True)
    return {"ok": True}
