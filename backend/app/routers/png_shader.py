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
    build_timeline,
    checkpoint_metadata,
    list_checkpoints,
    resolve_checkpoint,
    resolve_checkpoint_artifact,
    save_timeline,
)
from app.pipeline.graph import run_png_shader_pipeline
from app.pipeline.human_feedback import (
    MODES,
    FeedbackValidationError,
    build_human_feedback_notes,
    validate_feedback,
)
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
) -> None:
    """Run the PNG shader pipeline after the submit request has returned.

    ``upload_dir`` is the temp dir of an uploaded image and is removed on
    completion. Branch runs reuse the parent's reference image and pass
    ``upload_dir=None`` so the parent's ``run_dir`` is never deleted.
    ``pipeline_extra`` carries human-in-loop kwargs (human_feedback_notes,
    directed_acceptance, force_first_refinement_iteration, lineage,
    extra_artifacts) forwarded to ``run_png_shader_pipeline``.
    """
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
) -> None:
    """Register the run's model and launch the background pipeline thread.

    Shared by ``/run`` (uploaded image, ``upload_dir`` set) and
    ``/branch-refine`` (parent reference image, ``upload_dir=None``).
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


def _load_parent_input_spec(run_dir: "str | None") -> "dict | None":
    """Read a run's persisted input_spec.json, or None if unavailable.

    Used as the fallback source for branch inheritance when the parent's
    in-memory store entry no longer carries ``input_spec`` (e.g. a branch of a
    branch whose record predates the field).
    """
    if not run_dir:
        return None
    spec_path = Path(run_dir) / "input_spec.json"
    if not spec_path.exists():
        return None
    try:
        data = json.loads(spec_path.read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


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

    # Inherit the parent's input spec (target/candidates/quality) so the branch
    # stays comparable to its parent. A default-built spec would reset target
    # resolution, max shader chars, candidate config, etc. (P2). The parent
    # store is the primary source; run_dir/input_spec.json is the fallback.
    parent_spec = parent.get("input_spec")
    if not isinstance(parent_spec, dict):
        parent_spec = _load_parent_input_spec(parent_run_dir) or {}
    parent_target = parent_spec.get("target")
    parent_target = dict(parent_target) if isinstance(parent_target, dict) else {}
    parent_candidates = parent_spec.get("candidates")
    parent_candidates = dict(parent_candidates) if isinstance(parent_candidates, dict) else {}
    parent_quality = parent_spec.get("quality")
    parent_quality = dict(parent_quality) if isinstance(parent_quality, dict) else {}

    # Child quality: inherit the parent's, apply the user's overrides, then force
    # the directed closed loop on with at least one iteration.
    quality = {**parent_quality, **quality_overrides}
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

    child_input_spec = build_input_spec(
        str(reference_path),
        target=parent_target,
        candidates=parent_candidates,
        quality=quality,
    )
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
        "input_spec": child_input_spec,
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
        if stored.get("status") != "running":
            raise HTTPException(status_code=409, detail="run is not currently running")
        stored["stop_requested"] = True
        _run_store[run_id] = stored
        return {"stopping": True}
