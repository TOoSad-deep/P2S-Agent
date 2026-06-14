"""FastAPI router for PNG-to-Shader pipeline.

Endpoints:
  POST /png-shader/run             — submit image, get run_id + scoreboard
  GET  /png-shader/status/{run_id} — get cached result (in-memory store)
  POST /png-shader/refine/{run_id} — human-in-the-loop refinement
"""

from __future__ import annotations

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

from app.pipeline.graph import run_png_shader_pipeline
from app.pipeline.input_spec import build_input_spec, validate_input_spec
from app.services.langsmith_tracing import trace_context
from app.services.logging_config import attach_run_log

logger = logging.getLogger(__name__)

_run_store: dict[str, dict] = {}
_run_store_lock = threading.Lock()
_MAX_STORE_SIZE = 100

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
    upload_dir: Path,
    pipeline_input_spec: Optional[dict],
    seed_glsl: Optional[str],
    trace_input: dict,
    trace_metadata: dict,
) -> None:
    """Run the PNG shader pipeline after the submit request has returned."""
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

    def _publish_partial(partial: dict) -> None:
        _publish_partial_to_store(run_id, partial)

    run_log_path = Path("artifacts") / run_id / "run.log"
    with attach_run_log(run_id=run_id, log_file=run_log_path):
        logger.info("worker start: run_id=%s image=%s", run_id, image_path.name)
        try:
            with trace_context(
                "PNG Shader Pipeline",
                inputs=trace_input,
                metadata=trace_metadata,
                tags=["png-shader", run_id],
            ) as run_tree:
                pipeline_result = run_png_shader_pipeline(
                    image_path,
                    pipeline_input_spec,
                    run_id=run_id,
                    seed_glsl=seed_glsl,
                    progress_callback=_progress,
                    strategy_reader=_strategy_reader,
                    publish_partial=_publish_partial,
                )
                result_with_status = {**pipeline_result, "status": "completed", "current_phase": "done"}
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
                logger.info(
                    "worker done: run_id=%s selected=%s score=%s",
                    run_id,
                    pipeline_result.get("scoreboard", {}).get("selected_id"),
                    pipeline_result.get("quality_router", {}).get("final_score"),
                )
        except Exception as exc:
            logger.exception("worker failed: run_id=%s", run_id)
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
            shutil.rmtree(upload_dir, ignore_errors=True)


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
    logger.info(
        "request received: POST /png-shader/run run_id=%s filename=%s content_type=%s",
        run_id,
        image.filename,
        image.content_type,
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
        threading.Thread(
            target=_run_png_shader_background,
            kwargs={
                "run_id": run_id,
                "image_path": image_path,
                "upload_dir": upload_dir,
                "pipeline_input_spec": pipeline_input_spec,
                "seed_glsl": seed_glsl,
                "trace_input": trace_input,
                "trace_metadata": trace_metadata,
            },
            daemon=True,
        ).start()
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
    logger.info("refine request: run_id=%s feedback_len=%d", run_id, len(feedback or ""))
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

    if revised is None:
        raise HTTPException(status_code=502, detail="LLM returned no usable DSL")

    io_record = revised.pop("_io", None)

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
    return result


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
