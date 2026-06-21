"""PNG-shader router: initial run + lifecycle + artifacts (core domain).

Handlers for /run, /status, /checkpoints, /refine, /parameterize, /strategy,
/timeline, /branches, /metadata, /artifacts, /stop. Split out of the historical
app/routers/png_shader.py aggregator; behaviour is unchanged.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
import tempfile
from pathlib import Path
from time import time
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from p2s_agent.config import use_active_model
from p2s_agent.core.llm.model_resolver import ModelResolutionError, resolve_model_config
from p2s_agent.orchestration.checkpoints import (
    CheckpointError,
    build_timeline,
    list_checkpoints,
    resolve_checkpoint_artifact,
)
from p2s_agent.core.pipeline.input_spec import build_input_spec, validate_input_spec
from p2s_agent.orchestration.run_index import (
    RunIndexError,
    RunLineageRecord,
    build_branch_tree,
    load_run,
    load_run_family,
    update_run_metadata,
)
from p2s_agent.core.logging_config import log_event, logging_context
from app.api.guards import _check_content_length, _guard_upload
from app.api.routers._shared import (
    _enforce_text_cap,
    _MAX_FEEDBACK_CHARS,
    _MAX_INPUT_SPEC_CHARS,
    _MAX_MODIFIED_DSL_CHARS,
    _MAX_SEED_GLSL_CHARS,
    _REGION_ID_RE,
)

from p2s_agent import store
from p2s_agent.workers import WorkerCapacityError, _start_pipeline_worker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/png-shader", tags=["png-shader"])


@router.post("/run", dependencies=[Depends(_check_content_length)])
async def run_png_shader(
    request: Request,
    image: UploadFile = File(...),
    input_spec_json: Optional[str] = Form(default=None),
    seed_glsl: Optional[str] = Form(default=None),
) -> dict:
    """Submit an image and run the PNG-to-Shader pipeline in the background."""
    # Item 3 — cap free-text/code inputs. (Item 1's Content-Length guard runs as
    # a route dependency, before the multipart body is parsed.)
    _enforce_text_cap(input_spec_json, _MAX_INPUT_SPEC_CHARS, field="input_spec_json")
    _enforce_text_cap(seed_glsl, _MAX_SEED_GLSL_CHARS, field="seed_glsl")

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
        except HTTPException:
            raise
        except Exception as exc:
            shutil.rmtree(upload_dir, ignore_errors=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read uploaded image: {exc}",
            ) from exc

        # Item 1 — content-type allowlist + size cap + real-image verification.
        try:
            _guard_upload(request, image, contents)
        except HTTPException:
            shutil.rmtree(upload_dir, ignore_errors=True)
            raise

        try:
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
            from p2s_agent.core.pipeline.input_spec import build_input_spec as _default_spec
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
        store._store_run(run_id, initial_result)
        store._index_created(RunLineageRecord(
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
    except WorkerCapacityError as exc:
        # No worker slot was available → roll back the registered run and tell
        # the client to retry later (backpressure, not a server error).
        shutil.rmtree(upload_dir, ignore_errors=True)
        store._drop_run(run_id)
        raise HTTPException(
            status_code=429,
            detail=f"{exc}. Retry-After: a few seconds.",
        ) from exc
    except Exception as exc:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline submit error: {exc}",
        ) from exc


@router.get("/status/{run_id}")
async def get_status(run_id: str) -> dict:
    """Retrieve a cached pipeline result by run_id.

    Returns a deep copy taken under the store lock so a concurrent worker
    mutation can never trigger "dictionary changed size during iteration" while
    FastAPI serialises the response, and the caller never holds a live ref to
    the mutable store entry (Bug 4)."""
    result = store._snapshot_run(run_id)
    if result is None:
        # Run fell out of the in-memory store (LRU eviction at capacity, or a
        # process restart / uvicorn --reload). Rebuild it from the persisted
        # run_dir so the UI can still load the result instead of going blank.
        result = store.rehydrate_run(run_id)
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
    stored = store._snapshot_run(run_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"run_id '{run_id}' not found")
    return {
        "run_id": run_id,
        "status": stored.get("status"),
        "checkpoints": list_checkpoints(stored),
    }

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
    # Item 3 — cap free-text / code inputs before doing any work.
    _enforce_text_cap(feedback, _MAX_FEEDBACK_CHARS, field="feedback")
    _enforce_text_cap(modified_dsl_json, _MAX_MODIFIED_DSL_CHARS, field="modified_dsl_json")

    log_event(
        logger,
        "human_refine_request",
        run_id=run_id,
        feedback_len=len(feedback or ""),
        has_modified_dsl=bool(modified_dsl_json),
    )
    with store._run_store_lock:
        stored = store._run_store.get(run_id)
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
        from p2s_agent.core.candidates.llm_scene import generate_llm_refinement

        with use_active_model(store._get_run_model(run_id)):
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

    from p2s_agent.core.dsl.compiler import compile_dsl
    from p2s_agent.core.dsl.validator import validate_dsl

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
    store._store_run(run_id, result)
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
        from p2s_agent.core.candidates.llm_scene import parameterize_glsl

        with use_active_model(store._get_run_model(run_id)):
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

    from p2s_agent.core.render.shader_validator import validate_shader_static

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

    with store._run_store_lock:
        stored = store._run_store.get(run_id)
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
        store._store_run_locked(run_id, stored)
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
    stored = store._snapshot_run(run_id)
    if stored is not None:
        return {
            "run_id": run_id,
            "status": stored.get("status"),
            "timeline": build_timeline(stored, run_id=run_id),
        }

    # Not in store — look in the run index.
    rec = load_run(run_id, path=store._RUN_INDEX_PATH)  # targeted single-run read
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
    records = load_run_family(run_id, path=store._RUN_INDEX_PATH)  # by-root, not whole index
    if run_id in records:
        try:
            tree = build_branch_tree(records, run_id)
        except RunIndexError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        root_run_id = records[run_id].root_run_id
        return {"root_run_id": root_run_id, "active_run_id": run_id, "tree": tree}

    with store._run_store_lock:
        stored = store._run_store.get(run_id)
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
        updated = update_run_metadata(run_id, payload, path=store._RUN_INDEX_PATH)
    except RunIndexError as exc:
        # RunIndexError is a subclass of ValueError; catch it first for 404.
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Mirror allowed fields into the in-memory store (best-effort).
    mirror = {k: v for k, v in payload.items() if k in store._METADATA_MIRROR_KEYS}
    if mirror:
        with store._run_store_lock:
            stored = store._run_store.get(run_id)
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
      - ``mask:<region_id>``          → region_masks/<region_id>.json
        (matches what the ``region-mask`` endpoint advertises + persists)
    """
    # 1. Resolve run result + run_dir.
    with store._run_store_lock:
        stored = store._run_store.get(run_id)

    if stored is not None:
        result = stored
        run_dir = stored.get("run_dir")
    else:
        rec = load_run(run_id, path=store._RUN_INDEX_PATH)  # targeted single-run read
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

    # 1b. region-mask artifact (``mask:<region_id>``). The region-mask endpoint
    # advertises this id/url and writes ``region_masks/<region_id>.json``; serve
    # that exact file. id-safety mirrors the fusion artifact endpoint: the
    # region_id must match ``_REGION_ID_RE`` (blocks ``..``/``/``), and the
    # resolved path must stay contained in ``<run_dir>/region_masks``.
    if artifact_id.startswith("mask:"):
        region_id = artifact_id[len("mask:"):]
        if not region_id or not _REGION_ID_RE.match(region_id):
            raise HTTPException(
                status_code=422,
                detail=f"region_id {region_id!r} contains disallowed characters",
            )
        masks_dir = Path(run_dir) / "region_masks"
        target = masks_dir / f"{region_id}.json"
        try:
            resolved = target.resolve()
            resolved.relative_to(masks_dir.resolve())
        except (ValueError, OSError) as exc:
            raise HTTPException(status_code=422, detail="invalid artifact path") from exc
        if not resolved.exists():
            raise HTTPException(
                status_code=404,
                detail=f"artifact not found: {target.name}",
            )
        return FileResponse(resolved)

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
    with store._run_store_lock:
        stored = store._run_store.get(run_id)
        if stored is None:
            raise HTTPException(status_code=404, detail=f"run_id '{run_id}' not found")
        if stored.get("status") not in ("running", "queued"):
            raise HTTPException(status_code=409, detail="run is not currently running")
        stored["stop_requested"] = True
        store._store_run_locked(run_id, stored)
        return {"stopping": True}


# ---------------------------------------------------------------------------
# V3.1: Variant-group read/aggregate + action endpoints
# ---------------------------------------------------------------------------

