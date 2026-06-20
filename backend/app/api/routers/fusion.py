"""PNG-shader router: region-mask + Local Fusion endpoints (V4.2 / V4.5)."""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from time import time
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.pipeline.checkpoints import _selected_candidate
from app.pipeline.human_constraints import (
    HumanConstraintSpec,
    RegionConstraint,
    validate_constraint_spec,
)
from app.pipeline.region_metrics import compute_region_metrics
from app.pipeline.fusion_plans import (
    build_fusion_notes,
    load_plan,
    parse_fusion_plan,
    plan_to_dict,
    validate_fusion_plan,
)
from app.pipeline.image_composite import build_composite_target
from app.pipeline.artifacts import save_json
from app.pipeline.input_spec import build_input_spec, validate_input_spec
from app.pipeline.run_index import RunLineageRecord
from app.services.logging_config import log_event
from app.api.routers._shared import validate_safe_id, _coerce_int, _REGION_ID_RE

from p2s_agent import store
from p2s_agent.orchestration import sessions
from p2s_agent.workers import WorkerCapacityError, _start_pipeline_worker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/png-shader", tags=["png-shader"])


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
    with store._run_store_lock:
        stored = store._run_store.get(run_id)
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
# V4.5 Local Fusion endpoints
# ---------------------------------------------------------------------------


@router.post("/fusions")
async def create_fusion(payload: dict) -> dict:
    """Create a draft fusion plan from a base run + one or more source runs.

    Body: { base_run_id (required), draw_session_id?, feedback?, source_run_ids?,
            regions:[{id,label,source_run_id,instruction,geometry_type,geometry,
            strength,blend_mode,feather}] }
    The base/source runs are visual references; only the base contributes seed GLSL.
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="payload must be a JSON object")

    base_run_id = str(payload.get("base_run_id") or "")
    if not base_run_id:
        raise HTTPException(status_code=422, detail="base_run_id is required")

    base = store._touch_run(base_run_id)
    if base is None:
        raise HTTPException(status_code=404, detail=f"run_id '{base_run_id}' not found")

    base_run_dir = base.get("run_dir")
    if not base_run_dir:
        raise HTTPException(status_code=409, detail="base run_dir is not available yet")

    base_selected_glsl = base.get("selected_glsl")
    if not base_selected_glsl:
        raise HTTPException(
            status_code=422,
            detail="base run has no selected GLSL — cannot be a fusion base",
        )

    base_render = sessions._resolve_run_render(base, base_run_dir)
    if base_render is None:
        raise HTTPException(status_code=422, detail="base render not available")

    # Collect the source run-id set: payload source_run_ids ∪ every region.source_run_id.
    raw_sources = payload.get("source_run_ids")
    source_ids: list[str] = []
    seen: set[str] = set()

    def _add_source(sid: "str | None") -> None:
        if sid and sid not in seen:
            seen.add(sid)
            source_ids.append(sid)

    if isinstance(raw_sources, list):
        for s in raw_sources:
            _add_source(str(s) if s is not None else None)
    raw_regions = payload.get("regions")
    if isinstance(raw_regions, list):
        for r in raw_regions:
            if isinstance(r, dict):
                _add_source(str(r.get("source_run_id") or "") or None)

    # Every source run must resolve a render (sources are visual only — no GLSL needed).
    for sid in source_ids:
        with store._run_store_lock:
            src = store._run_store.get(sid)
        if src is None:
            raise HTTPException(
                status_code=422,
                detail=f"source run '{sid}' not found",
            )
        if sessions._resolve_run_render(src, src.get("run_dir")) is None:
            raise HTTPException(
                status_code=422,
                detail=f"source run '{sid}' render not available",
            )

    fusion_id = "fusion_" + uuid4().hex[:8]
    base_lineage = base.get("lineage") or {}
    root_run_id = base_lineage.get("root_run_id") or base_run_id

    record = parse_fusion_plan(
        payload,
        fusion_id=fusion_id,
        root_run_id=root_run_id,
        parent_run_id=base_run_id,
        created_at=time(),
    )
    # Force the resolved source set so validate_fusion_plan's membership check
    # passes even when the payload only listed sources via regions.
    record.source_run_ids = list(source_ids)

    errors = validate_fusion_plan(record)
    if errors:
        raise HTTPException(status_code=422, detail={"fusion_errors": errors})

    sessions._save_plan_best_effort(record)
    sessions._append_fusion_event_best_effort(fusion_id, {
        "event": "fusion_plan_created",
        "base_run_id": base_run_id,
        "source_run_ids": source_ids,
        "region_count": len(record.regions),
        "at": time(),
    })

    log_event(
        logger,
        "fusion_plan_created",
        fusion_id=fusion_id,
        base_run_id=base_run_id,
        source_count=len(source_ids),
        region_count=len(record.regions),
    )

    return {"fusion_id": fusion_id, "status": "draft"}


@router.get("/fusions/{fusion_id}")
async def get_fusion(fusion_id: str) -> dict:
    """Return a fusion plan's status + regions (FusionStatus shape)."""
    validate_safe_id(fusion_id, field="fusion_id")  # Item 2 — block path traversal
    record = load_plan(fusion_id, root=sessions._FUSIONS_ROOT)
    if record is None:
        raise HTTPException(status_code=404, detail=f"fusion_id '{fusion_id}' not found")

    composite_target_url = (
        f"/png-shader/fusions/{fusion_id}/artifacts/composite_target"
        if record.composite_target_artifact_id
        else None
    )
    return {
        "fusion_id": fusion_id,
        "status": record.status,
        "base_run_id": record.base_run_id,
        "source_run_ids": list(record.source_run_ids),
        "output_run_id": record.output_run_id,
        "composite_target_url": composite_target_url,
        "regions": [dataclasses.asdict(reg) for reg in record.regions],
        "error": record.metadata.get("error"),
    }


@router.post("/fusions/{fusion_id}/composite-target")
async def create_composite_target(fusion_id: str) -> dict:
    """Build the composite visual target by blending source renders into the base.

    On a build failure the fusion is marked ``failed`` and 200 is returned with
    the error (per design: composite failure → fusion status failed, not 500).
    """
    validate_safe_id(fusion_id, field="fusion_id")  # Item 2 — block path traversal
    record = load_plan(fusion_id, root=sessions._FUSIONS_ROOT)
    if record is None:
        raise HTTPException(status_code=404, detail=f"fusion_id '{fusion_id}' not found")

    # Re-resolve the base render.
    base = store._touch_run(record.base_run_id)
    if base is None:
        raise HTTPException(status_code=422, detail=f"base run '{record.base_run_id}' not found")
    base_render = sessions._resolve_run_render(base, base.get("run_dir"))
    if base_render is None:
        raise HTTPException(status_code=422, detail="base render not available")

    # Re-resolve every source render.
    source_render_paths: dict[str, Path] = {}
    for sid in record.source_run_ids:
        with store._run_store_lock:
            src = store._run_store.get(sid)
        render = sessions._resolve_run_render(src, src.get("run_dir")) if src is not None else None
        if render is None:
            raise HTTPException(
                status_code=422,
                detail=f"source run '{sid}' render not available",
            )
        source_render_paths[sid] = render

    output_dir = sessions._fusion_artifacts_dir(fusion_id)
    try:
        build_composite_target(
            base_render_path=base_render,
            source_render_paths=source_render_paths,
            regions=record.regions,
            output_dir=output_dir,
        )
    except Exception as exc:
        logger.warning("fusion composite build failed for fusion_id=%s", fusion_id, exc_info=True)
        record.status = "failed"
        record.metadata["error"] = str(exc)
        record.updated_at = time()
        sessions._save_plan_best_effort(record)
        sessions._append_fusion_event_best_effort(fusion_id, {
            "event": "fusion_composite_target_failed", "error": str(exc), "at": time(),
        })
        return {"fusion_id": fusion_id, "status": "failed", "error": str(exc)}

    record.composite_target_artifact_id = "composite_target"
    record.status = "target_ready"
    record.updated_at = time()
    sessions._save_plan_best_effort(record)
    sessions._append_fusion_event_best_effort(fusion_id, {
        "event": "fusion_composite_target_created", "at": time(),
    })

    return {
        "fusion_id": fusion_id,
        "status": "target_ready",
        "composite_target_url": f"/png-shader/fusions/{fusion_id}/artifacts/composite_target",
    }


@router.post("/fusions/{fusion_id}/run")
async def run_fusion(fusion_id: str, payload: dict) -> dict:
    """Launch a fusion child run seeded from the base GLSL, guided by the plan.

    Body: { quality?:{}, directed_acceptance?:{} }
    Mirrors branch-refine: an independent child run (own run_id / run_dir /
    lifecycle). The base and source runs are never overwritten.
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="payload must be a JSON object")

    validate_safe_id(fusion_id, field="fusion_id")  # Item 2 — block path traversal
    record = load_plan(fusion_id, root=sessions._FUSIONS_ROOT)
    if record is None:
        raise HTTPException(status_code=404, detail=f"fusion_id '{fusion_id}' not found")

    base_run_id = record.base_run_id
    base = store._touch_run(base_run_id)
    if base is None:
        raise HTTPException(status_code=422, detail=f"base run '{base_run_id}' not found")

    base_run_dir = base.get("run_dir")
    base_selected_glsl = base.get("selected_glsl")
    if not base_run_dir or not base_selected_glsl:
        raise HTTPException(status_code=422, detail="base run is no longer usable as a fusion base")
    if sessions._resolve_run_render(base, base_run_dir) is None:
        raise HTTPException(status_code=422, detail="base render not available")

    base_reference = Path(base_run_dir) / "reference_input.png"
    if not base_reference.exists():
        raise HTTPException(status_code=422, detail="base reference image is not available")

    child_run_id = "run_" + uuid4().hex[:8]
    notes = build_fusion_notes(record)

    quality_overrides = payload.get("quality") or {}
    if not isinstance(quality_overrides, dict):
        raise HTTPException(status_code=422, detail="quality must be an object")
    quality = dict(quality_overrides)
    quality["refinement_mode"] = "on"
    quality["max_refinement_iterations"] = max(
        _coerce_int(
            quality.get("max_refinement_iterations"),
            "max_refinement_iterations", 0, 0, 20,
        ),
        1,
    )
    quality["vlm_judge_enabled"] = 1

    _da_overrides = payload.get("directed_acceptance") or {}
    if not isinstance(_da_overrides, dict):
        raise HTTPException(status_code=422, detail="directed_acceptance must be an object")
    score_drop_tolerance = _da_overrides.get("score_drop_tolerance", 0.03)
    directed_acceptance = {
        "enabled": True,
        "feedback": record.feedback,
        "mode": "fusion",
        "score_drop_tolerance": score_drop_tolerance,
        "require_vlm_for_score_drop": True,
        "composite_target_artifact_id": record.composite_target_artifact_id,
        "fusion_id": fusion_id,
    }

    lineage = {
        "parent_run_id": base_run_id,
        "root_run_id": record.root_run_id,
        "source_checkpoint_id": "final:selected",
        "mode": "fusion",
        "feedback": record.feedback,
        "fusion_id": fusion_id,
        "base_run_id": base_run_id,
        "source_run_ids": list(record.source_run_ids),
    }

    child_input_spec = build_input_spec(str(base_reference), quality=quality)
    errors = validate_input_spec(child_input_spec)
    if errors:
        raise HTTPException(status_code=422, detail={"input_spec_errors": errors})

    extra_artifacts = {
        "fusion_plan.json": plan_to_dict(record),
        "lineage.json": lineage,
        "directed_acceptance.json": directed_acceptance,
        "human_feedback.txt": record.feedback,
    }

    initial_result = {
        "run_id": child_run_id,
        "status": "running",
        "parent_run_id": base_run_id,
        "source_checkpoint_id": "final:selected",
        "lineage": lineage,
        "fusion_id": fusion_id,
        "submitted_at": time(),
        "strategy": dict(child_input_spec.get("quality") or quality),
        "stop_requested": False,
        "strategy_revision": 1,
    }
    store._store_run(child_run_id, initial_result)
    store._index_created(RunLineageRecord(
        run_id=child_run_id,
        root_run_id=record.root_run_id,
        parent_run_id=base_run_id,
        source_checkpoint_id="final:selected",
        source_checkpoint_label=None,
        mode="fusion",
        feedback=record.feedback,
        title=None,
        status="running",
        run_dir=None,
        created_at=time(),
        fusion_id=fusion_id,
        base_run_id=base_run_id,
        source_run_ids=list(record.source_run_ids),
    ))

    try:
        _start_pipeline_worker(
            run_id=child_run_id,
            image_path=base_reference,
            upload_dir=None,  # reuse base reference; never delete the base run_dir
            pipeline_input_spec=child_input_spec,
            seed_glsl=base_selected_glsl,
            model_config=store._get_run_model(base_run_id),
            trace_input={
                "run_id": child_run_id,
                "base_run_id": base_run_id,
                "fusion_id": fusion_id,
            },
            trace_metadata={
                "run_id": child_run_id,
                "pipeline": "png-shader-fusion",
                "base_run_id": base_run_id,
                "fusion_id": fusion_id,
            },
            pipeline_extra={
                "human_feedback_notes": notes,
                "directed_acceptance": directed_acceptance,
                "force_first_refinement_iteration": True,
                "lineage": lineage,
                "extra_artifacts": extra_artifacts,
            },
        )
    except WorkerCapacityError as exc:
        # Saturated worker pool → roll back the child run, signal backpressure.
        # The fusion plan record is left untouched (status unchanged) so the
        # caller can retry the run.
        store._drop_run(child_run_id)
        raise HTTPException(
            status_code=429,
            detail=f"{exc}. Retry-After: a few seconds.",
        ) from exc

    record.output_run_id = child_run_id
    record.status = "running"
    record.updated_at = time()
    sessions._save_plan_best_effort(record)
    sessions._append_fusion_event_best_effort(fusion_id, {
        "event": "fusion_run_started", "output_run_id": child_run_id, "at": time(),
    })

    log_event(
        logger,
        "fusion_run_started",
        fusion_id=fusion_id,
        base_run_id=base_run_id,
        output_run_id=child_run_id,
    )

    return {"fusion_id": fusion_id, "status": "running", "output_run_id": child_run_id}


@router.get("/fusions/{fusion_id}/artifacts/{artifact_id:path}")
async def get_fusion_artifact(fusion_id: str, artifact_id: str) -> FileResponse:
    """Serve a fusion artifact from ``<root>/fusions/<fusion_id>/``.

    artifact_id forms:
      - ``composite_target``        → composite_target.png
      - ``region_mask:<region_id>`` → region_masks/<region_id>.png
      - ``fusion_plan``             → <root>/fusions/<fusion_id>.json
    Mirrors the checkpoint artifacts endpoint's suffix allowlist + path
    containment (``.resolve().relative_to(dir)``) safety.
    """
    validate_safe_id(fusion_id, field="fusion_id")  # Item 2 — block path traversal
    fusion_dir = sessions._fusion_artifacts_dir(fusion_id)

    if artifact_id == "composite_target":
        target = fusion_dir / "composite_target.png"
    elif artifact_id == "fusion_plan":
        target = sessions._fusions_results_root() / "fusions" / f"{fusion_id}.json"
    elif artifact_id.startswith("region_mask:"):
        region_id = artifact_id[len("region_mask:"):]
        if not region_id or not _REGION_ID_RE.match(region_id):
            raise HTTPException(
                status_code=422,
                detail=f"region_id {region_id!r} contains disallowed characters",
            )
        target = fusion_dir / "region_masks" / f"{region_id}.png"
    else:
        raise HTTPException(status_code=404, detail=f"unknown artifact id: {artifact_id!r}")

    # Suffix allowlist + path containment.
    if target.suffix.lower() not in (".png", ".json"):
        raise HTTPException(status_code=422, detail="disallowed artifact suffix")
    containment_root = (
        sessions._fusions_results_root() / "fusions"
        if artifact_id == "fusion_plan"
        else fusion_dir
    )
    try:
        resolved = target.resolve()
        resolved.relative_to(containment_root.resolve())
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=422, detail="invalid artifact path") from exc

    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"artifact not found: {target.name}")

    return FileResponse(resolved)


# ---------------------------------------------------------------------------
# V4.3 Preference CRUD endpoints
# ---------------------------------------------------------------------------


