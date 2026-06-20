"""PNG-shader router: branch-refine (directed refinement child run)."""

from __future__ import annotations

import logging
from pathlib import Path
from time import time
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from p2s_agent.orchestration.checkpoints import (
    CheckpointError,
    checkpoint_metadata,
    list_checkpoints,
    resolve_checkpoint,
)
from p2s_agent.orchestration.preferences import build_preference_notes, load_profile
from p2s_agent.orchestration.human_feedback import (
    MODES,
    FeedbackValidationError,
    build_human_feedback_notes,
    validate_feedback,
)
from p2s_agent.orchestration.human_constraints import (
    HumanConstraintSpec,
    build_constraint_notes,
    parse_constraint_spec,
    spec_to_dict,
    validate_constraint_spec,
)
from p2s_agent.core.pipeline.input_spec import build_input_spec, validate_input_spec
from p2s_agent.orchestration.run_index import RunLineageRecord
from p2s_agent.core.logging_config import log_event
from app.api.routers._shared import _coerce_int, _enforce_text_cap, _MAX_FEEDBACK_CHARS

from p2s_agent import store
from p2s_agent.orchestration import sessions
from p2s_agent.workers import WorkerCapacityError, _start_pipeline_worker

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/png-shader", tags=["png-shader"])


@router.post("/runs/{run_id}/branch-refine")
async def branch_refine(run_id: str, payload: dict) -> dict:
    """Create a directed-refinement child run seeded from a parent checkpoint.

    The child is an independent run (own run_id / run_dir / lifecycle) seeded
    with the checkpoint's GLSL and the user's feedback. The parent is never
    overwritten and its reference image is never deleted.
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="payload must be a JSON object")

    parent = store._touch_run(run_id)
    if parent is None:
        raise HTTPException(status_code=404, detail=f"run_id '{run_id}' not found")

    checkpoint_id = str(payload.get("checkpoint_id") or "final:selected")
    mode = str(payload.get("mode") or "refine")
    feedback = str(payload.get("feedback") or "")
    _enforce_text_cap(feedback, _MAX_FEEDBACK_CHARS, field="feedback")
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
    _pref_profile = load_profile(root=sessions._PREFERENCES_ROOT)
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
        _coerce_int(
            quality.get("max_refinement_iterations"),
            "max_refinement_iterations", 0, 0, 20,
        ),
        1,
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
    store._store_run(child_run_id, initial_result)
    store._index_created(RunLineageRecord(
        run_id=child_run_id, root_run_id=root_run_id, parent_run_id=run_id,
        source_checkpoint_id=checkpoint_id,
        source_checkpoint_label=checkpoint.label,
        mode=mode, feedback=feedback, title=None,
        status="pending", run_dir=None, created_at=time(),
    ))

    try:
        _start_pipeline_worker(
            run_id=child_run_id,
            image_path=reference_path,
            upload_dir=None,  # reuse parent reference; never delete the parent run_dir
            pipeline_input_spec=child_input_spec,
            seed_glsl=checkpoint.glsl,
            model_config=store._get_run_model(run_id),
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
                # V4.5 region veto: forward protect-mode regions so the pipeline
                # can hard-veto candidates that mutate them. Empty list when no
                # constraints → pipeline no-op (backward compatible).
                "protect_regions": (
                    [r for r in _constraint_spec.regions if r.mode == "protect"]
                    if _constraint_spec is not None else []
                ),
            },
        )
    except WorkerCapacityError as exc:
        # Saturated worker pool → roll back the child run, signal backpressure.
        store._drop_run(child_run_id)
        raise HTTPException(
            status_code=429,
            detail=f"{exc}. Retry-After: a few seconds.",
        ) from exc

    if stop_parent:
        with store._run_store_lock:
            stored_parent = store._run_store.get(run_id)
            if stored_parent is not None and stored_parent.get("status") == "running":
                stored_parent["stop_requested"] = True

    return initial_result


