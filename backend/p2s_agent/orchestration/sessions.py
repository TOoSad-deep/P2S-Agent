"""Session-level orchestration helpers for the PNG-to-Shader pipeline.

Framework-free home for the run/session lifecycle glue that sits *between* the
in-memory store and the persistent orchestration records. Extracted from
``app.routers.png_shader`` so the web layer holds no domain orchestration of its
own.

Dependency direction: this module imports ``p2s_agent.store`` (attribute
access), the sibling ``p2s_agent.orchestration.*`` records, the agent
``core.*`` helpers, and ``p2s_agent.workers`` (to spawn variant child runs). It
never imports any ``app.*`` / ``fastapi`` / ``starlette`` symbol (enforced by
test_agent_web_boundary), so HTTP-shaped failures are raised as agent domain
errors (``AgentInputError`` → 422, ``AgentConflictError`` → 409) that the web
layer (``app/main.py``) translates centrally.

Cycle note: the module-level ``from p2s_agent.workers import ...`` is safe
because the worker layer calls back into this module
(``_finalize_fusion_for_run``) via a function-body (lazy) import at its call
site, so the two modules do not close an import cycle at load time.
"""

from __future__ import annotations

import logging
from pathlib import Path
from time import time
from typing import Optional
from uuid import uuid4

from p2s_agent import store
from p2s_agent.core.errors import AgentConflictError, AgentInputError
from p2s_agent.core.pipeline.artifacts import DEFAULT_RESULTS_ROOT
from p2s_agent.core.pipeline.input_spec import build_input_spec, validate_input_spec
from p2s_agent.core.validation import coerce_int
from p2s_agent.orchestration.checkpoints import (
    CheckpointError,
    _selected_candidate,
    candidate_render_relative,
    checkpoint_metadata,
    list_checkpoints,
    resolve_checkpoint,
)
from p2s_agent.orchestration.draw_sessions import (
    load_session_events,
    plan_draw_batches,
)
from p2s_agent.orchestration.fusion_plans import (
    FusionPlanRecord,
    append_plan_event,
    save_plan,
    update_plan_status,
)
from p2s_agent.orchestration.human_constraints import (
    HumanConstraintSpec,
    build_constraint_notes,
    spec_to_dict,
)
from p2s_agent.orchestration.human_feedback import build_human_feedback_notes
from p2s_agent.orchestration.preferences import (
    build_preference_notes,
    load_profile,
)
from p2s_agent.orchestration.run_index import RunLineageRecord
from p2s_agent.orchestration.variant_groups import (
    VariantGroupRecord,
    append_group_event,
    build_variant_strategies,
    save_group,
)
from p2s_agent.workers import _start_pipeline_worker, _variant_worker_semaphore

logger = logging.getLogger(__name__)

# Single source of truth for the on-disk persistence roots. The web layer
# references these by ATTRIBUTE (e.g. ``sessions._FUSIONS_ROOT``) so a test that
# overrides one is seen by both the session helpers here and the route handlers
# / worker terminal finalize that read them.
_FUSIONS_ROOT: Optional[str] = None  # tests override to isolate (V4.5 fusion)
_VARIANT_GROUPS_ROOT: Optional[str] = None  # tests override to isolate
_DRAW_SESSIONS_ROOT: Optional[str] = None  # tests override to isolate
_PREFERENCES_ROOT: Optional[str] = None  # tests override to isolate


def _finalize_fusion_for_run(run_id: str, status: str) -> None:
    """Best-effort: close out the fusion plan a finished run belongs to (Bug 5).

    When a worker reaches a terminal state, if the run's lineage carries a
    ``fusion_id`` we mark the corresponding FusionPlanRecord ``completed`` /
    ``failed`` so ``GET /fusions/{id}`` stops reporting ``running`` forever and
    the frontend poll terminates. Fully wrapped so a failure here can never
    break the worker thread."""
    try:
        with store._run_store_lock:
            stored = store._run_store.get(run_id) or {}
            lineage = stored.get("lineage") or {}
            fusion_id = stored.get("fusion_id") or lineage.get("fusion_id")
        if not fusion_id:
            return
        update_plan_status(
            str(fusion_id), status, updated_at=time(), root=_FUSIONS_ROOT
        )
    except Exception:
        logger.warning("fusion plan finalize failed for run_id=%s", run_id, exc_info=True)


# ---------------------------------------------------------------------------
# Variant fan-out / draw-session creation helpers
# ---------------------------------------------------------------------------


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
            coerce_int(
                quality.get("max_refinement_iterations"),
                field="max_refinement_iterations", default=0, lo=0, hi=20,
            ),
            1,
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
            raise AgentInputError({"input_spec_errors": errors})

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
        store._store_run(child_run_id, initial_result)
        store._index_created(RunLineageRecord(
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
            model_config=store._get_run_model(parent_run_id),
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
                # V4.5 region veto: forward protect-mode regions (empty when no
                # constraints → pipeline no-op, backward compatible).
                "protect_regions": (
                    [r for r in constraint_spec.regions if r.mode == "protect"]
                    if constraint_spec is not None else []
                ),
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


def _resolve_draw_checkpoint(parent: dict, checkpoint_id: str):
    """Resolve a draw-session checkpoint + reference image from a parent run.

    Mirrors explore_variants' pre-checks. Returns ``(checkpoint, reference_path)``.

    Raises:
        AgentConflictError(409): no branchable checkpoint / no run_dir / no reference.
        AgentInputError(422): checkpoint_id cannot be resolved.
    """
    if not list_checkpoints(parent):
        raise AgentConflictError("no branchable checkpoint yet")
    try:
        checkpoint = resolve_checkpoint(parent, checkpoint_id)
    except CheckpointError as exc:
        raise AgentInputError(str(exc)) from exc

    parent_run_dir = parent.get("run_dir")
    if not parent_run_dir:
        raise AgentConflictError("parent run_dir is not available yet")
    reference_path = Path(parent_run_dir) / "reference_input.png"
    if not reference_path.exists():
        raise AgentConflictError("parent reference image is not available")
    return checkpoint, reference_path


def _prevalidate_draw_quality(reference_path: Path, quality: dict) -> None:
    """Validate ONCE up-front, before any child run is created, to avoid leaving
    orphaned/partial groups behind when an input-spec error would otherwise fire
    mid-loop inside ``_create_variant_group``.

    Builds the same probe quality dict the helper applies per child.

    Raises:
        AgentInputError(422): when the probe input-spec has validation errors.
    """
    probe = {
        **quality,
        "refinement_mode": "on",
        "max_refinement_iterations": max(
            coerce_int(
                quality.get("max_refinement_iterations"),
                field="max_refinement_iterations", default=0, lo=0, hi=20,
            ),
            1,
        ),
        "vlm_judge_enabled": 1,
    }
    probe_spec = build_input_spec(str(reference_path), quality=probe)
    errors = validate_input_spec(probe_spec)
    if errors:
        raise AgentInputError({"input_spec_errors": errors})


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


# ---------------------------------------------------------------------------
# Fusion path / persistence helpers
# ---------------------------------------------------------------------------


def _fusions_results_root() -> Path:
    """Resolve the on-disk root that holds the per-fusion artifacts dir.

    ``save_plan`` writes ``<root>/fusions/<fusion_id>.json``; this returns the
    same ``<root>`` so composite_target.png + region_masks/ land under
    ``<root>/fusions/<fusion_id>/``.
    """
    return Path(_FUSIONS_ROOT) if _FUSIONS_ROOT is not None else DEFAULT_RESULTS_ROOT


def _fusion_artifacts_dir(fusion_id: str) -> Path:
    """Per-fusion artifacts dir: ``<root>/fusions/<fusion_id>/``."""
    return _fusions_results_root() / "fusions" / fusion_id


def _resolve_run_render(stored: dict, run_dir: "str | None") -> "Path | None":
    """Resolve a run's selected render PNG, mirroring the region-mask endpoint.

    Returns the path to the selected candidate's render under
    ``<run_dir>/candidates/`` — accepting either render-backend spelling
    (``<id>_render.png`` for DSL-rasterized candidates or ``<id>_webgl.png``
    for WebGL-scored GLSL candidates, see ``candidate_render_relative``) — if it
    exists, else None.
    """
    if not run_dir:
        return None
    selected_cand = _selected_candidate(stored)
    if selected_cand is None:
        return None
    sid = selected_cand.get("id")
    if not sid:
        return None
    base = Path(run_dir)
    render_path = base / candidate_render_relative(base, sid)
    return render_path if render_path.exists() else None


def _save_plan_best_effort(record: FusionPlanRecord) -> None:
    try:
        save_plan(record, root=_FUSIONS_ROOT)
    except Exception:
        logger.warning("fusion save_plan failed for fusion_id=%s", record.fusion_id, exc_info=True)


def _append_fusion_event_best_effort(fusion_id: str, event: dict) -> None:
    try:
        append_plan_event(fusion_id, event, root=_FUSIONS_ROOT)
    except Exception:
        logger.warning("fusion append_plan_event failed for fusion_id=%s", fusion_id, exc_info=True)
