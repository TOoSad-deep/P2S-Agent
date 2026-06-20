"""PNG-shader router: explore-variants + variant-group read/action endpoints."""

from __future__ import annotations

import logging
from pathlib import Path
from time import time
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from p2s_agent.orchestration.checkpoints import CheckpointError, list_checkpoints, resolve_checkpoint
from p2s_agent.orchestration.variant_groups import (
    aggregate_group_status,
    append_group_event,
    build_variant_strategies,
    load_group,
    save_group,
)
from p2s_agent.orchestration.preferences import (
    PreferenceEvent,
    append_preference_event,
    load_profile,
    rank_variants_by_preference,
)
from p2s_agent.orchestration.human_constraints import (
    HumanConstraintSpec,
    parse_constraint_spec,
    validate_constraint_spec,
)
from p2s_agent.orchestration.run_index import RunIndexError, load_run_index, update_run_metadata
from p2s_agent.core.logging_config import log_event
from app.api.routers._shared import _coerce_int, _enforce_text_cap, _MAX_FEEDBACK_CHARS, _MAX_VARIANT_COUNT

from p2s_agent import store
from p2s_agent.orchestration import sessions

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/png-shader", tags=["png-shader"])


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

    parent = store._touch_run(run_id)
    if parent is None:
        raise HTTPException(status_code=404, detail=f"run_id '{run_id}' not found")

    variant_count = _coerce_int(
        payload.get("variant_count"), "variant_count", 4, 2, _MAX_VARIANT_COUNT
    )
    diversity = str(payload.get("diversity") or "medium")
    mode = str(payload.get("mode") or "explore")
    feedback = str(payload.get("feedback") or "")
    _enforce_text_cap(feedback, _MAX_FEEDBACK_CHARS, field="feedback")
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
    group_id, child_run_ids = sessions._create_variant_group(
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
        with store._run_store_lock:
            stored_parent = store._run_store.get(run_id)
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
# V3.5 draw-session (gacha-style batch draw) endpoints
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# V3.1: Variant-group read/aggregate + action endpoints
# ---------------------------------------------------------------------------

_STATUS_RANK = {"completed": 0, "running": 1, "queued": 2, "cancelled": 3, "failed": 4}


def _get_group_or_404(group_id: str):
    record = load_group(group_id, root=sessions._VARIANT_GROUPS_ROOT)
    if record is None:
        raise HTTPException(status_code=404, detail=f"group_id '{group_id}' not found")
    return record


@router.get("/variant-groups/{group_id}")
async def get_variant_group(group_id: str) -> dict:
    """Return aggregated status + sorted variants for a variant group."""
    record = _get_group_or_404(group_id)

    # Build per-child variant dicts.
    index_records = load_run_index(path=store._RUN_INDEX_PATH)
    variants = []
    with store._run_store_lock:
        for position, run_id in enumerate(record.child_run_ids):
            stored = store._run_store.get(run_id)
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

    # Annotate each variant with preference ranking (recommendation hint only;
    # winner is NEVER changed here).
    profile = load_profile(root=sessions._PREFERENCES_ROOT)
    ranking = rank_variants_by_preference(variants, profile)
    for v in variants:
        pref = ranking.get(v["run_id"], {"preference_score": 0.0, "recommended": False})
        v["preference_score"] = pref["preference_score"]
        v["recommended"] = pref["recommended"]

    status = aggregate_group_status([v["status"] for v in variants])

    return {
        "group_id": record.group_id,
        "parent_run_id": record.parent_run_id,
        "source_checkpoint_id": record.source_checkpoint_id,
        "feedback": record.feedback,
        "status": status,
        "winner_run_id": record.winner_run_id,
        "preference_enabled": bool(profile.get("enabled")),
        "variants": variants,
    }


@router.post("/variant-groups/{group_id}/stop")
async def stop_variant_group(group_id: str) -> dict:
    """Signal all queued/running children of a variant group to stop."""
    record = _get_group_or_404(group_id)

    with store._run_store_lock:
        for run_id in record.child_run_ids:
            stored = store._run_store.get(run_id)
            if stored is not None and stored.get("status") in ("queued", "running"):
                stored["stop_requested"] = True

    try:
        append_group_event(
            group_id,
            {"event": "stopped", "at": time()},
            root=sessions._VARIANT_GROUPS_ROOT,
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
        save_group(record, root=sessions._VARIANT_GROUPS_ROOT)
    except Exception as exc:
        logger.error("save_group failed for group_id=%s", group_id, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to persist winner") from exc

    # Mark the winner favorite in the run index (best-effort).
    try:
        update_run_metadata(winner_run_id, {"favorite": True}, path=store._RUN_INDEX_PATH)
    except RunIndexError:
        pass  # Run not in the index yet — silently ignore.
    except Exception:
        logger.warning(
            "update_run_metadata(favorite) failed for run_id=%s", winner_run_id, exc_info=True
        )

    # Mirror into the in-memory store (best-effort).
    with store._run_store_lock:
        stored = store._run_store.get(winner_run_id)
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
            root=sessions._VARIANT_GROUPS_ROOT,
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
            root=sessions._PREFERENCES_ROOT,
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
            root=sessions._VARIANT_GROUPS_ROOT,
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
            root=sessions._PREFERENCES_ROOT,
        )
    except Exception:
        logger.warning(
            "append_preference_event(variant_rated) failed for group_id=%s", group_id, exc_info=True
        )

    return {"group_id": group_id, "run_id": run_id, "rating": rating}

