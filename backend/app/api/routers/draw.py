"""PNG-shader router: batch draw-session endpoints (V3.5)."""

from __future__ import annotations

import logging
from time import time
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from p2s_agent.orchestration.variant_groups import build_variant_strategies
from p2s_agent.orchestration.draw_sessions import (
    DrawSessionRecord,
    aggregate_draw_status,
    append_session_event,
    load_session,
    save_session,
)
from p2s_agent.orchestration.human_constraints import (
    HumanConstraintSpec,
    parse_constraint_spec,
    validate_constraint_spec,
)
from p2s_agent.orchestration.run_index import RunIndexError, load_run_index, update_run_metadata
from p2s_agent.core.logging_config import log_event
from app.api.routers._shared import _coerce_int, _enforce_text_cap, _MAX_FEEDBACK_CHARS

from p2s_agent import store
from p2s_agent.orchestration import sessions

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/png-shader", tags=["png-shader"])

# Draw-session (V3.5 batch draw) card-event vocabulary.
_DRAW_CARD_EVENT_TYPES: frozenset[str] = frozenset({
    "favorite", "eliminate", "tag", "note", "use_as_fusion_base", "use_as_region_source",
})


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

    parent = store._touch_run(run_id)
    if parent is None:
        raise HTTPException(status_code=404, detail=f"run_id '{run_id}' not found")

    feedback = str(payload.get("feedback") or "")
    if not feedback.strip():
        raise HTTPException(status_code=422, detail="feedback is required for draw-session")
    _enforce_text_cap(feedback, _MAX_FEEDBACK_CHARS, field="feedback")

    card_count = _coerce_int(payload.get("card_count"), "card_count", 8, 2, 12)

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

    checkpoint, reference_path = sessions._resolve_draw_checkpoint(parent, checkpoint_id)
    # Pre-validate ONCE up-front so a bad spec can't leave partial groups behind.
    sessions._prevalidate_draw_quality(reference_path, quality)

    draw_id = "draw_" + uuid4().hex[:8]
    root_run_id = (parent.get("lineage") or {}).get("root_run_id") or run_id

    _ds_use_preferences: bool = _ds_constraint_spec.use_preferences if _ds_constraint_spec is not None else True
    group_ids, card_run_ids = sessions._create_draw_groups(
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
        save_session(record, root=sessions._DRAW_SESSIONS_ROOT)
        append_session_event(
            draw_id,
            {"event": "draw_session_created", "card_run_ids": card_run_ids,
             "group_ids": group_ids, "at": time()},
            root=sessions._DRAW_SESSIONS_ROOT,
        )
    except Exception:
        logger.warning("draw session persist failed for draw_id=%s", draw_id, exc_info=True)

    if stop_parent := bool(payload.get("stop_parent")):
        with store._run_store_lock:
            stored_parent = store._run_store.get(run_id)
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


@router.get("/draw-sessions/{draw_id}")
async def get_draw_session(draw_id: str) -> dict:
    """Return a draw session's aggregated status + per-card details."""
    record = load_session(draw_id, root=sessions._DRAW_SESSIONS_ROOT)
    if record is None:
        raise HTTPException(status_code=404, detail=f"draw_id '{draw_id}' not found")

    overlay = sessions._fold_draw_overlay(draw_id)
    index_records = load_run_index(path=store._RUN_INDEX_PATH)

    cards: list[dict] = []
    with store._run_store_lock:
        for position, run_id in enumerate(record.card_run_ids):
            ov = overlay.get(run_id, {})
            idx_rec = index_records.get(run_id)
            stored = store._run_store.get(run_id)
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
    record = load_session(draw_id, root=sessions._DRAW_SESSIONS_ROOT)
    if record is None:
        raise HTTPException(status_code=404, detail=f"draw_id '{draw_id}' not found")
    return record


def _draw_parent_or_409(record: DrawSessionRecord) -> dict:
    parent = store._touch_run(record.parent_run_id)
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

    card_count = _coerce_int(payload.get("card_count"), "card_count", 4, 2, 12)

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

    checkpoint, reference_path = sessions._resolve_draw_checkpoint(parent, checkpoint_id)
    sessions._prevalidate_draw_quality(reference_path, quality)

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
            save_session(record, root=sessions._DRAW_SESSIONS_ROOT)
        except Exception:
            logger.warning("draw-more incremental save failed for draw_id=%s", draw_id, exc_info=True)

    _dm_use_preferences: bool = _dm_constraint_spec.use_preferences if _dm_constraint_spec is not None else True
    sessions._create_draw_groups(
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
        save_session(record, root=sessions._DRAW_SESSIONS_ROOT)
        append_session_event(
            draw_id,
            {"event": "draw_more_requested", "card_count": card_count,
             "group_ids": new_group_ids, "card_run_ids": new_card_ids, "at": time()},
            root=sessions._DRAW_SESSIONS_ROOT,
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

    checkpoint, reference_path = sessions._resolve_draw_checkpoint(parent, checkpoint_id)
    sessions._prevalidate_draw_quality(reference_path, quality)

    # Build a ONE-element strategies list.
    # build_variant_strategies requires count>=2, so we request 2 and take only the first.
    strategies = build_variant_strategies(
        feedback=feedback, count=2, diversity=diversity, mode=mode,
    )[:1]
    for s in strategies:
        s["locks"] = {**(s.get("locks") or {}), **request_locks}

    gid, cids = sessions._create_variant_group(
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
    store._index_updated(new_run_id, {"replacement_of_run_id": target_run_id})
    with store._run_store_lock:
        stored_new = store._run_store.get(new_run_id)
        if stored_new is not None:
            stored_new["replacement_of_run_id"] = target_run_id

    record.group_ids.append(gid)
    record.card_run_ids.extend(cids)
    record.updated_at = time()
    try:
        save_session(record, root=sessions._DRAW_SESSIONS_ROOT)
        append_session_event(
            draw_id,
            {"event": "draw_card_eliminated", "run_id": target_run_id,
             "value": True, "auto": True, "at": time()},
            root=sessions._DRAW_SESSIONS_ROOT,
        )
        append_session_event(
            draw_id,
            {"event": "draw_card_redrawn", "run_id": target_run_id,
             "replacement_run_id": new_run_id, "reason": payload.get("reason"), "at": time()},
            root=sessions._DRAW_SESSIONS_ROOT,
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
            root=sessions._DRAW_SESSIONS_ROOT,
        )
    except Exception:
        logger.warning("append_session_event failed for draw_id=%s", draw_id, exc_info=True)

    if event_type == "favorite":
        favorite = bool(payload.get("value", True))
        try:
            update_run_metadata(run_id, {"favorite": favorite}, path=store._RUN_INDEX_PATH)
        except RunIndexError:
            pass
        except Exception:
            logger.warning("draw favorite mirror failed for run_id=%s", run_id, exc_info=True)
        # Mirror to run-store for the /status/{run_id} consumer; draw-session GET
        # treats the events overlay as authoritative.
        with store._run_store_lock:
            stored = store._run_store.get(run_id)
            if stored is not None:
                stored["favorite"] = favorite

    log_event(logger, "draw_card_event", draw_id=draw_id, run_id=run_id, event_type=event_type)

    return {"draw_id": draw_id, "run_id": run_id, "event_type": event_type, "ok": True}


