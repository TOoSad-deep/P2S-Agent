"""PNG-shader router: user-preference profile + events endpoints (V4.4)."""

from __future__ import annotations

import logging
from time import time
from uuid import uuid4

from fastapi import APIRouter, HTTPException

from p2s_agent.orchestration.preferences import (
    PreferenceEvent,
    append_preference_event,
    clear_preferences,
    default_profile,
    load_preference_events,
    load_profile,
    patch_profile,
    rebuild_profile,
    save_profile,
)

from p2s_agent.orchestration import sessions

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/png-shader", tags=["png-shader"])


@router.get("/preferences/profile")
async def get_preference_profile() -> dict:
    """Return the current preference profile (default if none saved)."""
    try:
        return load_profile(root=sessions._PREFERENCES_ROOT)
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
        return patch_profile(payload, updated_at=time(), root=sessions._PREFERENCES_ROOT)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception:
        logger.warning("patch_profile failed", exc_info=True)
        return load_profile(root=sessions._PREFERENCES_ROOT)


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
        append_preference_event(event, root=sessions._PREFERENCES_ROOT)
    except Exception:
        logger.warning("append_preference_event failed", exc_info=True)

    return {"event_id": event_id, "ok": True}


@router.post("/preferences/rebuild")
async def rebuild_preference_profile() -> dict:
    """Rebuild the preference profile from all stored events and save it."""
    try:
        events = load_preference_events(root=sessions._PREFERENCES_ROOT)
        base = load_profile(root=sessions._PREFERENCES_ROOT)
        profile = rebuild_profile(events, updated_at=time(), base_profile=base)
        save_profile(profile, root=sessions._PREFERENCES_ROOT)
        return profile
    except Exception:
        logger.warning("rebuild_preference_profile failed", exc_info=True)
        return load_profile(root=sessions._PREFERENCES_ROOT)


@router.post("/preferences/clear")
async def clear_preference_data() -> dict:
    """Clear all preference events and reset the profile to defaults."""
    try:
        clear_preferences(root=sessions._PREFERENCES_ROOT)
    except Exception:
        logger.warning("clear_preferences failed", exc_info=True)
    return {"ok": True}
