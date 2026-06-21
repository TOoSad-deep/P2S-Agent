"""Best-effort shadow-mirror helpers shared by orchestration persistence modules.

These let the file-based modules (variant_groups / draw_sessions / preferences /
fusion_plans) additionally mirror their writes into SQLite without changing the
authoritative file behavior. Every ``mirror_*`` helper wraps its work in
try/except so a DB issue NEVER breaks the file write path; failures log at DEBUG.

Shadow DB location mirrors run_index's convention: ``results_root=None`` →
canonical ``backend/data/p2s.db``; a directory → ``<dir>/p2s.db`` (per-test
isolation). Reads still come from the files — this only populates the tables so
the eventual read-cutover has live data.

Known shadow-mode divergences (resolved at read-cutover): file-side deletes /
rewrites are not mirrored; event ``event_type``/``ts`` are best-effort.
"""
from __future__ import annotations

import dataclasses
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_ENABLED = True
_INIT_LOCK = threading.Lock()
_INITED: set[str] = set()


def shadow_engine(results_root):
    """Engine for the shadow DB. results_root=None → canonical backend/data/p2s.db;
    a directory → <dir>/p2s.db (per-test isolation). Lazily init_db (idempotent)."""
    from p2s_agent.core.db.engine import get_engine, init_db
    eng = get_engine(None) if results_root is None else get_engine(Path(results_root))
    key = str(eng.url)
    if key not in _INITED:
        with _INIT_LOCK:
            if key not in _INITED:
                init_db(eng)  # create_all: only adds missing tables
                _INITED.add(key)
    return eng


def _ts(event: dict) -> float:
    """Best-effort timestamp extraction from an event dict (defaults to 0.0)."""
    v = event.get("ts", event.get("timestamp", 0.0))
    return float(v) if isinstance(v, (int, float)) else 0.0


def _mirror_event(results_root, entity_type, entity_id, event, event_type) -> None:
    if not _ENABLED:
        return
    try:
        from p2s_agent.core.db.repositories import events as _ev
        _ev.append_event(shadow_engine(results_root), entity_type=entity_type,
                         entity_id=entity_id, event_type=event_type,
                         payload=event, ts=_ts(event))
    except Exception:
        logger.debug("%s shadow event failed", entity_type, exc_info=True)


# --- variant_groups ---------------------------------------------------------
def mirror_group(results_root, record) -> None:
    if not _ENABLED:
        return
    try:
        from p2s_agent.core.db.repositories import variant_groups as _r
        _r.upsert_group(shadow_engine(results_root), dataclasses.asdict(record))
    except Exception:
        logger.debug("variant_group shadow upsert failed", exc_info=True)


def mirror_group_event(results_root, group_id, event) -> None:
    _mirror_event(results_root, "variant_group", group_id, event, str(event.get("type", "")))


# --- draw_sessions ----------------------------------------------------------
def mirror_session(results_root, record) -> None:
    if not _ENABLED:
        return
    try:
        from p2s_agent.core.db.repositories import draw_sessions as _r
        _r.upsert_session(shadow_engine(results_root), dataclasses.asdict(record))
    except Exception:
        logger.debug("draw_session shadow upsert failed", exc_info=True)


def mirror_session_event(results_root, draw_id, event) -> None:
    _mirror_event(results_root, "draw_session", draw_id, event, str(event.get("type", "")))


# --- preferences ------------------------------------------------------------
def mirror_profile(results_root, profile) -> None:
    if not _ENABLED:
        return
    try:
        from p2s_agent.core.db.repositories import preferences as _r
        _r.save_profile(shadow_engine(results_root), profile)
    except Exception:
        logger.debug("preference profile shadow save failed", exc_info=True)


def mirror_pref_event(results_root, event) -> None:
    if not _ENABLED:
        return
    try:
        from p2s_agent.core.db.repositories import events as _ev
        _ev.append_event(shadow_engine(results_root), entity_type="preference",
                         entity_id=None, event_type=event.event_type,
                         payload=dataclasses.asdict(event), ts=event.timestamp)
    except Exception:
        logger.debug("preference event shadow failed", exc_info=True)


# --- fusion_plans -----------------------------------------------------------
def mirror_fusion(results_root, record) -> None:
    if not _ENABLED:
        return
    try:
        from p2s_agent.core.db.repositories import fusions as _r
        _r.upsert_fusion(shadow_engine(results_root), dataclasses.asdict(record))
    except Exception:
        logger.debug("fusion shadow upsert failed", exc_info=True)


def mirror_fusion_event(results_root, fusion_id, event) -> None:
    _mirror_event(results_root, "fusion", fusion_id, event, str(event.get("type", "")))
