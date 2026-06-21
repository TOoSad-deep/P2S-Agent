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
