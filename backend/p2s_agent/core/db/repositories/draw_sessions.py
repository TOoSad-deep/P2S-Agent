"""draw_sessions table repository."""
from __future__ import annotations

from p2s_agent.core.db.repositories import _base
from p2s_agent.core.db.schema import draw_sessions as _ds


def upsert_session(engine, row: dict) -> None:
    _base.upsert(engine, _ds, "draw_id", row)


def get_session(engine, draw_id: str) -> "dict | None":
    return _base.get_by_pk(engine, _ds, "draw_id", draw_id)
