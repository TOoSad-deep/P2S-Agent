"""Generic append-only event stream over the `events` table."""
from __future__ import annotations

from sqlalchemy import select

from app.db.schema import events as _events


def append_event(engine, *, entity_type: str, entity_id, event_type: str,
                 payload: dict, ts: float) -> None:
    with engine.begin() as conn:
        conn.execute(_events.insert().values(
            entity_type=entity_type, entity_id=entity_id,
            event_type=event_type, payload=payload, ts=ts))


def load_events(engine, *, entity_type: str, entity_id) -> list:
    """Events for one entity, ordered by insertion (event_id). entity_id=None
    matches NULL-entity streams (e.g. preference)."""
    cond = _events.c.entity_type == entity_type
    cond = cond & (_events.c.entity_id.is_(None) if entity_id is None
                   else _events.c.entity_id == entity_id)
    with engine.connect() as conn:
        rows = conn.execute(
            select(_events).where(cond).order_by(_events.c.event_id)
        ).mappings().all()
    return [dict(r) for r in rows]
