"""Generic append-only event stream over the `events` table."""
from __future__ import annotations

from sqlalchemy import select

from p2s_agent.core.db.schema import events as _events


def append_event(engine, *, entity_type: str, entity_id, event_type: str,
                 payload: dict, ts: float) -> None:
    # Guard: an explicit payload=None would bind JSON ``null`` (defeating the
    # NOT NULL/'{}' default and round-tripping back as Python None).
    with engine.begin() as conn:
        conn.execute(_events.insert().values(
            entity_type=entity_type, entity_id=entity_id, event_type=event_type,
            payload=payload if payload is not None else {}, ts=ts))


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


def delete_events(engine, entity_type: str, entity_id) -> int:
    """Delete all events for an entity; returns rows removed. entity_id=None
    matches the NULL-entity stream (e.g. preference)."""
    from sqlalchemy import delete as _delete
    cond = _events.c.entity_type == entity_type
    cond = cond & (_events.c.entity_id.is_(None) if entity_id is None
                   else _events.c.entity_id == entity_id)
    with engine.begin() as conn:
        return conn.execute(_delete(_events).where(cond)).rowcount
