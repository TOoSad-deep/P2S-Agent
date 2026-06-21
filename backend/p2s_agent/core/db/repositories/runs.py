"""runs table repository (thin wrappers over _base)."""
from __future__ import annotations

from p2s_agent.core.db.repositories import _base
from p2s_agent.core.db.schema import runs as _runs


def upsert_run(engine, row: dict) -> None:
    _base.upsert(engine, _runs, "run_id", row)


def update_run(engine, run_id: str, fields: dict) -> int:
    return _base.update_by_pk(engine, _runs, "run_id", run_id, fields)


def get_run(engine, run_id: str) -> "dict | None":
    return _base.get_by_pk(engine, _runs, "run_id", run_id)


def get_all_runs(engine) -> dict:
    return _base.get_all(engine, _runs, "run_id")


def delete_runs(engine, run_ids) -> int:
    """Delete the given run_ids; returns the number of rows removed."""
    ids = list(run_ids)
    if not ids:
        return 0
    with engine.begin() as conn:
        res = conn.execute(_runs.delete().where(_runs.c.run_id.in_(ids)))
        return res.rowcount


def get_runs_by_root(engine, root_run_id) -> dict:
    """Runs sharing a root (indexed by root_run_id), keyed by run_id."""
    from sqlalchemy import select
    with engine.connect() as conn:
        rows = conn.execute(
            select(_runs).where(_runs.c.root_run_id == root_run_id)
        ).mappings().all()
    return {r["run_id"]: dict(r) for r in rows}
