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
