"""variant_groups table repository."""
from __future__ import annotations

from p2s_agent.core.db.repositories import _base
from p2s_agent.core.db.schema import variant_groups as _vg


def upsert_group(engine, row: dict) -> None:
    _base.upsert(engine, _vg, "group_id", row)


def get_group(engine, group_id: str) -> "dict | None":
    return _base.get_by_pk(engine, _vg, "group_id", group_id)
