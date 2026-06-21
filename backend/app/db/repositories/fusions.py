"""fusion_plans + fusion_regions repository.

The plan dict carries nested `regions` (list of region dicts using the
FusionRegion field `id`). On write we split regions into the child table
(adding ordinal); on read we re-attach them ordered by ordinal.
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.db.repositories import _base
from app.db.schema import fusion_plans as _fp
from app.db.schema import fusion_regions as _fr

_REGION_DEFAULTS = {
    "label": "", "source_run_id": "", "instruction": "",
    "geometry_type": "rect", "geometry": {},
    "strength": 0.5, "blend_mode": "soft", "feather": 0.08,
}


def upsert_fusion(engine, plan: dict) -> None:
    regions = plan.get("regions") or []
    plan_row = {k: v for k, v in plan.items() if k != "regions"}
    fusion_id = plan_row["fusion_id"]
    with engine.begin() as conn:
        # plan row upsert
        stmt = sqlite_insert(_fp).values(**plan_row)
        set_ = {k: stmt.excluded[k] for k in plan_row if k != "fusion_id"}
        if set_:
            stmt = stmt.on_conflict_do_update(index_elements=["fusion_id"], set_=set_)
        else:
            stmt = stmt.on_conflict_do_nothing(index_elements=["fusion_id"])
        conn.execute(stmt)
        # replace regions wholesale
        conn.execute(_fr.delete().where(_fr.c.fusion_id == fusion_id))
        for ordinal, reg in enumerate(regions):
            values = {"fusion_id": fusion_id, "region_id": reg["id"], "ordinal": ordinal}
            for key, default in _REGION_DEFAULTS.items():
                values[key] = reg.get(key, default)
            conn.execute(_fr.insert().values(**values))


def _region_row_to_dict(row) -> dict:
    return {
        "id": row["region_id"],
        "label": row["label"],
        "source_run_id": row["source_run_id"],
        "instruction": row["instruction"],
        "geometry_type": row["geometry_type"],
        "geometry": row["geometry"],
        "strength": row["strength"],
        "blend_mode": row["blend_mode"],
        "feather": row["feather"],
    }


def get_fusion(engine, fusion_id: str) -> "dict | None":
    plan = _base.get_by_pk(engine, _fp, "fusion_id", fusion_id)
    if plan is None:
        return None
    with engine.connect() as conn:
        rows = conn.execute(
            select(_fr).where(_fr.c.fusion_id == fusion_id).order_by(_fr.c.ordinal)
        ).mappings().all()
    plan["regions"] = [_region_row_to_dict(r) for r in rows]
    return plan
