import pytest
from sqlalchemy import insert, select, delete
from sqlalchemy.exc import IntegrityError


def _eng(tmp_path):
    from p2s_agent.core.db.engine import get_engine, init_db
    eng = get_engine(tmp_path)
    init_db(eng)
    return eng


def test_fusion_region_cascade_delete(tmp_path):
    from p2s_agent.core.db.schema import fusion_plans, fusion_regions
    eng = _eng(tmp_path)
    with eng.begin() as conn:
        conn.execute(insert(fusion_plans).values(
            fusion_id="f1", root_run_id="r", parent_run_id="p", created_at=1.0))
        conn.execute(insert(fusion_regions).values(
            fusion_id="f1", region_id="reg1", ordinal=0))
    with eng.begin() as conn:
        conn.execute(delete(fusion_plans).where(fusion_plans.c.fusion_id == "f1"))
    with eng.connect() as conn:
        remaining = conn.execute(select(fusion_regions)).fetchall()
    assert remaining == []  # 级联清理


def test_preference_profile_singleton(tmp_path):
    from p2s_agent.core.db.schema import preference_profile
    eng = _eng(tmp_path)
    with eng.begin() as conn:
        conn.execute(insert(preference_profile).values(id=1))
    with pytest.raises(IntegrityError):  # id=2 触发 CHECK(id=1)
        with eng.begin() as conn:
            conn.execute(insert(preference_profile).values(id=2))


def test_runs_defaults_applied(tmp_path):
    from p2s_agent.core.db.schema import runs
    eng = _eng(tmp_path)
    with eng.begin() as conn:
        conn.execute(insert(runs).values(
            run_id="x", root_run_id="x", created_at=1.0))
    with eng.connect() as conn:
        row = conn.execute(select(runs).where(runs.c.run_id == "x")).mappings().one()
    assert row["status"] == "unknown"
    assert row["favorite"] in (0, False)
    assert row["tags"] == []          # JSON server_default '[]'
    assert row["source_run_ids"] == []
