from sqlalchemy import create_engine, inspect


def _fresh_engine():
    # 纯内存库，仅验证 DDL 结构（不涉及 PRAGMA/FK 行为）
    eng = create_engine("sqlite://")
    from app.db.schema import metadata
    metadata.create_all(eng)
    return eng


def test_all_seven_tables_created():
    insp = inspect(_fresh_engine())
    expected = {
        "runs", "variant_groups", "draw_sessions",
        "fusion_plans", "fusion_regions", "preference_profile", "events",
    }
    assert expected.issubset(set(insp.get_table_names()))


def test_runs_has_key_columns():
    insp = inspect(_fresh_engine())
    cols = {c["name"] for c in insp.get_columns("runs")}
    for required in (
        "run_id", "root_run_id", "parent_run_id", "status", "final_score",
        "favorite", "tags", "variant_group_id", "draw_session_id",
        "fusion_id", "source_run_ids", "run_dir", "created_at",
    ):
        assert required in cols


def test_runs_indexes_present():
    insp = inspect(_fresh_engine())
    names = {ix["name"] for ix in insp.get_indexes("runs")}
    for required in (
        "idx_runs_root", "idx_runs_status", "idx_runs_score",
        "idx_runs_fav", "idx_runs_vgroup", "idx_runs_draw", "idx_runs_fusion",
    ):
        assert required in names


def test_events_table_shape():
    insp = inspect(_fresh_engine())
    cols = {c["name"] for c in insp.get_columns("events")}
    assert {"event_id", "entity_type", "entity_id", "event_type", "payload", "ts"}.issubset(cols)
