"""Read-cutover: the 4 file-based loaders read SQLite first (file fallback).

Each test seeds only the DB (no file) and asserts the loader returns the record
without the JSON file ever being created — proving the read came from SQLite.
"""


def test_load_group_reads_db_without_file(tmp_path):
    from p2s_agent.core.db.shadow import shadow_engine
    from p2s_agent.core.db.repositories import variant_groups as vg
    from p2s_agent.orchestration.variant_groups import load_group
    vg.upsert_group(shadow_engine(tmp_path), {
        "group_id": "g", "root_run_id": "r", "parent_run_id": "p",
        "source_checkpoint_id": "c", "variant_count": 2,
        "child_run_ids": ["a"], "created_at": 1.0})
    rec = load_group("g", root=tmp_path)
    assert rec is not None and rec.child_run_ids == ["a"]
    assert not (tmp_path / "variant_groups" / "g.json").exists()


def test_load_session_reads_db_without_file(tmp_path):
    from p2s_agent.core.db.shadow import shadow_engine
    from p2s_agent.core.db.repositories import draw_sessions as ds
    from p2s_agent.orchestration.draw_sessions import load_session
    ds.upsert_session(shadow_engine(tmp_path), {
        "draw_id": "d", "root_run_id": "r", "parent_run_id": "p",
        "source_checkpoint_id": "c", "requested_count": 4,
        "group_ids": ["g"], "created_at": 1.0})
    rec = load_session("d", root=tmp_path)
    assert rec is not None and rec.group_ids == ["g"]
    assert not (tmp_path / "draw_sessions" / "d.json").exists()


def test_load_plan_reads_db_without_file(tmp_path):
    from p2s_agent.core.db.shadow import shadow_engine
    from p2s_agent.core.db.repositories import fusions as fz
    from p2s_agent.orchestration.fusion_plans import load_plan
    fz.upsert_fusion(shadow_engine(tmp_path), {
        "fusion_id": "f", "root_run_id": "r", "parent_run_id": "p",
        "base_run_id": "b", "source_run_ids": ["s"], "feedback": "x",
        "status": "draft", "created_at": 1.0,
        "regions": [{"id": "reg1", "source_run_id": "s"}]})
    rec = load_plan("f", root=tmp_path)
    assert rec is not None and [r.id for r in rec.regions] == ["reg1"]
    assert not (tmp_path / "fusions" / "f.json").exists()


def test_load_profile_reads_db_without_file(tmp_path):
    from p2s_agent.core.db.shadow import shadow_engine
    from p2s_agent.core.db.repositories import preferences as pf
    from p2s_agent.orchestration.preferences import load_profile
    pf.save_profile(shadow_engine(tmp_path), {
        "updated_at": 5.0, "enabled": True, "positive_preferences": ["bright"]})
    prof = load_profile(root=tmp_path)
    assert prof["positive_preferences"] == ["bright"]
    assert "id" not in prof  # singleton id stripped
    assert not (tmp_path / "preferences" / "profile.json").exists()
