def test_shadow_engine_dir_inits_and_isolates(tmp_path):
    from p2s_agent.core.db.shadow import shadow_engine
    from sqlalchemy import inspect
    eng = shadow_engine(tmp_path)
    assert (tmp_path / "p2s.db").exists()
    assert "runs" in set(inspect(eng).get_table_names())


def test_shadow_engine_caches(tmp_path):
    from p2s_agent.core.db.shadow import shadow_engine
    assert shadow_engine(tmp_path) is shadow_engine(tmp_path)
