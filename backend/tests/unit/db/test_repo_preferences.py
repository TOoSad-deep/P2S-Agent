def test_profile_absent_then_save_load(repo_engine):
    from p2s_agent.core.db.repositories import preferences as pref
    assert pref.load_profile(repo_engine) is None
    pref.save_profile(repo_engine, {
        "schema_version": 1, "updated_at": 5.0, "enabled": True,
        "default_locks": {"palette": True}, "positive_preferences": ["bright"],
        "negative_preferences": [], "preferred_variant_labels": ["semantic"],
        "score_drop_tolerance_hint": 0.03, "summary_source_event_count": 4,
    })
    got = pref.load_profile(repo_engine)
    assert got["positive_preferences"] == ["bright"]
    assert got["default_locks"] == {"palette": True}
    assert got["enabled"] in (1, True)


def test_save_profile_is_singleton(repo_engine):
    from p2s_agent.core.db.repositories import preferences as pref
    from p2s_agent.core.db.schema import preference_profile
    from sqlalchemy import select, func
    pref.save_profile(repo_engine, {"updated_at": 1.0, "enabled": True})
    pref.save_profile(repo_engine, {"updated_at": 2.0, "enabled": False})
    with repo_engine.connect() as conn:
        n = conn.execute(select(func.count()).select_from(preference_profile)).scalar()
    assert n == 1                                   # 始终单行
    assert pref.load_profile(repo_engine)["updated_at"] == 2.0
