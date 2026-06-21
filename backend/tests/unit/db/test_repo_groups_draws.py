def test_variant_group_roundtrip(repo_engine):
    from app.db.repositories import variant_groups as vg
    vg.upsert_group(repo_engine, {
        "group_id": "g1", "root_run_id": "r", "parent_run_id": "p",
        "source_checkpoint_id": "c", "variant_count": 3,
        "child_run_ids": ["a", "b", "c"], "created_at": 1.0,
    })
    got = vg.get_group(repo_engine, "g1")
    assert got["child_run_ids"] == ["a", "b", "c"] and got["variant_count"] == 3


def test_draw_session_roundtrip(repo_engine):
    from app.db.repositories import draw_sessions as ds
    ds.upsert_session(repo_engine, {
        "draw_id": "d1", "root_run_id": "r", "parent_run_id": "p",
        "source_checkpoint_id": "c", "requested_count": 8,
        "group_ids": ["g1", "g2"], "metadata": {"k": "v"}, "created_at": 1.0,
    })
    got = ds.get_session(repo_engine, "d1")
    assert got["group_ids"] == ["g1", "g2"] and got["metadata"] == {"k": "v"}
