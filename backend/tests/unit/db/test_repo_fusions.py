def _plan(regions):
    return {
        "fusion_id": "f1", "root_run_id": "r", "parent_run_id": "p",
        "base_run_id": "b", "source_run_ids": ["s1"], "feedback": "x",
        "status": "draft", "created_at": 1.0, "regions": regions,
    }


def test_upsert_get_fusion_with_regions_ordered(repo_engine):
    from app.db.repositories import fusions
    fusions.upsert_fusion(repo_engine, _plan([
        {"id": "reg2", "source_run_id": "s1", "geometry": {"x": 0, "y": 0, "w": 1, "h": 1}},
        {"id": "reg1", "source_run_id": "s1", "strength": 0.7},
    ]))
    got = fusions.get_fusion(repo_engine, "f1")
    assert [r["id"] for r in got["regions"]] == ["reg2", "reg1"]   # 保序
    assert got["regions"][0]["geometry"] == {"x": 0, "y": 0, "w": 1, "h": 1}
    assert got["regions"][1]["strength"] == 0.7
    assert got["source_run_ids"] == ["s1"]


def test_reupsert_replaces_regions(repo_engine):
    from app.db.repositories import fusions
    fusions.upsert_fusion(repo_engine, _plan([{"id": "a", "source_run_id": "s1"},
                                              {"id": "b", "source_run_id": "s1"}]))
    fusions.upsert_fusion(repo_engine, _plan([{"id": "c", "source_run_id": "s1"}]))
    got = fusions.get_fusion(repo_engine, "f1")
    assert [r["id"] for r in got["regions"]] == ["c"]   # 整体替换


def test_get_missing_fusion_none(repo_engine):
    from app.db.repositories import fusions
    assert fusions.get_fusion(repo_engine, "nope") is None
