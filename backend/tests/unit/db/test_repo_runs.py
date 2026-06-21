def test_upsert_get_run_roundtrip_with_json(repo_engine):
    from app.db.repositories import runs
    runs.upsert_run(repo_engine, {
        "run_id": "r1", "root_run_id": "r1", "created_at": 1.0,
        "tags": ["a", "b"], "source_run_ids": ["s1"], "favorite": True,
    })
    got = runs.get_run(repo_engine, "r1")
    assert got["tags"] == ["a", "b"]          # JSON 往返
    assert got["source_run_ids"] == ["s1"]
    assert got["favorite"] in (1, True)


def test_update_run_fields(repo_engine):
    from app.db.repositories import runs
    runs.upsert_run(repo_engine, {"run_id": "r2", "root_run_id": "r2", "created_at": 1.0})
    assert runs.update_run(repo_engine, "r2", {"status": "completed", "final_score": 0.8}) == 1
    assert runs.get_run(repo_engine, "r2")["status"] == "completed"


def test_get_all_runs_keyed(repo_engine):
    from app.db.repositories import runs
    for rid in ("a", "b"):
        runs.upsert_run(repo_engine, {"run_id": rid, "root_run_id": rid, "created_at": 1.0})
    assert set(runs.get_all_runs(repo_engine)) == {"a", "b"}
