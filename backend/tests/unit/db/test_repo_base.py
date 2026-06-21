def test_upsert_insert_then_update(repo_engine):
    from app.db.repositories import _base
    from app.db.schema import runs
    _base.upsert(repo_engine, runs, "run_id",
                 {"run_id": "a", "root_run_id": "a", "created_at": 1.0, "status": "pending"})
    _base.upsert(repo_engine, runs, "run_id",
                 {"run_id": "a", "root_run_id": "a", "created_at": 1.0, "status": "completed"})
    got = _base.get_by_pk(repo_engine, runs, "run_id", "a")
    assert got["status"] == "completed"  # 第二次 upsert 覆盖


def test_update_by_pk_partial(repo_engine):
    from app.db.repositories import _base
    from app.db.schema import runs
    _base.upsert(repo_engine, runs, "run_id",
                 {"run_id": "b", "root_run_id": "b", "created_at": 1.0})
    n = _base.update_by_pk(repo_engine, runs, "run_id", "b", {"final_score": 0.9, "favorite": True})
    assert n == 1
    got = _base.get_by_pk(repo_engine, runs, "run_id", "b")
    assert got["final_score"] == 0.9 and got["favorite"] in (1, True)


def test_get_all_keyed_by_pk(repo_engine):
    from app.db.repositories import _base
    from app.db.schema import runs
    for rid in ("x", "y"):
        _base.upsert(repo_engine, runs, "run_id",
                     {"run_id": rid, "root_run_id": rid, "created_at": 1.0})
    allrows = _base.get_all(repo_engine, runs, "run_id")
    assert set(allrows) == {"x", "y"} and allrows["x"]["run_id"] == "x"


def test_get_by_pk_missing_returns_none(repo_engine):
    from app.db.repositories import _base
    from app.db.schema import runs
    assert _base.get_by_pk(repo_engine, runs, "run_id", "nope") is None
