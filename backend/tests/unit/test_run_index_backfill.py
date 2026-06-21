from p2s_agent.orchestration.run_index import (
    RunLineageRecord, append_run_created, backfill_runs_to_db, reconcile_runs_with_db)


def _rec(run_id, status="pending"):
    return RunLineageRecord(run_id=run_id, root_run_id=run_id, parent_run_id=None,
        source_checkpoint_id=None, source_checkpoint_label=None, mode=None, feedback=None,
        title=None, status=status, run_dir=None, created_at=1.0)


def test_backfill_populates_db_from_jsonl(tmp_path, monkeypatch):
    import p2s_agent.orchestration.run_index as ri
    monkeypatch.setattr(ri, "_SHADOW_DB_ENABLED", False)  # simulate pre-shadow history
    idx = tmp_path / "run_index.jsonl"
    append_run_created(_rec("h1"), path=idx)
    append_run_created(_rec("h2"), path=idx)
    from p2s_agent.core.db.engine import get_engine, init_db
    from p2s_agent.core.db.repositories import runs as runs_repo
    init_db(get_engine(tmp_path))
    assert runs_repo.get_run(get_engine(tmp_path), "h1") is None  # not mirrored (shadow off)
    assert backfill_runs_to_db(path=idx) == 2
    assert runs_repo.get_run(get_engine(tmp_path), "h1")["run_id"] == "h1"
    assert reconcile_runs_with_db(path=idx) == []  # parity


def test_backfill_idempotent(tmp_path, monkeypatch):
    import p2s_agent.orchestration.run_index as ri
    monkeypatch.setattr(ri, "_SHADOW_DB_ENABLED", False)
    idx = tmp_path / "run_index.jsonl"
    append_run_created(_rec("h1"), path=idx)
    backfill_runs_to_db(path=idx)
    assert backfill_runs_to_db(path=idx) == 1   # re-run: still 1, no dup
    assert reconcile_runs_with_db(path=idx) == []


def test_backfill_absent_jsonl_noop(tmp_path):
    assert backfill_runs_to_db(path=tmp_path / "nope.jsonl") == 0
