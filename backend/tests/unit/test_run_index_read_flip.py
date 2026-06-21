"""Read-cutover: load_run_index reads the SQLite runs table, falling back to
folding the JSONL only when the DB is empty/unavailable."""
from p2s_agent.orchestration.run_index import (
    RunLineageRecord, append_run_created, load_run_index)


def _rec(run_id, status="running"):
    return RunLineageRecord(run_id=run_id, root_run_id=run_id, parent_run_id=None,
        source_checkpoint_id=None, source_checkpoint_label=None, mode=None, feedback=None,
        title=None, status=status, run_dir=None, created_at=1.0)


def test_load_reads_from_db_without_touching_jsonl(tmp_path):
    """A run present only in the DB is returned; the JSONL is never read/created."""
    from p2s_agent.core.db.engine import get_engine, init_db
    from p2s_agent.core.db.repositories import runs as runs_repo
    eng = get_engine(tmp_path)
    init_db(eng)
    runs_repo.upsert_run(eng, {"run_id": "db1", "root_run_id": "db1",
                               "created_at": 1.0, "status": "completed"})
    idx = tmp_path / "run_index.jsonl"  # never created
    got = load_run_index(path=idx)
    assert "db1" in got and got["db1"].status == "completed"
    assert not idx.exists()  # JSONL untouched — the read came from the DB


def test_load_falls_back_to_jsonl_fold_when_db_empty(tmp_path, monkeypatch):
    """With the shadow off the DB stays empty, so load folds the JSONL."""
    import p2s_agent.orchestration.run_index as ri
    monkeypatch.setattr(ri, "_SHADOW_DB_ENABLED", False)  # no mirror → DB empty
    idx = tmp_path / "run_index.jsonl"
    append_run_created(_rec("j1", status="running"), path=idx)  # JSONL only
    got = load_run_index(path=idx)  # DB empty → fold JSONL fallback
    assert got["j1"].status == "running"
