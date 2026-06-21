"""Shadow dual-write: run_index also mirrors into SQLite (additive).

JSONL stays authoritative; these tests assert the shadow runs table is
populated and that a shadow failure never breaks the JSONL path.
"""
from p2s_agent.orchestration.run_index import (
    RunLineageRecord,
    append_run_created,
    append_run_updated,
    load_run_index,
)


def _rec(run_id="r1", status="pending"):
    return RunLineageRecord(
        run_id=run_id, root_run_id=run_id, parent_run_id=None,
        source_checkpoint_id=None, source_checkpoint_label=None, mode=None,
        feedback=None, title=None, status=status, run_dir=None, created_at=1.0,
        tags=["a"], source_run_ids=["s1"],
    )


def test_shadow_created_populates_runs_table(tmp_path):
    from p2s_agent.core.db.engine import get_engine
    from p2s_agent.core.db.repositories import runs as runs_repo
    idx = tmp_path / "run_index.jsonl"
    append_run_created(_rec("r1"), path=idx)
    # JSONL still authoritative
    assert idx.exists()
    assert load_run_index(path=idx)["r1"].run_id == "r1"
    # shadow DB (beside the JSONL) populated
    row = runs_repo.get_run(get_engine(tmp_path), "r1")
    assert row is not None and row["run_id"] == "r1"
    assert row["tags"] == ["a"] and row["source_run_ids"] == ["s1"]


def test_shadow_update_mirrors_fields(tmp_path):
    from p2s_agent.core.db.engine import get_engine
    from p2s_agent.core.db.repositories import runs as runs_repo
    idx = tmp_path / "run_index.jsonl"
    append_run_created(_rec("r2", status="pending"), path=idx)
    append_run_updated("r2", {"status": "completed", "final_score": 0.9}, path=idx)
    row = runs_repo.get_run(get_engine(tmp_path), "r2")
    assert row["status"] == "completed" and row["final_score"] == 0.9


def test_shadow_failure_never_breaks_jsonl(tmp_path, monkeypatch):
    # Force the shadow engine to raise; JSONL write+read must still work.
    import p2s_agent.orchestration.run_index as ri
    monkeypatch.setattr(ri, "_shadow_engine",
                        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")))
    idx = tmp_path / "run_index.jsonl"
    append_run_created(_rec("r3"), path=idx)
    assert idx.exists()
    assert load_run_index(path=idx)["r3"].run_id == "r3"  # JSONL unaffected


def test_shadow_disabled_flag_skips_db(tmp_path, monkeypatch):
    import p2s_agent.orchestration.run_index as ri
    from p2s_agent.core.db.engine import get_engine, init_db
    from p2s_agent.core.db.repositories import runs as runs_repo
    monkeypatch.setattr(ri, "_SHADOW_DB_ENABLED", False)
    init_db(get_engine(tmp_path))  # table exists, but shadow disabled
    idx = tmp_path / "run_index.jsonl"
    append_run_created(_rec("r4"), path=idx)
    assert runs_repo.get_run(get_engine(tmp_path), "r4") is None  # not mirrored
    assert idx.exists()  # JSONL still written
