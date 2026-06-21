"""Targeted reads: load_run (single, indexed) and load_run_family (by root,
indexed) avoid scanning the whole index to build a branch tree / fetch one run."""
from p2s_agent.orchestration.run_index import (
    RunLineageRecord, append_run_created, load_run, load_run_family)


def _rec(run_id, root=None, parent=None, status="pending"):
    return RunLineageRecord(run_id=run_id, root_run_id=root or run_id, parent_run_id=parent,
        source_checkpoint_id=None, source_checkpoint_label=None, mode=None, feedback=None,
        title=None, status=status, run_dir=None, created_at=1.0)


def test_get_runs_by_root_filters_to_one_family(tmp_path):
    from p2s_agent.core.db.engine import get_engine, init_db
    from p2s_agent.core.db.repositories import runs as r
    eng = get_engine(tmp_path)
    init_db(eng)
    r.upsert_run(eng, {"run_id": "a", "root_run_id": "R1", "created_at": 1.0})
    r.upsert_run(eng, {"run_id": "b", "root_run_id": "R1", "created_at": 2.0})
    r.upsert_run(eng, {"run_id": "c", "root_run_id": "R2", "created_at": 3.0})
    assert set(r.get_runs_by_root(eng, "R1")) == {"a", "b"}


def test_load_run_single(tmp_path):
    idx = tmp_path / "run_index.jsonl"
    append_run_created(_rec("x"), path=idx)
    rec = load_run("x", path=idx)
    assert rec is not None and rec.run_id == "x"
    assert load_run("nope", path=idx) is None


def test_load_run_family_returns_only_that_root(tmp_path):
    idx = tmp_path / "run_index.jsonl"
    append_run_created(_rec("a", root="R1"), path=idx)
    append_run_created(_rec("b", root="R1", parent="a"), path=idx)
    append_run_created(_rec("c", root="R2"), path=idx)
    fam = load_run_family("a", path=idx)  # resolve a's root (R1), query family
    assert set(fam) == {"a", "b"}
    # passing a non-root member still resolves to the whole family
    assert set(load_run_family("b", path=idx)) == {"a", "b"}
