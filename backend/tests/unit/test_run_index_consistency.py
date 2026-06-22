"""Regression tests for DB/JSONL read-consistency bugs found in review:
a run present in the JSONL but missing from the DB must NOT disappear from
reads, and prune must be able to remove a DB-only run."""
import p2s_agent.orchestration.run_index as ri
from p2s_agent.orchestration.run_index import (
    RunLineageRecord, append_run_created, load_run_index, load_run,
    load_run_family, build_branch_tree, prune_run_index)


def _rec(rid, root=None, parent=None, status="completed"):
    return RunLineageRecord(rid, root or rid, parent, None, None, None, None, None,
                            status, None, 1.0)


def test_jsonl_only_run_recovered_by_union_reads(tmp_path, monkeypatch):
    """A run whose shadow write was swallowed (JSONL written, DB not) must still
    appear in load_run_index (union) and load_run (per-id fold fallback)."""
    idx = tmp_path / "run_index.jsonl"
    append_run_created(_rec("R1"), path=idx)                          # JSONL + DB
    monkeypatch.setattr(ri, "_SHADOW_DB_ENABLED", False)              # simulate swallowed DB write
    append_run_created(_rec("C1", root="R1", parent="R1"), path=idx)  # JSONL only
    monkeypatch.setattr(ri, "_SHADOW_DB_ENABLED", True)

    assert set(load_run_index(path=idx)) == {"R1", "C1"}             # union recovers it
    got = load_run("C1", path=idx)
    assert got is not None and got.run_id == "C1"                    # per-id fold fallback


def test_load_run_family_fallback_on_anchor_db_miss(tmp_path, monkeypatch):
    """When the anchor run isn't in the DB, load_run_family folds the JSONL so the
    branch tree is still complete."""
    idx = tmp_path / "run_index.jsonl"
    monkeypatch.setattr(ri, "_SHADOW_DB_ENABLED", False)             # whole family JSONL-only
    append_run_created(_rec("ROOT"), path=idx)
    append_run_created(_rec("CH", root="ROOT", parent="ROOT"), path=idx)
    monkeypatch.setattr(ri, "_SHADOW_DB_ENABLED", True)
    fam = load_run_family("CH", path=idx)
    assert {"ROOT", "CH"}.issubset(set(fam))
    tree = build_branch_tree(load_run_family("ROOT", path=idx), "ROOT")
    assert any(c["run_id"] == "CH" for c in tree["children"])


def test_load_run_family_includes_jsonl_only_sibling(tmp_path, monkeypatch):
    """The branch tree must include a sibling that's in the JSONL but missing
    from the DB (anchor IS in the DB) — closes the by-root residual."""
    idx = tmp_path / "run_index.jsonl"
    append_run_created(_rec("R1"), path=idx)                          # JSONL + DB
    monkeypatch.setattr(ri, "_SHADOW_DB_ENABLED", False)
    append_run_created(_rec("C1", root="R1", parent="R1"), path=idx)  # JSONL only
    monkeypatch.setattr(ri, "_SHADOW_DB_ENABLED", True)
    fam = load_run_family("R1", path=idx)                             # anchor R1 in DB
    assert {"R1", "C1"}.issubset(set(fam))                            # union closes the gap
    tree = build_branch_tree(load_run_family("R1", path=idx), "R1")
    assert any(c["run_id"] == "C1" for c in tree["children"])


def test_prune_removes_db_only_run(tmp_path):
    """A run that exists only in the DB (not in the JSONL) must be prunable."""
    from p2s_agent.core.db.repositories import runs as r
    idx = tmp_path / "run_index.jsonl"
    r.upsert_run(ri._shadow_engine(idx),
                 {"run_id": "D1", "root_run_id": "D1", "created_at": 1.0})
    assert "D1" in load_run_index(path=idx)
    prune_run_index({"D1"}, path=idx)
    assert "D1" not in load_run_index(path=idx)   # gone from the DB too
