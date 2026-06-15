"""Unit tests for the run-lineage JSONL index (M3-1).

TDD: these tests are written before the implementation and must initially fail
for the right reason (ImportError or AttributeError), then pass after implementation.

Run with:
    cd backend && python3 -m pytest tests/unit/test_run_index.py -v
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from app.pipeline.run_index import (
    RunIndexError,
    RunLineageRecord,
    append_run_created,
    append_run_updated,
    build_branch_tree,
    load_run_index,
    update_run_metadata,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record(
    run_id: str = "run-a",
    root_run_id: str | None = None,
    parent_run_id: str | None = None,
    *,
    status: str = "pending",
    created_at: float | None = None,
    run_dir: str | None = None,
    **kwargs,
) -> RunLineageRecord:
    return RunLineageRecord(
        run_id=run_id,
        root_run_id=root_run_id if root_run_id is not None else run_id,
        parent_run_id=parent_run_id,
        source_checkpoint_id=None,
        source_checkpoint_label=None,
        mode=None,
        feedback=None,
        title=None,
        status=status,
        run_dir=run_dir,
        created_at=created_at if created_at is not None else time.time(),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# append_run_created / load_run_index: basic round-trip
# ---------------------------------------------------------------------------

def test_append_created_writes_one_jsonl_line(tmp_path):
    index_path = tmp_path / "run_index.jsonl"
    rec = _record("r1")
    append_run_created(rec, path=index_path)

    lines = index_path.read_text().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["event"] == "created"
    assert data["run_id"] == "r1"
    assert data["root_run_id"] == "r1"
    assert data["status"] == "pending"


def test_append_created_creates_parent_dir(tmp_path):
    deep_path = tmp_path / "nested" / "dir" / "run_index.jsonl"
    append_run_created(_record("r1"), path=deep_path)
    assert deep_path.exists()


def test_load_run_index_returns_empty_dict_for_missing_file(tmp_path):
    missing = tmp_path / "no_such_file.jsonl"
    result = load_run_index(path=missing)
    assert result == {}


# ---------------------------------------------------------------------------
# created then updated: later events override earlier ones
# ---------------------------------------------------------------------------

def test_updated_overrides_created_fields(tmp_path):
    index_path = tmp_path / "idx.jsonl"
    rec = _record("r1", status="pending")
    append_run_created(rec, path=index_path)
    append_run_updated("r1", {"status": "running", "final_score": 0.85}, path=index_path)

    index = load_run_index(path=index_path)
    assert "r1" in index
    assert index["r1"].status == "running"
    assert index["r1"].final_score == 0.85


def test_updated_fills_in_run_dir_after_created_with_none(tmp_path):
    """A created record with run_dir=None is later completed by an updated."""
    index_path = tmp_path / "idx.jsonl"
    rec = _record("r2", status="pending", run_dir=None)
    append_run_created(rec, path=index_path)
    append_run_updated("r2", {"run_dir": "/some/path", "status": "running"}, path=index_path)

    index = load_run_index(path=index_path)
    assert index["r2"].run_dir == "/some/path"
    assert index["r2"].status == "running"


def test_multiple_updates_accumulate_latest_wins(tmp_path):
    index_path = tmp_path / "idx.jsonl"
    append_run_created(_record("r3", status="pending"), path=index_path)
    append_run_updated("r3", {"status": "running"}, path=index_path)
    append_run_updated("r3", {"status": "completed", "final_score": 0.9}, path=index_path)

    index = load_run_index(path=index_path)
    assert index["r3"].status == "completed"
    assert index["r3"].final_score == 0.9


# ---------------------------------------------------------------------------
# append_run_updated before created: best-effort record creation
# ---------------------------------------------------------------------------

def test_orphan_updated_creates_best_effort_record(tmp_path):
    """An updated for a run_id with no prior created should not crash."""
    index_path = tmp_path / "idx.jsonl"
    append_run_updated("orphan-1", {"status": "running", "final_score": 0.5}, path=index_path)

    index = load_run_index(path=index_path)
    assert "orphan-1" in index
    assert index["orphan-1"].status == "running"
    assert index["orphan-1"].root_run_id == "orphan-1"  # defaults to own run_id


# ---------------------------------------------------------------------------
# Malformed / blank lines are skipped without crashing
# ---------------------------------------------------------------------------

def test_malformed_lines_are_skipped(tmp_path):
    index_path = tmp_path / "idx.jsonl"
    append_run_created(_record("r-good"), path=index_path)

    # Inject garbage after the valid line.
    with index_path.open("a") as fh:
        fh.write("\n")               # blank line
        fh.write("not json at all\n")
        fh.write("{truncated\n")

    index = load_run_index(path=index_path)
    assert "r-good" in index
    assert len(index) == 1


# ---------------------------------------------------------------------------
# build_branch_tree
# ---------------------------------------------------------------------------

def _setup_branch_records(tmp_path: Path) -> Path:
    """Write a root + two children + one grandchild to the index."""
    idx = tmp_path / "idx.jsonl"
    t = 1_000_000.0

    root = _record("root-a", root_run_id="root-a", parent_run_id=None, created_at=t)
    child1 = _record("child-1", root_run_id="root-a", parent_run_id="root-a", created_at=t + 10)
    child2 = _record("child-2", root_run_id="root-a", parent_run_id="root-a", created_at=t + 20)
    grandchild = _record("grand-1", root_run_id="root-a", parent_run_id="child-1", created_at=t + 30)

    for rec in [root, child1, child2, grandchild]:
        append_run_created(rec, path=idx)
    return idx


def test_build_branch_tree_root_shape(tmp_path):
    idx = _setup_branch_records(tmp_path)
    records = load_run_index(path=idx)
    tree = build_branch_tree(records, "root-a")

    assert tree["run_id"] == "root-a"
    assert tree["root_run_id"] == "root-a"
    assert tree["parent_run_id"] is None
    assert "children" in tree
    assert isinstance(tree["children"], list)


def test_build_branch_tree_child_nested_under_parent(tmp_path):
    idx = _setup_branch_records(tmp_path)
    records = load_run_index(path=idx)
    tree = build_branch_tree(records, "root-a")

    child_ids = [c["run_id"] for c in tree["children"]]
    assert "child-1" in child_ids
    assert "child-2" in child_ids


def test_build_branch_tree_grandchild_nested_under_child(tmp_path):
    idx = _setup_branch_records(tmp_path)
    records = load_run_index(path=idx)
    tree = build_branch_tree(records, "root-a")

    child1_node = next(c for c in tree["children"] if c["run_id"] == "child-1")
    grandchild_ids = [c["run_id"] for c in child1_node["children"]]
    assert "grand-1" in grandchild_ids


def test_build_branch_tree_siblings_sorted_by_created_at(tmp_path):
    idx = _setup_branch_records(tmp_path)
    records = load_run_index(path=idx)
    tree = build_branch_tree(records, "root-a")

    child_ids = [c["run_id"] for c in tree["children"]]
    assert child_ids == ["child-1", "child-2"]  # child-1 created earlier


def test_build_branch_tree_lookup_by_child_id(tmp_path):
    """build_branch_tree accepts any run_id in the tree, not just root."""
    idx = _setup_branch_records(tmp_path)
    records = load_run_index(path=idx)
    tree = build_branch_tree(records, "child-1")

    # Should resolve to root and return same full tree.
    assert tree["run_id"] == "root-a"


def test_build_branch_tree_raises_on_unknown_run_id(tmp_path):
    idx = _setup_branch_records(tmp_path)
    records = load_run_index(path=idx)

    with pytest.raises((KeyError, RunIndexError)):
        build_branch_tree(records, "does-not-exist")


def test_build_branch_tree_node_has_required_fields(tmp_path):
    """All required TypeScript-interface fields must be present in nodes."""
    idx = _setup_branch_records(tmp_path)
    records = load_run_index(path=idx)
    tree = build_branch_tree(records, "root-a")

    required = {
        "run_id", "root_run_id", "parent_run_id",
        "source_checkpoint_id", "source_checkpoint_label",
        "title", "mode", "feedback",
        "status", "final_score", "created_at", "completed_at",
        "favorite", "children",
    }
    assert required.issubset(tree.keys()), f"Missing keys: {required - tree.keys()}"


def test_build_branch_tree_preserves_root_run_id_in_child(tmp_path):
    """A child record's root_run_id must equal the parent root's run_id."""
    idx = _setup_branch_records(tmp_path)
    records = load_run_index(path=idx)
    tree = build_branch_tree(records, "root-a")

    child1_node = next(c for c in tree["children"] if c["run_id"] == "child-1")
    assert child1_node["root_run_id"] == "root-a"


# ---------------------------------------------------------------------------
# update_run_metadata
# ---------------------------------------------------------------------------

def test_update_run_metadata_updates_title(tmp_path):
    idx = tmp_path / "idx.jsonl"
    append_run_created(_record("r5"), path=idx)
    result = update_run_metadata("r5", {"title": "My shader"}, path=idx)
    assert result.title == "My shader"

    # Verify it persisted.
    reloaded = load_run_index(path=idx)
    assert reloaded["r5"].title == "My shader"


def test_update_run_metadata_updates_favorite(tmp_path):
    idx = tmp_path / "idx.jsonl"
    append_run_created(_record("r6"), path=idx)
    result = update_run_metadata("r6", {"favorite": True}, path=idx)
    assert result.favorite is True

    reloaded = load_run_index(path=idx)
    assert reloaded["r6"].favorite is True


def test_update_run_metadata_updates_tags(tmp_path):
    idx = tmp_path / "idx.jsonl"
    append_run_created(_record("r7"), path=idx)
    result = update_run_metadata("r7", {"tags": ["fire", "ice"]}, path=idx)
    assert result.tags == ["fire", "ice"]

    reloaded = load_run_index(path=idx)
    assert reloaded["r7"].tags == ["fire", "ice"]


def test_update_run_metadata_rejects_status(tmp_path):
    idx = tmp_path / "idx.jsonl"
    append_run_created(_record("r8"), path=idx)
    with pytest.raises(ValueError):
        update_run_metadata("r8", {"status": "hacked"}, path=idx)


def test_update_run_metadata_rejects_parent_run_id(tmp_path):
    idx = tmp_path / "idx.jsonl"
    append_run_created(_record("r9"), path=idx)
    with pytest.raises(ValueError):
        update_run_metadata("r9", {"parent_run_id": "some-other"}, path=idx)


def test_update_run_metadata_rejects_root_run_id(tmp_path):
    idx = tmp_path / "idx.jsonl"
    append_run_created(_record("r10"), path=idx)
    with pytest.raises(ValueError):
        update_run_metadata("r10", {"root_run_id": "fake-root"}, path=idx)


def test_update_run_metadata_rejects_run_dir(tmp_path):
    idx = tmp_path / "idx.jsonl"
    append_run_created(_record("r11"), path=idx)
    with pytest.raises(ValueError):
        update_run_metadata("r11", {"run_dir": "/evil"}, path=idx)


def test_update_run_metadata_raises_for_unknown_run_id(tmp_path):
    idx = tmp_path / "idx.jsonl"
    with pytest.raises((KeyError, RunIndexError)):
        update_run_metadata("nonexistent", {"title": "x"}, path=idx)


def test_update_run_metadata_rejects_disallowed_mixed_with_allowed(tmp_path):
    """Even if some keys are allowed, one disallowed key rejects the whole patch."""
    idx = tmp_path / "idx.jsonl"
    append_run_created(_record("r12"), path=idx)
    with pytest.raises(ValueError):
        update_run_metadata("r12", {"title": "ok", "status": "bad"}, path=idx)


def test_update_run_metadata_empty_patch_does_not_write(tmp_path):
    idx = tmp_path / "idx.jsonl"
    append_run_created(_record("r-noop"), path=idx)
    lines_before = idx.read_text().splitlines()
    result = update_run_metadata("r-noop", {}, path=idx)
    assert idx.read_text().splitlines() == lines_before
    assert result.run_id == "r-noop"


# ---------------------------------------------------------------------------
# Thread-safety smoke test (not a real concurrency test — just verifies
# multiple sequential writes all land in the file)
# ---------------------------------------------------------------------------

def test_multiple_appends_all_land(tmp_path):
    idx = tmp_path / "idx.jsonl"
    for i in range(10):
        append_run_created(_record(f"run-{i}", created_at=float(i)), path=idx)

    lines = idx.read_text().splitlines()
    assert len(lines) == 10

    index = load_run_index(path=idx)
    assert len(index) == 10
