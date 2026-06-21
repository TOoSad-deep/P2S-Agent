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

from p2s_agent.orchestration.run_index import (
    SCHEMA_VERSION,
    RunIndexError,
    RunLineageRecord,
    append_run_created,
    append_run_updated,
    build_branch_tree,
    compact_run_index,
    load_run_index,
    maybe_compact_run_index,
    prune_run_index,
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


# ---------------------------------------------------------------------------
# V3 variant fields: RunLineageRecord, fold, and build_branch_tree node
# ---------------------------------------------------------------------------

def test_variant_fields_survive_created_fold(tmp_path):
    """A created record with variant fields folds back with all three preserved."""
    idx = tmp_path / "idx.jsonl"
    rec = _record(
        "v-run-1",
        variant_group_id="group_1",
        variant_index=2,
        variant_label="lighting_color",
    )
    append_run_created(rec, path=idx)

    index = load_run_index(path=idx)
    assert "v-run-1" in index
    r = index["v-run-1"]
    assert r.variant_group_id == "group_1"
    assert r.variant_index == 2
    assert r.variant_label == "lighting_color"


def test_non_variant_record_folds_with_none_variant_fields(tmp_path):
    """A record without variant fields folds with all three as None (back-compat)."""
    idx = tmp_path / "idx.jsonl"
    rec = _record("plain-run")
    append_run_created(rec, path=idx)

    index = load_run_index(path=idx)
    assert "plain-run" in index
    r = index["plain-run"]
    assert r.variant_group_id is None
    assert r.variant_index is None
    assert r.variant_label is None


def test_variant_fields_survive_updated_merge(tmp_path):
    """An updated event carrying variant_group_id merges correctly."""
    idx = tmp_path / "idx.jsonl"
    rec = _record("v-run-2")
    append_run_created(rec, path=idx)
    append_run_updated(
        "v-run-2",
        {"variant_group_id": "group_x", "variant_index": 0, "variant_label": "baseline"},
        path=idx,
    )

    index = load_run_index(path=idx)
    r = index["v-run-2"]
    assert r.variant_group_id == "group_x"
    assert r.variant_index == 0
    assert r.variant_label == "baseline"


def test_build_branch_tree_variant_node_fields(tmp_path):
    """build_branch_tree node for a variant child includes the three variant fields."""
    idx = tmp_path / "idx.jsonl"
    t = 2_000_000.0
    root = _record("vroot", root_run_id="vroot", parent_run_id=None, created_at=t)
    variant_child = _record(
        "vchild-1",
        root_run_id="vroot",
        parent_run_id="vroot",
        created_at=t + 10,
        variant_group_id="group_1",
        variant_index=2,
        variant_label="lighting_color",
    )
    for rec in [root, variant_child]:
        append_run_created(rec, path=idx)

    records = load_run_index(path=idx)
    tree = build_branch_tree(records, "vroot")

    child_node = next(c for c in tree["children"] if c["run_id"] == "vchild-1")
    assert child_node["variant_group_id"] == "group_1"
    assert child_node["variant_index"] == 2
    assert child_node["variant_label"] == "lighting_color"


def test_build_branch_tree_non_variant_node_has_none_variant_fields(tmp_path):
    """build_branch_tree node for a non-variant run has all three variant fields as None."""
    idx = tmp_path / "idx.jsonl"
    t = 3_000_000.0
    root = _record("nvroot", root_run_id="nvroot", parent_run_id=None, created_at=t)
    child = _record("nvchild", root_run_id="nvroot", parent_run_id="nvroot", created_at=t + 5)
    for rec in [root, child]:
        append_run_created(rec, path=idx)

    records = load_run_index(path=idx)
    tree = build_branch_tree(records, "nvroot")

    child_node = next(c for c in tree["children"] if c["run_id"] == "nvchild")
    assert child_node["variant_group_id"] is None
    assert child_node["variant_index"] is None
    assert child_node["variant_label"] is None


# ---------------------------------------------------------------------------
# V3.5 draw-session lineage fields: RunLineageRecord, fold, and branch tree
# ---------------------------------------------------------------------------


def test_draw_session_lineage_fields_survive_created_fold(tmp_path):
    """A record with all three draw-session lineage fields folds back correctly."""
    idx = tmp_path / "idx.jsonl"
    rec = _record(
        "ds-run-1",
        draw_session_id="draw_session_42",
        draw_card_index=3,
        replacement_of_run_id="old-run-7",
    )
    append_run_created(rec, path=idx)

    index = load_run_index(path=idx)
    assert "ds-run-1" in index
    r = index["ds-run-1"]
    assert r.draw_session_id == "draw_session_42"
    assert r.draw_card_index == 3
    assert r.replacement_of_run_id == "old-run-7"


def test_draw_card_index_int_coercion_from_string(tmp_path):
    """draw_card_index stored as string in JSONL is coerced back to int on load."""
    idx = tmp_path / "idx.jsonl"
    rec = _record("ds-run-coerce")
    append_run_created(rec, path=idx)
    # Simulate a JSONL line where draw_card_index is serialised as a string.
    with idx.open("a", encoding="utf-8") as fh:
        import json as _json
        fh.write(_json.dumps({"event": "updated", "run_id": "ds-run-coerce", "draw_card_index": "5"}) + "\n")

    index = load_run_index(path=idx)
    assert index["ds-run-coerce"].draw_card_index == 5
    assert isinstance(index["ds-run-coerce"].draw_card_index, int)


def test_draw_session_lineage_defaults_to_none(tmp_path):
    """A plain record (no draw fields) folds with all three as None."""
    idx = tmp_path / "idx.jsonl"
    rec = _record("plain-ds-run")
    append_run_created(rec, path=idx)

    index = load_run_index(path=idx)
    r = index["plain-ds-run"]
    assert r.draw_session_id is None
    assert r.draw_card_index is None
    assert r.replacement_of_run_id is None


def test_build_branch_tree_draw_session_node_fields(tmp_path):
    """build_branch_tree node for a draw-session child includes all three draw fields."""
    idx = tmp_path / "idx.jsonl"
    t = 4_000_000.0
    root = _record("ds-root", root_run_id="ds-root", parent_run_id=None, created_at=t)
    ds_child = _record(
        "ds-child-1",
        root_run_id="ds-root",
        parent_run_id="ds-root",
        created_at=t + 10,
        draw_session_id="draw_session_42",
        draw_card_index=3,
        replacement_of_run_id="old-run-7",
    )
    for rec in [root, ds_child]:
        append_run_created(rec, path=idx)

    records = load_run_index(path=idx)
    tree = build_branch_tree(records, "ds-root")

    child_node = next(c for c in tree["children"] if c["run_id"] == "ds-child-1")
    assert child_node["draw_session_id"] == "draw_session_42"
    assert child_node["draw_card_index"] == 3
    assert child_node["replacement_of_run_id"] == "old-run-7"


def test_build_branch_tree_non_draw_session_node_has_none_draw_fields(tmp_path):
    """build_branch_tree node for a non-draw-session run has all three draw fields as None."""
    idx = tmp_path / "idx.jsonl"
    t = 5_000_000.0
    root = _record("nds-root", root_run_id="nds-root", parent_run_id=None, created_at=t)
    child = _record("nds-child", root_run_id="nds-root", parent_run_id="nds-root", created_at=t + 5)
    for rec in [root, child]:
        append_run_created(rec, path=idx)

    records = load_run_index(path=idx)
    tree = build_branch_tree(records, "nds-root")

    child_node = next(c for c in tree["children"] if c["run_id"] == "nds-child")
    assert child_node["draw_session_id"] is None
    assert child_node["draw_card_index"] is None
    assert child_node["replacement_of_run_id"] is None


# ---------------------------------------------------------------------------
# V4.5 fusion lineage fields: RunLineageRecord, fold, and branch tree
# ---------------------------------------------------------------------------


def test_fusion_lineage_fields_survive_created_fold(tmp_path):
    """A record with all three fusion lineage fields folds back correctly."""
    idx = tmp_path / "idx.jsonl"
    rec = _record(
        "fusion-run-1",
        fusion_id="fusion_abc123",
        base_run_id="run_base",
        source_run_ids=["run_src_a", "run_src_b"],
    )
    append_run_created(rec, path=idx)

    index = load_run_index(path=idx)
    assert "fusion-run-1" in index
    r = index["fusion-run-1"]
    assert r.fusion_id == "fusion_abc123"
    assert r.base_run_id == "run_base"
    assert r.source_run_ids == ["run_src_a", "run_src_b"]


def test_fusion_lineage_defaults_when_absent(tmp_path):
    """A plain record (no fusion fields) folds with None / empty list defaults."""
    idx = tmp_path / "idx.jsonl"
    rec = _record("plain-fusion-run")
    append_run_created(rec, path=idx)

    index = load_run_index(path=idx)
    r = index["plain-fusion-run"]
    assert r.fusion_id is None
    assert r.base_run_id is None
    assert r.source_run_ids == []


def test_build_branch_tree_fusion_node_fields(tmp_path):
    """build_branch_tree node for a fusion child surfaces all three fusion fields."""
    idx = tmp_path / "idx.jsonl"
    t = 6_000_000.0
    root = _record("f-root", root_run_id="f-root", parent_run_id=None, created_at=t)
    fusion_child = _record(
        "f-child-1",
        root_run_id="f-root",
        parent_run_id="f-root",
        created_at=t + 10,
        fusion_id="fusion_abc123",
        base_run_id="f-root",
        source_run_ids=["run_src_a", "run_src_b"],
    )
    for rec in [root, fusion_child]:
        append_run_created(rec, path=idx)

    records = load_run_index(path=idx)
    tree = build_branch_tree(records, "f-root")

    child_node = next(c for c in tree["children"] if c["run_id"] == "f-child-1")
    assert child_node["fusion_id"] == "fusion_abc123"
    assert child_node["base_run_id"] == "f-root"
    assert child_node["source_run_ids"] == ["run_src_a", "run_src_b"]


def test_build_branch_tree_non_fusion_node_has_default_fusion_fields(tmp_path):
    """build_branch_tree node for a non-fusion run has None / empty fusion fields."""
    idx = tmp_path / "idx.jsonl"
    t = 7_000_000.0
    root = _record("nf-root", root_run_id="nf-root", parent_run_id=None, created_at=t)
    child = _record("nf-child", root_run_id="nf-root", parent_run_id="nf-root", created_at=t + 5)
    for rec in [root, child]:
        append_run_created(rec, path=idx)

    records = load_run_index(path=idx)
    tree = build_branch_tree(records, "nf-root")

    child_node = next(c for c in tree["children"] if c["run_id"] == "nf-child")
    assert child_node["fusion_id"] is None
    assert child_node["base_run_id"] is None
    assert child_node["source_run_ids"] == []


# ---------------------------------------------------------------------------
# Performance: load_run_index caches the folded result by file mtime+size
# (Task 1 — perf/observability)
# ---------------------------------------------------------------------------


def test_load_run_index_caches_when_file_unchanged(tmp_path, monkeypatch):
    """Two consecutive loads with no file change must not re-read the file.

    We spy on json.loads (called once per JSONL line during a fold). The first
    load parses the lines; the second load with an unchanged file must reuse the
    cached fold and therefore parse nothing.
    """
    import p2s_agent.orchestration.run_index as ri

    idx = tmp_path / "idx.jsonl"
    append_run_created(_record("c1", status="running"), path=idx)
    append_run_created(_record("c2", status="running"), path=idx)

    calls = {"n": 0}
    real_loads = ri.json.loads

    def _counting_loads(s, *a, **kw):
        calls["n"] += 1
        return real_loads(s, *a, **kw)

    monkeypatch.setattr(ri.json, "loads", _counting_loads)

    first = load_run_index(path=idx)
    parsed_after_first = calls["n"]
    assert parsed_after_first >= 2  # at least one parse per JSONL line

    second = load_run_index(path=idx)
    # No file change => cache hit => zero additional parses.
    assert calls["n"] == parsed_after_first
    assert second == first


def test_load_run_index_refolds_after_append(tmp_path):
    """After an append (mtime/size change), the next load reflects the new record."""
    idx = tmp_path / "idx.jsonl"
    append_run_created(_record("a1", status="running"), path=idx)

    first = load_run_index(path=idx)
    assert set(first.keys()) == {"a1"}

    # Append a new record; size (and usually mtime) change.
    append_run_created(_record("a2", status="running"), path=idx)

    second = load_run_index(path=idx)
    assert set(second.keys()) == {"a1", "a2"}


def test_load_run_index_cache_isolated_per_path(tmp_path):
    """Caching must be keyed per path so two different files don't collide."""
    idx_a = tmp_path / "a.jsonl"
    idx_b = tmp_path / "b.jsonl"
    append_run_created(_record("only-a", status="running"), path=idx_a)
    append_run_created(_record("only-b", status="running"), path=idx_b)

    res_a = load_run_index(path=idx_a)
    res_b = load_run_index(path=idx_b)

    assert set(res_a.keys()) == {"only-a"}
    assert set(res_b.keys()) == {"only-b"}


def test_load_run_index_cached_result_is_not_mutated_by_caller(tmp_path):
    """A caller mutating the returned dict must not corrupt the cache."""
    idx = tmp_path / "idx.jsonl"
    append_run_created(_record("m1", status="running"), path=idx)

    first = load_run_index(path=idx)
    first["injected"] = first["m1"]  # mutate the returned mapping

    second = load_run_index(path=idx)  # cache hit, file unchanged
    assert "injected" not in second
    assert set(second.keys()) == {"m1"}


# ---------------------------------------------------------------------------
# Observability: malformed JSONL lines are logged, not silently dropped
# (Task 2)
# ---------------------------------------------------------------------------


def test_malformed_line_emits_warning(tmp_path, caplog):
    """A malformed JSONL line is logged as a warning and skipped, valid records kept."""
    import logging

    idx = tmp_path / "idx.jsonl"
    append_run_created(_record("good-1", status="running"), path=idx)
    with idx.open("a", encoding="utf-8") as fh:
        fh.write("{this is not valid json\n")

    with caplog.at_level(logging.WARNING, logger="p2s_agent.orchestration.run_index"):
        index = load_run_index(path=idx)

    assert "good-1" in index
    assert len(index) == 1
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "expected a warning for the malformed line"
    # The warning should help locate the corruption (line number or snippet).
    joined = " ".join(r.getMessage() for r in warnings)
    assert "run_index" in joined or "malformed" in joined.lower() or "line" in joined.lower()


# ---------------------------------------------------------------------------
# Terminal-status stickiness in the fold (Task 3)
# ---------------------------------------------------------------------------


def test_terminal_status_not_regressed_by_stale_updated(tmp_path):
    """[created(running), updated(completed), updated(running)] => completed (terminal)."""
    idx = tmp_path / "idx.jsonl"
    append_run_created(_record("t1", status="running"), path=idx)
    append_run_updated("t1", {"status": "completed", "final_score": 0.9}, path=idx)
    append_run_updated("t1", {"status": "running"}, path=idx)  # stale / out-of-order

    index = load_run_index(path=idx)
    assert index["t1"].status == "completed"
    # Non-status fields in the stale event are still allowed to merge.


def test_failed_terminal_status_sticks(tmp_path):
    """A failed run is not flipped back to running by a stale updated."""
    idx = tmp_path / "idx.jsonl"
    append_run_created(_record("t2", status="running"), path=idx)
    append_run_updated("t2", {"status": "failed"}, path=idx)
    append_run_updated("t2", {"status": "running"}, path=idx)

    index = load_run_index(path=idx)
    assert index["t2"].status == "failed"


def test_cancelled_terminal_status_sticks(tmp_path):
    """A cancelled run is not flipped back to running by a stale updated."""
    idx = tmp_path / "idx.jsonl"
    append_run_created(_record("t3", status="running"), path=idx)
    append_run_updated("t3", {"status": "cancelled"}, path=idx)
    append_run_updated("t3", {"status": "running"}, path=idx)

    index = load_run_index(path=idx)
    assert index["t3"].status == "cancelled"


def test_terminal_status_allows_non_status_field_updates(tmp_path):
    """A post-terminal updated may still patch non-status fields (e.g. title)."""
    idx = tmp_path / "idx.jsonl"
    append_run_created(_record("t4", status="running"), path=idx)
    append_run_updated("t4", {"status": "completed"}, path=idx)
    append_run_updated("t4", {"status": "running", "title": "late title"}, path=idx)

    index = load_run_index(path=idx)
    assert index["t4"].status == "completed"  # status sticks
    assert index["t4"].title == "late title"  # other fields still merge


def test_non_terminal_status_still_progresses(tmp_path):
    """Non-terminal -> non-terminal transitions are unaffected (pending -> running)."""
    idx = tmp_path / "idx.jsonl"
    append_run_created(_record("t5", status="pending"), path=idx)
    append_run_updated("t5", {"status": "running"}, path=idx)

    index = load_run_index(path=idx)
    assert index["t5"].status == "running"


# ---------------------------------------------------------------------------
# Task 8: JSONL schema version + atomic compaction / rotation
# ---------------------------------------------------------------------------


def test_created_line_has_schema_version(tmp_path):
    """Every created line carries the current schema version under "v"."""
    p = tmp_path / "idx.jsonl"
    append_run_created(_record("sv-1"), path=p)

    first_line = p.read_text().splitlines()[0]
    data = json.loads(first_line)
    assert data["v"] == SCHEMA_VERSION


def test_updated_line_has_schema_version(tmp_path):
    """Every updated line carries the current schema version under "v"."""
    p = tmp_path / "idx.jsonl"
    append_run_created(_record("sv-2"), path=p)
    append_run_updated("sv-2", {"status": "running"}, path=p)

    last_line = p.read_text().splitlines()[-1]
    data = json.loads(last_line)
    assert data["event"] == "updated"
    assert data["v"] == SCHEMA_VERSION


def test_fold_tolerates_legacy_line_without_version(tmp_path):
    """A legacy created line with no "v" key folds identically (back-compat)."""
    p = tmp_path / "idx.jsonl"
    legacy = {
        "event": "created",
        "run_id": "legacy-1",
        "root_run_id": "legacy-1",
        "status": "completed",
        "created_at": 100.0,
        "final_score": 0.7,
    }
    with p.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(legacy) + "\n")

    index = load_run_index(path=p)
    assert "legacy-1" in index
    assert index["legacy-1"].status == "completed"
    assert index["legacy-1"].final_score == 0.7


def test_fold_warns_on_future_version_but_still_folds(tmp_path, caplog):
    """A created line with a future schema version warns but is still folded."""
    import logging

    p = tmp_path / "idx.jsonl"
    future = {
        "event": "created",
        "v": 999,
        "run_id": "future-1",
        "root_run_id": "future-1",
        "status": "running",
        "created_at": 200.0,
    }
    with p.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(future) + "\n")

    with caplog.at_level(logging.WARNING, logger="p2s_agent.orchestration.run_index"):
        index = load_run_index(path=p)

    assert "future-1" in index
    assert index["future-1"].status == "running"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "expected a warning for the future schema version"
    joined = " ".join(r.getMessage() for r in warnings)
    assert "future schema version" in joined or "999" in joined


def test_compact_is_fold_equivalent_and_shrinks(tmp_path):
    """compact collapses N append events into 1 line per run, fold-equivalent."""
    p = tmp_path / "idx.jsonl"
    append_run_created(_record("rc1", status="pending"), path=p)
    append_run_updated("rc1", {"status": "running"}, path=p)
    append_run_updated("rc1", {"status": "completed", "final_score": 0.91}, path=p)
    append_run_created(_record("rc2", status="pending"), path=p)
    append_run_updated("rc2", {"status": "running", "final_score": 0.5}, path=p)

    before = load_run_index(path=p)
    assert set(before.keys()) == {"rc1", "rc2"}

    n = compact_run_index(path=p)
    assert n == 2

    lines = p.read_text().splitlines()
    assert len(lines) == 2
    # every surviving line is a "created" event with the schema version
    for line in lines:
        data = json.loads(line)
        assert data["event"] == "created"
        assert data["v"] == SCHEMA_VERSION

    after = load_run_index(path=p)
    assert set(after.keys()) == set(before.keys())
    for rid in before:
        assert after[rid].status == before[rid].status
        assert after[rid].final_score == before[rid].final_score
        assert after[rid].root_run_id == before[rid].root_run_id
        assert after[rid].created_at == before[rid].created_at

    # a single .bak snapshot of the prior file exists
    bak = p.with_suffix(p.suffix + ".bak")
    assert bak.exists()


def test_compact_no_op_for_missing_file(tmp_path):
    """compact on an absent file returns 0 and creates nothing."""
    missing = tmp_path / "no_such.jsonl"
    assert compact_run_index(path=missing) == 0
    assert not missing.exists()


def test_compact_invalidates_cache(tmp_path):
    """After compaction, load returns the compacted fold, not a stale cache."""
    p = tmp_path / "idx.jsonl"
    append_run_created(_record("ic1", status="pending"), path=p)
    append_run_updated("ic1", {"status": "completed", "final_score": 0.8}, path=p)

    before = load_run_index(path=p)  # populate cache
    compact_run_index(path=p)
    after = load_run_index(path=p)

    assert set(after.keys()) == set(before.keys())
    assert after["ic1"].status == before["ic1"].status
    assert after["ic1"].final_score == before["ic1"].final_score


def test_maybe_compact_respects_threshold(tmp_path):
    """maybe_compact is a no-op below min_lines, compacts when bloated."""
    p = tmp_path / "idx.jsonl"
    # 1 created + many updated events for a single run => high lines:runs ratio.
    append_run_created(_record("mc1", status="pending"), path=p)
    for i in range(20):
        append_run_updated("mc1", {"status": "running", "final_score": float(i)}, path=p)

    lines_before = len(p.read_text().splitlines())
    assert lines_before == 21

    # Below min_lines: no compaction, file untouched.
    assert maybe_compact_run_index(path=p, min_lines=10_000) is False
    assert len(p.read_text().splitlines()) == lines_before

    # Low min_lines + bloat ratio the data exceeds: compaction happens.
    assert maybe_compact_run_index(path=p, min_lines=5, bloat_ratio=3.0) is True
    assert len(p.read_text().splitlines()) == 1

    index = load_run_index(path=p)
    assert set(index.keys()) == {"mc1"}
    assert index["mc1"].final_score == 19.0


def test_maybe_compact_no_op_for_missing_file(tmp_path):
    """maybe_compact on an absent file is False and never raises."""
    missing = tmp_path / "no_such.jsonl"
    assert maybe_compact_run_index(path=missing) is False


# ---------------------------------------------------------------------------
# Graceful degradation: read path must not propagate PermissionError (OSError)
# ---------------------------------------------------------------------------


def test_load_run_index_empty_on_permission_error_during_read(tmp_path, monkeypatch):
    """A PermissionError on the underlying open() degrades to an empty index.

    A uvicorn --reload worker on macOS can lose TCC access to ~/Documents; every
    open() under it then raises PermissionError (errno 1). The read path must
    return the documented empty value instead of propagating into an HTTP 500.
    """
    idx = tmp_path / "idx.jsonl"
    append_run_created(_record("perm-read-1", status="running"), path=idx)

    real_open = Path.open

    def _denying_open(self, *args, **kwargs):
        if self == idx:
            raise PermissionError(1, "Operation not permitted")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", _denying_open)

    assert load_run_index(path=idx) == {}


def test_load_run_index_empty_on_permission_error_during_stat(tmp_path, monkeypatch):
    """A PermissionError on stat() degrades to an empty index instead of a 500."""
    idx = tmp_path / "idx.jsonl"
    append_run_created(_record("perm-stat-1", status="running"), path=idx)

    real_stat = Path.stat

    def _denying_stat(self, *args, **kwargs):
        if self == idx:
            raise PermissionError(1, "Operation not permitted")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", _denying_stat)

    assert load_run_index(path=idx) == {}


# ---------------------------------------------------------------------------
# prune_run_index: drop specific run_ids (retention cleanup)
# ---------------------------------------------------------------------------


def test_prune_removes_only_named_runs(tmp_path):
    """prune drops the named run_ids and leaves the rest fold-equivalent."""
    p = tmp_path / "idx.jsonl"
    append_run_created(_record("pr1", status="completed", created_at=1.0), path=p)
    append_run_created(_record("pr2", status="completed", created_at=2.0), path=p)
    append_run_created(_record("pr3", status="completed", created_at=3.0), path=p)

    removed = prune_run_index({"pr1", "pr3"}, path=p)

    assert removed == 2
    after = load_run_index(path=p)
    assert set(after.keys()) == {"pr2"}
    # surviving lines are folded "created" events with the schema version
    for line in p.read_text().splitlines():
        data = json.loads(line)
        assert data["event"] == "created"
        assert data["v"] == SCHEMA_VERSION


def test_prune_ignores_unknown_run_ids(tmp_path):
    """run_ids not present in the index contribute 0 to the removed count."""
    p = tmp_path / "idx.jsonl"
    append_run_created(_record("pk1", status="completed"), path=p)

    removed = prune_run_index({"nope", "missing"}, path=p)

    assert removed == 0
    assert set(load_run_index(path=p).keys()) == {"pk1"}


def test_prune_empty_set_is_noop(tmp_path):
    """An empty run_id set writes nothing and returns 0."""
    p = tmp_path / "idx.jsonl"
    append_run_created(_record("pe1", status="completed"), path=p)
    before_bytes = p.read_bytes()

    assert prune_run_index(set(), path=p) == 0
    assert p.read_bytes() == before_bytes


def test_prune_missing_file_is_noop(tmp_path):
    """prune on an absent index returns 0 and creates nothing."""
    missing = tmp_path / "no_such.jsonl"
    assert prune_run_index({"x"}, path=missing) == 0
    assert not missing.exists()


def test_prune_invalidates_cache(tmp_path):
    """After prune, load returns the pruned fold, not a stale cache."""
    p = tmp_path / "idx.jsonl"
    append_run_created(_record("pc1", status="completed"), path=p)
    append_run_created(_record("pc2", status="completed"), path=p)

    load_run_index(path=p)  # populate cache
    prune_run_index({"pc1"}, path=p)

    assert set(load_run_index(path=p).keys()) == {"pc2"}
