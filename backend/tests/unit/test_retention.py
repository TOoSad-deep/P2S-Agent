"""Unit tests for run retention/cleanup (data-root housekeeping).

TDD: written before the implementation; each test must first fail for the right
reason (ImportError / AttributeError), then pass after implementation.

Run with:
    cd backend && python3 -m pytest tests/unit/test_retention.py -v
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from p2s_agent.orchestration import retention
from p2s_agent.orchestration.run_index import (
    RunLineageRecord,
    append_run_created,
    load_run_index,
)


# ---------------------------------------------------------------------------
# parse_size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1024", 1024),
        ("2KB", 2 * 1024),
        ("500MB", 500 * 1024**2),
        ("2GB", 2 * 1024**3),
        ("1TB", 1024**4),
        ("2g", 2 * 1024**3),  # lowercase + short unit
        ("1.5GB", int(1.5 * 1024**3)),
        ("  3 MB ", 3 * 1024**2),  # whitespace tolerant
    ],
)
def test_parse_size_valid(text, expected):
    assert retention.parse_size(text) == expected


@pytest.mark.parametrize("text", [None, "", "   "])
def test_parse_size_blank_is_none(text):
    assert retention.parse_size(text) is None


@pytest.mark.parametrize("text", ["abc", "10 furlongs", "GB"])
def test_parse_size_garbage_raises(text):
    with pytest.raises(ValueError):
        retention.parse_size(text)


# ---------------------------------------------------------------------------
# policy_from_env
# ---------------------------------------------------------------------------


def test_policy_defaults_when_env_empty():
    pol = retention.policy_from_env({})
    assert pol.enabled is True
    assert pol.max_runs == 1000
    assert pol.max_age_days is None
    assert pol.max_bytes is None
    assert pol.any_limit_active is True


def test_policy_enabled_flag_parses_falsey():
    for falsey in ("false", "0", "no", "off", "FALSE"):
        assert retention.policy_from_env({"P2S_RETENTION_ENABLED": falsey}).enabled is False
    for truthy in ("true", "1", "yes", "on", "TRUE"):
        assert retention.policy_from_env({"P2S_RETENTION_ENABLED": truthy}).enabled is True


def test_policy_max_runs_zero_or_negative_disables_that_limit():
    assert retention.policy_from_env({"P2S_RETENTION_MAX_RUNS": "0"}).max_runs is None
    assert retention.policy_from_env({"P2S_RETENTION_MAX_RUNS": "-5"}).max_runs is None
    assert retention.policy_from_env({"P2S_RETENTION_MAX_RUNS": "250"}).max_runs == 250


def test_policy_age_and_bytes_parse():
    pol = retention.policy_from_env(
        {"P2S_RETENTION_MAX_AGE_DAYS": "30", "P2S_RETENTION_MAX_BYTES": "2GB"}
    )
    assert pol.max_age_days == 30
    assert pol.max_bytes == 2 * 1024**3


def test_policy_all_limits_off_means_no_active_limit():
    pol = retention.policy_from_env({"P2S_RETENTION_MAX_RUNS": "0"})
    assert pol.any_limit_active is False


# ---------------------------------------------------------------------------
# plan_cleanup helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 6, 21, 12, 0, 0, tzinfo=timezone.utc)


def _at(days_ago: float) -> float:
    return (_NOW - timedelta(days=days_ago)).timestamp()


def _make_run(
    root: Path,
    run_id: str,
    *,
    created_at: float,
    status: str = "completed",
    nbytes: int = 0,
    **rec_kwargs,
) -> RunLineageRecord:
    """Create a run directory on disk + return its lineage record."""
    d = root / f"2026-06-01_png-shader_single_run_{run_id}"
    d.mkdir(parents=True)
    if nbytes:
        (d / "candidates").mkdir()
        (d / "candidates" / "r.png").write_bytes(b"x" * nbytes)
    else:
        (d / "manifest.json").write_text("{}")
    return RunLineageRecord(
        run_id=run_id,
        root_run_id=rec_kwargs.pop("root_run_id", run_id),
        parent_run_id=rec_kwargs.pop("parent_run_id", None),
        source_checkpoint_id=None,
        source_checkpoint_label=None,
        mode=None,
        feedback=None,
        title=None,
        status=status,
        run_dir=str(d),
        created_at=created_at,
        **rec_kwargs,
    )


def _index(*records: RunLineageRecord) -> dict[str, RunLineageRecord]:
    return {r.run_id: r for r in records}


def _deleted_ids(plan) -> set[str]:
    return {t.run_id for t in plan.delete}


# ---------------------------------------------------------------------------
# plan_cleanup: max_runs
# ---------------------------------------------------------------------------


def test_plan_max_runs_keeps_newest_deletes_rest(tmp_path):
    recs = [_make_run(tmp_path, f"r{i}", created_at=_at(10 - i)) for i in range(5)]
    # r0 oldest (10d) ... r4 newest (6d)
    pol = retention.RetentionPolicy(max_runs=3)
    plan = retention.plan_cleanup(tmp_path, pol, now=_NOW, index_records=_index(*recs))
    assert _deleted_ids(plan) == {"r0", "r1"}
    assert plan.kept_count == 3
    assert all(t.reason for t in plan.delete)


def test_plan_max_runs_geq_total_deletes_nothing(tmp_path):
    recs = [_make_run(tmp_path, f"r{i}", created_at=_at(i)) for i in range(3)]
    pol = retention.RetentionPolicy(max_runs=5)
    plan = retention.plan_cleanup(tmp_path, pol, now=_NOW, index_records=_index(*recs))
    assert _deleted_ids(plan) == set()
    assert plan.kept_count == 3


def test_plan_disabled_policy_deletes_nothing(tmp_path):
    recs = [_make_run(tmp_path, f"r{i}", created_at=_at(i)) for i in range(3)]
    pol = retention.RetentionPolicy(enabled=False, max_runs=1)
    plan = retention.plan_cleanup(tmp_path, pol, now=_NOW, index_records=_index(*recs))
    assert _deleted_ids(plan) == set()


def test_plan_no_active_limit_deletes_nothing(tmp_path):
    recs = [_make_run(tmp_path, f"r{i}", created_at=_at(i)) for i in range(3)]
    pol = retention.RetentionPolicy(max_runs=None, max_age_days=None, max_bytes=None)
    plan = retention.plan_cleanup(tmp_path, pol, now=_NOW, index_records=_index(*recs))
    assert _deleted_ids(plan) == set()


def test_plan_missing_root_deletes_nothing(tmp_path):
    pol = retention.RetentionPolicy(max_runs=1)
    plan = retention.plan_cleanup(
        tmp_path / "absent", pol, now=_NOW, index_records={}
    )
    assert _deleted_ids(plan) == set()


# ---------------------------------------------------------------------------
# plan_cleanup: max_age_days
# ---------------------------------------------------------------------------


def test_plan_max_age_deletes_old(tmp_path):
    young = _make_run(tmp_path, "young", created_at=_at(10))
    old = _make_run(tmp_path, "old", created_at=_at(40))
    pol = retention.RetentionPolicy(max_runs=None, max_age_days=30)
    plan = retention.plan_cleanup(
        tmp_path, pol, now=_NOW, index_records=_index(young, old)
    )
    assert _deleted_ids(plan) == {"old"}


def test_plan_age_boundary_exactly_d_days_not_deleted(tmp_path):
    boundary = _make_run(tmp_path, "edge", created_at=_at(30))
    pol = retention.RetentionPolicy(max_runs=None, max_age_days=30)
    plan = retention.plan_cleanup(
        tmp_path, pol, now=_NOW, index_records=_index(boundary)
    )
    assert _deleted_ids(plan) == set()


# ---------------------------------------------------------------------------
# plan_cleanup: invariant 2 — terminal status / orphan dirs
# ---------------------------------------------------------------------------


def test_plan_skips_non_terminal_run(tmp_path):
    running = _make_run(tmp_path, "running", created_at=_at(99), status="running")
    done = _make_run(tmp_path, "done", created_at=_at(1))
    pol = retention.RetentionPolicy(max_runs=1)
    plan = retention.plan_cleanup(
        tmp_path, pol, now=_NOW, index_records=_index(running, done)
    )
    # "running" is the oldest and beyond max_runs, but non-terminal → never deleted
    assert _deleted_ids(plan) == set()


def test_plan_reports_orphan_dir_without_record(tmp_path):
    rec = _make_run(tmp_path, "kept", created_at=_at(1))
    orphan = tmp_path / "2026-06-01_png-shader_single_run_orphan"
    orphan.mkdir()
    (orphan / "x").write_text("y")
    pol = retention.RetentionPolicy(max_runs=1)
    plan = retention.plan_cleanup(tmp_path, pol, now=_NOW, index_records=_index(rec))
    assert _deleted_ids(plan) == set()
    assert orphan.name in plan.skipped_no_record


# ---------------------------------------------------------------------------
# plan_cleanup: invariant 3 — session membership / fusion references
# ---------------------------------------------------------------------------


def test_plan_protects_variant_member(tmp_path):
    member = _make_run(tmp_path, "vm", created_at=_at(99), variant_group_id="g1")
    newer = _make_run(tmp_path, "new", created_at=_at(1))
    pol = retention.RetentionPolicy(max_runs=1)
    plan = retention.plan_cleanup(
        tmp_path, pol, now=_NOW, index_records=_index(member, newer)
    )
    assert "vm" not in _deleted_ids(plan)


def test_plan_protects_fusion_source(tmp_path):
    src = _make_run(tmp_path, "src", created_at=_at(99))
    fusion_out = _make_run(
        tmp_path, "fout", created_at=_at(1), fusion_id="f1",
        base_run_id="src", source_run_ids=["src"],
    )
    pol = retention.RetentionPolicy(max_runs=1)
    plan = retention.plan_cleanup(
        tmp_path, pol, now=_NOW, index_records=_index(src, fusion_out)
    )
    assert "src" not in _deleted_ids(plan)


# ---------------------------------------------------------------------------
# plan_cleanup: invariant 4 — ancestor protection (leaf-up deletion)
# ---------------------------------------------------------------------------


def test_plan_protects_ancestor_of_surviving_child(tmp_path):
    parent = _make_run(tmp_path, "P", created_at=_at(50))
    child = _make_run(tmp_path, "C", created_at=_at(1), parent_run_id="P", root_run_id="P")
    pol = retention.RetentionPolicy(max_runs=1)  # only newest (C) kept by count
    plan = retention.plan_cleanup(
        tmp_path, pol, now=_NOW, index_records=_index(parent, child)
    )
    assert _deleted_ids(plan) == set()
    assert "P" in plan.protected


def test_plan_deletes_parent_when_child_also_deleted(tmp_path):
    parent = _make_run(tmp_path, "P", created_at=_at(50))
    child = _make_run(tmp_path, "C", created_at=_at(40), parent_run_id="P", root_run_id="P")
    newest = _make_run(tmp_path, "N", created_at=_at(1))
    pol = retention.RetentionPolicy(max_runs=1)  # keep only N; P and C both candidates
    plan = retention.plan_cleanup(
        tmp_path, pol, now=_NOW, index_records=_index(parent, child, newest)
    )
    assert _deleted_ids(plan) == {"P", "C"}


# ---------------------------------------------------------------------------
# plan_cleanup: max_bytes backstop
# ---------------------------------------------------------------------------


def test_plan_max_bytes_deletes_oldest_until_under_cap(tmp_path):
    a = _make_run(tmp_path, "a", created_at=_at(30), nbytes=1000)
    b = _make_run(tmp_path, "b", created_at=_at(20), nbytes=1000)
    c = _make_run(tmp_path, "c", created_at=_at(10), nbytes=1000)
    # total ~3000B; cap 1500 → must drop the two oldest (a, b) to get under.
    pol = retention.RetentionPolicy(max_runs=None, max_bytes=1500)
    plan = retention.plan_cleanup(
        tmp_path, pol, now=_NOW, index_records=_index(a, b, c)
    )
    assert _deleted_ids(plan) == {"a", "b"}
    assert plan.freed_bytes == 2000


# ---------------------------------------------------------------------------
# apply_cleanup: rmtree + prune index
# ---------------------------------------------------------------------------


def _seed_index(tmp_path, *records: RunLineageRecord) -> Path:
    idx = tmp_path / "run_index.jsonl"
    for r in records:
        append_run_created(r, path=idx)
    return idx


def test_apply_deletes_dirs_and_prunes_index(tmp_path):
    recs = [_make_run(tmp_path, f"r{i}", created_at=_at(10 - i), nbytes=100) for i in range(4)]
    idx = _seed_index(tmp_path, *recs)
    records = load_run_index(path=idx)

    pol = retention.RetentionPolicy(max_runs=2)
    plan = retention.plan_cleanup(
        tmp_path, pol, now=_NOW, index_records=records, compute_sizes=True
    )
    result = retention.apply_cleanup(plan, index_path=idx)

    assert set(result.deleted) == {"r0", "r1"}
    assert result.freed_bytes == 200
    assert result.errors == []
    # oldest two dirs are gone, newest two remain
    assert not (tmp_path / "2026-06-01_png-shader_single_run_r0").exists()
    assert (tmp_path / "2026-06-01_png-shader_single_run_r3").exists()
    # index no longer references the deleted runs
    assert set(load_run_index(path=idx).keys()) == {"r2", "r3"}


def test_apply_empty_plan_is_noop(tmp_path):
    rec = _make_run(tmp_path, "solo", created_at=_at(1))
    idx = _seed_index(tmp_path, rec)
    plan = retention.plan_cleanup(
        tmp_path, retention.RetentionPolicy(max_runs=10), now=_NOW,
        index_records=load_run_index(path=idx),
    )
    result = retention.apply_cleanup(plan, index_path=idx)
    assert result.deleted == []
    assert (tmp_path / "2026-06-01_png-shader_single_run_solo").exists()
    assert set(load_run_index(path=idx).keys()) == {"solo"}


def test_apply_continues_past_a_failed_rmtree(tmp_path):
    recs = [_make_run(tmp_path, f"r{i}", created_at=_at(10 - i), nbytes=50) for i in range(3)]
    idx = _seed_index(tmp_path, *recs)
    pol = retention.RetentionPolicy(max_runs=1)  # delete r0, r1
    plan = retention.plan_cleanup(
        tmp_path, pol, now=_NOW, index_records=load_run_index(path=idx),
        compute_sizes=True,
    )
    # Sabotage one target: remove it from disk first so rmtree raises.
    import shutil as _sh
    _sh.rmtree(tmp_path / "2026-06-01_png-shader_single_run_r0")

    result = retention.apply_cleanup(plan, index_path=idx)

    assert "r1" in result.deleted  # the healthy target still got deleted
    assert any(rid == "r0" for rid, _ in result.errors)
    # pruning only drops the runs we actually deleted
    assert "r0" in load_run_index(path=idx)  # not pruned (delete failed)
    assert "r1" not in load_run_index(path=idx)


# ---------------------------------------------------------------------------
# orphan_targets: record-less directories (CLI --include-orphans)
# ---------------------------------------------------------------------------


def test_orphan_targets_filters_by_mtime(tmp_path):
    import os

    old = tmp_path / "2026-01-01_png-shader_single_run_o1"
    old.mkdir()
    (old / "f").write_bytes(b"x" * 10)
    young = tmp_path / "2026-06-20_png-shader_single_run_o2"
    young.mkdir()
    (young / "f").write_bytes(b"x" * 10)
    os.utime(old, (_at(100), _at(100)))
    os.utime(young, (_at(1), _at(1)))

    targets = retention.orphan_targets(
        tmp_path, [old.name, young.name], now=_NOW, max_age_days=30
    )
    names = {t.run_dir.name for t in targets}
    assert names == {old.name}
    assert targets[0].size_bytes == 10


def test_orphan_targets_without_age_bound_is_empty(tmp_path):
    d = tmp_path / "2026-01-01_png-shader_single_run_o1"
    d.mkdir()
    assert retention.orphan_targets(tmp_path, [d.name], now=_NOW, max_age_days=None) == []


def test_session_referenced_run_ids_scans_subsystem_json(tmp_path):
    (tmp_path / "variant_groups").mkdir()
    (tmp_path / "variant_groups" / "g.json").write_text(
        json.dumps({"child_run_ids": ["run_aaa", "run_bbb"]})
    )
    (tmp_path / "fusions").mkdir()
    (tmp_path / "fusions" / "f.json").write_text(
        json.dumps({"base_run_id": "run_ccc", "source_run_ids": ["run_ddd"]})
    )
    refs = retention.session_referenced_run_ids(tmp_path)
    assert {"run_aaa", "run_bbb", "run_ccc", "run_ddd"} <= refs


def test_orphan_targets_skips_session_referenced_dir(tmp_path):
    import os

    keep = tmp_path / "2026-01-01_png-shader_single_run_keepme"
    keep.mkdir()
    (keep / "f").write_text("x")
    os.utime(keep, (_at(100), _at(100)))

    targets = retention.orphan_targets(
        tmp_path,
        [keep.name],
        now=_NOW,
        max_age_days=30,
        protected_run_ids={"run_keepme"},
    )
    assert targets == []


# ---------------------------------------------------------------------------
# cleanup_at_startup: the never-raises seam called by the FastAPI startup hook
# ---------------------------------------------------------------------------


def test_cleanup_at_startup_applies_policy(tmp_path):
    recs = [_make_run(tmp_path, f"r{i}", created_at=_at(10 - i)) for i in range(3)]
    idx = _seed_index(tmp_path, *recs)
    assert idx.exists()

    result = retention.cleanup_at_startup(
        env={"P2S_RETENTION_MAX_RUNS": "1"}, root=tmp_path, now=_NOW
    )

    assert result is not None
    assert set(result.deleted) == {"r0", "r1"}
    assert not (tmp_path / "2026-06-01_png-shader_single_run_r0").exists()


def test_cleanup_at_startup_disabled_is_noop(tmp_path):
    recs = [_make_run(tmp_path, f"r{i}", created_at=_at(i)) for i in range(3)]
    _seed_index(tmp_path, *recs)

    result = retention.cleanup_at_startup(
        env={"P2S_RETENTION_ENABLED": "false", "P2S_RETENTION_MAX_RUNS": "1"},
        root=tmp_path,
        now=_NOW,
    )

    assert result is None
    assert all(
        (tmp_path / f"2026-06-01_png-shader_single_run_r{i}").exists() for i in range(3)
    )


def test_cleanup_at_startup_never_raises_on_bad_root(tmp_path):
    # A non-existent root must not blow up server startup.
    result = retention.cleanup_at_startup(
        env={"P2S_RETENTION_MAX_RUNS": "1"}, root=tmp_path / "nope", now=_NOW
    )
    assert result is None or result.deleted == []
