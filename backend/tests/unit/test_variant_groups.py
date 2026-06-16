"""Tests for backend/app/pipeline/variant_groups.py (V3-1, TDD).

Run with:
    cd backend && python3 -m pytest tests/unit/test_variant_groups.py -v
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from app.pipeline.variant_groups import (
    VariantGroupRecord,
    aggregate_group_status,
    append_group_event,
    build_variant_strategies,
    load_group,
    load_group_events,
    save_group,
)


# ---------------------------------------------------------------------------
# build_variant_strategies — label / count
# ---------------------------------------------------------------------------

class TestBuildVariantStrategiesLabels:
    def test_count_4_medium_returns_4_dicts(self):
        result = build_variant_strategies(
            feedback="make it warmer", count=4, diversity="medium", mode="refine"
        )
        assert len(result) == 4

    def test_count_4_medium_label_order(self):
        result = build_variant_strategies(
            feedback="f", count=4, diversity="medium", mode="refine"
        )
        labels = [s["label"] for s in result]
        assert labels == ["conservative", "semantic", "lighting_color", "detail_texture"]

    def test_count_2_returns_first_2_labels(self):
        result = build_variant_strategies(
            feedback="f", count=2, diversity="medium", mode="refine"
        )
        labels = [s["label"] for s in result]
        assert labels == ["conservative", "semantic"]

    def test_count_6_returns_all_6_labels(self):
        result = build_variant_strategies(
            feedback="f", count=6, diversity="medium", mode="refine"
        )
        labels = [s["label"] for s in result]
        assert labels == [
            "conservative",
            "semantic",
            "lighting_color",
            "detail_texture",
            "structure_form",
            "alt_technique",
        ]

    def test_count_1_raises_value_error(self):
        with pytest.raises(ValueError):
            build_variant_strategies(
                feedback="f", count=1, diversity="medium", mode="refine"
            )

    def test_count_7_raises_value_error(self):
        with pytest.raises(ValueError):
            build_variant_strategies(
                feedback="f", count=7, diversity="medium", mode="refine"
            )


# ---------------------------------------------------------------------------
# build_variant_strategies — required fields on each strategy dict
# ---------------------------------------------------------------------------

class TestBuildVariantStrategiesFields:
    def _strats(self, **kw):
        defaults = dict(feedback="make it brighter", count=4, diversity="medium", mode="refine")
        defaults.update(kw)
        return build_variant_strategies(**defaults)

    def test_each_strategy_has_prompt_focus(self):
        for s in self._strats():
            assert "prompt_focus" in s
            assert isinstance(s["prompt_focus"], str)
            assert s["prompt_focus"]  # non-empty

    def test_each_strategy_has_float_score_drop_tolerance(self):
        for s in self._strats():
            assert "score_drop_tolerance" in s
            assert isinstance(s["score_drop_tolerance"], float)

    def test_each_strategy_has_notes_list(self):
        for s in self._strats():
            assert "notes" in s
            assert isinstance(s["notes"], list)

    def test_each_strategy_notes_contain_variant_line(self):
        for s in self._strats():
            has_variant_line = any("[VARIANT]" in note for note in s["notes"])
            assert has_variant_line, f"No [VARIANT] note in strategy {s['label']!r}: {s['notes']}"

    def test_each_strategy_has_locks_dict(self):
        for s in self._strats():
            assert "locks" in s
            assert isinstance(s["locks"], dict)

    def test_each_strategy_has_diversity_field(self):
        for s in self._strats(diversity="medium"):
            assert s["diversity"] == "medium"


# ---------------------------------------------------------------------------
# build_variant_strategies — diversity rules
# ---------------------------------------------------------------------------

class TestBuildVariantStrategiesDiversity:
    def test_low_diversity_all_have_small_edits_only_lock(self):
        result = build_variant_strategies(
            feedback="f", count=4, diversity="low", mode="refine"
        )
        for s in result:
            assert s["locks"].get("small_edits_only") is True, (
                f"strategy {s['label']!r} missing small_edits_only lock"
            )

    def test_low_diversity_score_drop_tolerance_clamped_at_0_01(self):
        result = build_variant_strategies(
            feedback="f", count=6, diversity="low", mode="refine"
        )
        for s in result:
            assert s["score_drop_tolerance"] <= 0.01, (
                f"strategy {s['label']!r} tolerance {s['score_drop_tolerance']} > 0.01"
            )

    def test_medium_diversity_no_small_edits_lock(self):
        result = build_variant_strategies(
            feedback="f", count=4, diversity="medium", mode="refine"
        )
        for s in result:
            assert "small_edits_only" not in s["locks"]

    def test_high_diversity_note_mentions_different_rendering_technique(self):
        result = build_variant_strategies(
            feedback="f", count=6, diversity="high", mode="refine"
        )
        found = any(
            "different rendering technique" in note
            for s in result
            for note in s["notes"]
        )
        assert found, "No 'different rendering technique' note found in any strategy"

    def test_high_diversity_tolerance_capped_at_0_05(self):
        result = build_variant_strategies(
            feedback="f", count=6, diversity="high", mode="refine"
        )
        for s in result:
            assert s["score_drop_tolerance"] <= 0.05, (
                f"strategy {s['label']!r} tolerance {s['score_drop_tolerance']} > 0.05"
            )

    def test_unknown_diversity_treated_as_medium(self):
        result_unknown = build_variant_strategies(
            feedback="f", count=4, diversity="unknown_xyz", mode="refine"
        )
        result_medium = build_variant_strategies(
            feedback="f", count=4, diversity="medium", mode="refine"
        )
        assert result_unknown == result_medium


# ---------------------------------------------------------------------------
# build_variant_strategies — determinism
# ---------------------------------------------------------------------------

class TestBuildVariantStrategiesDeterminism:
    def test_identical_args_produce_identical_output(self):
        args = dict(feedback="test feedback", count=4, diversity="medium", mode="refine")
        r1 = build_variant_strategies(**args)
        r2 = build_variant_strategies(**args)
        assert r1 == r2

    def test_determinism_with_high_diversity(self):
        args = dict(feedback="test feedback", count=6, diversity="high", mode="explore")
        r1 = build_variant_strategies(**args)
        r2 = build_variant_strategies(**args)
        assert r1 == r2


# ---------------------------------------------------------------------------
# save_group / load_group round-trip
# ---------------------------------------------------------------------------

class TestGroupPersistence:
    def _make_record(self, **overrides) -> VariantGroupRecord:
        defaults = dict(
            group_id="grp-001",
            root_run_id="run-root",
            parent_run_id="run-parent",
            source_checkpoint_id="ckpt-42",
            feedback="add more contrast",
            mode="refine",
            variant_count=3,
            diversity="medium",
            status="queued",
            child_run_ids=["run-c1", "run-c2"],
            winner_run_id=None,
            created_at=1_700_000_000.0,
            completed_at=None,
        )
        defaults.update(overrides)
        return VariantGroupRecord(**defaults)

    def test_save_returns_path(self, tmp_path):
        rec = self._make_record()
        p = save_group(rec, root=tmp_path)
        assert isinstance(p, Path)
        assert p.exists()

    def test_load_after_save_restores_all_fields(self, tmp_path):
        rec = self._make_record()
        save_group(rec, root=tmp_path)
        loaded = load_group(rec.group_id, root=tmp_path)
        assert loaded is not None
        assert asdict(loaded) == asdict(rec)

    def test_load_preserves_child_run_ids(self, tmp_path):
        rec = self._make_record(child_run_ids=["x", "y", "z"])
        save_group(rec, root=tmp_path)
        loaded = load_group(rec.group_id, root=tmp_path)
        assert loaded.child_run_ids == ["x", "y", "z"]

    def test_load_preserves_winner_run_id_when_set(self, tmp_path):
        rec = self._make_record(winner_run_id="run-c2")
        save_group(rec, root=tmp_path)
        loaded = load_group(rec.group_id, root=tmp_path)
        assert loaded.winner_run_id == "run-c2"

    def test_load_missing_group_returns_none(self, tmp_path):
        result = load_group("does-not-exist", root=tmp_path)
        assert result is None

    def test_load_malformed_json_returns_none(self, tmp_path):
        grp_dir = tmp_path / "variant_groups"
        grp_dir.mkdir(parents=True, exist_ok=True)
        (grp_dir / "bad-group.json").write_text("not json{{{", encoding="utf-8")
        result = load_group("bad-group", root=tmp_path)
        assert result is None

    def test_save_uses_variant_groups_subdir(self, tmp_path):
        rec = self._make_record(group_id="grp-sub-test")
        p = save_group(rec, root=tmp_path)
        assert p.parent.name == "variant_groups"

    def test_overwrite_with_updated_record(self, tmp_path):
        rec = self._make_record(status="queued")
        save_group(rec, root=tmp_path)
        rec2 = self._make_record(status="completed", winner_run_id="run-c1")
        save_group(rec2, root=tmp_path)
        loaded = load_group(rec.group_id, root=tmp_path)
        assert loaded.status == "completed"
        assert loaded.winner_run_id == "run-c1"


# ---------------------------------------------------------------------------
# append_group_event / load_group_events
# ---------------------------------------------------------------------------

class TestGroupEvents:
    def test_append_then_load_returns_event_in_order(self, tmp_path):
        ev1 = {"type": "status_changed", "status": "running", "ts": 1.0}
        ev2 = {"type": "child_started", "run_id": "run-c1", "ts": 2.0}
        append_group_event("grp-ev-1", ev1, root=tmp_path)
        append_group_event("grp-ev-1", ev2, root=tmp_path)
        events = load_group_events("grp-ev-1", root=tmp_path)
        assert events == [ev1, ev2]

    def test_load_events_missing_file_returns_empty_list(self, tmp_path):
        events = load_group_events("nonexistent-grp", root=tmp_path)
        assert events == []

    def test_malformed_jsonl_line_is_skipped(self, tmp_path):
        grp_dir = tmp_path / "variant_groups"
        grp_dir.mkdir(parents=True, exist_ok=True)
        ev_path = grp_dir / "grp-mal_events.jsonl"
        ev_path.write_text(
            '{"type": "good"}\n'
            'NOT JSON {\n'
            '{"type": "also_good"}\n',
            encoding="utf-8",
        )
        events = load_group_events("grp-mal", root=tmp_path)
        assert len(events) == 2
        assert events[0]["type"] == "good"
        assert events[1]["type"] == "also_good"

    def test_blank_lines_skipped(self, tmp_path):
        grp_dir = tmp_path / "variant_groups"
        grp_dir.mkdir(parents=True, exist_ok=True)
        ev_path = grp_dir / "grp-blank_events.jsonl"
        ev_path.write_text(
            '{"type": "ev1"}\n'
            '\n'
            '   \n'
            '{"type": "ev2"}\n',
            encoding="utf-8",
        )
        events = load_group_events("grp-blank", root=tmp_path)
        assert [e["type"] for e in events] == ["ev1", "ev2"]

    def test_events_stored_in_variant_groups_subdir(self, tmp_path):
        append_group_event("grp-path-check", {"x": 1}, root=tmp_path)
        expected = tmp_path / "variant_groups" / "grp-path-check_events.jsonl"
        assert expected.exists()

    def test_no_real_test_results_pollution(self, tmp_path):
        """Ensure tests never touch the real DEFAULT_RESULTS_ROOT."""
        from app.pipeline.artifacts import DEFAULT_RESULTS_ROOT
        append_group_event("grp-isolation", {"ts": 0}, root=tmp_path)
        real_dir = DEFAULT_RESULTS_ROOT / "variant_groups"
        # The file should not exist there (or if it does, our group shouldn't be in it).
        if real_dir.exists():
            assert not (real_dir / "grp-isolation_events.jsonl").exists()


# ---------------------------------------------------------------------------
# aggregate_group_status
# ---------------------------------------------------------------------------

class TestAggregateGroupStatus:
    def test_empty_list_returns_queued(self):
        assert aggregate_group_status([]) == "queued"

    def test_all_queued_returns_queued(self):
        assert aggregate_group_status(["queued", "queued", "queued"]) == "queued"

    def test_all_completed_returns_completed(self):
        assert aggregate_group_status(["completed", "completed"]) == "completed"

    def test_mixed_completed_and_failed_returns_partial_failed(self):
        assert aggregate_group_status(["completed", "failed"]) == "partial_failed"

    def test_all_failed_returns_failed(self):
        assert aggregate_group_status(["failed", "failed"]) == "failed"

    def test_running_and_completed_returns_running(self):
        assert aggregate_group_status(["running", "completed"]) == "running"

    def test_queued_and_running_returns_running(self):
        assert aggregate_group_status(["queued", "running"]) == "running"

    def test_cancelled_and_failed_returns_cancelled(self):
        assert aggregate_group_status(["cancelled", "failed"]) == "cancelled"

    def test_all_cancelled_returns_cancelled(self):
        assert aggregate_group_status(["cancelled", "cancelled"]) == "cancelled"

    def test_completed_and_cancelled_returns_partial_failed(self):
        # some completed + some non-completed terminal → partial_failed
        assert aggregate_group_status(["completed", "cancelled"]) == "partial_failed"

    def test_single_running_returns_running(self):
        assert aggregate_group_status(["running"]) == "running"

    def test_single_queued_returns_queued(self):
        assert aggregate_group_status(["queued"]) == "queued"

    def test_single_completed_returns_completed(self):
        assert aggregate_group_status(["completed"]) == "completed"
