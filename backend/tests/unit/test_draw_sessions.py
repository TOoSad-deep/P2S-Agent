"""Tests for backend/app/pipeline/draw_sessions.py (V3.5, TDD).

Run with:
    cd backend && python3 -m pytest tests/unit/test_draw_sessions.py -v
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from app.pipeline.draw_sessions import (
    DrawSessionRecord,
    aggregate_draw_status,
    append_session_event,
    load_session,
    load_session_events,
    plan_draw_batches,
    save_session,
)


# ---------------------------------------------------------------------------
# plan_draw_batches — exact examples required by spec
# ---------------------------------------------------------------------------


class TestPlanDrawBatchesExamples:
    def test_count_2_returns_2(self):
        assert plan_draw_batches(2) == [2]

    def test_count_4_returns_4(self):
        assert plan_draw_batches(4) == [4]

    def test_count_6_returns_6(self):
        assert plan_draw_batches(6) == [6]

    def test_count_7_returns_4_3(self):
        assert plan_draw_batches(7) == [4, 3]

    def test_count_8_returns_4_4(self):
        assert plan_draw_batches(8) == [4, 4]

    def test_count_11_returns_6_5(self):
        assert plan_draw_batches(11) == [6, 5]

    def test_count_12_returns_6_6(self):
        assert plan_draw_batches(12) == [6, 6]


class TestPlanDrawBatchesValidation:
    def test_count_1_raises_value_error(self):
        with pytest.raises(ValueError):
            plan_draw_batches(1)

    def test_count_0_raises_value_error(self):
        with pytest.raises(ValueError):
            plan_draw_batches(0)

    def test_count_13_raises_value_error(self):
        with pytest.raises(ValueError):
            plan_draw_batches(13)

    def test_count_negative_raises_value_error(self):
        with pytest.raises(ValueError):
            plan_draw_batches(-1)


class TestPlanDrawBatchesConstraints:
    def test_all_batches_in_valid_range(self):
        for count in range(2, 13):
            batches = plan_draw_batches(count)
            for b in batches:
                assert 2 <= b <= 6, f"count={count}: batch {b} outside [2,6]"

    def test_batches_sum_to_count(self):
        for count in range(2, 13):
            batches = plan_draw_batches(count)
            assert sum(batches) == count, f"count={count}: sum={sum(batches)} != {count}"

    def test_deterministic(self):
        for count in range(2, 13):
            assert plan_draw_batches(count) == plan_draw_batches(count)

    def test_count_3_valid_split(self):
        # 3 cards: 1 group → [3]  (3 is in [2,6])
        batches = plan_draw_batches(3)
        assert sum(batches) == 3
        for b in batches:
            assert 2 <= b <= 6

    def test_count_5_valid_split(self):
        batches = plan_draw_batches(5)
        assert sum(batches) == 5
        for b in batches:
            assert 2 <= b <= 6

    def test_count_9_valid_split(self):
        batches = plan_draw_batches(9)
        assert sum(batches) == 9
        for b in batches:
            assert 2 <= b <= 6

    def test_count_10_valid_split(self):
        batches = plan_draw_batches(10)
        assert sum(batches) == 10
        for b in batches:
            assert 2 <= b <= 6


# ---------------------------------------------------------------------------
# DrawSessionRecord — default fields
# ---------------------------------------------------------------------------


class TestDrawSessionRecordDefaults:
    def _make_record(self, **overrides) -> DrawSessionRecord:
        defaults = dict(
            draw_id="draw-001",
            root_run_id="run-root",
            parent_run_id="run-parent",
            source_checkpoint_id="ckpt-42",
            feedback="add glow effect",
            status="queued",
            requested_count=6,
            diversity="medium",
        )
        defaults.update(overrides)
        return DrawSessionRecord(**defaults)

    def test_mode_defaults_to_batch_draw(self):
        rec = self._make_record()
        assert rec.mode == "batch_draw"

    def test_group_ids_defaults_to_empty_list(self):
        rec = self._make_record()
        assert rec.group_ids == []

    def test_card_run_ids_defaults_to_empty_list(self):
        rec = self._make_record()
        assert rec.card_run_ids == []

    def test_winner_run_id_defaults_to_none(self):
        rec = self._make_record()
        assert rec.winner_run_id is None

    def test_created_at_defaults_to_zero(self):
        rec = self._make_record()
        assert rec.created_at == 0.0

    def test_updated_at_defaults_to_none(self):
        rec = self._make_record()
        assert rec.updated_at is None

    def test_completed_at_defaults_to_none(self):
        rec = self._make_record()
        assert rec.completed_at is None

    def test_metadata_defaults_to_empty_dict(self):
        rec = self._make_record()
        assert rec.metadata == {}

    def test_metadata_instances_are_independent(self):
        r1 = self._make_record()
        r2 = self._make_record()
        r1.metadata["key"] = "val"
        assert "key" not in r2.metadata


# ---------------------------------------------------------------------------
# save_session / load_session round-trip
# ---------------------------------------------------------------------------


class TestSessionPersistence:
    def _make_record(self, **overrides) -> DrawSessionRecord:
        defaults = dict(
            draw_id="draw-001",
            root_run_id="run-root",
            parent_run_id="run-parent",
            source_checkpoint_id="ckpt-42",
            feedback="add glow effect",
            status="queued",
            requested_count=6,
            diversity="medium",
            group_ids=["grp-1", "grp-2"],
            card_run_ids=["run-c1", "run-c2"],
            winner_run_id=None,
            created_at=1_700_000_000.0,
            updated_at=None,
            completed_at=None,
            metadata={"source": "test"},
        )
        defaults.update(overrides)
        return DrawSessionRecord(**defaults)

    def test_save_returns_path(self, tmp_path):
        rec = self._make_record()
        p = save_session(rec, root=tmp_path)
        assert isinstance(p, Path)
        assert p.exists()

    def test_save_uses_draw_sessions_subdir(self, tmp_path):
        rec = self._make_record(draw_id="draw-sub-test")
        p = save_session(rec, root=tmp_path)
        assert p.parent.name == "draw_sessions"

    def test_load_after_save_restores_all_fields(self, tmp_path):
        rec = self._make_record()
        save_session(rec, root=tmp_path)
        loaded = load_session(rec.draw_id, root=tmp_path)
        assert loaded is not None
        assert asdict(loaded) == asdict(rec)

    def test_load_preserves_group_ids(self, tmp_path):
        rec = self._make_record(group_ids=["g1", "g2", "g3"])
        save_session(rec, root=tmp_path)
        loaded = load_session(rec.draw_id, root=tmp_path)
        assert loaded.group_ids == ["g1", "g2", "g3"]

    def test_load_preserves_card_run_ids(self, tmp_path):
        rec = self._make_record(card_run_ids=["r1", "r2", "r3"])
        save_session(rec, root=tmp_path)
        loaded = load_session(rec.draw_id, root=tmp_path)
        assert loaded.card_run_ids == ["r1", "r2", "r3"]

    def test_load_preserves_winner_run_id_when_set(self, tmp_path):
        rec = self._make_record(winner_run_id="run-c2")
        save_session(rec, root=tmp_path)
        loaded = load_session(rec.draw_id, root=tmp_path)
        assert loaded.winner_run_id == "run-c2"

    def test_load_preserves_metadata(self, tmp_path):
        rec = self._make_record(metadata={"version": 3, "tag": "v3.5"})
        save_session(rec, root=tmp_path)
        loaded = load_session(rec.draw_id, root=tmp_path)
        assert loaded.metadata == {"version": 3, "tag": "v3.5"}

    def test_load_preserves_timestamps(self, tmp_path):
        rec = self._make_record(
            created_at=1_700_000_000.0,
            updated_at=1_700_000_100.0,
            completed_at=1_700_000_200.0,
        )
        save_session(rec, root=tmp_path)
        loaded = load_session(rec.draw_id, root=tmp_path)
        assert loaded.created_at == 1_700_000_000.0
        assert loaded.updated_at == 1_700_000_100.0
        assert loaded.completed_at == 1_700_000_200.0

    def test_load_missing_returns_none(self, tmp_path):
        result = load_session("does-not-exist", root=tmp_path)
        assert result is None

    def test_load_malformed_json_returns_none(self, tmp_path):
        sess_dir = tmp_path / "draw_sessions"
        sess_dir.mkdir(parents=True, exist_ok=True)
        (sess_dir / "bad-draw.json").write_text("not json{{{", encoding="utf-8")
        result = load_session("bad-draw", root=tmp_path)
        assert result is None

    def test_overwrite_with_updated_record(self, tmp_path):
        rec = self._make_record(status="queued")
        save_session(rec, root=tmp_path)
        rec2 = self._make_record(status="completed", winner_run_id="run-c1")
        save_session(rec2, root=tmp_path)
        loaded = load_session(rec.draw_id, root=tmp_path)
        assert loaded.status == "completed"
        assert loaded.winner_run_id == "run-c1"

    def test_load_tolerates_string_requested_count(self, tmp_path):
        # simulate JSON with a string-typed requested_count (field-by-field coercion)
        sess_dir = tmp_path / "draw_sessions"
        sess_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "draw_id": "draw-str",
            "root_run_id": "r",
            "parent_run_id": "p",
            "source_checkpoint_id": "c",
            "feedback": "f",
            "status": "queued",
            "requested_count": "8",  # string, should coerce to int
            "diversity": "medium",
        }
        (sess_dir / "draw-str.json").write_text(json.dumps(data), encoding="utf-8")
        loaded = load_session("draw-str", root=tmp_path)
        assert loaded is not None
        assert loaded.requested_count == 8

    def test_load_tolerates_string_created_at(self, tmp_path):
        sess_dir = tmp_path / "draw_sessions"
        sess_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "draw_id": "draw-float",
            "root_run_id": "r",
            "parent_run_id": "p",
            "source_checkpoint_id": "c",
            "feedback": "f",
            "status": "queued",
            "requested_count": 4,
            "diversity": "medium",
            "created_at": "1700000000.0",  # string, should coerce to float
        }
        (sess_dir / "draw-float.json").write_text(json.dumps(data), encoding="utf-8")
        loaded = load_session("draw-float", root=tmp_path)
        assert loaded is not None
        assert loaded.created_at == 1_700_000_000.0

    def test_no_real_test_results_pollution(self, tmp_path):
        """Ensure tests never touch the real DEFAULT_RESULTS_ROOT."""
        from app.pipeline.artifacts import DEFAULT_RESULTS_ROOT

        rec = DrawSessionRecord(
            draw_id="draw-isolation",
            root_run_id="r",
            parent_run_id="p",
            source_checkpoint_id="c",
            feedback="f",
            status="queued",
            requested_count=2,
            diversity="medium",
        )
        save_session(rec, root=tmp_path)
        real_dir = DEFAULT_RESULTS_ROOT / "draw_sessions"
        if real_dir.exists():
            assert not (real_dir / "draw-isolation.json").exists()


# ---------------------------------------------------------------------------
# append_session_event / load_session_events
# ---------------------------------------------------------------------------


class TestSessionEvents:
    def test_append_then_load_returns_events_in_order(self, tmp_path):
        ev1 = {"type": "status_changed", "status": "running", "ts": 1.0}
        ev2 = {"type": "card_started", "run_id": "run-c1", "ts": 2.0}
        append_session_event("draw-ev-1", ev1, root=tmp_path)
        append_session_event("draw-ev-1", ev2, root=tmp_path)
        events = load_session_events("draw-ev-1", root=tmp_path)
        assert events == [ev1, ev2]

    def test_load_events_missing_file_returns_empty_list(self, tmp_path):
        events = load_session_events("nonexistent-draw", root=tmp_path)
        assert events == []

    def test_malformed_jsonl_line_is_skipped(self, tmp_path):
        sess_dir = tmp_path / "draw_sessions"
        sess_dir.mkdir(parents=True, exist_ok=True)
        ev_path = sess_dir / "draw-mal_events.jsonl"
        ev_path.write_text(
            '{"type": "good"}\n'
            "NOT JSON {\n"
            '{"type": "also_good"}\n',
            encoding="utf-8",
        )
        events = load_session_events("draw-mal", root=tmp_path)
        assert len(events) == 2
        assert events[0]["type"] == "good"
        assert events[1]["type"] == "also_good"

    def test_blank_lines_skipped(self, tmp_path):
        sess_dir = tmp_path / "draw_sessions"
        sess_dir.mkdir(parents=True, exist_ok=True)
        ev_path = sess_dir / "draw-blank_events.jsonl"
        ev_path.write_text(
            '{"type": "ev1"}\n'
            "\n"
            "   \n"
            '{"type": "ev2"}\n',
            encoding="utf-8",
        )
        events = load_session_events("draw-blank", root=tmp_path)
        assert [e["type"] for e in events] == ["ev1", "ev2"]

    def test_non_dict_json_line_is_skipped(self, tmp_path):
        sess_dir = tmp_path / "draw_sessions"
        sess_dir.mkdir(parents=True, exist_ok=True)
        ev_path = sess_dir / "draw-nondict_events.jsonl"
        ev_path.write_text(
            '{"type": "ok"}\n'
            '"just a string"\n'
            "[1, 2, 3]\n"
            '{"type": "also_ok"}\n',
            encoding="utf-8",
        )
        events = load_session_events("draw-nondict", root=tmp_path)
        assert len(events) == 2
        assert events[0]["type"] == "ok"
        assert events[1]["type"] == "also_ok"

    def test_events_stored_in_draw_sessions_subdir(self, tmp_path):
        append_session_event("draw-path-check", {"x": 1}, root=tmp_path)
        expected = tmp_path / "draw_sessions" / "draw-path-check_events.jsonl"
        assert expected.exists()

    def test_no_real_test_results_pollution(self, tmp_path):
        """Ensure tests never touch the real DEFAULT_RESULTS_ROOT."""
        from app.pipeline.artifacts import DEFAULT_RESULTS_ROOT

        append_session_event("draw-isolation-ev", {"ts": 0}, root=tmp_path)
        real_dir = DEFAULT_RESULTS_ROOT / "draw_sessions"
        if real_dir.exists():
            assert not (real_dir / "draw-isolation-ev_events.jsonl").exists()


# ---------------------------------------------------------------------------
# aggregate_draw_status — delegate to aggregate_group_status
# ---------------------------------------------------------------------------


class TestAggregateDrawStatus:
    def test_empty_list_returns_queued(self):
        assert aggregate_draw_status([]) == "queued"

    def test_all_queued_returns_queued(self):
        assert aggregate_draw_status(["queued", "queued", "queued"]) == "queued"

    def test_mixed_running_returns_running(self):
        assert aggregate_draw_status(["running", "queued"]) == "running"

    def test_all_completed_returns_completed(self):
        assert aggregate_draw_status(["completed", "completed"]) == "completed"

    def test_completed_and_failed_returns_partial_failed(self):
        assert aggregate_draw_status(["completed", "failed"]) == "partial_failed"

    def test_all_failed_returns_failed(self):
        assert aggregate_draw_status(["failed", "failed"]) == "failed"

    def test_no_completed_with_cancelled_returns_cancelled(self):
        assert aggregate_draw_status(["cancelled", "failed"]) == "cancelled"

    def test_all_cancelled_returns_cancelled(self):
        assert aggregate_draw_status(["cancelled", "cancelled"]) == "cancelled"
