"""Unit tests for p2s_agent.orchestration.preferences (V4.3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from p2s_agent.orchestration.preferences import (
    PreferenceEvent,
    append_preference_event,
    build_preference_notes,
    clear_preferences,
    default_profile,
    load_preference_events,
    load_profile,
    patch_profile,
    rank_variants_by_preference,
    rebuild_profile,
    save_profile,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(
    event_id: str = "evt-1",
    event_type: str = "winner_selected",
    timestamp: float = 1000.0,
    **kwargs,
) -> PreferenceEvent:
    return PreferenceEvent(event_id=event_id, event_type=event_type, timestamp=timestamp, **kwargs)


# ---------------------------------------------------------------------------
# append + load round-trip
# ---------------------------------------------------------------------------

class TestAppendAndLoad:
    def test_round_trip_single(self, tmp_path):
        ev = _make_event(event_id="e1", reason="bright colors", tags=["vivid"])
        append_preference_event(ev, root=tmp_path)
        events = load_preference_events(root=tmp_path)
        assert len(events) == 1
        assert events[0].event_id == "e1"
        assert events[0].reason == "bright colors"
        assert events[0].tags == ["vivid"]

    def test_round_trip_multiple(self, tmp_path):
        for i in range(5):
            append_preference_event(_make_event(event_id=f"e{i}", timestamp=float(i)), root=tmp_path)
        events = load_preference_events(root=tmp_path)
        assert len(events) == 5
        assert [e.event_id for e in events] == [f"e{i}" for i in range(5)]

    def test_limit_returns_last_n(self, tmp_path):
        for i in range(5):
            append_preference_event(_make_event(event_id=f"e{i}", timestamp=float(i)), root=tmp_path)
        events = load_preference_events(limit=3, root=tmp_path)
        assert len(events) == 3
        # last 3 are e2, e3, e4
        assert [e.event_id for e in events] == ["e2", "e3", "e4"]

    def test_limit_larger_than_total(self, tmp_path):
        for i in range(2):
            append_preference_event(_make_event(event_id=f"e{i}"), root=tmp_path)
        events = load_preference_events(limit=10, root=tmp_path)
        assert len(events) == 2

    def test_empty_when_file_missing(self, tmp_path):
        events = load_preference_events(root=tmp_path)
        assert events == []

    def test_defensive_load_skips_blank_lines(self, tmp_path):
        prefs_dir = tmp_path / "preferences"
        prefs_dir.mkdir(parents=True)
        (prefs_dir / "events.jsonl").write_text("\n\n\n", encoding="utf-8")
        events = load_preference_events(root=tmp_path)
        assert events == []

    def test_defensive_load_skips_malformed_json(self, tmp_path):
        prefs_dir = tmp_path / "preferences"
        prefs_dir.mkdir(parents=True)
        (prefs_dir / "events.jsonl").write_text(
            'not-json\n{"event_id": "ok", "event_type": "manual_note", "timestamp": 1.0}\n',
            encoding="utf-8",
        )
        events = load_preference_events(root=tmp_path)
        assert len(events) == 1
        assert events[0].event_id == "ok"

    def test_defensive_load_skips_non_dict_lines(self, tmp_path):
        prefs_dir = tmp_path / "preferences"
        prefs_dir.mkdir(parents=True)
        (prefs_dir / "events.jsonl").write_text(
            '[1, 2, 3]\n{"event_id": "ok2", "event_type": "manual_note", "timestamp": 2.0}\n',
            encoding="utf-8",
        )
        events = load_preference_events(root=tmp_path)
        assert len(events) == 1
        assert events[0].event_id == "ok2"

    def test_defensive_load_tolerates_missing_keys(self, tmp_path):
        prefs_dir = tmp_path / "preferences"
        prefs_dir.mkdir(parents=True)
        # Only required minimal fields
        (prefs_dir / "events.jsonl").write_text(
            '{"event_id": "min", "event_type": "manual_note", "timestamp": 5.0}\n',
            encoding="utf-8",
        )
        events = load_preference_events(root=tmp_path)
        assert len(events) == 1
        assert events[0].reason is None
        assert events[0].tags == []

    def test_all_fields_preserved(self, tmp_path):
        ev = PreferenceEvent(
            event_id="full",
            event_type="winner_selected",
            timestamp=99.9,
            run_id="run-1",
            group_id="grp-1",
            feedback="great job",
            winner_run_id="run-1",
            loser_run_ids=["run-2", "run-3"],
            rating=1,
            reason="sharp lines",
            tags=["detail", "clarity"],
            context={"variant_label": "detail_texture", "locks": {"small_edits_only": True}},
        )
        append_preference_event(ev, root=tmp_path)
        loaded = load_preference_events(root=tmp_path)
        assert len(loaded) == 1
        r = loaded[0]
        assert r.run_id == "run-1"
        assert r.group_id == "grp-1"
        assert r.loser_run_ids == ["run-2", "run-3"]
        assert r.rating == 1
        assert r.context == {"variant_label": "detail_texture", "locks": {"small_edits_only": True}}

    def test_load_preference_events_returns_empty_on_permission_error(self, tmp_path, monkeypatch):
        # A valid events.jsonl exists, but the read open() is denied (TCC/EPERM).
        # read-cutover: bypass the DB so this asserts the file-read degradation path
        monkeypatch.setattr("p2s_agent.core.db.shadow._ENABLED", False)
        append_preference_event(_make_event(event_id="e1"), root=tmp_path)
        events_path = tmp_path / "preferences" / "events.jsonl"

        real_open = Path.open

        def fake_open(self, *args, **kwargs):
            if self == events_path:
                raise PermissionError(1, "Operation not permitted")
            return real_open(self, *args, **kwargs)

        monkeypatch.setattr(Path, "open", fake_open)
        assert load_preference_events(root=tmp_path) == []


# ---------------------------------------------------------------------------
# Profile persistence
# ---------------------------------------------------------------------------

class TestLoadProfile:
    def test_returns_default_when_missing(self, tmp_path):
        profile = load_profile(root=tmp_path)
        expected = default_profile()
        assert profile == expected

    def test_returns_default_on_malformed_json(self, tmp_path):
        prefs_dir = tmp_path / "preferences"
        prefs_dir.mkdir(parents=True)
        (prefs_dir / "profile.json").write_text("not json {{{}}", encoding="utf-8")
        profile = load_profile(root=tmp_path)
        assert profile == default_profile()

    def test_returns_default_when_json_is_not_dict(self, tmp_path):
        prefs_dir = tmp_path / "preferences"
        prefs_dir.mkdir(parents=True)
        (prefs_dir / "profile.json").write_text("[1, 2, 3]", encoding="utf-8")
        profile = load_profile(root=tmp_path)
        assert profile == default_profile()

    def test_loads_saved_profile(self, tmp_path):
        prof = default_profile()
        prof["enabled"] = False
        save_profile(prof, root=tmp_path)
        loaded = load_profile(root=tmp_path)
        assert loaded["enabled"] is False

    def test_save_returns_path(self, tmp_path):
        p = save_profile(default_profile(), root=tmp_path)
        assert isinstance(p, Path)
        assert p.exists()


# ---------------------------------------------------------------------------
# patch_profile
# ---------------------------------------------------------------------------

class TestPatchProfile:
    def test_applies_editable_fields(self, tmp_path):
        save_profile(default_profile(), root=tmp_path)
        result = patch_profile(
            {"enabled": False, "positive_preferences": ["warm tones"]},
            updated_at=42.0,
            root=tmp_path,
        )
        assert result["enabled"] is False
        assert result["positive_preferences"] == ["warm tones"]

    def test_sets_updated_at_from_arg(self, tmp_path):
        save_profile(default_profile(), root=tmp_path)
        result = patch_profile({"enabled": True}, updated_at=999.5, root=tmp_path)
        assert result["updated_at"] == 999.5

    def test_rejects_disallowed_key(self, tmp_path):
        save_profile(default_profile(), root=tmp_path)
        with pytest.raises(ValueError, match="disallowed patch keys"):
            patch_profile({"schema_version": 99}, updated_at=1.0, root=tmp_path)

    def test_rejects_multiple_disallowed_keys(self, tmp_path):
        save_profile(default_profile(), root=tmp_path)
        with pytest.raises(ValueError, match="disallowed patch keys"):
            patch_profile(
                {"schema_version": 2, "summary_source_event_count": 5},
                updated_at=1.0,
                root=tmp_path,
            )

    def test_persists_patch(self, tmp_path):
        save_profile(default_profile(), root=tmp_path)
        patch_profile({"enabled": False}, updated_at=7.0, root=tmp_path)
        loaded = load_profile(root=tmp_path)
        assert loaded["enabled"] is False
        assert loaded["updated_at"] == 7.0

    def test_applies_all_editable_keys(self, tmp_path):
        save_profile(default_profile(), root=tmp_path)
        result = patch_profile(
            {
                "enabled": False,
                "default_locks": {"small_edits_only": True},
                "positive_preferences": ["bright"],
                "negative_preferences": ["dark"],
                "score_drop_tolerance_hint": 0.05,
            },
            updated_at=1.0,
            root=tmp_path,
        )
        assert result["default_locks"] == {"small_edits_only": True}
        assert result["score_drop_tolerance_hint"] == 0.05


# ---------------------------------------------------------------------------
# clear_preferences
# ---------------------------------------------------------------------------

class TestClearPreferences:
    def test_resets_profile_to_default(self, tmp_path):
        prof = default_profile()
        prof["enabled"] = False
        prof["positive_preferences"] = ["test"]
        save_profile(prof, root=tmp_path)
        clear_preferences(root=tmp_path)
        loaded = load_profile(root=tmp_path)
        assert loaded == default_profile()

    def test_empties_events(self, tmp_path):
        append_preference_event(_make_event(), root=tmp_path)
        clear_preferences(root=tmp_path)
        events = load_preference_events(root=tmp_path)
        assert events == []

    def test_does_not_crash_when_files_absent(self, tmp_path):
        # Should not raise even if nothing has been written
        clear_preferences(root=tmp_path)

    def test_events_file_removed(self, tmp_path):
        append_preference_event(_make_event(), root=tmp_path)
        prefs_dir = tmp_path / "preferences"
        assert (prefs_dir / "events.jsonl").exists()
        clear_preferences(root=tmp_path)
        assert not (prefs_dir / "events.jsonl").exists()


# ---------------------------------------------------------------------------
# rebuild_profile
# ---------------------------------------------------------------------------

class TestRebuildProfile:
    def _winner(self, event_id, reason=None, tags=None, context=None, timestamp=1.0):
        return PreferenceEvent(
            event_id=event_id,
            event_type="winner_selected",
            timestamp=timestamp,
            reason=reason,
            tags=tags or [],
            context=context or {},
        )

    def _rating(self, event_id, rating, reason=None, tags=None, timestamp=1.0):
        return PreferenceEvent(
            event_id=event_id,
            event_type="variant_rated",
            timestamp=timestamp,
            rating=rating,
            reason=reason,
            tags=tags or [],
        )

    def test_empty_events_gives_default_shape(self):
        result = rebuild_profile([], updated_at=5.0)
        assert result["summary_source_event_count"] == 0
        assert result["updated_at"] == 5.0
        assert result["positive_preferences"] == []
        assert result["negative_preferences"] == []

    def test_deterministic_same_events_twice(self):
        events = [
            self._winner("e1", reason="warm tones", tags=["vivid"], context={"variant_label": "lighting_color"}),
            self._winner("e2", reason="sharp edges", tags=["detail"], context={"variant_label": "detail_texture"}),
        ]
        r1 = rebuild_profile(events, updated_at=10.0)
        r2 = rebuild_profile(events, updated_at=10.0)
        assert r1 == r2

    def test_winner_reason_added_to_positive(self):
        events = [self._winner("e1", reason="warm tones")]
        result = rebuild_profile(events, updated_at=1.0)
        assert "warm tones" in result["positive_preferences"]

    def test_winner_tags_added_to_positive(self):
        events = [self._winner("e1", tags=["vivid", "colorful"])]
        result = rebuild_profile(events, updated_at=1.0)
        assert "vivid" in result["positive_preferences"]
        assert "colorful" in result["positive_preferences"]

    def test_positive_rating_adds_to_positive(self):
        events = [self._rating("e1", rating=1, reason="bright", tags=["light"])]
        result = rebuild_profile(events, updated_at=1.0)
        assert "bright" in result["positive_preferences"]
        assert "light" in result["positive_preferences"]

    def test_negative_rating_adds_to_negative(self):
        events = [self._rating("e1", rating=-1, reason="too dark", tags=["dark"])]
        result = rebuild_profile(events, updated_at=1.0)
        assert "too dark" in result["negative_preferences"]
        assert "dark" in result["negative_preferences"]
        assert "too dark" not in result["positive_preferences"]

    def test_positive_deduplication_preserves_first_seen(self):
        events = [
            self._winner("e1", reason="bright", tags=["vivid"]),
            self._winner("e2", reason="bright", tags=["vivid"]),  # duplicate
        ]
        result = rebuild_profile(events, updated_at=1.0)
        assert result["positive_preferences"].count("bright") == 1
        assert result["positive_preferences"].count("vivid") == 1

    def test_preferred_variant_labels_by_frequency(self):
        events = [
            self._winner("e1", context={"variant_label": "lighting_color"}),
            self._winner("e2", context={"variant_label": "lighting_color"}),
            self._winner("e3", context={"variant_label": "detail_texture"}),
        ]
        result = rebuild_profile(events, updated_at=1.0)
        labels = result["preferred_variant_labels"]
        assert labels[0] == "lighting_color"  # highest frequency
        assert "detail_texture" in labels

    def test_preferred_variant_labels_alpha_tiebreak(self):
        # Same frequency → alphabetical order
        events = [
            self._winner("e1", context={"variant_label": "zzz"}),
            self._winner("e2", context={"variant_label": "aaa"}),
        ]
        result = rebuild_profile(events, updated_at=1.0)
        labels = result["preferred_variant_labels"]
        assert labels == ["aaa", "zzz"]

    def test_preferred_variant_labels_max_4(self):
        events = [
            self._winner(f"e{i}", context={"variant_label": f"label_{i}"})
            for i in range(6)
        ]
        result = rebuild_profile(events, updated_at=1.0)
        assert len(result["preferred_variant_labels"]) <= 4

    def test_preferred_variant_labels_falls_back_to_tags(self):
        # No variant_label in context → use tags
        events = [self._winner("e1", tags=["semantic", "semantic", "alt_technique"])]
        result = rebuild_profile(events, updated_at=1.0)
        # tags contribute but semantic only appears once (deduped within the event)
        # "semantic" comes from tags list as each tag is counted individually
        labels = result["preferred_variant_labels"]
        assert "semantic" in labels

    def test_majority_lock_goes_into_default_locks(self):
        events = [
            self._winner("e1", context={"locks": {"small_edits_only": True}}),
            self._winner("e2", context={"locks": {"small_edits_only": True}}),
            self._winner("e3", context={"locks": {"small_edits_only": False}}),
        ]
        result = rebuild_profile(events, updated_at=1.0)
        # 2/3 True → majority
        assert result["default_locks"].get("small_edits_only") is True

    def test_minority_lock_excluded(self):
        events = [
            self._winner("e1", context={"locks": {"small_edits_only": True}}),
            self._winner("e2", context={"locks": {"small_edits_only": False}}),
            self._winner("e3", context={"locks": {"small_edits_only": False}}),
        ]
        result = rebuild_profile(events, updated_at=1.0)
        # 1/3 True → not majority
        assert "small_edits_only" not in result["default_locks"]

    def test_exactly_half_lock_is_majority(self):
        events = [
            self._winner("e1", context={"locks": {"small_edits_only": True}}),
            self._winner("e2", context={"locks": {"small_edits_only": False}}),
        ]
        result = rebuild_profile(events, updated_at=1.0)
        # 1/2 True → ≥ half, included
        assert result["default_locks"].get("small_edits_only") is True

    def test_summary_source_event_count(self):
        events = [self._winner(f"e{i}") for i in range(7)]
        result = rebuild_profile(events, updated_at=1.0)
        assert result["summary_source_event_count"] == 7

    def test_updated_at_from_arg(self):
        result = rebuild_profile([], updated_at=12345.6)
        assert result["updated_at"] == 12345.6

    def test_preserves_base_profile_enabled_flag(self):
        base = default_profile()
        base["enabled"] = False
        result = rebuild_profile([], updated_at=1.0, base_profile=base)
        assert result["enabled"] is False

    def test_base_profile_none_uses_default(self):
        result = rebuild_profile([], updated_at=1.0, base_profile=None)
        assert result["schema_version"] == 1

    def test_schema_version_preserved(self):
        base = default_profile()
        base["schema_version"] = 2
        result = rebuild_profile([], updated_at=1.0, base_profile=base)
        assert result["schema_version"] == 2

    def test_winner_tags_not_added_to_negative(self):
        events = [self._winner("e1", reason="great", tags=["tag1"])]
        result = rebuild_profile(events, updated_at=1.0)
        assert result["negative_preferences"] == []

    def test_all_event_types_combined(self):
        events = [
            self._winner("e1", reason="warm", tags=["vivid"]),
            self._rating("e2", rating=1, reason="bright"),
            self._rating("e3", rating=-1, reason="too dark", tags=["muddy"]),
        ]
        result = rebuild_profile(events, updated_at=1.0)
        assert "warm" in result["positive_preferences"]
        assert "vivid" in result["positive_preferences"]
        assert "bright" in result["positive_preferences"]
        assert "too dark" in result["negative_preferences"]
        assert "muddy" in result["negative_preferences"]


# ---------------------------------------------------------------------------
# build_preference_notes
# ---------------------------------------------------------------------------

class TestBuildPreferenceNotes:
    def test_enabled_false_returns_empty(self):
        profile = default_profile()
        profile["enabled"] = False
        profile["positive_preferences"] = ["bright"]
        assert build_preference_notes(profile) == []

    def test_enabled_missing_treated_as_falsy(self):
        profile = {}
        assert build_preference_notes(profile) == []

    def test_positives_emitted(self):
        profile = default_profile()
        profile["positive_preferences"] = ["warm tones", "sharp lines"]
        notes = build_preference_notes(profile)
        assert "[PREFERENCE+] warm tones" in notes
        assert "[PREFERENCE+] sharp lines" in notes

    def test_negatives_emitted(self):
        profile = default_profile()
        profile["negative_preferences"] = ["too dark", "blurry"]
        notes = build_preference_notes(profile)
        assert "[PREFERENCE-] too dark" in notes
        assert "[PREFERENCE-] blurry" in notes

    def test_locks_emitted(self):
        profile = default_profile()
        profile["default_locks"] = {"small_edits_only": True}
        notes = build_preference_notes(profile)
        assert "[PREFERENCE LOCK] small_edits_only" in notes

    def test_false_lock_not_emitted(self):
        profile = default_profile()
        profile["default_locks"] = {"small_edits_only": False}
        notes = build_preference_notes(profile)
        assert not any("PREFERENCE LOCK" in n for n in notes)

    def test_labels_emitted(self):
        profile = default_profile()
        profile["preferred_variant_labels"] = ["lighting_color", "detail_texture"]
        notes = build_preference_notes(profile)
        label_notes = [n for n in notes if "PREFERENCE LABELS" in n]
        assert len(label_notes) == 1
        assert "lighting_color" in label_notes[0]
        assert "detail_texture" in label_notes[0]

    def test_empty_labels_no_label_note(self):
        profile = default_profile()
        profile["preferred_variant_labels"] = []
        notes = build_preference_notes(profile)
        assert not any("PREFERENCE LABELS" in n for n in notes)

    def test_all_notes_deterministic_order(self):
        profile = default_profile()
        profile["positive_preferences"] = ["a", "b"]
        profile["negative_preferences"] = ["c"]
        profile["default_locks"] = {"lock1": True}
        profile["preferred_variant_labels"] = ["lbl"]
        n1 = build_preference_notes(profile)
        n2 = build_preference_notes(profile)
        assert n1 == n2

    def test_empty_profile_enabled_no_prefs(self):
        profile = default_profile()  # enabled=True but all empty
        notes = build_preference_notes(profile)
        assert notes == []

    def test_full_combined(self):
        profile = default_profile()
        profile["positive_preferences"] = ["bright colors"]
        profile["negative_preferences"] = ["dark shadows"]
        profile["default_locks"] = {"small_edits_only": True}
        profile["preferred_variant_labels"] = ["lighting_color"]
        notes = build_preference_notes(profile)
        assert "[PREFERENCE+] bright colors" in notes
        assert "[PREFERENCE-] dark shadows" in notes
        assert "[PREFERENCE LOCK] small_edits_only" in notes
        assert any("PREFERENCE LABELS" in n and "lighting_color" in n for n in notes)


# ---------------------------------------------------------------------------
# default_profile
# ---------------------------------------------------------------------------

class TestDefaultProfile:
    def test_schema(self):
        p = default_profile()
        assert p["schema_version"] == 1
        assert p["updated_at"] == 0.0
        assert p["enabled"] is True
        assert p["default_locks"] == {}
        assert p["positive_preferences"] == []
        assert p["negative_preferences"] == []
        assert p["preferred_variant_labels"] == []
        assert p["score_drop_tolerance_hint"] == 0.02
        assert p["summary_source_event_count"] == 0

    def test_returns_fresh_copy_each_call(self):
        p1 = default_profile()
        p2 = default_profile()
        p1["positive_preferences"].append("x")
        assert p2["positive_preferences"] == []


# ---------------------------------------------------------------------------
# Isolation: real test_results never touched
# ---------------------------------------------------------------------------

class TestIsolation:
    def test_operations_stay_in_tmp_path(self, tmp_path):
        from p2s_agent.core.pipeline.artifacts import DEFAULT_RESULTS_ROOT
        real_prefs = DEFAULT_RESULTS_ROOT / "preferences"
        append_preference_event(_make_event(), root=tmp_path)
        load_preference_events(root=tmp_path)
        load_profile(root=tmp_path)
        save_profile(default_profile(), root=tmp_path)
        clear_preferences(root=tmp_path)
        # Real preferences dir should NOT have been touched
        real_events = real_prefs / "events.jsonl"
        if real_events.exists():
            # If it existed before, it should not have our test data
            content = real_events.read_text()
            assert "evt-1" not in content


# ---------------------------------------------------------------------------
# V4.4: rank_variants_by_preference
# ---------------------------------------------------------------------------

def _make_variant(run_id: str, label: str, changes_summary: str | None = None) -> dict:
    return {"run_id": run_id, "label": label, "changes_summary": changes_summary}


def _enabled_profile(**kwargs) -> dict:
    p = default_profile()
    p["enabled"] = True
    p.update(kwargs)
    return p


class TestRankVariantsByPreference:
    # ------------------------------------------------------------------
    # Preferred label → recommended + score ≥ 1.0
    # ------------------------------------------------------------------

    def test_label_match_recommended_true(self):
        variants = [
            _make_variant("r1", "conservative"),
            _make_variant("r2", "semantic"),
        ]
        profile = _enabled_profile(preferred_variant_labels=["conservative"])
        result = rank_variants_by_preference(variants, profile)

        assert result["r1"]["recommended"] is True
        assert result["r1"]["preference_score"] >= 1.0
        assert result["r2"]["recommended"] is False
        assert result["r2"]["preference_score"] == 0.0

    # ------------------------------------------------------------------
    # Disabled profile → all zero / False regardless of label match
    # ------------------------------------------------------------------

    def test_disabled_profile_all_zero(self):
        variants = [
            _make_variant("r1", "conservative"),
            _make_variant("r2", "semantic"),
        ]
        profile = _enabled_profile(preferred_variant_labels=["conservative"])
        profile["enabled"] = False
        result = rank_variants_by_preference(variants, profile)

        for v in variants:
            assert result[v["run_id"]]["preference_score"] == 0.0
            assert result[v["run_id"]]["recommended"] is False

    # ------------------------------------------------------------------
    # No preferred labels & no keyword hits → all 0 / False
    # ------------------------------------------------------------------

    def test_no_labels_no_keywords_all_zero(self):
        variants = [
            _make_variant("r1", "conservative", "made lighting brighter"),
            _make_variant("r2", "semantic", "fixed colors"),
        ]
        profile = _enabled_profile(preferred_variant_labels=[], positive_preferences=[])
        result = rank_variants_by_preference(variants, profile)

        for v in variants:
            assert result[v["run_id"]]["preference_score"] == 0.0
            assert result[v["run_id"]]["recommended"] is False

    # ------------------------------------------------------------------
    # Determinism: same inputs → same output; inputs not mutated
    # ------------------------------------------------------------------

    def test_deterministic_and_no_mutation(self):
        variants = [
            _make_variant("r1", "conservative", "improved brightness"),
            _make_variant("r2", "semantic"),
        ]
        profile = _enabled_profile(
            preferred_variant_labels=["conservative"],
            positive_preferences=["brightness"],
        )
        # Snapshot original state
        original_variants = [dict(v) for v in variants]
        original_profile = dict(profile)

        result1 = rank_variants_by_preference(variants, profile)
        result2 = rank_variants_by_preference(variants, profile)

        # Same output both times.
        assert result1 == result2

        # Inputs not mutated.
        for orig, v in zip(original_variants, variants):
            assert v == orig
        assert profile == original_profile

    # ------------------------------------------------------------------
    # Ties: two variants with the same top score → both recommended
    # ------------------------------------------------------------------

    def test_ties_both_recommended(self):
        variants = [
            _make_variant("r1", "bold"),
            _make_variant("r2", "elegant"),
            _make_variant("r3", "minimal"),
        ]
        profile = _enabled_profile(preferred_variant_labels=["bold", "elegant"])
        result = rank_variants_by_preference(variants, profile)

        assert result["r1"]["recommended"] is True
        assert result["r2"]["recommended"] is True
        assert result["r3"]["recommended"] is False
        assert result["r1"]["preference_score"] == result["r2"]["preference_score"]
        assert result["r1"]["preference_score"] >= 1.0

    # ------------------------------------------------------------------
    # Keyword contribution from changes_summary
    # ------------------------------------------------------------------

    def test_keyword_in_changes_summary_adds_score(self):
        variants = [
            _make_variant("r1", "variant_a", "improved brightness and contrast"),
            _make_variant("r2", "variant_b"),
        ]
        profile = _enabled_profile(
            preferred_variant_labels=[],
            positive_preferences=["brightness"],
        )
        result = rank_variants_by_preference(variants, profile)

        assert result["r1"]["preference_score"] > 0.0
        assert result["r1"]["recommended"] is True
        assert result["r2"]["preference_score"] == 0.0
        assert result["r2"]["recommended"] is False

    def test_keyword_contribution_capped_at_half(self):
        variants = [
            _make_variant("r1", "x", "brightness contrast saturation hue clarity"),
        ]
        profile = _enabled_profile(
            preferred_variant_labels=[],
            positive_preferences=["brightness", "contrast", "saturation", "hue", "clarity"],
        )
        result = rank_variants_by_preference(variants, profile)
        # Capped at 0.5 total keyword contribution.
        assert result["r1"]["preference_score"] == pytest.approx(0.5)

    # ------------------------------------------------------------------
    # Return values are JSON-friendly (float, bool)
    # ------------------------------------------------------------------

    def test_result_json_friendly(self):
        variants = [_make_variant("r1", "x")]
        profile = _enabled_profile(preferred_variant_labels=["x"])
        result = rank_variants_by_preference(variants, profile)

        score = result["r1"]["preference_score"]
        rec = result["r1"]["recommended"]
        assert isinstance(score, float)
        assert isinstance(rec, bool)

    # ------------------------------------------------------------------
    # Empty variants list → empty dict
    # ------------------------------------------------------------------

    def test_empty_variants(self):
        profile = _enabled_profile(preferred_variant_labels=["x"])
        result = rank_variants_by_preference([], profile)
        assert result == {}
