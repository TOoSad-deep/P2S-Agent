"""Tests for human_constraints.py — V4.1 Structured Constraints (TDD).

Written BEFORE the implementation so every test starts red.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.pipeline.human_constraints import (
    HumanConstraintSpec,
    RegionConstraint,
    build_constraint_notes,
    constraint_to_artifacts,
    parse_constraint_spec,
    spec_to_dict,
    validate_constraint_spec,
)


# ---------------------------------------------------------------------------
# parse_constraint_spec
# ---------------------------------------------------------------------------


class TestParseConstraintSpec:
    def test_none_returns_default(self):
        spec = parse_constraint_spec(None)
        assert isinstance(spec, HumanConstraintSpec)
        assert spec.locks == {}
        assert spec.targets == {}
        assert spec.edit_strength == 0.5
        assert spec.regions == []
        assert spec.use_preferences is True

    def test_empty_dict_returns_default(self):
        spec = parse_constraint_spec({})
        assert spec.locks == {}
        assert spec.targets == {}
        assert spec.edit_strength == 0.5
        assert spec.regions == []
        assert spec.use_preferences is True

    def test_full_payload_round_trips(self):
        payload = {
            "locks": {"preserve_layout": True, "preserve_palette": False},
            "targets": {"brightness": "increase", "contrast": "decrease"},
            "edit_strength": 0.3,
            "regions": [
                {
                    "id": "r1",
                    "label": "Sky area",
                    "mode": "modify",
                    "instruction": "Brighten sky",
                    "geometry_type": "rect",
                    "geometry": {"x": 0.0, "y": 0.0, "w": 1.0, "h": 0.5},
                    "strength": 0.7,
                }
            ],
            "use_preferences": False,
        }
        spec = parse_constraint_spec(payload)
        assert spec.locks == {"preserve_layout": True, "preserve_palette": False}
        assert spec.targets == {"brightness": "increase", "contrast": "decrease"}
        assert spec.edit_strength == 0.3
        assert spec.use_preferences is False
        assert len(spec.regions) == 1
        r = spec.regions[0]
        assert r.id == "r1"
        assert r.label == "Sky area"
        assert r.mode == "modify"
        assert r.instruction == "Brighten sky"
        assert r.geometry_type == "rect"
        assert r.geometry == {"x": 0.0, "y": 0.0, "w": 1.0, "h": 0.5}
        assert r.strength == 0.7

    def test_unknown_keys_ignored(self):
        spec = parse_constraint_spec({"foo": "bar", "extra": 42})
        assert spec.locks == {}
        assert spec.targets == {}

    def test_non_dict_locks_coerced_to_empty(self):
        spec = parse_constraint_spec({"locks": "not-a-dict"})
        assert spec.locks == {}

    def test_non_list_regions_coerced_to_empty(self):
        spec = parse_constraint_spec({"regions": "not-a-list"})
        assert spec.regions == []

    def test_region_defaults_applied(self):
        """Strength defaults to 0.5 when omitted from region dict."""
        payload = {
            "regions": [
                {
                    "id": "r2",
                    "label": "Ground",
                    "mode": "protect",
                    "instruction": "Keep as-is",
                    "geometry_type": "rect",
                    "geometry": {"x": 0.0, "y": 0.5, "w": 1.0, "h": 0.5},
                }
            ]
        }
        spec = parse_constraint_spec(payload)
        assert spec.regions[0].strength == 0.5

    def test_edit_strength_default(self):
        spec = parse_constraint_spec({"edit_strength": 0.8})
        assert spec.edit_strength == 0.8

    def test_edit_strength_missing_defaults_to_half(self):
        spec = parse_constraint_spec({})
        assert spec.edit_strength == 0.5

    def test_non_dict_targets_coerced_to_empty(self):
        spec = parse_constraint_spec({"targets": ["bad"]})
        assert spec.targets == {}

    def test_edit_strength_bad_string_falls_back_to_half(self):
        """parse_constraint_spec must NOT raise on a bad edit_strength string."""
        spec = parse_constraint_spec({"edit_strength": "bad"})
        assert spec.edit_strength == 0.5

    def test_region_strength_bad_string_falls_back_to_half(self):
        """_parse_region must NOT raise on a bad strength string."""
        payload = {
            "regions": [
                {
                    "id": "r",
                    "label": "l",
                    "mode": "modify",
                    "instruction": "i",
                    "geometry_type": "rect",
                    "geometry": {"x": 0, "y": 0, "w": 1, "h": 1},
                    "strength": "bad",
                }
            ]
        }
        spec = parse_constraint_spec(payload)
        assert spec.regions[0].strength == 0.5


# ---------------------------------------------------------------------------
# validate_constraint_spec
# ---------------------------------------------------------------------------


class TestValidateConstraintSpec:
    def test_valid_default_spec_no_errors(self):
        spec = parse_constraint_spec(None)
        errors = validate_constraint_spec(spec)
        assert errors == []

    def test_edit_strength_too_high(self):
        spec = HumanConstraintSpec(edit_strength=1.5)
        errors = validate_constraint_spec(spec)
        assert any("edit_strength" in e for e in errors)

    def test_edit_strength_negative(self):
        spec = HumanConstraintSpec(edit_strength=-0.1)
        errors = validate_constraint_spec(spec)
        assert any("edit_strength" in e for e in errors)

    def test_edit_strength_boundary_zero_valid(self):
        spec = HumanConstraintSpec(edit_strength=0.0)
        errors = validate_constraint_spec(spec)
        assert errors == []

    def test_edit_strength_boundary_one_valid(self):
        spec = HumanConstraintSpec(edit_strength=1.0)
        errors = validate_constraint_spec(spec)
        assert errors == []

    def test_region_strength_out_of_range(self):
        region = RegionConstraint(
            id="r1", label="test", mode="modify", instruction="x",
            geometry_type="rect", geometry={"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5},
            strength=1.5,
        )
        spec = HumanConstraintSpec(regions=[region])
        errors = validate_constraint_spec(spec)
        assert any("r1" in e for e in errors)

    def test_bad_target_direction_names_attr(self):
        spec = HumanConstraintSpec(targets={"brightness": "invalid_dir"})
        errors = validate_constraint_spec(spec)
        assert any("brightness" in e for e in errors)

    def test_valid_target_directions(self):
        spec = HumanConstraintSpec(
            targets={"brightness": "increase", "contrast": "decrease", "saturation": "keep"}
        )
        errors = validate_constraint_spec(spec)
        assert errors == []

    def test_locks_non_bool_error(self):
        spec = HumanConstraintSpec(locks={"preserve_layout": "yes"})
        errors = validate_constraint_spec(spec)
        assert any("preserve_layout" in e for e in errors)

    def test_locks_bool_valid(self):
        spec = HumanConstraintSpec(locks={"preserve_layout": True, "preserve_palette": False})
        errors = validate_constraint_spec(spec)
        assert errors == []

    def test_region_bad_mode(self):
        region = RegionConstraint(
            id="r1", label="test", mode="replace", instruction="x",
            geometry_type="rect", geometry={"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5},
        )
        spec = HumanConstraintSpec(regions=[region])
        errors = validate_constraint_spec(spec)
        assert any("r1" in e for e in errors)

    def test_region_bad_geometry_type(self):
        region = RegionConstraint(
            id="r1", label="test", mode="modify", instruction="x",
            geometry_type="circle", geometry={"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5},
        )
        spec = HumanConstraintSpec(regions=[region])
        errors = validate_constraint_spec(spec)
        assert any("r1" in e for e in errors)

    def test_rect_out_of_bounds_x_plus_w(self):
        """x + w > 1.0 must error and name the region id."""
        region = RegionConstraint(
            id="region_abc", label="test", mode="modify", instruction="x",
            geometry_type="rect", geometry={"x": 0.8, "y": 0.0, "w": 0.5, "h": 0.5},
        )
        spec = HumanConstraintSpec(regions=[region])
        errors = validate_constraint_spec(spec)
        assert any("region_abc" in e for e in errors)

    def test_rect_out_of_bounds_y_plus_h(self):
        """y + h > 1.0 must error and name the region id."""
        region = RegionConstraint(
            id="region_xyz", label="test", mode="protect", instruction="y",
            geometry_type="rect", geometry={"x": 0.0, "y": 0.7, "w": 0.5, "h": 0.5},
        )
        spec = HumanConstraintSpec(regions=[region])
        errors = validate_constraint_spec(spec)
        assert any("region_xyz" in e for e in errors)

    def test_rect_zero_width_error(self):
        region = RegionConstraint(
            id="r_zero_w", label="test", mode="modify", instruction="x",
            geometry_type="rect", geometry={"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.5},
        )
        spec = HumanConstraintSpec(regions=[region])
        errors = validate_constraint_spec(spec)
        assert any("r_zero_w" in e for e in errors)

    def test_rect_negative_x_error(self):
        region = RegionConstraint(
            id="r_neg", label="test", mode="modify", instruction="x",
            geometry_type="rect", geometry={"x": -0.1, "y": 0.0, "w": 0.5, "h": 0.5},
        )
        spec = HumanConstraintSpec(regions=[region])
        errors = validate_constraint_spec(spec)
        assert any("r_neg" in e for e in errors)

    def test_duplicate_region_id_error(self):
        region1 = RegionConstraint(
            id="dup", label="first", mode="modify", instruction="x",
            geometry_type="rect", geometry={"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5},
        )
        region2 = RegionConstraint(
            id="dup", label="second", mode="protect", instruction="y",
            geometry_type="rect", geometry={"x": 0.5, "y": 0.0, "w": 0.5, "h": 0.5},
        )
        spec = HumanConstraintSpec(regions=[region1, region2])
        errors = validate_constraint_spec(spec)
        assert any("dup" in e for e in errors)

    def test_empty_region_id_error(self):
        region = RegionConstraint(
            id="", label="test", mode="modify", instruction="x",
            geometry_type="rect", geometry={"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5},
        )
        spec = HumanConstraintSpec(regions=[region])
        errors = validate_constraint_spec(spec)
        assert len(errors) > 0

    def test_valid_full_spec(self):
        region = RegionConstraint(
            id="sky", label="Sky", mode="modify", instruction="Brighten",
            geometry_type="rect", geometry={"x": 0.0, "y": 0.0, "w": 1.0, "h": 0.4},
            strength=0.8,
        )
        spec = HumanConstraintSpec(
            locks={"preserve_layout": True},
            targets={"brightness": "increase"},
            edit_strength=0.6,
            regions=[region],
            use_preferences=True,
        )
        errors = validate_constraint_spec(spec)
        assert errors == []

    def test_image_dimensions_accepted_no_crash(self):
        """image_width/height are accepted for API symmetry."""
        spec = parse_constraint_spec(None)
        errors = validate_constraint_spec(spec, image_width=1920, image_height=1080)
        assert errors == []

    def test_rect_bounds_float_epsilon_no_false_positive(self):
        """x=0.1, w=0.9 sums to 1.0 within float error — must NOT produce a bounds error."""
        region = RegionConstraint(
            id="r_eps",
            label="test",
            mode="modify",
            instruction="i",
            geometry_type="rect",
            geometry={"x": 0.1, "y": 0.0, "w": 0.9, "h": 1.0},
        )
        spec = HumanConstraintSpec(regions=[region])
        errors = validate_constraint_spec(spec)
        assert not any("r_eps" in e and "exceeds" in e for e in errors)


# ---------------------------------------------------------------------------
# build_constraint_notes
# ---------------------------------------------------------------------------


class TestBuildConstraintNotes:
    def test_preserve_layout_lock(self):
        spec = HumanConstraintSpec(locks={"preserve_layout": True})
        notes = build_constraint_notes(spec)
        assert any("[GLOBAL LOCK]" in n and "layout" in n.lower() for n in notes)

    def test_preserve_palette_lock(self):
        spec = HumanConstraintSpec(locks={"preserve_palette": True})
        notes = build_constraint_notes(spec)
        assert any("[GLOBAL LOCK]" in n and "palette" in n.lower() for n in notes)

    def test_preserve_background_lock(self):
        spec = HumanConstraintSpec(locks={"preserve_background": True})
        notes = build_constraint_notes(spec)
        assert any("[GLOBAL LOCK]" in n and "background" in n.lower() for n in notes)

    def test_small_edits_only_lock(self):
        spec = HumanConstraintSpec(locks={"small_edits_only": True})
        notes = build_constraint_notes(spec)
        assert any("[GLOBAL LOCK]" in n for n in notes)

    def test_falsy_lock_not_emitted(self):
        spec = HumanConstraintSpec(locks={"preserve_layout": False})
        notes = build_constraint_notes(spec)
        assert not any("[GLOBAL LOCK]" in n and "layout" in n.lower() for n in notes)

    def test_target_increase_emits_target_note(self):
        spec = HumanConstraintSpec(targets={"reflection_strength": "increase"})
        notes = build_constraint_notes(spec)
        assert any("[TARGET]" in n and "Increase" in n and "reflection_strength" in n for n in notes)

    def test_target_decrease_emits_target_note(self):
        spec = HumanConstraintSpec(targets={"noise": "decrease"})
        notes = build_constraint_notes(spec)
        assert any("[TARGET]" in n and "Decrease" in n and "noise" in n for n in notes)

    def test_target_keep_not_emitted(self):
        spec = HumanConstraintSpec(targets={"brightness": "keep"})
        notes = build_constraint_notes(spec)
        assert not any("[TARGET]" in n for n in notes)

    def test_edit_strength_always_emitted(self):
        spec = HumanConstraintSpec(edit_strength=0.4)
        notes = build_constraint_notes(spec)
        assert any("[EDIT STRENGTH]" in n and "0.4" in n for n in notes)

    def test_edit_strength_emitted_even_at_one(self):
        spec = HumanConstraintSpec(edit_strength=1.0)
        notes = build_constraint_notes(spec)
        assert any("[EDIT STRENGTH]" in n for n in notes)

    def test_region_modify_emits_modify_note_with_coords(self):
        region = RegionConstraint(
            id="sky", label="Sky", mode="modify", instruction="Brighten the sky",
            geometry_type="rect", geometry={"x": 0.0, "y": 0.0, "w": 1.0, "h": 0.4},
        )
        spec = HumanConstraintSpec(regions=[region])
        notes = build_constraint_notes(spec)
        assert any(
            "[REGION MODIFY sky]" in n
            and "Brighten the sky" in n
            and "x=0.0" in n
            for n in notes
        )

    def test_region_protect_emits_protect_note_with_coords(self):
        region = RegionConstraint(
            id="ground", label="Ground", mode="protect", instruction="Keep ground stable",
            geometry_type="rect", geometry={"x": 0.0, "y": 0.6, "w": 1.0, "h": 0.4},
        )
        spec = HumanConstraintSpec(regions=[region])
        notes = build_constraint_notes(spec)
        assert any(
            "[REGION PROTECT ground]" in n
            and "Keep ground stable" in n
            for n in notes
        )

    def test_non_rect_region_emits_text_only_note(self):
        region = RegionConstraint(
            id="mask1", label="Custom mask", mode="modify", instruction="Apply here",
            geometry_type="mask", geometry={},
        )
        spec = HumanConstraintSpec(regions=[region])
        notes = build_constraint_notes(spec)
        assert any("[REGION MODIFY mask1]" in n for n in notes)

    def test_notes_are_deterministic(self):
        """Calling twice returns identical lists."""
        spec = HumanConstraintSpec(
            locks={"preserve_layout": True},
            targets={"brightness": "increase"},
            edit_strength=0.6,
            regions=[
                RegionConstraint(
                    id="r1", label="L", mode="modify", instruction="Do something",
                    geometry_type="rect", geometry={"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5},
                )
            ],
        )
        assert build_constraint_notes(spec) == build_constraint_notes(spec)

    def test_empty_spec_still_emits_edit_strength(self):
        spec = HumanConstraintSpec()
        notes = build_constraint_notes(spec)
        assert any("[EDIT STRENGTH]" in n for n in notes)


# ---------------------------------------------------------------------------
# spec_to_dict
# ---------------------------------------------------------------------------


class TestSpecToDict:
    def test_default_spec_to_dict(self):
        spec = HumanConstraintSpec()
        d = spec_to_dict(spec)
        assert d["locks"] == {}
        assert d["targets"] == {}
        assert d["edit_strength"] == 0.5
        assert d["regions"] == []
        assert d["use_preferences"] is True

    def test_round_trip_through_parse(self):
        """parse_constraint_spec(spec_to_dict(spec)) ≈ spec."""
        spec = HumanConstraintSpec(
            locks={"preserve_layout": True},
            targets={"brightness": "increase"},
            edit_strength=0.7,
            regions=[
                RegionConstraint(
                    id="r1", label="Sky", mode="modify", instruction="Brighten",
                    geometry_type="rect", geometry={"x": 0.0, "y": 0.0, "w": 1.0, "h": 0.5},
                    strength=0.9,
                )
            ],
            use_preferences=False,
        )
        d = spec_to_dict(spec)
        restored = parse_constraint_spec(d)
        assert restored.locks == spec.locks
        assert restored.targets == spec.targets
        assert restored.edit_strength == spec.edit_strength
        assert restored.use_preferences == spec.use_preferences
        assert len(restored.regions) == 1
        r = restored.regions[0]
        assert r.id == "r1"
        assert r.mode == "modify"
        assert r.strength == 0.9

    def test_dict_is_json_serializable(self):
        spec = HumanConstraintSpec(
            locks={"preserve_layout": True},
            targets={"x": "keep"},
            regions=[
                RegionConstraint(
                    id="r1", label="L", mode="modify", instruction="i",
                    geometry_type="rect", geometry={"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5},
                )
            ],
        )
        d = spec_to_dict(spec)
        # Should not raise
        json.dumps(d)


# ---------------------------------------------------------------------------
# constraint_to_artifacts
# ---------------------------------------------------------------------------


class TestConstraintToArtifacts:
    def test_writes_constraints_json(self, tmp_path):
        spec = HumanConstraintSpec(
            locks={"preserve_layout": True},
            edit_strength=0.6,
        )
        result = constraint_to_artifacts(tmp_path, spec)
        assert result is not None
        assert result.name == "constraints.json"
        assert result.exists()
        data = json.loads(result.read_text(encoding="utf-8"))
        assert data["locks"] == {"preserve_layout": True}
        assert data["edit_strength"] == 0.6

    def test_none_run_dir_returns_none(self):
        spec = HumanConstraintSpec()
        result = constraint_to_artifacts(None, spec)
        assert result is None

    def test_falsy_string_run_dir_returns_none(self):
        spec = HumanConstraintSpec()
        result = constraint_to_artifacts("", spec)
        assert result is None

    def test_returned_path_points_to_file(self, tmp_path):
        spec = HumanConstraintSpec()
        result = constraint_to_artifacts(tmp_path, spec)
        assert isinstance(result, Path)
        assert result.is_file()
