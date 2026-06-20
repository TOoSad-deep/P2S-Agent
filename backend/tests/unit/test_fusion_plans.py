"""Tests for backend/app/pipeline/fusion_plans.py (V4.5, TDD).

Run with:
    cd backend && python3 -m pytest tests/unit/test_fusion_plans.py -v
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from app.pipeline.fusion_plans import (
    FusionPlanRecord,
    FusionRegion,
    append_plan_event,
    build_fusion_notes,
    load_plan,
    load_plan_events,
    parse_fusion_plan,
    plan_to_dict,
    save_plan,
    validate_fusion_plan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_region(
    id: str = "r1",
    label: str = "sky",
    source_run_id: str = "run-src-1",
    instruction: str = "soft light",
    geometry_type: str = "rect",
    geometry: dict | None = None,
    **kwargs,
) -> FusionRegion:
    return FusionRegion(
        id=id,
        label=label,
        source_run_id=source_run_id,
        instruction=instruction,
        geometry_type=geometry_type,
        geometry=geometry if geometry is not None else {"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5},
        **kwargs,
    )


def _make_record(**overrides) -> FusionPlanRecord:
    defaults = dict(
        fusion_id="fus-001",
        root_run_id="run-root",
        parent_run_id="run-parent",
        base_run_id="run-base",
        source_run_ids=["run-src-1"],
        draw_session_id=None,
        feedback="borrow sky glow",
        status="draft",
        regions=[_make_region()],
        created_at=1_700_000_000.0,
    )
    defaults.update(overrides)
    return FusionPlanRecord(**defaults)


# ---------------------------------------------------------------------------
# parse_fusion_plan
# ---------------------------------------------------------------------------


class TestParseFusionPlan:
    def _parse(self, payload: dict, **kwargs) -> FusionPlanRecord:
        return parse_fusion_plan(
            payload,
            fusion_id=kwargs.get("fusion_id", "fus-parse-1"),
            root_run_id=kwargs.get("root_run_id", "run-root"),
            parent_run_id=kwargs.get("parent_run_id", "run-parent"),
            created_at=kwargs.get("created_at", 1_700_000_000.0),
        )

    def test_minimal_payload_with_one_region(self):
        payload = {
            "base_run_id": "run-base",
            "regions": [
                {
                    "id": "r1",
                    "label": "sky",
                    "source_run_id": "run-src-1",
                    "instruction": "soft glow",
                    "geometry_type": "rect",
                    "geometry": {"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5},
                }
            ],
        }
        record = self._parse(payload)
        assert record.base_run_id == "run-base"
        assert len(record.regions) == 1
        assert record.regions[0].id == "r1"
        assert record.status == "draft"

    def test_source_run_ids_derived_from_regions_when_absent(self):
        payload = {
            "base_run_id": "run-base",
            "regions": [
                {
                    "id": "r1",
                    "label": "sky",
                    "source_run_id": "run-src-A",
                    "instruction": "glow",
                    "geometry_type": "rect",
                    "geometry": {"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5},
                },
                {
                    "id": "r2",
                    "label": "ground",
                    "source_run_id": "run-src-B",
                    "instruction": "texture",
                    "geometry_type": "rect",
                    "geometry": {"x": 0.0, "y": 0.5, "w": 0.5, "h": 0.5},
                },
            ],
        }
        record = self._parse(payload)
        # Both source run IDs derived, dedup-stable order
        assert "run-src-A" in record.source_run_ids
        assert "run-src-B" in record.source_run_ids
        assert record.source_run_ids.index("run-src-A") < record.source_run_ids.index("run-src-B")

    def test_source_run_ids_deduped_from_regions(self):
        payload = {
            "base_run_id": "run-base",
            "regions": [
                {
                    "id": "r1",
                    "label": "sky",
                    "source_run_id": "run-src-1",
                    "instruction": "glow",
                    "geometry_type": "rect",
                    "geometry": {"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5},
                },
                {
                    "id": "r2",
                    "label": "horizon",
                    "source_run_id": "run-src-1",  # same source
                    "instruction": "blur",
                    "geometry_type": "rect",
                    "geometry": {"x": 0.0, "y": 0.5, "w": 0.5, "h": 0.5},
                },
            ],
        }
        record = self._parse(payload)
        assert record.source_run_ids == ["run-src-1"]

    def test_source_run_ids_from_payload_takes_precedence(self):
        payload = {
            "base_run_id": "run-base",
            "source_run_ids": ["explicit-src-1", "explicit-src-2"],
            "regions": [
                {
                    "id": "r1",
                    "label": "sky",
                    "source_run_id": "other-src",
                    "instruction": "glow",
                    "geometry_type": "rect",
                    "geometry": {"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5},
                },
            ],
        }
        record = self._parse(payload)
        assert record.source_run_ids == ["explicit-src-1", "explicit-src-2"]

    def test_region_defaults_applied(self):
        payload = {
            "base_run_id": "run-base",
            "regions": [
                {
                    "id": "r1",
                    "label": "sky",
                    "source_run_id": "run-src-1",
                    "instruction": "glow",
                    # no geometry_type, blend_mode, strength, feather
                    "geometry": {"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5},
                }
            ],
        }
        record = self._parse(payload)
        region = record.regions[0]
        assert region.geometry_type == "rect"
        assert region.blend_mode == "soft"
        assert region.strength == 0.5
        assert region.feather == 0.08

    def test_bad_strength_coerced_to_default(self):
        payload = {
            "base_run_id": "run-base",
            "regions": [
                {
                    "id": "r1",
                    "label": "sky",
                    "source_run_id": "run-src-1",
                    "instruction": "glow",
                    "geometry_type": "rect",
                    "geometry": {"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5},
                    "strength": "not-a-float",
                }
            ],
        }
        record = self._parse(payload)
        assert record.regions[0].strength == 0.5

    def test_bad_feather_coerced_to_default(self):
        payload = {
            "base_run_id": "run-base",
            "regions": [
                {
                    "id": "r1",
                    "label": "sky",
                    "source_run_id": "run-src-1",
                    "instruction": "glow",
                    "geometry_type": "rect",
                    "geometry": {"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5},
                    "feather": [1, 2, 3],  # non-scalar
                }
            ],
        }
        record = self._parse(payload)
        assert record.regions[0].feather == 0.08

    def test_status_always_draft(self):
        payload = {"base_run_id": "run-base", "status": "completed"}
        record = self._parse(payload)
        assert record.status == "draft"

    def test_metadata_passed_through(self):
        payload = {"base_run_id": "run-base", "metadata": {"version": 1}}
        record = self._parse(payload)
        assert record.metadata == {"version": 1}

    def test_draw_session_id_optional(self):
        payload = {"base_run_id": "run-base", "draw_session_id": "draw-42"}
        record = self._parse(payload)
        assert record.draw_session_id == "draw-42"

    def test_created_at_preserved(self):
        payload = {"base_run_id": "run-base"}
        record = self._parse(payload, created_at=9_999_999.5)
        assert record.created_at == 9_999_999.5

    def test_fusion_id_set_from_kwarg(self):
        payload = {"base_run_id": "run-base"}
        record = self._parse(payload, fusion_id="fus-xyz")
        assert record.fusion_id == "fus-xyz"

    # ------------------------------------------------------------------
    # Tolerance: non-dict / non-list / None payload fields must not raise
    # ------------------------------------------------------------------

    def test_string_metadata_does_not_raise_and_coerces_to_empty_dict(self):
        """metadata='not-a-dict' must not raise; record.metadata == {}."""
        payload = {
            "base_run_id": "b",
            "metadata": "not-a-dict",
            "regions": [
                {
                    "id": "r1",
                    "label": "sky",
                    "source_run_id": "run-src-1",
                    "instruction": "glow",
                    "geometry_type": "rect",
                    "geometry": {"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5},
                }
            ],
        }
        record = parse_fusion_plan(
            payload,
            fusion_id="f",
            root_run_id="r",
            parent_run_id="p",
            created_at=1.0,
        )
        assert record.metadata == {}

    def test_string_geometry_does_not_raise_and_coerces_to_empty_dict(self):
        """A region with geometry='not-a-dict' must not raise; geometry == {}."""
        payload = {
            "base_run_id": "b",
            "regions": [
                {
                    "id": "r1",
                    "label": "sky",
                    "source_run_id": "run-src-1",
                    "instruction": "glow",
                    "geometry_type": "rect",
                    "geometry": "not-a-dict",
                }
            ],
        }
        record = parse_fusion_plan(
            payload,
            fusion_id="f",
            root_run_id="r",
            parent_run_id="p",
            created_at=1.0,
        )
        assert len(record.regions) == 1
        assert record.regions[0].geometry == {}

    def test_string_source_run_ids_does_not_raise_and_coerces_to_empty(self):
        """source_run_ids='notalist' must not raise; derived from regions (empty → [])."""
        payload = {
            "base_run_id": "b",
            "source_run_ids": "notalist",
        }
        record = parse_fusion_plan(
            payload,
            fusion_id="f",
            root_run_id="r",
            parent_run_id="p",
            created_at=1.0,
        )
        # No regions → derived list is empty
        assert record.source_run_ids == []

    def test_none_payload_does_not_raise(self):
        """parse_fusion_plan(None, ...) must not raise; treats payload as {}."""
        record = parse_fusion_plan(
            None,  # type: ignore[arg-type]
            fusion_id="f",
            root_run_id="r",
            parent_run_id="p",
            created_at=1.0,
        )
        assert record.base_run_id == ""
        assert record.metadata == {}
        assert record.regions == []

    def test_non_dict_region_entry_is_skipped(self):
        """A region entry that is not a dict must be silently skipped."""
        payload = {
            "base_run_id": "b",
            "regions": [
                "not-a-dict-region",
                {
                    "id": "r1",
                    "label": "sky",
                    "source_run_id": "run-src-1",
                    "instruction": "glow",
                    "geometry_type": "rect",
                    "geometry": {"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5},
                },
            ],
        }
        record = parse_fusion_plan(
            payload,
            fusion_id="f",
            root_run_id="r",
            parent_run_id="p",
            created_at=1.0,
        )
        # Only the valid dict region is kept
        assert len(record.regions) == 1
        assert record.regions[0].id == "r1"


# ---------------------------------------------------------------------------
# validate_fusion_plan
# ---------------------------------------------------------------------------


class TestValidateFusionPlan:
    def _valid_record(self, **overrides) -> FusionPlanRecord:
        return _make_record(**overrides)

    def test_valid_plan_returns_empty_list(self):
        record = self._valid_record()
        errors = validate_fusion_plan(record)
        assert errors == []

    def test_missing_base_run_id_returns_error(self):
        record = self._valid_record(base_run_id="")
        errors = validate_fusion_plan(record)
        assert any("base_run_id is required" in e for e in errors)

    def test_region_source_run_id_not_in_source_run_ids(self):
        region = _make_region(id="r1", source_run_id="run-not-listed")
        record = self._valid_record(
            source_run_ids=["run-src-1"],
            regions=[region],
        )
        errors = validate_fusion_plan(record)
        assert any("r1" in e and "run-not-listed" in e for e in errors)

    def test_region_source_run_id_empty_is_error(self):
        region = _make_region(id="r1", source_run_id="")
        record = self._valid_record(
            source_run_ids=["run-src-1"],
            regions=[region],
        )
        errors = validate_fusion_plan(record)
        assert any("r1" in e for e in errors)

    def test_duplicate_region_id_returns_error(self):
        r1a = _make_region(id="r1", geometry={"x": 0.0, "y": 0.0, "w": 0.3, "h": 0.3})
        r1b = _make_region(id="r1", geometry={"x": 0.5, "y": 0.5, "w": 0.3, "h": 0.3})
        record = self._valid_record(regions=[r1a, r1b])
        errors = validate_fusion_plan(record)
        assert any("duplicate" in e.lower() and "r1" in e for e in errors)

    def test_rect_out_of_bounds_x_plus_w_gt_1(self):
        region = _make_region(
            id="r1",
            geometry={"x": 0.8, "y": 0.0, "w": 0.5, "h": 0.5},
        )
        record = self._valid_record(regions=[region])
        errors = validate_fusion_plan(record)
        assert any("r1" in e for e in errors)

    def test_rect_out_of_bounds_y_plus_h_gt_1(self):
        region = _make_region(
            id="r1",
            geometry={"x": 0.0, "y": 0.8, "w": 0.5, "h": 0.5},
        )
        record = self._valid_record(regions=[region])
        errors = validate_fusion_plan(record)
        assert any("r1" in e for e in errors)

    def test_rect_zero_width_is_error(self):
        region = _make_region(
            id="r1",
            geometry={"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.5},
        )
        record = self._valid_record(regions=[region])
        errors = validate_fusion_plan(record)
        assert any("r1" in e for e in errors)

    def test_rect_negative_x_is_error(self):
        region = _make_region(
            id="r1",
            geometry={"x": -0.1, "y": 0.0, "w": 0.5, "h": 0.5},
        )
        record = self._valid_record(regions=[region])
        errors = validate_fusion_plan(record)
        assert any("r1" in e for e in errors)

    def test_rect_exactly_at_boundary_is_valid(self):
        region = _make_region(
            id="r1",
            geometry={"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
        )
        record = self._valid_record(regions=[region])
        errors = validate_fusion_plan(record)
        assert errors == []

    def test_bad_blend_mode_returns_error(self):
        region = _make_region(id="r1", blend_mode="invalid_mode")
        record = self._valid_record(regions=[region])
        errors = validate_fusion_plan(record)
        assert any("blend_mode" in e and "r1" in e for e in errors)

    def test_valid_blend_modes(self):
        for mode in ["soft", "replace_target", "protect_base"]:
            region = _make_region(id="r1", blend_mode=mode)
            record = self._valid_record(regions=[region])
            errors = validate_fusion_plan(record)
            blend_errors = [e for e in errors if "blend_mode" in e]
            assert blend_errors == [], f"mode {mode!r} should be valid"

    def test_strength_out_of_range_is_error(self):
        region = _make_region(id="r1", strength=1.5)
        record = self._valid_record(regions=[region])
        errors = validate_fusion_plan(record)
        assert any("strength" in e and "r1" in e for e in errors)

    def test_feather_out_of_range_is_error(self):
        region = _make_region(id="r1", feather=-0.1)
        record = self._valid_record(regions=[region])
        errors = validate_fusion_plan(record)
        assert any("feather" in e and "r1" in e for e in errors)

    def test_geometry_type_invalid_is_error(self):
        region = _make_region(id="r1", geometry_type="hexagon")
        record = self._valid_record(regions=[region])
        errors = validate_fusion_plan(record)
        assert any("geometry_type" in e and "r1" in e for e in errors)

    def test_polygon_geometry_type_not_bounds_checked(self):
        # polygon/mask are valid types but don't trigger rect bounds checks
        region = _make_region(
            id="r1",
            geometry_type="polygon",
            geometry={"points": [[0, 0], [1, 0], [0.5, 1]]},
        )
        record = self._valid_record(regions=[region])
        errors = validate_fusion_plan(record)
        # No bounds error for polygon
        geo_errors = [e for e in errors if "bounds" in e.lower() or "geometry" in e.lower()]
        assert geo_errors == []

    def test_multiple_errors_returned(self):
        # Missing base_run_id AND bad region
        region = _make_region(id="r1", blend_mode="INVALID", source_run_id="not-listed")
        record = self._valid_record(base_run_id="", regions=[region])
        errors = validate_fusion_plan(record)
        assert len(errors) >= 2


# ---------------------------------------------------------------------------
# build_fusion_notes
# ---------------------------------------------------------------------------


class TestBuildFusionNotes:
    def test_includes_fusion_goal_header(self):
        record = _make_record()
        notes = build_fusion_notes(record)
        assert any("[FUSION GOAL]" in n for n in notes)

    def test_includes_base_run_id_in_base_note(self):
        record = _make_record(base_run_id="run-base-42")
        notes = build_fusion_notes(record)
        assert any("[BASE]" in n and "run-base-42" in n for n in notes)

    def test_includes_region_source_note_with_id(self):
        region = _make_region(id="r1", source_run_id="run-src-1", instruction="soft glow")
        record = _make_record(regions=[region])
        notes = build_fusion_notes(record)
        assert any("[REGION SOURCE r1]" in n for n in notes)

    def test_region_note_contains_source_run_id(self):
        region = _make_region(id="r1", source_run_id="run-src-1", instruction="soft glow")
        record = _make_record(regions=[region])
        notes = build_fusion_notes(record)
        region_notes = [n for n in notes if "[REGION SOURCE r1]" in n]
        assert len(region_notes) == 1
        assert "run-src-1" in region_notes[0]

    def test_region_note_contains_instruction(self):
        region = _make_region(
            id="r1",
            source_run_id="run-src-1",
            instruction="warm sunlight",
        )
        record = _make_record(regions=[region])
        notes = build_fusion_notes(record)
        region_notes = [n for n in notes if "[REGION SOURCE r1]" in n]
        assert "warm sunlight" in region_notes[0]

    def test_rect_region_note_contains_coords(self):
        region = _make_region(
            id="r1",
            geometry={"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
        )
        record = _make_record(regions=[region])
        notes = build_fusion_notes(record)
        region_notes = [n for n in notes if "[REGION SOURCE r1]" in n]
        note = region_notes[0]
        assert "x=0.1" in note
        assert "y=0.2" in note
        assert "w=0.3" in note
        assert "h=0.4" in note

    def test_region_note_contains_blend_mode_and_strength(self):
        region = _make_region(
            id="r1",
            blend_mode="replace_target",
            strength=0.75,
        )
        record = _make_record(regions=[region])
        notes = build_fusion_notes(record)
        region_notes = [n for n in notes if "[REGION SOURCE r1]" in n]
        note = region_notes[0]
        assert "replace_target" in note
        assert "0.75" in note

    def test_includes_important_footer(self):
        record = _make_record()
        notes = build_fusion_notes(record)
        assert any("[IMPORTANT]" in n for n in notes)

    def test_notes_order_goal_base_regions_important(self):
        region = _make_region(id="r1")
        record = _make_record(regions=[region])
        notes = build_fusion_notes(record)
        # Find indices
        goal_idx = next(i for i, n in enumerate(notes) if "[FUSION GOAL]" in n)
        base_idx = next(i for i, n in enumerate(notes) if "[BASE]" in n)
        region_idx = next(i for i, n in enumerate(notes) if "[REGION SOURCE" in n)
        important_idx = next(i for i, n in enumerate(notes) if "[IMPORTANT]" in n)
        assert goal_idx < base_idx < region_idx < important_idx

    def test_deterministic_output(self):
        record = _make_record()
        assert build_fusion_notes(record) == build_fusion_notes(record)

    def test_multiple_regions_each_get_note(self):
        r1 = _make_region(id="r1", geometry={"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5})
        r2 = _make_region(
            id="r2",
            source_run_id="run-src-2",
            geometry={"x": 0.5, "y": 0.5, "w": 0.5, "h": 0.5},
        )
        record = _make_record(
            source_run_ids=["run-src-1", "run-src-2"],
            regions=[r1, r2],
        )
        notes = build_fusion_notes(record)
        assert any("[REGION SOURCE r1]" in n for n in notes)
        assert any("[REGION SOURCE r2]" in n for n in notes)


# ---------------------------------------------------------------------------
# Persistence round-trip: save_plan / load_plan
# ---------------------------------------------------------------------------


class TestPlanPersistence:
    def test_save_returns_path(self, tmp_path):
        record = _make_record()
        p = save_plan(record, root=tmp_path)
        assert isinstance(p, Path)
        assert p.exists()

    def test_save_uses_fusions_subdir(self, tmp_path):
        record = _make_record(fusion_id="fus-subdir-test")
        p = save_plan(record, root=tmp_path)
        assert p.parent.name == "fusions"

    def test_load_after_save_restores_all_scalar_fields(self, tmp_path):
        record = _make_record(
            fusion_id="fus-rt-1",
            base_run_id="run-base",
            source_run_ids=["run-src-1"],
            feedback="test feedback",
            status="draft",
            created_at=1_700_000_000.0,
            metadata={"tag": "v4.5"},
        )
        save_plan(record, root=tmp_path)
        loaded = load_plan("fus-rt-1", root=tmp_path)
        assert loaded is not None
        assert loaded.fusion_id == record.fusion_id
        assert loaded.base_run_id == record.base_run_id
        assert loaded.source_run_ids == record.source_run_ids
        assert loaded.feedback == record.feedback
        assert loaded.status == record.status
        assert loaded.created_at == record.created_at
        assert loaded.metadata == record.metadata

    def test_load_after_save_preserves_regions(self, tmp_path):
        region = _make_region(
            id="r1",
            label="sky",
            source_run_id="run-src-1",
            instruction="warm glow",
            geometry={"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
            blend_mode="replace_target",
            strength=0.75,
            feather=0.12,
        )
        record = _make_record(regions=[region])
        save_plan(record, root=tmp_path)
        loaded = load_plan(record.fusion_id, root=tmp_path)
        assert loaded is not None
        assert len(loaded.regions) == 1
        lr = loaded.regions[0]
        assert lr.id == "r1"
        assert lr.label == "sky"
        assert lr.source_run_id == "run-src-1"
        assert lr.instruction == "warm glow"
        assert lr.geometry == {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}
        assert lr.blend_mode == "replace_target"
        assert lr.strength == 0.75
        assert lr.feather == 0.12

    def test_load_preserves_optional_none_fields(self, tmp_path):
        record = _make_record(
            draw_session_id=None,
            composite_target_artifact_id=None,
            output_run_id=None,
            updated_at=None,
        )
        save_plan(record, root=tmp_path)
        loaded = load_plan(record.fusion_id, root=tmp_path)
        assert loaded is not None
        assert loaded.draw_session_id is None
        assert loaded.composite_target_artifact_id is None
        assert loaded.output_run_id is None
        assert loaded.updated_at is None

    def test_load_preserves_set_optional_fields(self, tmp_path):
        record = _make_record(
            draw_session_id="draw-99",
            composite_target_artifact_id="art-42",
            output_run_id="run-out-1",
            updated_at=1_700_000_100.0,
        )
        save_plan(record, root=tmp_path)
        loaded = load_plan(record.fusion_id, root=tmp_path)
        assert loaded is not None
        assert loaded.draw_session_id == "draw-99"
        assert loaded.composite_target_artifact_id == "art-42"
        assert loaded.output_run_id == "run-out-1"
        assert loaded.updated_at == 1_700_000_100.0

    def test_load_missing_returns_none(self, tmp_path):
        result = load_plan("does-not-exist", root=tmp_path)
        assert result is None

    def test_load_malformed_json_returns_none(self, tmp_path):
        fus_dir = tmp_path / "fusions"
        fus_dir.mkdir(parents=True, exist_ok=True)
        (fus_dir / "bad-fus.json").write_text("not json{{{", encoding="utf-8")
        result = load_plan("bad-fus", root=tmp_path)
        assert result is None

    def test_load_non_dict_json_list_returns_none(self, tmp_path):
        fus_dir = tmp_path / "fusions"
        fus_dir.mkdir(parents=True, exist_ok=True)
        (fus_dir / "fus-list.json").write_text("[1, 2, 3]", encoding="utf-8")
        result = load_plan("fus-list", root=tmp_path)
        assert result is None

    def test_load_non_dict_json_string_returns_none(self, tmp_path):
        fus_dir = tmp_path / "fusions"
        fus_dir.mkdir(parents=True, exist_ok=True)
        (fus_dir / "fus-str.json").write_text('"just a string"', encoding="utf-8")
        result = load_plan("fus-str", root=tmp_path)
        assert result is None

    def test_load_non_dict_json_number_returns_none(self, tmp_path):
        fus_dir = tmp_path / "fusions"
        fus_dir.mkdir(parents=True, exist_ok=True)
        (fus_dir / "fus-num.json").write_text("42", encoding="utf-8")
        result = load_plan("fus-num", root=tmp_path)
        assert result is None

    def test_load_tolerates_string_created_at(self, tmp_path):
        fus_dir = tmp_path / "fusions"
        fus_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "fusion_id": "fus-float",
            "root_run_id": "r",
            "parent_run_id": "p",
            "base_run_id": "b",
            "source_run_ids": [],
            "draw_session_id": None,
            "feedback": "",
            "status": "draft",
            "created_at": "1700000000.0",
        }
        (fus_dir / "fus-float.json").write_text(json.dumps(data), encoding="utf-8")
        loaded = load_plan("fus-float", root=tmp_path)
        assert loaded is not None
        assert loaded.created_at == 1_700_000_000.0

    def test_overwrite_with_updated_record(self, tmp_path):
        record = _make_record(status="draft")
        save_plan(record, root=tmp_path)
        record2 = _make_record(status="completed", output_run_id="run-out-99")
        save_plan(record2, root=tmp_path)
        loaded = load_plan(record.fusion_id, root=tmp_path)
        assert loaded.status == "completed"
        assert loaded.output_run_id == "run-out-99"

    def test_full_asdict_round_trip(self, tmp_path):
        region = _make_region(id="r1")
        record = _make_record(regions=[region])
        save_plan(record, root=tmp_path)
        loaded = load_plan(record.fusion_id, root=tmp_path)
        assert asdict(loaded) == asdict(record)

    def test_no_real_test_results_pollution(self, tmp_path):
        """Ensure tests never touch the real DEFAULT_RESULTS_ROOT."""
        from p2s_agent.core.pipeline.artifacts import DEFAULT_RESULTS_ROOT

        record = _make_record(fusion_id="fus-isolation")
        save_plan(record, root=tmp_path)
        real_dir = DEFAULT_RESULTS_ROOT / "fusions"
        if real_dir.exists():
            assert not (real_dir / "fus-isolation.json").exists()


# ---------------------------------------------------------------------------
# append_plan_event / load_plan_events
# ---------------------------------------------------------------------------


class TestPlanEvents:
    def test_append_then_load_returns_events_in_order(self, tmp_path):
        ev1 = {"type": "status_changed", "status": "running", "ts": 1.0}
        ev2 = {"type": "region_ready", "region_id": "r1", "ts": 2.0}
        append_plan_event("fus-ev-1", ev1, root=tmp_path)
        append_plan_event("fus-ev-1", ev2, root=tmp_path)
        events = load_plan_events("fus-ev-1", root=tmp_path)
        assert events == [ev1, ev2]

    def test_load_events_missing_file_returns_empty_list(self, tmp_path):
        events = load_plan_events("nonexistent-fus", root=tmp_path)
        assert events == []

    def test_malformed_jsonl_line_is_skipped(self, tmp_path):
        fus_dir = tmp_path / "fusions"
        fus_dir.mkdir(parents=True, exist_ok=True)
        ev_path = fus_dir / "fus-mal_events.jsonl"
        ev_path.write_text(
            '{"type": "good"}\n'
            "NOT JSON {\n"
            '{"type": "also_good"}\n',
            encoding="utf-8",
        )
        events = load_plan_events("fus-mal", root=tmp_path)
        assert len(events) == 2
        assert events[0]["type"] == "good"
        assert events[1]["type"] == "also_good"

    def test_blank_lines_skipped(self, tmp_path):
        fus_dir = tmp_path / "fusions"
        fus_dir.mkdir(parents=True, exist_ok=True)
        ev_path = fus_dir / "fus-blank_events.jsonl"
        ev_path.write_text(
            '{"type": "ev1"}\n'
            "\n"
            "   \n"
            '{"type": "ev2"}\n',
            encoding="utf-8",
        )
        events = load_plan_events("fus-blank", root=tmp_path)
        assert [e["type"] for e in events] == ["ev1", "ev2"]

    def test_non_dict_json_line_is_skipped(self, tmp_path):
        fus_dir = tmp_path / "fusions"
        fus_dir.mkdir(parents=True, exist_ok=True)
        ev_path = fus_dir / "fus-nondict_events.jsonl"
        ev_path.write_text(
            '{"type": "ok"}\n'
            '"just a string"\n'
            "[1, 2, 3]\n"
            '{"type": "also_ok"}\n',
            encoding="utf-8",
        )
        events = load_plan_events("fus-nondict", root=tmp_path)
        assert len(events) == 2
        assert events[0]["type"] == "ok"
        assert events[1]["type"] == "also_ok"

    def test_events_stored_in_fusions_subdir(self, tmp_path):
        append_plan_event("fus-path-check", {"x": 1}, root=tmp_path)
        expected = tmp_path / "fusions" / "fus-path-check_events.jsonl"
        assert expected.exists()

    def test_events_ensure_ascii_false(self, tmp_path):
        # Non-ASCII content must survive
        event = {"type": "note", "text": "你好世界"}
        append_plan_event("fus-unicode", event, root=tmp_path)
        events = load_plan_events("fus-unicode", root=tmp_path)
        assert events[0]["text"] == "你好世界"

    def test_no_real_test_results_pollution(self, tmp_path):
        """Ensure tests never touch the real DEFAULT_RESULTS_ROOT."""
        from p2s_agent.core.pipeline.artifacts import DEFAULT_RESULTS_ROOT

        append_plan_event("fus-isolation-ev", {"ts": 0}, root=tmp_path)
        real_dir = DEFAULT_RESULTS_ROOT / "fusions"
        if real_dir.exists():
            assert not (real_dir / "fus-isolation-ev_events.jsonl").exists()

    def test_load_plan_events_returns_empty_on_permission_error(
        self, tmp_path, monkeypatch
    ):
        """If open() raises PermissionError (TCC loss on macOS Documents),
        load_plan_events must degrade gracefully and return []."""
        append_plan_event("fus-perm", {"type": "ev1", "ts": 1.0}, root=tmp_path)
        ev_path = tmp_path / "fusions" / "fus-perm_events.jsonl"
        assert ev_path.exists()

        real_open = Path.open

        def fake_open(self, *args, **kwargs):
            if self == ev_path:
                raise PermissionError(1, "Operation not permitted")
            return real_open(self, *args, **kwargs)

        monkeypatch.setattr(Path, "open", fake_open)
        assert load_plan_events("fus-perm", root=tmp_path) == []


# ---------------------------------------------------------------------------
# plan_to_dict
# ---------------------------------------------------------------------------


class TestPlanToDict:
    def test_returns_dict(self):
        record = _make_record()
        result = plan_to_dict(record)
        assert isinstance(result, dict)

    def test_json_serializable(self):
        record = _make_record()
        result = plan_to_dict(record)
        # Should not raise
        serialized = json.dumps(result)
        assert isinstance(serialized, str)

    def test_regions_nested_as_dicts(self):
        region = _make_region(id="r1")
        record = _make_record(regions=[region])
        result = plan_to_dict(record)
        assert isinstance(result["regions"], list)
        assert isinstance(result["regions"][0], dict)
        assert result["regions"][0]["id"] == "r1"

    def test_preserves_fusion_id(self):
        record = _make_record(fusion_id="fus-dict-42")
        result = plan_to_dict(record)
        assert result["fusion_id"] == "fus-dict-42"

    def test_matches_asdict(self):
        region = _make_region(id="r1")
        record = _make_record(regions=[region])
        from dataclasses import asdict
        assert plan_to_dict(record) == asdict(record)
