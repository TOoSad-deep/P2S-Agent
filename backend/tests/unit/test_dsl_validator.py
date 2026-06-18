"""Unit tests for dsl_validator.py — Phase 3 DSL validation."""

from __future__ import annotations

import copy

import pytest

from app.dsl.schema import (
    FIXTURE_BOX_GRADIENT,
    FIXTURE_CIRCLE_SOLID,
    FIXTURE_GLOW_RING,
    FIXTURE_ROUNDEDBOX_VIGNETTE,
)
from app.dsl.validator import ValidationResult, validate_dsl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clone(fixture: dict) -> dict:
    return copy.deepcopy(fixture)


# ---------------------------------------------------------------------------
# Valid fixtures
# ---------------------------------------------------------------------------

def test_validate_valid_circle_solid():
    result = validate_dsl(FIXTURE_CIRCLE_SOLID)
    assert isinstance(result, ValidationResult)
    assert result.valid is True
    assert result.errors == []


def test_validate_valid_box_gradient():
    result = validate_dsl(FIXTURE_BOX_GRADIENT)
    assert result.valid is True, result.errors


def test_validate_valid_glow_ring():
    result = validate_dsl(FIXTURE_GLOW_RING)
    assert result.valid is True, result.errors


def test_validate_valid_roundedbox_vignette():
    result = validate_dsl(FIXTURE_ROUNDEDBOX_VIGNETTE)
    assert result.valid is True, result.errors


# ---------------------------------------------------------------------------
# Missing top-level fields
# ---------------------------------------------------------------------------

def test_validate_missing_canvas():
    dsl = _clone(FIXTURE_CIRCLE_SOLID)
    del dsl["canvas"]
    result = validate_dsl(dsl)
    assert result.valid is False
    assert len(result.errors) > 0


def test_validate_missing_layers():
    dsl = _clone(FIXTURE_CIRCLE_SOLID)
    del dsl["layers"]
    result = validate_dsl(dsl)
    assert result.valid is False
    assert len(result.errors) > 0


def test_validate_empty_layers():
    dsl = _clone(FIXTURE_CIRCLE_SOLID)
    dsl["layers"] = []
    result = validate_dsl(dsl)
    assert result.valid is False
    assert len(result.errors) > 0


# ---------------------------------------------------------------------------
# Layer-level errors
# ---------------------------------------------------------------------------

def test_validate_bad_primitive_type():
    dsl = _clone(FIXTURE_CIRCLE_SOLID)
    dsl["layers"][0]["type"] = "triangle"
    result = validate_dsl(dsl)
    assert result.valid is False
    assert any("triangle" in e or "type" in e for e in result.errors)


def test_validate_bad_fill_type():
    dsl = _clone(FIXTURE_CIRCLE_SOLID)
    dsl["layers"][0]["fill"]["type"] = "texture"
    result = validate_dsl(dsl)
    assert result.valid is False
    assert any("texture" in e or "fill" in e for e in result.errors)


def test_validate_bad_opacity():
    dsl = _clone(FIXTURE_CIRCLE_SOLID)
    dsl["layers"][0]["opacity"] = 1.5
    result = validate_dsl(dsl)
    assert result.valid is False
    assert any("opacity" in e for e in result.errors)


def test_validate_bad_effect_type():
    dsl = _clone(FIXTURE_CIRCLE_SOLID)
    dsl["layers"][0]["effects"] = [{"type": "blur"}]
    result = validate_dsl(dsl)
    assert result.valid is False
    assert any("blur" in e or "effect" in e.lower() for e in result.errors)


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------

def test_validate_warns_duplicate_ids():
    dsl = _clone(FIXTURE_CIRCLE_SOLID)
    # Add a second layer with the same id
    second_layer = copy.deepcopy(dsl["layers"][0])
    dsl["layers"].append(second_layer)
    result = validate_dsl(dsl)
    assert result.valid is True  # duplicate id is a warning, not an error
    assert any("duplicate" in w.lower() or "id" in w.lower() for w in result.warnings)


def test_validate_warns_missing_params():
    dsl = _clone(FIXTURE_CIRCLE_SOLID)
    del dsl["layers"][0]["params"]
    result = validate_dsl(dsl)
    # Should still be valid (params is optional) but produce a warning
    assert result.valid is True
    assert any("params" in w.lower() for w in result.warnings)


# ---------------------------------------------------------------------------
# Value-level param validation (Bug 1)
# ---------------------------------------------------------------------------

def test_validate_center_must_be_two_numbers():
    dsl = _clone(FIXTURE_CIRCLE_SOLID)
    dsl["layers"][0]["params"]["center"] = 0.5
    result = validate_dsl(dsl)
    assert result.valid is False
    assert any("center" in e for e in result.errors)


def test_validate_radius_must_be_number():
    dsl = _clone(FIXTURE_CIRCLE_SOLID)
    dsl["layers"][0]["params"]["radius"] = "big"
    result = validate_dsl(dsl)
    assert result.valid is False
    assert any("radius" in e for e in result.errors)


def test_validate_polygon_sides_must_be_int_ge_3():
    dsl = _clone(FIXTURE_CIRCLE_SOLID)
    dsl["layers"][0]["type"] = "polygon"
    dsl["layers"][0]["params"] = {"center": [0.5, 0.5], "radius": 0.3, "sides": 2}
    result = validate_dsl(dsl)
    assert result.valid is False
    assert any("sides" in e for e in result.errors)


def test_validate_radius_must_be_finite():
    dsl = _clone(FIXTURE_CIRCLE_SOLID)
    dsl["layers"][0]["params"]["radius"] = float("inf")
    result = validate_dsl(dsl)
    assert result.valid is False
    assert any("radius" in e or "finite" in e for e in result.errors)


def test_validate_gradient_stop_position_must_be_numeric():
    dsl = _clone(FIXTURE_BOX_GRADIENT)
    dsl["layers"][0]["fill"]["stops"][0]["position"] = "start"
    result = validate_dsl(dsl)
    assert result.valid is False
    assert any("position" in e for e in result.errors)


def test_validate_gradient_stop_position_must_be_in_unit_range():
    dsl = _clone(FIXTURE_BOX_GRADIENT)
    dsl["layers"][0]["fill"]["stops"][0]["position"] = 1.5
    result = validate_dsl(dsl)
    assert result.valid is False
    assert any("position" in e for e in result.errors)


def test_validate_size_must_be_two_numbers():
    dsl = _clone(FIXTURE_CIRCLE_SOLID)
    dsl["layers"][0]["type"] = "box"
    dsl["layers"][0]["params"] = {"center": [0.5, 0.5], "size": [0.3, "wide"]}
    result = validate_dsl(dsl)
    assert result.valid is False
    assert any("size" in e for e in result.errors)


def test_validate_scale_transform_must_be_numeric():
    dsl = _clone(FIXTURE_CIRCLE_SOLID)
    dsl["layers"][0]["transform"] = {"type": "scale", "x": "nope", "y": 1.0}
    result = validate_dsl(dsl)
    assert result.valid is False
    assert any("scale" in e or "transform" in e for e in result.errors)


def test_validate_good_polygon_still_valid():
    dsl = _clone(FIXTURE_CIRCLE_SOLID)
    dsl["layers"][0]["type"] = "polygon"
    dsl["layers"][0]["params"] = {"center": [0.5, 0.5], "radius": 0.3, "sides": 6}
    result = validate_dsl(dsl)
    assert result.valid is True, result.errors
