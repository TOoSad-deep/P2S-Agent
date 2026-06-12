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
