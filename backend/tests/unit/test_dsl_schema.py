"""Unit tests for dsl_schema.py — Phase 3 DSL schema definitions."""

from __future__ import annotations

import pytest

from p2s_agent.core.dsl.schema import (
    DSL_SCHEMA_VERSION,
    EFFECT_TYPES,
    FILL_TYPES,
    FIXTURE_BOX_GRADIENT,
    FIXTURE_CIRCLE_SOLID,
    FIXTURE_GLOW_RING,
    FIXTURE_ROUNDEDBOX_VIGNETTE,
    PRIMITIVE_TYPES,
)


# ---------------------------------------------------------------------------
# Fixture completeness
# ---------------------------------------------------------------------------

def test_fixture_circle_solid_is_complete():
    assert "schema_version" in FIXTURE_CIRCLE_SOLID
    assert "canvas" in FIXTURE_CIRCLE_SOLID
    assert "layers" in FIXTURE_CIRCLE_SOLID
    layer = FIXTURE_CIRCLE_SOLID["layers"][0]
    assert layer["type"] == "circle"
    assert layer["fill"]["type"] == "solid"
    assert "params" in layer


def test_fixture_box_gradient_is_complete():
    assert "schema_version" in FIXTURE_BOX_GRADIENT
    assert "canvas" in FIXTURE_BOX_GRADIENT
    assert "layers" in FIXTURE_BOX_GRADIENT
    layer = FIXTURE_BOX_GRADIENT["layers"][0]
    assert layer["type"] == "box"
    assert layer["fill"]["type"] == "linearGradient"
    assert "stops" in layer["fill"]
    assert len(layer["fill"]["stops"]) >= 2


def test_all_fixtures_have_schema_version():
    fixtures = [
        FIXTURE_CIRCLE_SOLID,
        FIXTURE_BOX_GRADIENT,
        FIXTURE_GLOW_RING,
        FIXTURE_ROUNDEDBOX_VIGNETTE,
    ]
    for f in fixtures:
        assert "schema_version" in f, f"Missing schema_version in fixture: {f}"
        assert isinstance(f["schema_version"], int)


def test_all_fixtures_have_layers():
    fixtures = [
        FIXTURE_CIRCLE_SOLID,
        FIXTURE_BOX_GRADIENT,
        FIXTURE_GLOW_RING,
        FIXTURE_ROUNDEDBOX_VIGNETTE,
    ]
    for f in fixtures:
        assert "layers" in f
        assert isinstance(f["layers"], list)
        assert len(f["layers"]) >= 1


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

def test_dsl_schema_version_is_int():
    assert isinstance(DSL_SCHEMA_VERSION, int)
    assert DSL_SCHEMA_VERSION >= 1


def test_primitive_types_contains_required():
    required = {"circle", "ellipse", "box", "roundedBox", "ring", "polygon"}
    assert required.issubset(set(PRIMITIVE_TYPES))


def test_fill_types_contains_required():
    required = {"solid", "linearGradient", "radialGradient"}
    assert required.issubset(set(FILL_TYPES))


def test_effect_types_contains_required():
    required = {"glow", "vignette", "grain"}
    assert required.issubset(set(EFFECT_TYPES))
