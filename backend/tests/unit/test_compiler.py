"""Unit tests for compiler.py — Phase 3 DSL-to-GLSL compiler."""

from __future__ import annotations

import copy

import pytest

from app.dsl.compiler import CompileResult, compile_dsl
from app.dsl.schema import (
    DSL_SCHEMA_VERSION,
    FIXTURE_BOX_GRADIENT,
    FIXTURE_CIRCLE_SOLID,
    FIXTURE_GLOW_RING,
    FIXTURE_ROUNDEDBOX_VIGNETTE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clone(fixture: dict) -> dict:
    return copy.deepcopy(fixture)


# ---------------------------------------------------------------------------
# Basic success tests
# ---------------------------------------------------------------------------

def test_compile_circle_solid_succeeds():
    result = compile_dsl(FIXTURE_CIRCLE_SOLID)
    assert isinstance(result, CompileResult)
    assert result.success is True
    assert len(result.glsl) > 0


def test_compile_box_gradient_succeeds():
    result = compile_dsl(FIXTURE_BOX_GRADIENT)
    assert result.success is True, result.errors
    assert len(result.glsl) > 0


def test_compile_glow_ring_succeeds():
    result = compile_dsl(FIXTURE_GLOW_RING)
    assert result.success is True, result.errors
    assert len(result.glsl) > 0


def test_compile_roundedbox_vignette_succeeds():
    result = compile_dsl(FIXTURE_ROUNDEDBOX_VIGNETTE)
    assert result.success is True, result.errors
    assert len(result.glsl) > 0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_compile_is_deterministic():
    result1 = compile_dsl(FIXTURE_CIRCLE_SOLID)
    result2 = compile_dsl(FIXTURE_CIRCLE_SOLID)
    assert result1.glsl == result2.glsl


# ---------------------------------------------------------------------------
# GLSL structural checks
# ---------------------------------------------------------------------------

def test_compile_glsl_has_main():
    result = compile_dsl(FIXTURE_CIRCLE_SOLID)
    assert "void main" in result.glsl


def test_compile_glsl_has_version():
    result = compile_dsl(FIXTURE_CIRCLE_SOLID)
    assert result.glsl.strip().startswith("#version 300 es")


def test_compile_glsl_has_frag_color():
    result = compile_dsl(FIXTURE_CIRCLE_SOLID)
    assert "fragColor" in result.glsl


def test_compile_glsl_has_resolution_uniform():
    result = compile_dsl(FIXTURE_CIRCLE_SOLID)
    assert "iResolution" in result.glsl


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_compile_empty_layers_still_produces_glsl():
    dsl = {
        "schema_version": DSL_SCHEMA_VERSION,
        "canvas": {"width": 512, "height": 512, "background": "#222222"},
        "layers": [],
    }
    result = compile_dsl(dsl)
    # No layers → background-only shader; should still produce valid GLSL
    assert len(result.glsl) > 0
    assert "fragColor" in result.glsl
    assert "void main" in result.glsl


# ---------------------------------------------------------------------------
# Fill type coverage
# ---------------------------------------------------------------------------

def test_compile_linear_gradient_layer():
    dsl = _clone(FIXTURE_BOX_GRADIENT)
    result = compile_dsl(dsl)
    assert result.success is True, result.errors
    # linearGradient uses mix and dot product
    assert "mix(" in result.glsl or "mix (" in result.glsl


def test_compile_radial_gradient_layer():
    dsl = {
        "schema_version": DSL_SCHEMA_VERSION,
        "canvas": {"width": 512, "height": 512, "background": "#000000"},
        "layers": [
            {
                "id": "radial_01",
                "type": "circle",
                "fill": {
                    "type": "radialGradient",
                    "stops": [
                        {"color": "#ffffff", "position": 0.0},
                        {"color": "#000000", "position": 1.0},
                    ],
                    "center": [0.5, 0.5],
                },
                "params": {"center": [0.5, 0.5], "radius": 0.4},
                "opacity": 1.0,
                "transform": None,
                "effects": [],
            }
        ],
    }
    result = compile_dsl(dsl)
    assert result.success is True, result.errors
    assert "length(" in result.glsl


def test_two_stop_gradient_uses_smoothstep_like_renderer():
    """Parity: the Pillow scoring renderer interpolates EVERY gradient segment
    with smoothstep; a 2-stop gradient must not silently ship a linear mix."""
    from app.dsl.compiler import compile_dsl
    from app.dsl.schema import FIXTURE_BOX_GRADIENT

    result = compile_dsl(FIXTURE_BOX_GRADIENT)

    assert result.success
    assert "smoothstep(L0_stop_0_pos, L0_stop_1_pos" in result.glsl
