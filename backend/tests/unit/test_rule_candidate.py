"""Unit tests for rule_candidate.py — Phase 3 rule-based DSL generator."""

from __future__ import annotations

import pytest

from app.candidates.rule import generate_rule_candidate
from app.dsl.compiler import compile_dsl
from app.dsl.validator import validate_dsl


# ---------------------------------------------------------------------------
# Preprocess fixtures
# ---------------------------------------------------------------------------

PREPROCESS_HIGH_GRADIENT = {
    "gradient_score": 0.8,
    "alpha_coverage": 0.0,
    "has_alpha": False,
    "photo_like_score": 0.1,
    "palette": ["#ff0000", "#0000ff"],
}

PREPROCESS_ALPHA = {
    "gradient_score": 0.1,
    "alpha_coverage": 0.5,
    "has_alpha": True,
    "photo_like_score": 0.1,
    "palette": ["#00ff00"],
}

PREPROCESS_DEFAULT = {
    "gradient_score": 0.0,
    "alpha_coverage": 0.0,
    "has_alpha": False,
    "photo_like_score": 0.0,
    "palette": ["#aabbcc"],
}

PREPROCESS_PHOTO = {
    "gradient_score": 0.0,
    "alpha_coverage": 0.0,
    "has_alpha": False,
    "photo_like_score": 0.9,
    "palette": ["#334455"],
}


# ---------------------------------------------------------------------------
# Rule-selection tests
# ---------------------------------------------------------------------------

def test_rule_candidate_gradient_preprocess():
    dsl = generate_rule_candidate(PREPROCESS_HIGH_GRADIENT)
    layers = dsl["layers"]
    assert len(layers) >= 1
    layer = layers[0]
    assert layer["fill"]["type"] == "radialGradient"


def test_rule_candidate_alpha_preprocess():
    # alpha_coverage=0.5 (>= 0.5) → large-foreground branch → roundedBox
    dsl = generate_rule_candidate(PREPROCESS_ALPHA)
    layers = dsl["layers"]
    assert len(layers) >= 1
    layer = layers[0]
    assert layer["type"] == "roundedBox"
    assert layer["fill"]["type"] == "solid"


def test_rule_candidate_small_alpha_gives_circle():
    # alpha_coverage < 0.5 with clear edges → small compact icon → circle
    preprocess = {
        "gradient_score": 0.1,
        "alpha_coverage": 0.25,
        "edge_sharpness": 0.15,
        "has_alpha": True,
        "photo_like_score": 0.1,
        "palette": ["#00ff00"],
        "color_count_estimate": 5,
    }
    dsl = generate_rule_candidate(preprocess)
    layer = dsl["layers"][0]
    assert layer["type"] == "circle"


def test_rule_candidate_default():
    dsl = generate_rule_candidate(PREPROCESS_DEFAULT)
    layers = dsl["layers"]
    assert len(layers) >= 1
    layer = layers[0]
    assert layer["type"] == "box"
    assert layer["fill"]["type"] == "solid"


def test_rule_candidate_adds_vignette_for_photo_like():
    dsl = generate_rule_candidate(PREPROCESS_PHOTO)
    layers = dsl["layers"]
    effects = layers[0].get("effects", [])
    effect_types = [e["type"] for e in effects]
    assert "vignette" in effect_types


def test_rule_candidate_no_vignette_for_non_photo():
    dsl = generate_rule_candidate(PREPROCESS_DEFAULT)
    layers = dsl["layers"]
    effects = layers[0].get("effects", [])
    effect_types = [e["type"] for e in effects]
    assert "vignette" not in effect_types


# ---------------------------------------------------------------------------
# Validity and compilability
# ---------------------------------------------------------------------------

def test_rule_candidate_output_is_valid_dsl():
    for preprocess in [
        PREPROCESS_HIGH_GRADIENT,
        PREPROCESS_ALPHA,
        PREPROCESS_DEFAULT,
        PREPROCESS_PHOTO,
    ]:
        dsl = generate_rule_candidate(preprocess)
        result = validate_dsl(dsl)
        assert result.valid is True, (
            f"DSL invalid for preprocess={preprocess}: {result.errors}"
        )


def test_rule_candidate_is_compilable():
    for preprocess in [
        PREPROCESS_HIGH_GRADIENT,
        PREPROCESS_ALPHA,
        PREPROCESS_DEFAULT,
        PREPROCESS_PHOTO,
    ]:
        dsl = generate_rule_candidate(preprocess)
        result = compile_dsl(dsl)
        assert result.success is True, (
            f"Compile failed for preprocess={preprocess}: {result.errors}"
        )
        assert len(result.glsl) > 0


def test_rule_candidate_canvas_dimensions():
    dsl = generate_rule_candidate(PREPROCESS_DEFAULT, canvas_width=1024, canvas_height=768)
    assert dsl["canvas"]["width"] == 1024
    assert dsl["canvas"]["height"] == 768


def test_rule_candidate_palette_color_used():
    preprocess = {**PREPROCESS_DEFAULT, "palette": ["#deadbe"]}
    dsl = generate_rule_candidate(preprocess)
    layer = dsl["layers"][0]
    fill = layer["fill"]
    # Solid fill should use the top palette color
    assert fill.get("color", "").lower() == "#deadbe"
