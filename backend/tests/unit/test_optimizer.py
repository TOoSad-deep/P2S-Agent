"""Unit tests for app.png_shader.optimizer (Phase 6).

No LLM, no browser. Uses a mock render_fn that writes a synthetic PNG.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
from PIL import Image

from app.dsl.schema import FIXTURE_CIRCLE_SOLID, FIXTURE_BOX_GRADIENT
from app.dsl.validator import validate_dsl
from app.pipeline.optimizer import (
    OptimizeResult,
    OptimizeStep,
    _collect_optimizable_params,
    _get_nested,
    _perturb_scalar,
    _perturb_vec2,
    _set_nested,
    build_optimization_artifacts,
    optimize_candidate,
    score_dsl,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURE_CIRCLE_GLOW = {
    "schema_version": 1,
    "canvas": {"width": 512, "height": 512, "background": "#000000"},
    "layers": [
        {
            "id": "c1",
            "type": "circle",
            "fill": {"type": "solid", "color": "#ffffff"},
            "params": {"center": [0.5, 0.5], "radius": 0.3},
            "opacity": 0.9,
            "transform": None,
            "effects": [
                {"type": "glow", "intensity": 2.0, "radius": 0.1, "color": "#ffffff"}
            ],
        }
    ],
}

FIXTURE_CIRCLE_OPACITY = {
    "schema_version": 1,
    "canvas": {"width": 512, "height": 512, "background": "#000000"},
    "layers": [
        {
            "id": "c_op",
            "type": "circle",
            "fill": {"type": "solid", "color": "#aaaaaa"},
            "params": {"center": [0.5, 0.5], "radius": 0.25},
            "opacity": 0.8,
            "transform": None,
            "effects": [],
        }
    ],
}


# ---------------------------------------------------------------------------
# Mock render helpers
# ---------------------------------------------------------------------------

def make_mock_render_fn(tmp_path: Path, color=(128, 128, 128, 255)):
    """Returns a render_fn that saves a solid-color PNG and returns its path."""
    counter = [0]

    def render_fn(glsl: str) -> Path:
        counter[0] += 1
        img = Image.new("RGBA", (64, 64), color)
        p = tmp_path / f"render_{counter[0]}.png"
        img.save(p)
        return p

    return render_fn


def make_mock_render_fn_none(tmp_path: Path):
    """Returns a render_fn that always returns None (simulates render failure)."""
    def render_fn(glsl: str):
        return None
    return render_fn


def make_ref_image(tmp_path: Path, color=(200, 200, 200, 255)) -> Path:
    """Create a solid-color reference PNG and return its path."""
    ref_path = tmp_path / "ref.png"
    img = Image.new("RGBA", (64, 64), color)
    img.save(ref_path)
    return ref_path


# ---------------------------------------------------------------------------
# Tests: score_dsl
# ---------------------------------------------------------------------------

def test_score_dsl_returns_float_with_mock_render(tmp_path):
    """score_dsl should return a float in [0.0, 1.0] with a working mock renderer."""
    ref_path = make_ref_image(tmp_path)
    render_fn = make_mock_render_fn(tmp_path)
    result = score_dsl(FIXTURE_CIRCLE_SOLID, ref_path, render_fn)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_score_dsl_returns_zero_on_none_render(tmp_path):
    """score_dsl should return 0.0 when the render_fn returns None."""
    ref_path = make_ref_image(tmp_path)
    render_fn = make_mock_render_fn_none(tmp_path)
    result = score_dsl(FIXTURE_CIRCLE_SOLID, ref_path, render_fn)
    assert result == 0.0


# ---------------------------------------------------------------------------
# Tests: perturbation helpers
# ---------------------------------------------------------------------------

def test_perturb_scalar_stays_in_range():
    """100 random perturbations of 0.5 should all stay in [0.0, 1.0]."""
    for _ in range(100):
        result = _perturb_scalar(0.5)
        assert 0.0 <= result <= 1.0, f"Out of range: {result}"


def test_perturb_vec2_stays_in_range():
    """100 perturbations of [0.5, 0.5] should all stay in [0.0, 1.0]."""
    for _ in range(100):
        result = _perturb_vec2([0.5, 0.5])
        assert len(result) == 2
        for v in result:
            assert 0.0 <= v <= 1.0, f"Out of range: {v}"


# ---------------------------------------------------------------------------
# Tests: _collect_optimizable_params
# ---------------------------------------------------------------------------

def test_collect_params_circle_solid():
    """FIXTURE_CIRCLE_SOLID should yield at least 1 optimizable param."""
    params = _collect_optimizable_params(FIXTURE_CIRCLE_SOLID)
    assert len(params) >= 1, "Expected at least 1 optimizable param"


def test_collect_params_includes_opacity():
    """A DSL with opacity=0.8 should include opacity in collected params."""
    params = _collect_optimizable_params(FIXTURE_CIRCLE_OPACITY)
    paths = [p for p, _, _ in params]
    assert any("opacity" in path for path in paths), (
        f"opacity not found in paths: {paths}"
    )


def test_collect_params_glow_effect():
    """A DSL with a glow effect should collect glow radius and/or intensity."""
    params = _collect_optimizable_params(FIXTURE_CIRCLE_GLOW)
    paths = [p for p, _, _ in params]
    glow_params = [p for p in paths if "effects" in p]
    assert len(glow_params) >= 1, f"No glow params found in: {paths}"


# ---------------------------------------------------------------------------
# Tests: _set_nested / _get_nested roundtrip
# ---------------------------------------------------------------------------

def test_set_nested_roundtrip():
    """set then get on nested value should round-trip correctly."""
    dsl = copy.deepcopy(FIXTURE_CIRCLE_SOLID)
    accessor = ["layers", 0, "params", "radius"]
    new_val = 0.42
    updated = _set_nested(dsl, accessor, new_val)
    retrieved = _get_nested(updated, accessor)
    assert retrieved == new_val


def test_set_nested_does_not_mutate_original():
    """_set_nested should return a deep copy without mutating the original."""
    dsl = copy.deepcopy(FIXTURE_CIRCLE_SOLID)
    original_radius = dsl["layers"][0]["params"]["radius"]
    accessor = ["layers", 0, "params", "radius"]
    _set_nested(dsl, accessor, 0.99)
    assert dsl["layers"][0]["params"]["radius"] == original_radius


# ---------------------------------------------------------------------------
# Tests: optimize_candidate — basic
# ---------------------------------------------------------------------------

def test_optimize_random_runs_without_error(tmp_path):
    """optimize_candidate with random strategy should return an OptimizeResult."""
    ref_path = make_ref_image(tmp_path)
    render_fn = make_mock_render_fn(tmp_path)
    result = optimize_candidate(
        FIXTURE_CIRCLE_SOLID, ref_path, render_fn, max_iterations=5, seed=42
    )
    assert isinstance(result, OptimizeResult)


def test_optimize_iterations_respected(tmp_path):
    """iterations_run in the result should equal max_iterations."""
    ref_path = make_ref_image(tmp_path)
    render_fn = make_mock_render_fn(tmp_path)
    result = optimize_candidate(
        FIXTURE_CIRCLE_SOLID, ref_path, render_fn, max_iterations=5, seed=0
    )
    assert result.iterations_run == 5


def test_optimize_loss_curve_non_empty(tmp_path):
    """loss_curve should always contain at least 1 entry (the initial score)."""
    ref_path = make_ref_image(tmp_path)
    render_fn = make_mock_render_fn(tmp_path)
    result = optimize_candidate(
        FIXTURE_CIRCLE_SOLID, ref_path, render_fn, max_iterations=5, seed=1
    )
    assert len(result.loss_curve) >= 1


def test_optimize_best_dsl_is_valid_dsl(tmp_path):
    """best_dsl returned by the optimizer should pass DSL validation."""
    ref_path = make_ref_image(tmp_path)
    render_fn = make_mock_render_fn(tmp_path)
    result = optimize_candidate(
        FIXTURE_CIRCLE_SOLID, ref_path, render_fn, max_iterations=5, seed=2
    )
    validation = validate_dsl(result.best_dsl)
    assert validation.valid, f"Validation errors: {validation.errors}"


# ---------------------------------------------------------------------------
# Tests: optimizer must NOT alter structure
# ---------------------------------------------------------------------------

def test_optimize_does_not_change_layer_type(tmp_path):
    """Layer primitive type must remain unchanged after optimization."""
    ref_path = make_ref_image(tmp_path)
    render_fn = make_mock_render_fn(tmp_path)
    original_type = FIXTURE_CIRCLE_SOLID["layers"][0]["type"]
    result = optimize_candidate(
        FIXTURE_CIRCLE_SOLID, ref_path, render_fn, max_iterations=5, seed=3
    )
    assert result.best_dsl["layers"][0]["type"] == original_type


def test_optimize_does_not_change_layer_count(tmp_path):
    """Layer count must remain unchanged after optimization."""
    ref_path = make_ref_image(tmp_path)
    render_fn = make_mock_render_fn(tmp_path)
    original_count = len(FIXTURE_CIRCLE_SOLID["layers"])
    result = optimize_candidate(
        FIXTURE_CIRCLE_SOLID, ref_path, render_fn, max_iterations=5, seed=4
    )
    assert len(result.best_dsl["layers"]) == original_count


# ---------------------------------------------------------------------------
# Tests: coordinate_descent strategy
# ---------------------------------------------------------------------------

def test_optimize_coordinate_descent_strategy(tmp_path):
    """coordinate_descent strategy should run without errors and return OptimizeResult."""
    ref_path = make_ref_image(tmp_path)
    render_fn = make_mock_render_fn(tmp_path)
    result = optimize_candidate(
        FIXTURE_CIRCLE_SOLID,
        ref_path,
        render_fn,
        max_iterations=6,
        strategy="coordinate_descent",
        seed=5,
    )
    assert isinstance(result, OptimizeResult)
    assert result.iterations_run == 6


# ---------------------------------------------------------------------------
# Tests: build_optimization_artifacts
# ---------------------------------------------------------------------------

def test_build_optimization_artifacts_has_keys(tmp_path):
    """build_optimization_artifacts should return a dict with all required keys."""
    ref_path = make_ref_image(tmp_path)
    render_fn = make_mock_render_fn(tmp_path)
    result = optimize_candidate(
        FIXTURE_CIRCLE_SOLID, ref_path, render_fn, max_iterations=3, seed=6
    )
    artifacts = build_optimization_artifacts(result)

    required_keys = {
        "initial_score",
        "best_score",
        "improved",
        "iterations_run",
        "loss_curve",
        "steps_accepted",
        "steps_rejected",
        "protected_aspects_violations",
    }
    assert required_keys.issubset(artifacts.keys()), (
        f"Missing keys: {required_keys - artifacts.keys()}"
    )


def test_build_optimization_artifacts_step_counts(tmp_path):
    """steps_accepted + steps_rejected should equal total log entries."""
    ref_path = make_ref_image(tmp_path)
    render_fn = make_mock_render_fn(tmp_path)
    result = optimize_candidate(
        FIXTURE_CIRCLE_SOLID, ref_path, render_fn, max_iterations=8, seed=7
    )
    artifacts = build_optimization_artifacts(result)
    total = artifacts["steps_accepted"] + artifacts["steps_rejected"]
    assert total == len(result.optimizer_log)


# ---------------------------------------------------------------------------
# Tests: improved flag
# ---------------------------------------------------------------------------

def test_optimize_improved_flag(tmp_path):
    """When mock render always returns the same image, improved may be False."""
    ref_path = make_ref_image(tmp_path, color=(128, 128, 128, 255))
    # Same color as ref so scores are very similar — improved is expected False/True
    render_fn = make_mock_render_fn(tmp_path, color=(128, 128, 128, 255))
    result = optimize_candidate(
        FIXTURE_CIRCLE_SOLID, ref_path, render_fn, max_iterations=5, seed=8
    )
    # improved is a bool — just confirm it exists and is bool
    assert isinstance(result.improved, bool)
    # Since render always produces an identical image, score won't improve
    assert result.improved is False


# ---------------------------------------------------------------------------
# Tests: gradient DSL
# ---------------------------------------------------------------------------

def test_optimize_with_gradient_dsl(tmp_path):
    """FIXTURE_BOX_GRADIENT should yield at least 1 optimizable param."""
    params = _collect_optimizable_params(FIXTURE_BOX_GRADIENT)
    assert len(params) >= 1, f"Expected params from gradient DSL, got: {params}"


def test_optimize_gradient_dsl_runs(tmp_path):
    """optimize_candidate on FIXTURE_BOX_GRADIENT should run without errors."""
    ref_path = make_ref_image(tmp_path)
    render_fn = make_mock_render_fn(tmp_path)
    result = optimize_candidate(
        FIXTURE_BOX_GRADIENT, ref_path, render_fn, max_iterations=5, seed=9
    )
    assert isinstance(result, OptimizeResult)
    assert result.best_dsl["layers"][0]["type"] == "box"
