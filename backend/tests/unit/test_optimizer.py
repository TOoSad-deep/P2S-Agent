"""Unit tests for app.pipeline.optimizer (Phase 6).

No LLM, no browser. Uses a mock render_fn that writes a synthetic PNG.
"""

from __future__ import annotations

import copy
import random
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

# linearGradient with a negative-axis (right-to-left) direction.
FIXTURE_BOX_NEG_GRADIENT = {
    "schema_version": 1,
    "canvas": {"width": 512, "height": 512, "background": "#000000"},
    "layers": [
        {
            "id": "ng_box",
            "type": "box",
            "fill": {
                "type": "linearGradient",
                "stops": [
                    {"color": "#ff0000", "position": 0.0},
                    {"color": "#0000ff", "position": 1.0},
                ],
                "direction": [-1.0, 0.0],
            },
            "params": {"center": [0.5, 0.5], "size": [0.4, 0.3]},
            "opacity": 1.0,
            "transform": None,
            "effects": [],
        }
    ],
}

# radialGradient using the canonical 'center' fill accessor.
FIXTURE_RADIAL_GRADIENT = {
    "schema_version": 1,
    "canvas": {"width": 512, "height": 512, "background": "#000000"},
    "layers": [
        {
            "id": "rg_box",
            "type": "box",
            "fill": {
                "type": "radialGradient",
                "stops": [
                    {"color": "#ffffff", "position": 0.0},
                    {"color": "#000000", "position": 1.0},
                ],
                "center": [0.4, 0.6],
            },
            "params": {"center": [0.5, 0.5], "size": [0.4, 0.3]},
            "opacity": 1.0,
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
# Tests: convergence / early-stop (no-improvement) + acceptance epsilon
# ---------------------------------------------------------------------------

def test_coordinate_descent_early_stops_when_no_improvement(tmp_path):
    """A constant score_fn must early-stop after one full sweep, not run to cap.

    With a render that always produces the SAME image, no perturbation can ever
    improve the score. The optimizer must detect a full no-improvement sweep and
    stop early — bounding compile/render/score calls to ~one sweep (2 trials per
    param) instead of churning through every max_iterations evaluation.
    """
    ref_path = make_ref_image(tmp_path, color=(200, 200, 200, 255))

    # Constant render: every trial scores identically => nothing ever improves.
    call_count = {"n": 0}

    def render_fn(glsl: str) -> Path:
        call_count["n"] += 1
        img = Image.new("RGBA", (64, 64), (128, 128, 128, 255))
        p = tmp_path / f"const_{call_count['n']}.png"
        img.save(p)
        return p

    num_params = len(_collect_optimizable_params(FIXTURE_CIRCLE_SOLID))
    # A generous cap so that running to the cap is clearly distinguishable from
    # stopping after one sweep.
    cap = num_params * 2 * 5
    result = optimize_candidate(
        FIXTURE_CIRCLE_SOLID,
        ref_path,
        render_fn,
        max_iterations=cap,
        strategy="coordinate_descent",
        seed=5,
    )

    # One full sweep = at most 2 trials per param. The optimizer must stop after
    # that single no-improvement sweep instead of consuming the full cap.
    one_sweep_trials = num_params * 2
    assert result.iterations_run <= one_sweep_trials, (
        f"expected early-stop within one sweep ({one_sweep_trials} trials), "
        f"got {result.iterations_run}"
    )
    assert result.iterations_run < cap, "optimizer ran to the max cap, no early-stop"
    # Renders performed (1 initial + per-trial) must also be bounded to one sweep.
    assert call_count["n"] <= one_sweep_trials + 1
    # The result must record WHY it stopped.
    assert result.stop_reason == "converged_no_improvement", (
        f"unexpected stop_reason: {result.stop_reason!r}"
    )


def test_coordinate_descent_stop_reason_max_iterations(tmp_path):
    """When improvements keep landing, the optimizer runs to the cap and records it."""
    ref_path = make_ref_image(tmp_path, color=(200, 200, 200, 255))

    # Monotonically improving render so every step is accepted and early-stop
    # never fires — the optimizer exhausts max_iterations.
    state = {"n": 0}

    def render_fn(glsl: str) -> Path:
        shade = min(40 + state["n"] * 8, 200)
        img = Image.new("RGBA", (64, 64), (shade, shade, shade, 255))
        p = tmp_path / f"imp_{state['n']}.png"
        state["n"] += 1
        img.save(p)
        return p

    result = optimize_candidate(
        FIXTURE_CIRCLE_SOLID,
        ref_path,
        render_fn,
        max_iterations=6,
        strategy="coordinate_descent",
        seed=5,
    )
    assert result.iterations_run == 6
    assert result.stop_reason == "max_iterations"


def test_coordinate_descent_rejects_sub_epsilon_improvement(tmp_path):
    """A perturbation improving the score by less than epsilon is NOT accepted.

    The first trial nudges the score up by a hair (< epsilon). That must be
    treated as noise and rejected, so best_score stays at the initial score and
    the step is logged accepted=False.
    """
    ref_path = make_ref_image(tmp_path, color=(200, 200, 200, 255))

    # Render 0 (initial) and every trial render must yield scores that differ by
    # less than the acceptance epsilon. Identical images give identical scores,
    # which trivially fall below ANY positive epsilon, so a previously-accepted
    # ``> best`` step (floating-point noise) is now rejected.
    def render_fn(glsl: str) -> Path:
        img = Image.new("RGBA", (64, 64), (150, 150, 150, 255))
        p = tmp_path / f"eps_{random.random()}.png"
        img.save(p)
        return p

    big_epsilon = 0.5  # No real render can clear a 0.5 score jump here.
    result = optimize_candidate(
        FIXTURE_CIRCLE_SOLID,
        ref_path,
        render_fn,
        max_iterations=4,
        strategy="coordinate_descent",
        seed=5,
        accept_epsilon=big_epsilon,
    )
    # No trial cleared the epsilon bar => nothing accepted, score unchanged.
    assert result.best_score == result.initial_score
    assert all(not step.accepted for step in result.optimizer_log)
    assert result.improved is False


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


# ---------------------------------------------------------------------------
# Bug 1: gradient direction must keep its sign (not clamped to [0, 1])
# ---------------------------------------------------------------------------

def test_perturb_vec2_preserves_negative_direction_sign():
    """A 'direction' vec2 with a negative component must NOT be clamped to 0.0.

    Clamping a -1.0 component to 0.0 silently degenerates the gradient axis.
    """
    random.seed(0)
    # Perturb a [-1.0, 0.0] direction many times with a small scale; the first
    # component must remain negative (sign preserved), never clamped to 0.0.
    for _ in range(50):
        result = _perturb_vec2([-1.0, 0.0], is_direction=True)
        assert len(result) == 2
        assert result[0] < 0.0, f"direction x lost its sign: {result}"


def test_coordinate_descent_direction_keeps_sign(tmp_path):
    """Coordinate-descent perturbation of a negative direction keeps it negative.

    The render improves on every trial so the direction perturbation IS
    accepted and written into best_dsl — exposing the [0, 1] clamp that
    degenerates the -1.0 x component to 0.0.
    """
    ref_path = make_ref_image(tmp_path)

    # Monotonically improving render so coordinate-descent always accepts and
    # the perturbed direction lands in best_dsl.
    state = {"n": 0}

    def render_fn(glsl: str) -> Path:
        shade = min(60 + state["n"] * 12, 200)
        img = Image.new("RGBA", (64, 64), (shade, shade, shade, 255))
        p = tmp_path / f"dir_{state['n']}.png"
        state["n"] += 1
        img.save(p)
        return p

    # Enough iterations for coordinate descent to reach the fill 'direction'
    # param (collected after params/opacity) and accept a perturbation.
    result = optimize_candidate(
        FIXTURE_BOX_NEG_GRADIENT,
        ref_path,
        render_fn,
        max_iterations=12,
        strategy="coordinate_descent",
        seed=11,
    )
    direction = result.best_dsl["layers"][0]["fill"]["direction"]
    # The x component started at -1.0; a +/-0.05 coordinate step must not flip
    # it across zero into a degenerate/positive axis.
    assert direction[0] < 0.0, f"direction x lost its sign: {direction}"


# ---------------------------------------------------------------------------
# Bug 2: radialGradient 'center' must actually be collected/optimized
# ---------------------------------------------------------------------------

def test_collect_params_radial_gradient_center(tmp_path):
    """radialGradient 'center' must appear in the collected optimizable params."""
    params = _collect_optimizable_params(FIXTURE_RADIAL_GRADIENT)
    paths = [p for p, _, _ in params]
    assert any("fill.center" in path for path in paths), (
        f"radialGradient center not collected: {paths}"
    )


# ---------------------------------------------------------------------------
# Bug 3: optimizer must NOT seed the process-global random module
# ---------------------------------------------------------------------------

def test_optimize_does_not_disturb_global_random(tmp_path):
    """optimize_candidate(seed=...) must not reseed the global random module."""
    ref_path = make_ref_image(tmp_path)
    render_fn = make_mock_render_fn(tmp_path)

    random.seed(1234)
    expected_sequence = [random.random() for _ in range(3)]

    random.seed(1234)
    optimize_candidate(
        FIXTURE_CIRCLE_SOLID, ref_path, render_fn, max_iterations=5, seed=42
    )
    after_optimize = [random.random() for _ in range(3)]

    assert after_optimize == expected_sequence, (
        "optimize_candidate disturbed the process-global random state"
    )


def test_optimize_seed_is_deterministic(tmp_path):
    """A fixed seed must give reproducible optimizer logs via a local RNG."""
    ref_path = make_ref_image(tmp_path, color=(10, 200, 50, 255))
    render_fn = make_mock_render_fn(tmp_path, color=(200, 10, 50, 255))

    r1 = optimize_candidate(
        FIXTURE_CIRCLE_GLOW, ref_path, render_fn, max_iterations=8, seed=7
    )
    r2 = optimize_candidate(
        FIXTURE_CIRCLE_GLOW, ref_path, render_fn, max_iterations=8, seed=7
    )
    assert [s.param_path for s in r1.optimizer_log] == [
        s.param_path for s in r2.optimizer_log
    ]
    assert [s.new_value for s in r1.optimizer_log] == [
        s.new_value for s in r2.optimizer_log
    ]


# ---------------------------------------------------------------------------
# Bug 4: coordinate-descent log score_before must reflect the actual pre-step score
# ---------------------------------------------------------------------------

def test_coordinate_descent_score_before_matches_accept_decision(tmp_path):
    """The logged score_before must be the baseline the accept decision used.

    Within one coordinate-descent step the +scale trial can be accepted first;
    the -scale trial is then judged against the accepted score, not the stale
    pre-step best_score. The log must stay internally consistent:
    accepted == (score_after > score_before).
    """
    ref_path = make_ref_image(tmp_path)

    # Trial order per step: [+scale, -scale]. Make +scale the best (closest),
    # and -scale better than the pre-step start but worse than +scale. With the
    # buggy stale score_before, the 2nd trial logs before=start, after>start,
    # yet accepted=False -> inconsistent.
    colors = [
        (60, 60, 60, 255),     # initial render (farthest from ref 200)
        (199, 199, 199, 255),  # +scale trial — closest => accepted
        (120, 120, 120, 255),  # -scale trial — better than start, worse than +scale
    ]
    state = {"n": 0}

    def render_fn(glsl: str) -> Path:
        idx = min(state["n"], len(colors) - 1)
        img = Image.new("RGBA", (64, 64), colors[idx])
        p = tmp_path / f"cd_{state['n']}.png"
        state["n"] += 1
        img.save(p)
        return p

    result = optimize_candidate(
        FIXTURE_CIRCLE_SOLID,
        ref_path,
        render_fn,
        max_iterations=2,
        strategy="coordinate_descent",
        seed=3,
    )

    # Every logged step must satisfy accepted == (score_after > score_before).
    for step in result.optimizer_log:
        assert step.accepted == (step.score_after > step.score_before), (
            f"inconsistent log: before={step.score_before} "
            f"after={step.score_after} accepted={step.accepted}"
        )
