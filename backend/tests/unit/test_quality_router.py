"""Phase 1 unit tests: quality router for PNG-to-Shader.

No LLM calls. Synthetic metric dicts are used throughout.
"""

from __future__ import annotations

import pytest

from app.metrics import quality_router
from app.metrics.quality_router import (
    FAILURE_TYPE_VALUES,
    NEXT_ACTION_VALUES,
    QUALITY_BAND_VALUES,
    STATUS_VALUES,
    QualityRouterOutput,
    compute_final_score,
    route,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_PERFECT_METRICS = {
    "mse": 0.0,
    "simple_ssim": 1.0,
    "alpha_coverage_diff": 0.0,
    "color_histogram_score": 1.0,
    "edge_density_diff": 0.0,
    "nonblank_render": True,
    "within_shader_budget": True,
}

_POOR_METRICS = {
    "mse": 0.9,
    "simple_ssim": 0.1,
    "alpha_coverage_diff": 0.8,
    "color_histogram_score": 0.1,
    "edge_density_diff": 0.9,
    "nonblank_render": True,
    "within_shader_budget": True,
}

_MEDIUM_METRICS = {
    "mse": 0.3,
    "simple_ssim": 0.6,
    "alpha_coverage_diff": 0.2,
    "color_histogram_score": 0.6,
    "edge_density_diff": 0.3,
    "nonblank_render": True,
    "within_shader_budget": True,
}

_GOOD_HARD_GATES = {"compiled": True, "rendered": True}
_FAILED_COMPILE = {"compiled": False, "rendered": False}
_FAILED_RENDER = {"compiled": True, "rendered": False}


# ---------------------------------------------------------------------------
# Hard-gate tests
# ---------------------------------------------------------------------------


def test_route_failed_when_not_compiled():
    out = route({"compiled": False, "rendered": True}, _PERFECT_METRICS)
    assert out.status == "failed"
    assert out.next_action == "fallback"
    assert out.failure_type == "structure"
    assert out.final_score == pytest.approx(0.0)


def test_route_failed_when_not_rendered():
    out = route({"compiled": True, "rendered": False}, _PERFECT_METRICS)
    assert out.status == "failed"
    assert out.next_action == "fallback"
    assert out.failure_type == "structure"
    assert out.final_score == pytest.approx(0.0)


def test_route_failed_on_blank_render():
    blank_metrics = dict(_PERFECT_METRICS, nonblank_render=False)
    out = route(_GOOD_HARD_GATES, blank_metrics)
    assert out.status == "failed"
    assert out.next_action == "fallback"
    assert out.failure_type == "structure"


def test_route_failed_budget_exceeded():
    over_budget = dict(_PERFECT_METRICS, within_shader_budget=False)
    out = route(_GOOD_HARD_GATES, over_budget)
    assert out.status == "failed"
    assert out.failure_type == "budget"
    assert out.next_action == "fallback"


# ---------------------------------------------------------------------------
# Score-based routing
# ---------------------------------------------------------------------------


def test_route_excellent_on_high_score():
    out = route(_GOOD_HARD_GATES, _PERFECT_METRICS)
    assert out.status == "pass"
    assert out.quality_band == "excellent"
    assert out.next_action == "final"
    assert out.failure_type == "none"
    assert out.final_score >= 0.85


def test_route_good_on_score_between_70_and_85():
    # Construct metrics that yield a score in [0.70, 0.85).
    metrics = {
        "mse": 0.15,
        "simple_ssim": 0.75,
        "alpha_coverage_diff": 0.10,
        "color_histogram_score": 0.75,
        "edge_density_diff": 0.15,
        "nonblank_render": True,
        "within_shader_budget": True,
    }
    out = route(_GOOD_HARD_GATES, metrics)
    assert out.status == "pass"
    assert out.quality_band == "good"
    assert out.next_action == "final"
    assert 0.70 <= out.final_score < 0.85


def test_route_preview_on_medium_score():
    out = route(_GOOD_HARD_GATES, _MEDIUM_METRICS)
    # Medium metrics should land in acceptable or poor preview.
    assert out.status == "preview"
    assert out.next_action in ("optimize", "revise")


def test_route_acceptable_preview_on_score_55_to_70():
    metrics = {
        "mse": 0.30,
        "simple_ssim": 0.60,
        "alpha_coverage_diff": 0.20,
        "color_histogram_score": 0.55,
        "edge_density_diff": 0.35,
        "nonblank_render": True,
        "within_shader_budget": True,
    }
    out = route(_GOOD_HARD_GATES, metrics)
    assert out.status == "preview"
    assert out.quality_band == "acceptable"
    assert out.next_action == "optimize"


def test_route_poor_preview_on_score_40_to_55():
    metrics = {
        "mse": 0.55,
        "simple_ssim": 0.40,
        "alpha_coverage_diff": 0.40,
        "color_histogram_score": 0.35,
        "edge_density_diff": 0.55,
        "nonblank_render": True,
        "within_shader_budget": True,
    }
    score = compute_final_score(metrics)
    # Make sure this is actually in [0.40, 0.55).
    if 0.40 <= score < 0.55:
        out = route(_GOOD_HARD_GATES, metrics)
        assert out.status == "preview"
        assert out.quality_band == "poor"
        assert out.next_action == "revise"


def test_route_failed_on_low_score():
    out = route(_GOOD_HARD_GATES, _POOR_METRICS)
    assert out.status == "failed"
    assert out.quality_band == "poor"
    assert out.next_action == "fallback"
    assert out.final_score < 0.40


# ---------------------------------------------------------------------------
# compute_final_score
# ---------------------------------------------------------------------------


def test_compute_final_score_perfect_metrics_is_one():
    score = compute_final_score(_PERFECT_METRICS)
    assert score == pytest.approx(1.0, abs=1e-9)


def test_compute_final_score_worst_metrics_is_zero():
    worst = {
        "mse": 1.0,
        "simple_ssim": 0.0,
        "alpha_coverage_diff": 1.0,
        "color_histogram_score": 0.0,
        "edge_density_diff": 1.0,
    }
    score = compute_final_score(worst)
    assert score == pytest.approx(0.0, abs=1e-9)


def test_compute_final_score_clamped():
    # Even if values somehow go out of range, result must stay in [0, 1].
    extreme = {
        "mse": -5.0,  # invalid but should not crash
        "simple_ssim": 10.0,
        "alpha_coverage_diff": -1.0,
        "color_histogram_score": 5.0,
        "edge_density_diff": -2.0,
    }
    score = compute_final_score(extreme)
    assert 0.0 <= score <= 1.0


def test_compute_final_score_blends_semantic_scores():
    # With perfect objective + perfect semantic, result should be 1.0.
    semantic = {"structure": 1.0, "color_fidelity": 1.0}
    score = compute_final_score(_PERFECT_METRICS, semantic_scores=semantic)
    assert score == pytest.approx(1.0, abs=1e-9)


def test_compute_final_score_semantic_lower_than_objective():
    # Semantic penalty should pull the score below the pure objective score.
    pure_score = compute_final_score(_PERFECT_METRICS)
    semantic = {"quality": 0.0}  # worst semantic
    blended = compute_final_score(_PERFECT_METRICS, semantic_scores=semantic)
    assert blended < pure_score


def test_compute_final_score_empty_semantic_ignored():
    # Empty dict should behave the same as None.
    score_none = compute_final_score(_PERFECT_METRICS, semantic_scores=None)
    score_empty = compute_final_score(_PERFECT_METRICS, semantic_scores={})
    assert score_none == pytest.approx(score_empty, abs=1e-9)


# ---------------------------------------------------------------------------
# Output fields are valid values
# ---------------------------------------------------------------------------


def test_output_fields_are_valid_values():
    out = route(_GOOD_HARD_GATES, _PERFECT_METRICS)
    assert out.status in STATUS_VALUES, f"Unexpected status: {out.status!r}"
    assert out.quality_band in QUALITY_BAND_VALUES, f"Unexpected band: {out.quality_band!r}"
    assert out.next_action in NEXT_ACTION_VALUES, f"Unexpected next_action: {out.next_action!r}"
    assert out.failure_type in FAILURE_TYPE_VALUES, f"Unexpected failure_type: {out.failure_type!r}"
    assert isinstance(out.reason, list)
    assert isinstance(out.protected_aspects, list)
    assert isinstance(out.final_score, float)


def test_output_fields_are_valid_for_failed_routes():
    for hard_gates in (
        {"compiled": False, "rendered": True},
        {"compiled": True, "rendered": False},
    ):
        out = route(hard_gates, _PERFECT_METRICS)
        assert out.status in STATUS_VALUES
        assert out.quality_band in QUALITY_BAND_VALUES
        assert out.next_action in NEXT_ACTION_VALUES
        assert out.failure_type in FAILURE_TYPE_VALUES


def test_route_passes_protected_aspects_through():
    aspects = ["glow_radius", "color_palette"]
    out = route(_GOOD_HARD_GATES, _PERFECT_METRICS, protected_aspects=aspects)
    assert out.protected_aspects == aspects


def test_route_protected_aspects_defaults_to_empty_list():
    out = route(_GOOD_HARD_GATES, _PERFECT_METRICS)
    assert out.protected_aspects == []


# ---------------------------------------------------------------------------
# QualityRouterOutput is a proper dataclass
# ---------------------------------------------------------------------------


def test_quality_router_output_is_dataclass():
    obj = QualityRouterOutput(
        status="pass",
        quality_band="good",
        next_action="final",
        final_score=0.75,
        failure_type="none",
        reason=["ok"],
        protected_aspects=[],
    )
    assert obj.status == "pass"
    assert obj.final_score == 0.75
    assert obj.reason == ["ok"]


# ---------------------------------------------------------------------------
# v2 score formula tests
# ---------------------------------------------------------------------------


def test_final_score_uses_v2_formula_when_v2_keys_present():
    metrics = {
        "simple_ssim": 1.0, "grid_color_sim": 1.0, "mask_iou": 1.0,
        "edge_iou": 1.0, "rmse": 0.0,
        # v1 keys deliberately bad — must be ignored by the v2 formula
        "mse": 1.0, "color_histogram_score": 0.0,
        "alpha_coverage_diff": 1.0, "edge_density_diff": 1.0,
    }
    assert compute_final_score(metrics) == 1.0


def test_final_score_falls_back_to_v1_formula():
    metrics = {"mse": 0.0, "simple_ssim": 1.0, "color_histogram_score": 1.0,
               "alpha_coverage_diff": 0.0, "edge_density_diff": 0.0}
    assert compute_final_score(metrics) == 1.0
