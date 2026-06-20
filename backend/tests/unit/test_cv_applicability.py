"""Unit tests for cv_features.py — CV applicability scoring."""

from __future__ import annotations

import pytest

from p2s_agent.core.utils.cv_features import (
    CV_APPLICABILITY_THRESHOLD_HIGH,
    CV_APPLICABILITY_THRESHOLD_LOW,
    compute_cv_applicability_score,
    get_cv_applicability_report,
)


# ---------------------------------------------------------------------------
# Preprocess fixtures
# ---------------------------------------------------------------------------

PREPROCESS_ICON = {
    "has_alpha": True,
    "alpha_coverage": 0.4,
    "edge_sharpness": 0.2,
    "color_count_estimate": 8,
    "component_count_estimate": 2,
    "texture_score": 0.1,
}

PREPROCESS_PHOTO = {
    "has_alpha": False,
    "alpha_coverage": 1.0,
    "edge_sharpness": 0.3,
    "color_count_estimate": 200,
    "component_count_estimate": 1,
    "texture_score": 0.9,
}

PREPROCESS_MEDIUM = {
    # has_alpha=True → +0.25 (has_alpha_bonus)
    # alpha_coverage=0.95 (> 0.85) → alpha_signal=0.5 → +0.10
    # edge_sharpness=0.6 (> 0.40) → sharpness_signal=0.4 → +0.08
    # color_count=60 (> 50) → color_signal=0.2 → +0.03
    # component_count=12 ([9,20]) → component_signal=0.5 → +0.05
    # texture_score=0.7 → texture_penalty=0.3 → +0.03
    # Total ≈ 0.54 → in [0.45, 0.75)
    "has_alpha": True,
    "alpha_coverage": 0.95,
    "edge_sharpness": 0.6,
    "color_count_estimate": 60,
    "component_count_estimate": 12,
    "texture_score": 0.7,
}


# ---------------------------------------------------------------------------
# Score range tests
# ---------------------------------------------------------------------------

def test_high_score_transparent_icon():
    """Transparent icon features → score >= 0.75."""
    score = compute_cv_applicability_score(PREPROCESS_ICON)
    assert score >= CV_APPLICABILITY_THRESHOLD_HIGH, (
        f"Expected score >= {CV_APPLICABILITY_THRESHOLD_HIGH}, got {score}"
    )


def test_low_score_photo_like():
    """Photo-like features (no alpha, many colors, high texture) → score < 0.45."""
    score = compute_cv_applicability_score(PREPROCESS_PHOTO)
    assert score < CV_APPLICABILITY_THRESHOLD_LOW, (
        f"Expected score < {CV_APPLICABILITY_THRESHOLD_LOW}, got {score}"
    )


def test_medium_score_mixed():
    """Mixed features → 0.45 <= score < 0.75."""
    score = compute_cv_applicability_score(PREPROCESS_MEDIUM)
    assert CV_APPLICABILITY_THRESHOLD_LOW <= score < CV_APPLICABILITY_THRESHOLD_HIGH, (
        f"Expected {CV_APPLICABILITY_THRESHOLD_LOW} <= score < "
        f"{CV_APPLICABILITY_THRESHOLD_HIGH}, got {score}"
    )


# ---------------------------------------------------------------------------
# Report enabled/disabled tests
# ---------------------------------------------------------------------------

def test_disabled_when_below_threshold():
    """Photo-like preprocess → report['enabled'] is False."""
    report = get_cv_applicability_report(PREPROCESS_PHOTO)
    assert report["enabled"] is False


def test_enabled_when_above_threshold():
    """Icon preprocess → report['enabled'] is True."""
    report = get_cv_applicability_report(PREPROCESS_ICON)
    assert report["enabled"] is True


# ---------------------------------------------------------------------------
# Priority tests
# ---------------------------------------------------------------------------

def test_priority_high():
    """Icon preprocess → report['priority'] == 'high'."""
    report = get_cv_applicability_report(PREPROCESS_ICON)
    assert report["priority"] == "high"


def test_priority_low():
    """Medium preprocess → report['priority'] == 'low'."""
    report = get_cv_applicability_report(PREPROCESS_MEDIUM)
    assert report["priority"] == "low"


# ---------------------------------------------------------------------------
# Report structure tests
# ---------------------------------------------------------------------------

def test_report_has_required_fields():
    """Report must contain score, priority, enabled, signals, reason."""
    report = get_cv_applicability_report(PREPROCESS_ICON)
    required_keys = {"score", "priority", "enabled", "signals", "reason"}
    assert required_keys.issubset(report.keys()), (
        f"Missing keys: {required_keys - set(report.keys())}"
    )
    # signals sub-keys
    signal_keys = {
        "has_alpha_bonus", "alpha_signal", "sharpness_signal",
        "color_signal", "component_signal", "texture_penalty",
    }
    assert signal_keys.issubset(report["signals"].keys()), (
        f"Missing signal keys: {signal_keys - set(report['signals'].keys())}"
    )
    # score type
    assert isinstance(report["score"], float)
    # priority values
    assert report["priority"] in ("high", "low", "disabled")
    # enabled is bool
    assert isinstance(report["enabled"], bool)
    # reason is non-empty string
    assert isinstance(report["reason"], str) and len(report["reason"]) > 0


# ---------------------------------------------------------------------------
# Clamping and edge-case tests
# ---------------------------------------------------------------------------

def test_score_clamped_to_0_1():
    """Extreme inputs should not produce a score outside [0.0, 1.0]."""
    extreme_high = {
        "has_alpha": True,
        "alpha_coverage": 0.5,
        "edge_sharpness": 0.2,
        "color_count_estimate": 1,
        "component_count_estimate": 1,
        "texture_score": 0.0,
    }
    extreme_low = {
        "has_alpha": False,
        "alpha_coverage": 0.0,
        "edge_sharpness": 1.0,
        "color_count_estimate": 10000,
        "component_count_estimate": 1000,
        "texture_score": 1.0,
    }
    for preprocess in (extreme_high, extreme_low):
        score = compute_cv_applicability_score(preprocess)
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"


def test_no_alpha_reduces_score():
    """has_alpha=False vs has_alpha=True (all other fields equal) → lower score."""
    base = {
        "has_alpha": True,
        "alpha_coverage": 0.4,
        "edge_sharpness": 0.2,
        "color_count_estimate": 8,
        "component_count_estimate": 2,
        "texture_score": 0.1,
    }
    no_alpha = {**base, "has_alpha": False}
    score_with_alpha = compute_cv_applicability_score(base)
    score_no_alpha = compute_cv_applicability_score(no_alpha)
    assert score_no_alpha < score_with_alpha, (
        f"Expected no-alpha score ({score_no_alpha}) < alpha score ({score_with_alpha})"
    )
