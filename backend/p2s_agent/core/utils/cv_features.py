"""CV applicability scoring for PNG-to-Shader.

Computes cv_applicability_score from preprocess features.
No LLM, no browser, no numpy. Only Pillow + stdlib.
"""

from __future__ import annotations

from dataclasses import dataclass

CV_APPLICABILITY_THRESHOLD_HIGH = 0.75
CV_APPLICABILITY_THRESHOLD_LOW = 0.45


@dataclass
class _CVSignals:
    has_alpha_bonus: float
    alpha_signal: float
    sharpness_signal: float
    color_signal: float
    component_signal: float
    texture_penalty: float

    @property
    def weighted_score(self) -> float:
        raw = (
            self.has_alpha_bonus * 0.25
            + self.alpha_signal * 0.20
            + self.sharpness_signal * 0.20
            + self.color_signal * 0.15
            + self.component_signal * 0.10
            + self.texture_penalty * 0.10
        )
        return max(0.0, min(1.0, raw))

    @property
    def contributions(self) -> dict[str, float]:
        return {
            "has_alpha_bonus": round(self.has_alpha_bonus * 0.25, 4),
            "alpha_signal": round(self.alpha_signal * 0.20, 4),
            "sharpness_signal": round(self.sharpness_signal * 0.20, 4),
            "color_signal": round(self.color_signal * 0.15, 4),
            "component_signal": round(self.component_signal * 0.10, 4),
            "texture_penalty": round(self.texture_penalty * 0.10, 4),
        }


def _compute_signals(preprocess: dict) -> _CVSignals:
    has_alpha = bool(preprocess.get("has_alpha", False))
    alpha_coverage = float(preprocess.get("alpha_coverage", 0.0))
    edge_sharpness = float(preprocess.get("edge_sharpness", 0.0))
    color_count = int(preprocess.get("color_count_estimate", 0))
    component_count = int(preprocess.get("component_count_estimate", 1))
    texture_score = float(preprocess.get("texture_score", 0.0))

    has_alpha_bonus = 1.0 if has_alpha else 0.0

    if not has_alpha:
        alpha_signal = 0.0
    elif alpha_coverage < 0.05:
        alpha_signal = 0.2
    elif alpha_coverage <= 0.85:
        alpha_signal = 1.0
    else:
        alpha_signal = 0.5

    if edge_sharpness < 0.05:
        sharpness_signal = 0.3
    elif edge_sharpness <= 0.40:
        sharpness_signal = 1.0
    else:
        sharpness_signal = 0.4

    if color_count <= 5:
        color_signal = 1.0
    elif color_count <= 20:
        color_signal = 0.8
    elif color_count <= 50:
        color_signal = 0.5
    else:
        color_signal = 0.2

    if 1 <= component_count <= 8:
        component_signal = 1.0
    elif component_count <= 20:
        component_signal = 0.5
    else:
        component_signal = 0.1

    texture_penalty = 1.0 - min(1.0, texture_score)

    return _CVSignals(
        has_alpha_bonus=has_alpha_bonus,
        alpha_signal=alpha_signal,
        sharpness_signal=sharpness_signal,
        color_signal=color_signal,
        component_signal=component_signal,
        texture_penalty=texture_penalty,
    )


def compute_cv_applicability_score(preprocess: dict) -> float:
    """Compute a CV applicability score in [0.0, 1.0] from preprocess features."""
    return _compute_signals(preprocess).weighted_score


def get_cv_applicability_report(preprocess: dict) -> dict:
    """Return a structured report of CV applicability."""
    signals = _compute_signals(preprocess)
    score = signals.weighted_score

    if score >= CV_APPLICABILITY_THRESHOLD_HIGH:
        priority = "high"
    elif score >= CV_APPLICABILITY_THRESHOLD_LOW:
        priority = "low"
    else:
        priority = "disabled"

    enabled = score >= CV_APPLICABILITY_THRESHOLD_LOW

    has_alpha = bool(preprocess.get("has_alpha", False))
    alpha_coverage = float(preprocess.get("alpha_coverage", 0.0))
    edge_sharpness = float(preprocess.get("edge_sharpness", 0.0))
    color_count = int(preprocess.get("color_count_estimate", 0))
    texture_score = float(preprocess.get("texture_score", 0.0))

    reason_parts: list[str] = []
    if has_alpha:
        reason_parts.append("has alpha")
        if alpha_coverage < 0.05:
            reason_parts.append("very low coverage")
        elif alpha_coverage <= 0.85:
            reason_parts.append("moderate coverage")
        else:
            reason_parts.append("high coverage")
    else:
        reason_parts.append("no alpha channel")

    if edge_sharpness < 0.05:
        reason_parts.append("blurry edges")
    elif edge_sharpness <= 0.40:
        reason_parts.append("clear edges")
    else:
        reason_parts.append("highly complex edges")

    if color_count <= 5:
        reason_parts.append("very few colors")
    elif color_count <= 20:
        reason_parts.append("few colors")
    elif color_count <= 50:
        reason_parts.append("moderate colors")
    else:
        reason_parts.append("many colors")

    if texture_score > 0.6:
        reason_parts.append("high texture (penalized)")

    if priority == "high":
        prefix = "High CV applicability"
    elif priority == "low":
        prefix = "Moderate CV applicability"
    else:
        prefix = "Low CV applicability"

    reason = f"{prefix}: {', '.join(reason_parts)}."

    return {
        "score": round(score, 4),
        "priority": priority,
        "enabled": enabled,
        "signals": signals.contributions,
        "reason": reason,
    }
