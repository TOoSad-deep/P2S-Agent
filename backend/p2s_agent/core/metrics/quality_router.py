"""Quality router for PNG-to-Shader candidates.

Routes a candidate through quality tiers based on hard gates and
objective (+ optional semantic) metrics. No LLM or browser calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Allowed values for enum-like fields
# ---------------------------------------------------------------------------

STATUS_VALUES = ["pass", "preview", "failed", "unsupported"]
QUALITY_BAND_VALUES = ["excellent", "good", "acceptable", "poor"]
NEXT_ACTION_VALUES = [
    "final",
    "optimize",
    "revise",
    "rollback",
    "reroute",
    "fallback",
    "return_report",
]
FAILURE_TYPE_VALUES = [
    "none",
    "structure",
    "parameter",
    "color",
    "layer_order",
    "budget",
    "unsupported",
]


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------


@dataclass
class QualityRouterOutput:
    """Structured output produced by :func:`route`.

    Attributes:
        status:             Overall routing status ("pass", "preview", …).
        quality_band:       Qualitative tier ("excellent", "good", …).
        next_action:        Recommended next step for the pipeline.
        final_score:        Weighted composite score in [0, 1].
        failure_type:       Primary failure category, or "none".
        reason:             Human-readable list of reasons / observations.
        protected_aspects:  Aspects that must not be changed in subsequent
                            iterations (passed through from caller).
    """

    status: str
    quality_band: str
    next_action: str
    final_score: float
    failure_type: str
    reason: list[str] = field(default_factory=list)
    protected_aspects: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------


def compute_final_score(
    objective_metrics: dict,
    semantic_scores: Optional[dict] = None,
) -> float:
    """Return a weighted composite quality score in [0, 1].

    v2 formula (used when 'mask_iou' key is present — position-aware):
        simple_ssim        × 0.30
        grid_color_sim     × 0.25
        mask_iou           × 0.20
        edge_iou           × 0.15
        (1 - rmse)         × 0.10

    v1 formula (legacy fallback when v2 keys are absent):
        (1 - mse)                  × 0.25
        simple_ssim                × 0.25
        color_histogram_score      × 0.20
        (1 - alpha_coverage_diff)  × 0.15
        (1 - edge_density_diff)    × 0.15

    When *semantic_scores* is provided the final score blends:
        objective × 0.7 + mean(semantic_scores.values()) × 0.3
    """
    if "mask_iou" in objective_metrics:
        # v2 formula: position-aware metrics dominate
        ssim = float(objective_metrics.get("simple_ssim", 0.0))
        grid = float(objective_metrics.get("grid_color_sim", 0.0))
        miou = float(objective_metrics.get("mask_iou", 0.0))
        eiou = float(objective_metrics.get("edge_iou", 0.0))
        rmse = float(objective_metrics.get("rmse", 1.0))
        objective_score = (
            ssim * 0.30
            + grid * 0.25
            + miou * 0.20
            + eiou * 0.15
            + (1.0 - rmse) * 0.10
        )
    else:
        # legacy v1 formula (old artifacts / tests without v2 keys)
        mse = float(objective_metrics.get("mse", 0.0))
        ssim = float(objective_metrics.get("simple_ssim", 0.0))
        hist = float(objective_metrics.get("color_histogram_score", 0.0))
        alpha_diff = float(objective_metrics.get("alpha_coverage_diff", 0.0))
        edge_diff = float(objective_metrics.get("edge_density_diff", 0.0))
        objective_score = (
            (1.0 - mse) * 0.25
            + ssim * 0.25
            + hist * 0.20
            + (1.0 - alpha_diff) * 0.15
            + (1.0 - edge_diff) * 0.15
        )

    if semantic_scores is not None and len(semantic_scores) > 0:
        values = list(semantic_scores.values())
        semantic_mean = sum(float(v) for v in values) / len(values)
        final = objective_score * 0.7 + semantic_mean * 0.3
    else:
        final = objective_score

    return float(max(0.0, min(1.0, final)))


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def route(
    hard_gates: dict,
    objective_metrics: dict,
    semantic_scores: Optional[dict] = None,
    *,
    protected_aspects: Optional[list[str]] = None,
) -> QualityRouterOutput:
    """Route a candidate to a quality tier.

    Hard gates are checked first (short-circuit); metric-based routing
    runs only when all gates pass.

    Args:
        hard_gates:         Dict with boolean keys ``compiled`` and
                            ``rendered`` (both must be True to proceed).
        objective_metrics:  Output of ``compute_objective_metrics``.
        semantic_scores:    Optional dict of semantic dimension scores,
                            each in [0, 1].
        protected_aspects:  Aspects that must not be modified downstream;
                            passed through verbatim to the output.

    Returns:
        :class:`QualityRouterOutput` with routing decision.
    """
    _protected = list(protected_aspects) if protected_aspects else []

    # --- Hard gate: compilation ---
    if not hard_gates.get("compiled", False):
        return QualityRouterOutput(
            status="failed",
            quality_band="poor",
            next_action="fallback",
            final_score=0.0,
            failure_type="structure",
            reason=["Shader failed to compile."],
            protected_aspects=_protected,
        )

    # --- Hard gate: rendering ---
    if not hard_gates.get("rendered", False):
        return QualityRouterOutput(
            status="failed",
            quality_band="poor",
            next_action="fallback",
            final_score=0.0,
            failure_type="structure",
            reason=["Shader failed to render."],
            protected_aspects=_protected,
        )

    # --- Hard gate: shader budget ---
    if not objective_metrics.get("within_shader_budget", True):
        return QualityRouterOutput(
            status="failed",
            quality_band="poor",
            next_action="fallback",
            final_score=0.0,
            failure_type="budget",
            reason=["Shader character count exceeds budget."],
            protected_aspects=_protected,
        )

    # --- Hard gate: non-blank render ---
    if not objective_metrics.get("nonblank_render", True):
        return QualityRouterOutput(
            status="failed",
            quality_band="poor",
            next_action="fallback",
            final_score=0.0,
            failure_type="structure",
            reason=["Render is blank (all transparent or all black)."],
            protected_aspects=_protected,
        )

    # --- Metric-based routing ---
    final_score = compute_final_score(objective_metrics, semantic_scores)
    reasons: list[str] = []

    if final_score >= 0.85:
        status = "pass"
        quality_band = "excellent"
        next_action = "final"
        failure_type = "none"
        reasons.append(f"Score {final_score:.3f} meets excellent threshold (≥0.85).")
    elif final_score >= 0.70:
        status = "pass"
        quality_band = "good"
        next_action = "final"
        failure_type = "none"
        reasons.append(f"Score {final_score:.3f} meets good threshold (≥0.70).")
    elif final_score >= 0.55:
        status = "preview"
        quality_band = "acceptable"
        next_action = "optimize"
        failure_type = "none"
        reasons.append(
            f"Score {final_score:.3f} is acceptable (≥0.55); optimization recommended."
        )
    elif final_score >= 0.40:
        status = "preview"
        quality_band = "poor"
        next_action = "revise"
        failure_type = "parameter"
        reasons.append(
            f"Score {final_score:.3f} is marginal (≥0.40); revision required."
        )
    else:
        status = "failed"
        quality_band = "poor"
        next_action = "fallback"
        failure_type = "structure"
        reasons.append(
            f"Score {final_score:.3f} is below minimum threshold (<0.40)."
        )

    return QualityRouterOutput(
        status=status,
        quality_band=quality_band,
        next_action=next_action,
        final_score=final_score,
        failure_type=failure_type,
        reason=reasons,
        protected_aspects=_protected,
    )
