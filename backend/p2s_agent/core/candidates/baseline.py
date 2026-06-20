"""Baseline candidate: the rule-based candidate labeled as 'baseline'.

The baseline is the reference point all other candidates are compared against.
It is always enabled and always runs first.
"""

from __future__ import annotations

from p2s_agent.core.candidates.rule import generate_rule_candidate


def generate_baseline_candidate(
    preprocess: dict,
    canvas_width: int = 512,
    canvas_height: int = 512,
) -> dict:
    """Generate the baseline DSL candidate by delegating to the rule candidate.

    The baseline is always run first and serves as the reference point that
    all other candidates are compared against.

    Args:
        preprocess: Dict of preprocessed image features.
        canvas_width: Output canvas width in pixels.
        canvas_height: Output canvas height in pixels.

    Returns:
        A valid DSL dict with ``_meta`` indicating source="baseline".
    """
    dsl = generate_rule_candidate(preprocess, canvas_width, canvas_height)
    dsl["_meta"] = {"source": "baseline", "priority": 0}
    return dsl
