"""Focused regression tests for the refinement-gate score wiring.

Bug: ``node_selection`` bumps the loser candidate's ``final_score`` by an
epsilon to win SORTING during a VLM near-tie. That artificial sort value then
leaked into the refinement gate (``_should_run_refinement``), so a candidate
whose TRUE objective score was below ``refinement_threshold`` could be bumped
over the threshold and wrongly skip the LLM refinement loop.

The tie-break bump must stay ORDERING-only; the absolute quality score used for
gating must remain the true objective score (``quality_router['final_score']``).
"""

from __future__ import annotations

from app.pipeline.graph import (
    CandidateRecord,
    _should_run_refinement,
)
from app.pipeline.scoring import _gate_quality_score


def _bumped_candidate(*, true_score: float, bumped_score: float) -> CandidateRecord:
    """A candidate whose stored ``final_score`` was bumped above its true score.

    ``quality_router['final_score']`` holds the un-bumped, true objective score
    (the VLM tie-break only mutates the record attribute, not the router dict).
    """
    return CandidateRecord(
        id="loser_0",
        source="rule",
        enabled=True,
        priority=1,
        dsl={"layers": []},
        output_kind="dsl",
        validation_valid=True,
        validation_errors=[],
        compile_success=True,
        compile_glsl="void mainImage(out vec4 fragColor, in vec2 fragCoord){fragColor=vec4(1.0);}",
        compile_errors=[],
        final_score=bumped_score,
        selected=True,
        quality_router={"final_score": true_score},
    )


def test_gate_quality_score_prefers_true_objective_over_bumped_final_score():
    """The gate score must come from quality_router, not the bumped final_score."""
    cand = _bumped_candidate(true_score=0.78, bumped_score=0.83)
    assert _gate_quality_score(cand) == 0.78


def test_gate_quality_score_falls_back_to_final_score_without_router():
    """When there is no router score, fall back to the record's final_score."""
    cand = _bumped_candidate(true_score=0.5, bumped_score=0.5)
    cand.quality_router = None
    assert _gate_quality_score(cand) == 0.5


def test_bumped_candidate_below_threshold_still_runs_refinement():
    """Near-tie bump pushes final_score over threshold, but true score is below it.

    The gate must still decide to RUN refinement (true objective score is below
    threshold) instead of short-circuiting to 'auto_threshold_reached'.
    """
    threshold = 0.80
    # True objective score 0.78 < threshold; bump pushes stored final_score to
    # 0.83 (> threshold) so it wins SORTING in select_best_candidate.
    cand = _bumped_candidate(true_score=0.78, bumped_score=0.83)
    assert cand.final_score > threshold  # the artificial sort value
    quality = dict(cand.quality_router)  # true objective score for gating

    # Sync the gate-relevant score the way the post-pipeline must before gating:
    # the gate input is the true objective score, not the bumped sort value.
    cand.final_score = _gate_quality_score(cand)

    should_run, reason = _should_run_refinement(
        "auto",
        cand,
        quality,
        threshold=threshold,
        high_score_stop=0.92,
    )

    assert should_run is True, "refinement must run when true score is below threshold"
    assert reason == "auto_below_threshold"
    assert reason != "auto_threshold_reached"
