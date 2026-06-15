"""Tests for the human-in-loop checkpoint resolver (V1)."""

from __future__ import annotations

import pytest

from app.pipeline.checkpoints import (
    CheckpointError,
    PipelineCheckpoint,
    checkpoint_metadata,
    list_checkpoints,
    resolve_checkpoint,
)

SELECTED_GLSL = "#define R 0.72\nvoid mainImage(out vec4 c, in vec2 p){ c = vec4(R); }"
CV_GLSL = "#define R 0.40\nvoid mainImage(out vec4 c, in vec2 p){ c = vec4(R); }"
ITER1_GLSL = "#define R 0.65\nvoid mainImage(out vec4 c, in vec2 p){ c = vec4(R); }"
ITER2_GLSL = "#define R 0.67\nvoid mainImage(out vec4 c, in vec2 p){ c = vec4(R); }"


def _result() -> dict:
    """A pipeline result/store-entry with candidates, iterations, and a final."""
    return {
        "selected_candidate_id": "llm_0",
        "selected_glsl": SELECTED_GLSL,
        "quality_router": {"final_score": 0.72},
        "scoreboard": {
            "selected_id": "llm_0",
            "candidates": [
                {
                    "id": "llm_0",
                    "source": "llm",
                    "selected": True,
                    "previewable": True,
                    "compile_glsl": SELECTED_GLSL,
                    "final_score": 0.61,
                    "objective_metrics": {"mse": 0.2},
                    "quality_router": {"final_score": 0.61},
                },
                {
                    "id": "cv_0",
                    "source": "cv",
                    "selected": False,
                    "previewable": True,
                    "compile_glsl": CV_GLSL,
                    "final_score": 0.40,
                    "objective_metrics": {},
                    "quality_router": {},
                },
                {
                    "id": "rule_0",
                    "source": "rule",
                    "selected": False,
                    "previewable": False,
                    "compile_glsl": "",
                    "final_score": None,
                    "objective_metrics": {},
                    "quality_router": {},
                },
            ],
        },
        "refinement_history": [
            {
                "iteration": 1,
                "compile_glsl": ITER1_GLSL,
                "score_after": 0.65,
                "improved": True,
                "accepted": True,
                "changes_summary": "1 changed line",
            },
            {
                "iteration": 2,
                "compile_glsl": ITER2_GLSL,
                "score_after": 0.67,
                "improved": True,
                "accepted": True,
            },
            {
                "iteration": 3,
                "compile_glsl": None,
                "score_after": None,
                "error": "LLM returned no usable GLSL",
            },
        ],
    }


# ---------------------------------------------------------------------------
# list_checkpoints
# ---------------------------------------------------------------------------

def test_list_includes_previewable_candidates_only():
    ids = {cp["id"] for cp in list_checkpoints(_result())}
    assert "candidate:llm_0" in ids
    assert "candidate:cv_0" in ids
    # rule_0 has no compiled GLSL -> not a branchable checkpoint
    assert "candidate:rule_0" not in ids


def test_list_marks_selected_candidate_as_baseline():
    by_id = {cp["id"]: cp for cp in list_checkpoints(_result())}
    selected = by_id["candidate:llm_0"]
    assert selected["accepted"] is True
    assert selected["label"] == "Selected baseline"
    assert by_id["candidate:cv_0"]["accepted"] is False


def test_list_includes_refinement_iterations_with_glsl():
    by_id = {cp["id"]: cp for cp in list_checkpoints(_result())}
    assert by_id["refinement:iter:1"]["kind"] == "refinement_iter"
    assert by_id["refinement:iter:1"]["iteration"] == 1
    assert by_id["refinement:iter:2"]["score"] == pytest.approx(0.67)
    # iteration 3 produced no GLSL -> excluded
    assert "refinement:iter:3" not in by_id


def test_list_includes_final_selected():
    by_id = {cp["id"]: cp for cp in list_checkpoints(_result())}
    final = by_id["final:selected"]
    assert final["kind"] == "final"
    assert final["score"] == pytest.approx(0.72)
    assert final["has_glsl"] is True


def test_list_omits_raw_glsl_payload():
    for cp in list_checkpoints(_result()):
        assert "glsl" not in cp
        assert cp["has_glsl"] is True
        assert set(cp) == {"id", "kind", "label", "score", "iteration", "accepted", "has_glsl"}


def test_list_handles_empty_result():
    assert list_checkpoints({}) == []


# ---------------------------------------------------------------------------
# resolve_checkpoint
# ---------------------------------------------------------------------------

def test_resolve_candidate_by_id():
    cp = resolve_checkpoint(_result(), "candidate:cv_0")
    assert isinstance(cp, PipelineCheckpoint)
    assert cp.kind == "candidate"
    assert cp.glsl == CV_GLSL
    assert cp.candidate_id == "cv_0"


def test_resolve_candidate_selected_alias():
    cp = resolve_checkpoint(_result(), "candidate:selected")
    assert cp.glsl == SELECTED_GLSL
    assert cp.kind == "candidate"
    assert cp.score == pytest.approx(0.61)


def test_resolve_refinement_iteration():
    cp = resolve_checkpoint(_result(), "refinement:iter:2")
    assert cp.kind == "refinement_iter"
    assert cp.iteration == 2
    assert cp.glsl == ITER2_GLSL


def test_resolve_final_selected():
    cp = resolve_checkpoint(_result(), "final:selected")
    assert cp.kind == "final"
    assert cp.glsl == SELECTED_GLSL
    assert cp.score == pytest.approx(0.72)


def test_resolve_unknown_candidate_raises():
    with pytest.raises(CheckpointError):
        resolve_checkpoint(_result(), "candidate:does_not_exist")


def test_resolve_unknown_iteration_raises():
    with pytest.raises(CheckpointError):
        resolve_checkpoint(_result(), "refinement:iter:99")


def test_resolve_malformed_id_raises():
    with pytest.raises(CheckpointError):
        resolve_checkpoint(_result(), "totally-bogus")


def test_resolve_checkpoint_without_glsl_raises():
    # rule_0 compiled to empty GLSL -> cannot seed a branch
    with pytest.raises(CheckpointError):
        resolve_checkpoint(_result(), "candidate:rule_0")


def test_resolve_final_without_selected_glsl_raises():
    result = _result()
    result["selected_glsl"] = None
    with pytest.raises(CheckpointError):
        resolve_checkpoint(result, "final:selected")


# ---------------------------------------------------------------------------
# checkpoint_metadata
# ---------------------------------------------------------------------------

def test_checkpoint_metadata_summary_shape():
    cp = resolve_checkpoint(_result(), "refinement:iter:1")
    meta = checkpoint_metadata(cp)
    assert meta["id"] == "refinement:iter:1"
    assert meta["has_glsl"] is True
    assert "glsl" not in meta
    assert set(meta) == {"id", "kind", "label", "score", "iteration", "accepted", "has_glsl"}
