"""Tests for human-in-loop feedback note construction + validation (V1)."""

from __future__ import annotations

import pytest

from p2s_agent.orchestration.checkpoints import PipelineCheckpoint
from p2s_agent.orchestration.human_feedback import (
    FeedbackValidationError,
    build_human_feedback_notes,
    validate_feedback,
)


def _checkpoint() -> PipelineCheckpoint:
    return PipelineCheckpoint(
        id="refinement:iter:2",
        kind="refinement_iter",
        label="Iteration 2 proposal",
        shader_kind="glsl",
        glsl="void mainImage(out vec4 c, in vec2 p){ c = vec4(0.5); }",
        score=0.67,
        iteration=2,
    )


def test_feedback_is_first_note_as_human_goal():
    notes = build_human_feedback_notes(
        feedback="make the water reflection stronger, do not darken the image",
        mode="refine",
        locks={},
        checkpoint=_checkpoint(),
    )
    assert notes[0].startswith("[HUMAN GOAL] make the water reflection stronger")
    assert any(n.startswith("[START CHECKPOINT] id=refinement:iter:2") for n in notes)


def test_locks_emit_lock_notes():
    notes = build_human_feedback_notes(
        feedback="brighter",
        mode="refine",
        locks={
            "preserve_layout": True,
            "preserve_palette": False,
            "preserve_background": True,
            "small_edits_only": True,
        },
        checkpoint=_checkpoint(),
    )
    joined = "\n".join(notes)
    assert "[LOCK] Preserve layout" in joined
    assert "[LOCK] Preserve background" in joined
    assert "[LOCK] Make small, targeted edits" in joined
    # preserve_palette is False -> no palette lock note
    assert "color palette" not in joined


def test_continue_mode_allows_empty_feedback():
    notes = build_human_feedback_notes(
        feedback="",
        mode="continue",
        locks={},
        checkpoint=_checkpoint(),
    )
    assert notes[0].startswith("[MODE] Continue automatic optimization")
    assert not any(n.startswith("[HUMAN GOAL]") for n in notes)


def test_polish_mode_adds_polish_note():
    notes = build_human_feedback_notes(
        feedback="cleaner edges",
        mode="polish",
        locks={},
        checkpoint=_checkpoint(),
    )
    assert any("[MODE] Polish only" in n for n in notes)


def test_validate_feedback_rejects_empty_for_refine():
    with pytest.raises(FeedbackValidationError):
        validate_feedback("   ", "refine")


def test_validate_feedback_rejects_empty_for_polish():
    with pytest.raises(FeedbackValidationError):
        validate_feedback("", "polish")


def test_validate_feedback_allows_empty_for_continue():
    # should not raise
    validate_feedback("", "continue")


def test_validate_feedback_allows_nonempty_for_refine():
    validate_feedback("make it brighter", "refine")
