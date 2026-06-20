"""Human feedback -> LLM prompt notes for branch refinement (V1).

Turns a user's natural-language goal, branch mode, and lock toggles into the
``extra_feedback`` notes injected at the start of the refinement loop, and
validates that a goal is present when the mode requires one.
"""

from __future__ import annotations

from p2s_agent.orchestration.checkpoints import PipelineCheckpoint

MODES = ("continue", "refine", "polish")

# Modes that pursue a stated user goal and therefore require non-empty feedback.
_FEEDBACK_REQUIRED_MODES = ("refine", "polish")


class FeedbackValidationError(ValueError):
    """Raised when a branch mode requires feedback but none was provided."""


def validate_feedback(feedback: str, mode: str) -> None:
    """Raise ``FeedbackValidationError`` when ``mode`` needs a goal but ``feedback`` is blank."""
    if mode in _FEEDBACK_REQUIRED_MODES and not (feedback or "").strip():
        raise FeedbackValidationError(f"feedback is required for mode '{mode}'")


def build_human_feedback_notes(
    *,
    feedback: str,
    mode: str,
    locks: dict | None,
    checkpoint: PipelineCheckpoint,
) -> list[str]:
    """Build the ordered prompt notes describing the human's intent.

    The human goal (or a continue-mode marker) is the first note so it leads the
    LLM prompt; the start checkpoint, mode hints, and lock constraints follow.
    """
    notes: list[str] = [
        f"[START CHECKPOINT] id={checkpoint.id}; score={checkpoint.score}",
    ]

    goal = (feedback or "").strip()
    if goal:
        notes.insert(0, "[HUMAN GOAL] " + goal)
    elif mode == "continue":
        notes.insert(
            0, "[MODE] Continue automatic optimization from the selected checkpoint."
        )

    if mode == "polish":
        notes.append(
            "[MODE] Polish only: keep composition and major shader structure stable."
        )

    locks = locks or {}
    if locks.get("preserve_layout"):
        notes.append("[LOCK] Preserve layout/composition; do not move major visual elements.")
    if locks.get("preserve_palette"):
        notes.append("[LOCK] Preserve the current color palette unless required by the human goal.")
    if locks.get("preserve_background"):
        notes.append("[LOCK] Preserve background and large-scale lighting.")
    if locks.get("small_edits_only"):
        notes.append("[LOCK] Make small, targeted edits; avoid rewriting the shader from scratch.")

    return notes
