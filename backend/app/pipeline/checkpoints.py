"""Checkpoint resolver for human-in-loop branch refinement (V1).

A *checkpoint* is a branchable point inside a run: a candidate, a refinement
iteration proposal, or the final selected shader. ``list_checkpoints`` produces
lightweight metadata for the UI; ``resolve_checkpoint`` maps a checkpoint id to
the GLSL that should seed a new branch run.

The functions operate on either a completed pipeline ``result`` dict or a
running ``_run_store`` entry — both expose ``scoreboard`` / ``refinement_history``
/ ``selected_glsl`` with the same shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

CheckpointKind = Literal["candidate", "refinement_iter", "final"]
ShaderKind = Literal["glsl"]


class CheckpointError(ValueError):
    """Raised when a checkpoint id is malformed, unknown, or has no GLSL."""


@dataclass
class PipelineCheckpoint:
    id: str
    kind: CheckpointKind
    label: str
    shader_kind: ShaderKind
    glsl: str
    score: float | None = None
    metrics: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    iteration: int | None = None
    candidate_id: str | None = None
    accepted: bool | None = None
    changes_summary: str | None = None
    source: str | None = None


def _has_glsl(value: str | None) -> bool:
    return bool((value or "").strip())


def _previewable(candidate: dict) -> bool:
    if candidate.get("previewable") is not None:
        return bool(candidate["previewable"])
    return bool(candidate.get("compile_success")) and _has_glsl(candidate.get("compile_glsl"))


def _selected_candidate(result: dict) -> dict | None:
    scoreboard = result.get("scoreboard") or {}
    candidates = scoreboard.get("candidates") or []
    selected_id = scoreboard.get("selected_id") or result.get("selected_candidate_id")
    for cand in candidates:
        if cand.get("id") == selected_id:
            return cand
    for cand in candidates:
        if cand.get("selected"):
            return cand
    return None


def _final_score(result: dict) -> float | None:
    quality = result.get("quality_router") or {}
    if quality.get("final_score") is not None:
        return quality.get("final_score")
    selected = _selected_candidate(result)
    if selected is not None:
        return selected.get("final_score")
    return None


def list_checkpoints(result: dict) -> list[dict]:
    """Summarize the branchable checkpoints in a run.

    Returns lightweight metadata dicts (``id, kind, label, score, iteration,
    accepted, has_glsl``) — never the raw GLSL, which can be large and is
    re-resolved on demand by ``resolve_checkpoint``.

    Order: previewable candidates, then refinement iteration proposals that
    produced GLSL, then the final selected shader.
    """
    checkpoints: list[dict] = []

    scoreboard = result.get("scoreboard") or {}
    for cand in scoreboard.get("candidates") or []:
        if not _previewable(cand):
            continue
        selected = bool(cand.get("selected"))
        checkpoints.append({
            "id": f"candidate:{cand.get('id')}",
            "kind": "candidate",
            "label": "Selected baseline" if selected else f"Candidate {cand.get('id')}",
            "score": cand.get("final_score"),
            "iteration": None,
            "accepted": selected,
            "has_glsl": True,
        })

    for entry in result.get("refinement_history") or []:
        if not _has_glsl(entry.get("compile_glsl")):
            continue
        iteration = entry.get("iteration")
        accepted = entry.get("accepted")
        if accepted is None:
            accepted = entry.get("improved")
        checkpoints.append({
            "id": f"refinement:iter:{iteration}",
            "kind": "refinement_iter",
            "label": f"Iteration {iteration} proposal",
            "score": entry.get("score_after"),
            "iteration": iteration,
            "accepted": accepted,
            "has_glsl": True,
        })

    if _has_glsl(result.get("selected_glsl")):
        checkpoints.append({
            "id": "final:selected",
            "kind": "final",
            "label": "Current best",
            "score": _final_score(result),
            "iteration": None,
            "accepted": True,
            "has_glsl": True,
        })

    return checkpoints


def checkpoint_metadata(cp: PipelineCheckpoint) -> dict:
    """The same lightweight metadata shape ``list_checkpoints`` emits."""
    return {
        "id": cp.id,
        "kind": cp.kind,
        "label": cp.label,
        "score": cp.score,
        "iteration": cp.iteration,
        "accepted": cp.accepted,
        "has_glsl": _has_glsl(cp.glsl),
    }


def resolve_checkpoint(result: dict, checkpoint_id: str) -> PipelineCheckpoint:
    """Resolve a checkpoint id to its seed GLSL and metadata.

    Raises ``CheckpointError`` when the id is malformed, unknown, or resolves to
    a checkpoint with no compiled GLSL (which cannot seed a branch run).
    """
    if not checkpoint_id or ":" not in checkpoint_id:
        raise CheckpointError(f"malformed checkpoint id: {checkpoint_id!r}")

    if checkpoint_id == "final:selected":
        return _resolve_final(result)
    if checkpoint_id == "candidate:selected":
        return _resolve_selected_candidate(result)
    if checkpoint_id.startswith("candidate:"):
        return _resolve_candidate(result, checkpoint_id.split(":", 1)[1])
    if checkpoint_id.startswith("refinement:iter:"):
        return _resolve_iteration(result, checkpoint_id)

    raise CheckpointError(f"unknown checkpoint id: {checkpoint_id!r}")


def _resolve_final(result: dict) -> PipelineCheckpoint:
    glsl = result.get("selected_glsl")
    if not _has_glsl(glsl):
        raise CheckpointError("final:selected has no GLSL")
    return PipelineCheckpoint(
        id="final:selected",
        kind="final",
        label="Current best",
        shader_kind="glsl",
        glsl=glsl,
        score=_final_score(result),
        metrics=dict(result.get("objective_metrics") or {}),
        quality=dict(result.get("quality_router") or {}),
        accepted=True,
        source="selected_glsl",
    )


def _resolve_selected_candidate(result: dict) -> PipelineCheckpoint:
    cand = _selected_candidate(result)
    glsl = (cand.get("compile_glsl") if cand else None) or result.get("selected_glsl")
    if not _has_glsl(glsl):
        raise CheckpointError("candidate:selected has no GLSL")
    return PipelineCheckpoint(
        id="candidate:selected",
        kind="candidate",
        label="Selected baseline",
        shader_kind="glsl",
        glsl=glsl,
        score=(cand.get("final_score") if cand else _final_score(result)),
        metrics=dict((cand or {}).get("objective_metrics") or {}),
        quality=dict((cand or {}).get("quality_router") or {}),
        candidate_id=(cand.get("id") if cand else None),
        accepted=True,
        source="scoreboard",
    )


def _resolve_candidate(result: dict, candidate_id: str) -> PipelineCheckpoint:
    scoreboard = result.get("scoreboard") or {}
    for cand in scoreboard.get("candidates") or []:
        if cand.get("id") != candidate_id:
            continue
        glsl = cand.get("compile_glsl")
        if not _has_glsl(glsl):
            raise CheckpointError(f"candidate {candidate_id!r} has no GLSL")
        return PipelineCheckpoint(
            id=f"candidate:{candidate_id}",
            kind="candidate",
            label="Selected baseline" if cand.get("selected") else f"Candidate {candidate_id}",
            shader_kind="glsl",
            glsl=glsl,
            score=cand.get("final_score"),
            metrics=dict(cand.get("objective_metrics") or {}),
            quality=dict(cand.get("quality_router") or {}),
            candidate_id=candidate_id,
            accepted=bool(cand.get("selected")),
            source=cand.get("source"),
        )
    raise CheckpointError(f"unknown candidate checkpoint: {candidate_id!r}")


def _resolve_iteration(result: dict, checkpoint_id: str) -> PipelineCheckpoint:
    suffix = checkpoint_id.split("refinement:iter:", 1)[1]
    try:
        iteration = int(suffix)
    except ValueError as exc:
        raise CheckpointError(f"malformed iteration checkpoint: {checkpoint_id!r}") from exc
    for entry in result.get("refinement_history") or []:
        if entry.get("iteration") != iteration:
            continue
        glsl = entry.get("compile_glsl")
        if not _has_glsl(glsl):
            raise CheckpointError(f"iteration {iteration} has no GLSL")
        accepted = entry.get("accepted")
        if accepted is None:
            accepted = entry.get("improved")
        return PipelineCheckpoint(
            id=checkpoint_id,
            kind="refinement_iter",
            label=f"Iteration {iteration} proposal",
            shader_kind="glsl",
            glsl=glsl,
            score=entry.get("score_after"),
            iteration=iteration,
            accepted=accepted,
            changes_summary=entry.get("changes_summary"),
            source="refinement_history",
        )
    raise CheckpointError(f"unknown refinement iteration: {iteration}")
