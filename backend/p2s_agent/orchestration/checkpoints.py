"""Checkpoint resolver for human-in-loop branch refinement (V1 + V2.1).

A *checkpoint* is a branchable point inside a run: a candidate, a refinement
iteration proposal, or the final selected shader. ``list_checkpoints`` produces
lightweight metadata for the UI; ``resolve_checkpoint`` maps a checkpoint id to
the GLSL that should seed a new branch run.

V2.1 additions:
    - ``build_timeline`` — rich timeline entries with artifact_ids and deltas.
    - ``save_timeline`` — write timeline.json into a run directory.
    - ``resolve_checkpoint_artifact`` — safely map checkpoint+kind to a file Path.

The functions operate on either a completed pipeline ``result`` dict or a
running ``_run_store`` entry — both expose ``scoreboard`` / ``refinement_history``
/ ``selected_glsl`` with the same shape.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from p2s_agent.core.pipeline.artifacts import save_json

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
    selected_cand = _selected_candidate(result)
    selected_id = selected_cand.get("id") if selected_cand else None
    for cand in scoreboard.get("candidates") or []:
        if not _previewable(cand):
            continue
        selected = cand.get("id") == selected_id
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


# ---------------------------------------------------------------------------
# V2.1 — Timeline, save_timeline, resolve_checkpoint_artifact
# ---------------------------------------------------------------------------

_CANDIDATE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SUFFIX_ALLOWLIST = {".png", ".json", ".glsl", ".txt"}
_VALID_ARTIFACT_KINDS = {"shader", "render", "llm_io"}


def build_timeline(result: dict, *, run_id: str | None = None) -> list[dict]:
    """Build a rich ordered list of CheckpointTimelineEntry dicts.

    Each entry carries the full timeline interface — all fields present,
    unused ones set to None. Entries are ordered: selected baseline,
    other previewable candidates, refinement iterations (including
    rejected), then final.

    Args:
        result: Pipeline result or run-store dict.
        run_id: Override for the run_id field; falls back to
            ``result.get("run_id")``.
    """
    effective_run_id: str | None = run_id if run_id is not None else result.get("run_id")

    def _entry(
        *,
        id: str,
        kind: str,
        label: str,
        iteration: int | None = None,
        score: float | None = None,
        score_before: float | None = None,
        delta: float | None = None,
        accepted: bool | None = None,
        human_goal_override: str | None = None,
        changes_summary: str | None = None,
        artifact_ids: dict | None = None,
    ) -> dict:
        return {
            "id": id,
            "run_id": effective_run_id,
            "kind": kind,
            "label": label,
            "iteration": iteration,
            "score": score,
            "score_before": score_before,
            "delta": delta,
            "accepted": accepted,
            "human_goal_override": human_goal_override,
            "changes_summary": changes_summary,
            "has_glsl": True,
            "artifact_ids": artifact_ids if artifact_ids is not None else {},
        }

    timeline: list[dict] = []

    scoreboard = result.get("scoreboard") or {}
    selected_cand = _selected_candidate(result)
    selected_id = selected_cand.get("id") if selected_cand else None

    selected_entry = None
    other_entries = []
    for cand in scoreboard.get("candidates") or []:
        if not _previewable(cand):
            continue
        cid = cand.get("id")
        if cid == selected_id:
            e = _entry(
                id="candidate:selected",
                kind="candidate",
                label="Selected baseline",
                score=cand.get("final_score"),
                accepted=True,
                artifact_ids={
                    "shader": "checkpoint:candidate:selected:shader",
                    "render": "checkpoint:candidate:selected:render",
                },
            )
            e["has_glsl"] = _has_glsl(cand.get("compile_glsl"))
            selected_entry = e
        else:
            e = _entry(
                id=f"candidate:{cid}",
                kind="candidate",
                label=f"Candidate {cid}",
                score=cand.get("final_score"),
                accepted=False,
                artifact_ids={
                    "render": f"checkpoint:candidate:{cid}:render",
                    "llm_io": f"checkpoint:candidate:{cid}:llm_io",
                },
            )
            e["has_glsl"] = _has_glsl(cand.get("compile_glsl"))
            other_entries.append(e)
    if selected_entry is not None:
        timeline.append(selected_entry)
    timeline.extend(other_entries)

    for entry in result.get("refinement_history") or []:
        if not _has_glsl(entry.get("compile_glsl")):
            continue
        iteration = entry.get("iteration")
        score_after = entry.get("score_after")
        score_before = entry.get("score_before")
        if isinstance(score_after, (int, float)) and isinstance(score_before, (int, float)):
            delta: float | None = score_after - score_before
        else:
            delta = None
        accepted = entry.get("accepted")
        if accepted is None:
            accepted = entry.get("improved")
        timeline.append(_entry(
            id=f"refinement:iter:{iteration}",
            kind="refinement_iter",
            label=f"Iteration {iteration} proposal",
            iteration=iteration,
            score=score_after,
            score_before=score_before,
            delta=delta,
            accepted=accepted,
            human_goal_override=entry.get("human_goal_override"),
            changes_summary=entry.get("changes_summary"),
            artifact_ids={},
        ))

    if _has_glsl(result.get("selected_glsl")):
        timeline.append(_entry(
            id="final:selected",
            kind="final",
            label="Current best",
            score=_final_score(result),
            accepted=True,
            artifact_ids={"shader": "checkpoint:final:selected:shader"},
        ))

    return timeline


def save_timeline(run_dir, result: dict, *, run_id: str | None = None) -> Path:
    """Write ``timeline.json`` into *run_dir* and return the written path.

    Args:
        run_dir: Destination directory. Accepts ``RunDir`` (has ``.path``),
            ``Path``, or ``str``.
        result: Pipeline result dict.
        run_id: Optional run_id override; falls back to ``result.get("run_id")``.
    """
    # Accept RunDir (has .path), Path, or str
    if hasattr(run_dir, "path"):
        dest = Path(run_dir.path)
    else:
        dest = Path(run_dir)

    effective_run_id: str | None = run_id if run_id is not None else result.get("run_id")
    data = {
        "run_id": effective_run_id,
        "timeline": build_timeline(result, run_id=effective_run_id),
    }
    return save_json(dest / "timeline.json", data)


def resolve_checkpoint_artifact(
    result: dict,
    checkpoint_id: str,
    kind: str,
    *,
    run_dir=None,
) -> Path:
    """Resolve a checkpoint + artifact kind to a safe absolute file path.

    This function is SECURITY-CRITICAL. It validates the kind, candidate id
    format, candidate existence in the scoreboard, path containment inside
    run_dir, and suffix allowlist before returning the path. The file need
    not exist — existence checking is the caller's responsibility.

    Args:
        result: Pipeline result dict (provides scoreboard for id validation).
        checkpoint_id: Checkpoint identifier (e.g. ``"final:selected"``).
        kind: Artifact kind — ``"shader"``, ``"render"``, or ``"llm_io"``.
        run_dir: Base directory for the run. Falls back to
            ``result.get("run_dir")``. **Must not be None.**

    Returns:
        Resolved, containment-checked, allowlisted absolute Path.

    Raises:
        CheckpointError: On any validation failure (kind, id, traversal, etc.).
    """
    if kind not in _VALID_ARTIFACT_KINDS:
        raise CheckpointError(
            f"unknown artifact kind {kind!r}; must be one of {sorted(_VALID_ARTIFACT_KINDS)}"
        )

    effective_run_dir = run_dir if run_dir is not None else result.get("run_dir")
    if not effective_run_dir:
        raise CheckpointError("run_dir is required to resolve a checkpoint artifact")

    if hasattr(effective_run_dir, "path"):
        base = Path(effective_run_dir.path).resolve()
    else:
        base = Path(effective_run_dir).resolve()

    # Determine relative path based on checkpoint_id + kind
    if checkpoint_id in ("final:selected", "candidate:selected"):
        if kind == "shader":
            relative = Path("selected_shader.glsl")
        elif kind == "render":
            selected = _selected_candidate(result)
            if selected is None:
                raise CheckpointError(
                    f"{checkpoint_id}+render: no selected candidate found in scoreboard"
                )
            sid = selected.get("id")
            relative = Path("candidates") / f"{sid}_render.png"
        else:  # llm_io
            raise CheckpointError(
                f"{checkpoint_id}+{kind}: no llm_io file for selected candidate"
            )

    elif checkpoint_id.startswith("candidate:"):
        candidate_id = checkpoint_id[len("candidate:"):]
        # Validate id format first (catches traversal patterns)
        if not _CANDIDATE_ID_RE.match(candidate_id):
            raise CheckpointError(
                f"candidate id {candidate_id!r} contains disallowed characters"
            )
        # Validate id is known in the scoreboard
        scoreboard = result.get("scoreboard") or {}
        known_ids = {c.get("id") for c in scoreboard.get("candidates") or []}
        if candidate_id not in known_ids:
            raise CheckpointError(
                f"unknown candidate id {candidate_id!r} (not in scoreboard)"
            )
        if kind == "render":
            relative = Path("candidates") / f"{candidate_id}_render.png"
        elif kind == "llm_io":
            relative = Path("candidates") / f"{candidate_id}.json"
        else:  # shader — candidate GLSL is inline, no file
            raise CheckpointError(
                f"candidate:{candidate_id}+shader: candidate GLSL is inline, no file artifact"
            )

    elif checkpoint_id.startswith("refinement:iter:"):
        raise CheckpointError(
            "no file artifact for refinement iterations (GLSL is inline)"
        )

    else:
        raise CheckpointError(f"unknown checkpoint id {checkpoint_id!r}")

    # Build and contain the path
    candidate_path = (base / relative).resolve()
    try:
        candidate_path.relative_to(base)
    except ValueError:
        raise CheckpointError(
            f"resolved path {candidate_path} escapes run_dir {base}"
        )

    # Suffix allowlist
    if candidate_path.suffix not in _SUFFIX_ALLOWLIST:
        raise CheckpointError(
            f"disallowed file suffix {candidate_path.suffix!r}; "
            f"allowed: {sorted(_SUFFIX_ALLOWLIST)}"
        )

    return candidate_path
