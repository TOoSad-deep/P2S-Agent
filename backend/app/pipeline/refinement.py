"""LLM-driven DSL refinement loop and revision-patch construction for
PNG-to-Shader. Split out of graph.py (2026-06-11); behavior is unchanged."""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

from app.pipeline.artifacts import save_json
from app.dsl.compiler import compile_dsl
from app.dsl.renderer import render_dsl_to_image
from app.pipeline.pool import CandidateRecord
from app.pipeline.revision import PatchOp, RevisionPatch
from app.pipeline.scoring import _evaluate_dsl, _metric_render_size
from app.services.logging_config import log_event


def _short_exception(exc: Exception) -> str:
    message = str(exc).strip()
    if len(message) > 500:
        message = message[:497] + "..."
    return f"{exc.__class__.__name__}: {message}" if message else exc.__class__.__name__


def _diff_dsl_summary(old_dsl: dict, new_dsl: dict) -> str:
    """Return a concise human-readable summary of what changed between two DSLs."""
    changes: list[str] = []
    old_layers: dict[str, dict] = {}
    new_layers: dict[str, dict] = {}
    for layer in old_dsl.get("layers", []):
        if isinstance(layer, dict) and layer.get("id"):
            old_layers[layer["id"]] = layer
    for layer in new_dsl.get("layers", []):
        if isinstance(layer, dict) and layer.get("id"):
            new_layers[layer["id"]] = layer

    for lid in sorted(set(old_layers) | set(new_layers)):
        if lid not in old_layers:
            changes.append(f"added layer '{lid}' ({new_layers[lid].get('type', '?')})")
        elif lid not in new_layers:
            changes.append(f"removed layer '{lid}'")
        else:
            old_l = old_layers[lid]
            new_l = new_layers[lid]
            if old_l.get("type") != new_l.get("type"):
                changes.append(f"{lid}: type {old_l.get('type')} → {new_l.get('type')}")
            old_p = old_l.get("params") or {}
            new_p = new_l.get("params") or {}
            for k in sorted(set(old_p) | set(new_p)):
                if old_p.get(k) != new_p.get(k):
                    changes.append(f"{lid}.{k}: {old_p.get(k)} → {new_p.get(k)}")
            old_fill = old_l.get("fill") or {}
            new_fill = new_l.get("fill") or {}
            if old_fill.get("type") != new_fill.get("type"):
                changes.append(f"{lid}.fill: {old_fill.get('type')} → {new_fill.get('type')}")
            elif old_fill.get("color") != new_fill.get("color") and old_fill.get("type") == "solid":
                changes.append(f"{lid}.fill.color: {old_fill.get('color')} → {new_fill.get('color')}")

    old_bg = (old_dsl.get("canvas") or {}).get("background")
    new_bg = (new_dsl.get("canvas") or {}).get("background")
    if old_bg != new_bg:
        changes.append(f"background: {old_bg} → {new_bg}")

    return "; ".join(changes[:8]) if changes else "no changes"


def build_semantic_notes(rubric: dict) -> list[str]:
    """Convert judge_rubric differences and hints into LLM feedback lines."""
    notes: list[str] = []
    for diff in list(rubric.get("differences", []))[:4]:
        notes.append(f"[VISUAL ISSUE] {diff}")
    for hint in list(rubric.get("revision_hints", []))[:3]:
        notes.append(f"[VISUAL GOAL] {hint}")
    return notes


def build_recent_history_notes(history: list[dict], max_entries: int = 3) -> list[str]:
    """Summarize recent iterations without embedding shader or DSL bodies."""
    notes: list[str] = []
    for h in history[-max_entries:]:
        if h.get("improved"):
            outcome = "accepted"
        elif h.get("error"):
            outcome = f"failed ({h.get('error_type') or 'error'})"
        else:
            outcome = "rejected"
        notes.append(
            f"[HISTORY iter {h.get('iteration')}] score "
            f"{h.get('score_before')} -> {h.get('score_after')} ({outcome}); "
            f"changes: {(h.get('changes_summary') or 'n/a')[:160]}"
        )
    return notes


def run_dsl_refinement_loop(
    preprocess: dict,
    initial_dsl: dict,
    initial_score: float,
    initial_metrics: dict,
    initial_quality: dict,
    reference_path: Path,
    *,
    canvas_width: int,
    canvas_height: int,
    max_shader_chars: int,
    protected_aspects: list[str] | None = None,
    max_iterations: int = 3,
    threshold: float = 0.80,
    high_score_stop: float = 0.92,
    min_improvement: float = 0.01,
    no_improvement_patience: int = 2,
    force_first_iteration: bool = False,
    initial_extra_feedback: "list[str] | None" = None,
    directed_acceptance: "dict | None" = None,
    directed_pairwise_judge: "Callable[[Path, Path], str | None] | None" = None,
    loop_dir: Path,
    strategy_reader: "Callable[[], dict] | None" = None,
    pairwise_judge: "Callable[[Path, Path], str | None] | None" = None,
    rubric_judge: "Callable[[Path], dict | None] | None" = None,
    on_iteration: "Callable[[dict], None] | None" = None,
    region_veto_fn: "Callable[[Path], object] | None" = None,
) -> dict:
    """Drive LLM to iteratively revise a DSL candidate.

    Each iteration: render current DSL → compute metrics → ask LLM to revise
    → validate/compile → render revised DSL → accept if improved.

    Returns dict with keys: best_dsl, best_glsl, best_score, best_metrics,
    best_quality, history (list of iteration records).
    """
    if protected_aspects is None:
        protected_aspects = ["layer_count", "primitive_types", "background"]
    from app.candidates.llm_scene import generate_llm_refinement
    from app.dsl.compiler import compile_dsl
    from app.dsl.validator import validate_dsl as _validate_dsl

    loop_dir.mkdir(parents=True, exist_ok=True)
    render_dir = loop_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "refinement start: initial_score=%.4f threshold=%.2f high_stop=%.2f max_iter=%d",
        float(initial_score),
        float(threshold),
        float(high_score_stop),
        int(max_iterations),
    )
    log_event(
        logger,
        "dsl_refinement_start",
        initial_score=float(initial_score),
        threshold=float(threshold),
        high_score_stop=float(high_score_stop),
        max_iterations=int(max_iterations),
        loop_dir=str(loop_dir),
    )

    best_dsl = initial_dsl
    best_score = initial_score
    best_metrics = dict(initial_metrics)
    best_quality = dict(initial_quality)
    _cr = compile_dsl(initial_dsl)
    best_glsl = _cr.glsl if _cr.success else ""

    # Render the starting DSL once so the LLM can see "what we have now" on
    # iter 1. We deliberately re-render here (rather than reusing the candidate
    # selection render) so this loop owns its own artifact lifecycle.
    baseline_render_path: Path | None = render_dir / "iter_0_baseline.png"
    render_width, render_height = _metric_render_size(canvas_width, canvas_height)
    try:
        render_dsl_to_image(
            initial_dsl,
            baseline_render_path,
            width=render_width,
            height=render_height,
        )
    except Exception:
        baseline_render_path = None
    current_render_path: Path | None = baseline_render_path

    history: list[dict] = []
    stop_reason = "max_iterations"
    no_improvement_count = 0
    # Persistent human-goal notes: prepended to every LLM call so a directed
    # branch keeps pursuing the user's intent even after transient feedback
    # resets on an accepted improvement.
    persistent_feedback: list[str] = list(initial_extra_feedback or [])
    extra_feedback: list[str] = []

    def _record(entry: dict) -> None:
        history.append(entry)
        if entry.get("llm_io") is not None:
            save_json(loop_dir / f"iter_{i + 1}_llm_io.json", entry["llm_io"])
        save_json(loop_dir / f"iter_{i + 1}.json", entry)
        if on_iteration is None:
            return
        try:
            on_iteration({
                "best_dsl": best_dsl,
                "best_glsl": best_glsl,
                "best_score": best_score,
                "best_metrics": best_metrics,
                "best_quality": best_quality,
                "history": list(history),
            })
        except Exception:
            logger.warning("on_iteration publish failed", exc_info=True)

    for i in range(max_iterations):
        # P2: read latest strategy + stop flag (one-shot per iteration)
        if strategy_reader is not None:
            try:
                live = strategy_reader()
            except Exception:
                live = {}
            if live.get("stop_requested"):
                stop_reason = "user_stop"
                break
            live_strategy = live.get("strategy") or {}
            if "refinement_threshold" in live_strategy:
                threshold = float(live_strategy["refinement_threshold"])
            if "refinement_high_score_stop" in live_strategy:
                high_score_stop = float(live_strategy["refinement_high_score_stop"])
            if "refinement_min_improvement" in live_strategy:
                min_improvement = float(live_strategy["refinement_min_improvement"])
            if "refinement_patience" in live_strategy:
                no_improvement_patience = int(live_strategy["refinement_patience"])
            if "max_refinement_iterations" in live_strategy:
                live_cap = int(live_strategy["max_refinement_iterations"])
                if i >= live_cap:
                    stop_reason = "user_lowered_cap"
                    break

        if best_score >= high_score_stop and not (force_first_iteration and not history):
            stop_reason = "high_score_stop"
            break
        if best_score >= threshold and not (force_first_iteration and not history):
            stop_reason = "threshold_reached"
            break

        scored_entries = [h["score_after"] for h in history if h.get("score_after") is not None]
        if len(scored_entries) >= 3:
            recent = scored_entries[-3:]
            if max(recent) - min(recent) < 0.02:
                stop_reason = "stagnation"
                break

        entry: dict = {
            "iteration": i + 1,
            "score_before": round(best_score, 4),
            "score_after": None,
            "delta": None,
            "improved": False,
            "meaningful_improvement": False,
            "changes_summary": None,
            "llm_io": None,
            "llm_duration_ms": None,
            "error": None,
            "error_type": None,
            "compile_glsl": None,
        }

        region_notes: list[str] = []
        if current_render_path is not None:
            try:
                from app.metrics.compute import grid_color_report
                region_notes = grid_color_report(reference_path, current_render_path)
            except Exception:
                logger.warning("grid_color_report failed", exc_info=True)

        semantic_notes: list[str] = []
        if rubric_judge is not None and current_render_path is not None:
            try:
                rubric = rubric_judge(current_render_path)
            except Exception:
                rubric = None
                logger.warning("rubric judge failed", exc_info=True)
            if rubric:
                semantic_notes = build_semantic_notes(rubric)

        history_notes = build_recent_history_notes(history)

        llm_start = time.monotonic()
        try:
            revised = generate_llm_refinement(
                preprocess=preprocess,
                current_dsl=best_dsl,
                metrics=best_metrics,
                quality_router=best_quality,
                canvas_width=canvas_width,
                canvas_height=canvas_height,
                reference_image_path=reference_path,
                current_render_path=current_render_path,
                extra_feedback=(
                    persistent_feedback + extra_feedback + history_notes
                    + semantic_notes + region_notes
                ) or None,
            )
        except Exception as exc:
            entry["llm_duration_ms"] = int((time.monotonic() - llm_start) * 1000)
            entry["error_type"] = exc.__class__.__name__
            entry["error"] = f"LLM call failed: {_short_exception(exc)}"
            log_event(
                logger,
                "dsl_refinement_llm_call_failed",
                level=logging.ERROR,
                iteration=i + 1,
                duration_ms=entry["llm_duration_ms"],
                error=entry["error"],
            )
            _record(entry)
            stop_reason = "llm_call_failed"
            break

        entry["llm_duration_ms"] = int((time.monotonic() - llm_start) * 1000)
        parse_error = None
        if revised and isinstance(revised, dict):
            entry["llm_io"] = revised.pop("_io", None)
            parse_error = revised.pop("_parse_error", None)

        if revised is None or parse_error:
            entry["error_type"] = "llm_returned_none"
            entry["error"] = parse_error or (
                "LLM returned no usable DSL: response content was empty, was not "
                "valid JSON, or did not contain a 'layers' field. Check provider "
                "support for response_format=json_object and the raw_response in "
                "this iteration's llm_io."
            )
            log_event(
                logger,
                "dsl_refinement_llm_parse_failed",
                level=logging.WARNING,
                iteration=i + 1,
                duration_ms=entry["llm_duration_ms"],
                error=entry["error"],
                attempts=len((entry.get("llm_io") or {}).get("attempts") or []),
                raw_response_len=len(str((entry.get("llm_io") or {}).get("raw_response") or "")),
            )
            _record(entry)
            stop_reason = "llm_returned_none"
            break

        val = _validate_dsl(revised)
        if not val.valid:
            entry["error"] = f"DSL invalid: {val.errors[:2]}"
            log_event(
                logger,
                "dsl_refinement_validation_failed",
                level=logging.WARNING,
                iteration=i + 1,
                errors=val.errors[:3],
            )
            _record(entry)
            no_improvement_count += 1
            if no_improvement_count >= no_improvement_patience:
                stop_reason = "no_improvement_patience"
                break
            continue

        cr = compile_dsl(revised)
        if not cr.success:
            entry["error"] = f"Compile failed: {cr.errors[:1]}"
            log_event(
                logger,
                "dsl_refinement_compile_failed",
                level=logging.WARNING,
                iteration=i + 1,
                errors=cr.errors[:3],
            )
            _record(entry)
            no_improvement_count += 1
            if no_improvement_count >= no_improvement_patience:
                stop_reason = "no_improvement_patience"
                break
            continue

        # Capture per-iteration GLSL so the frontend can preview exactly this
        # iteration's shader (independent of whether it ends up improving best).
        entry["compile_glsl"] = cr.glsl

        render_path = render_dir / f"iter_{i + 1}.png"
        new_metrics, new_quality, new_score, _ = _evaluate_dsl(
            revised,
            cr.glsl,
            reference_path,
            render_path,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            max_shader_chars=max_shader_chars,
            protected_aspects=protected_aspects,
        )

        if region_veto_fn is not None and render_path.exists():
            veto = region_veto_fn(render_path)
            if getattr(veto, "vetoed", False):
                entry["score_after"] = round(new_score, 4)
                entry["delta"] = round(new_score - best_score, 4)
                entry["rejected_reason"] = "protect_region_veto"
                entry["region_veto"] = getattr(veto, "regions", [])
                entry["constraint_score"] = getattr(veto, "constraint_score", None)
                entry["accepted"] = False
                entry["best_score_after"] = round(best_score, 4)
                extra_feedback = [
                    "[PROTECT VIOLATION] Your last revision degraded a protected "
                    f"region ({getattr(veto, 'reason', None) or 'protected region'}). "
                    "You MUST keep those regions unchanged; revise elsewhere."
                ]
                no_improvement_count += 1
                _record(entry)
                if no_improvement_count >= no_improvement_patience:
                    stop_reason = "no_improvement_patience"
                    break
                continue

        entry["score_after"] = round(new_score, 4)
        delta = new_score - best_score
        entry["delta"] = round(delta, 4)
        entry["improved"] = delta > 0.0
        entry["meaningful_improvement"] = delta >= min_improvement
        entry["changes_summary"] = _diff_dsl_summary(best_dsl, revised)
        logger.info(
            "refinement iter=%d before=%.4f after=%.4f delta=%+.4f improved=%s changes=%s",
            i + 1,
            float(entry["score_before"]),
            float(new_score),
            float(delta),
            bool(entry["improved"]),
            (entry["changes_summary"] or "")[:120],
        )
        log_event(
            logger,
            "dsl_refinement_iteration_scored",
            iteration=i + 1,
            score_before=float(entry["score_before"]),
            score_after=float(new_score),
            delta=float(delta),
            improved=bool(entry["improved"]),
            meaningful_improvement=bool(entry["meaningful_improvement"]),
            changes_summary=entry["changes_summary"],
        )

        # Arbitrate noise-level gains: objective metrics can't tell 0.005
        # improvement from rendering noise — let the judge veto.
        if (
            pairwise_judge is not None
            and 0.0 < delta < min_improvement
            and current_render_path is not None
            and render_path.exists()
        ):
            verdict = pairwise_judge(current_render_path, render_path)
            if verdict == "A":  # judge prefers the previous best
                entry["vlm_override"] = "veto_small_gain"
                delta = 0.0
                entry["improved"] = False

        accept = delta > 0.0

        # Directed acceptance (human-in-loop): allow a small score drop when the
        # goal-aware VLM judge prefers the candidate (B) for the user's goal.
        directed = directed_acceptance or {}
        score_drop_tolerance = float(directed.get("score_drop_tolerance", 0.0))
        if (
            not accept
            and directed_pairwise_judge is not None
            and current_render_path is not None
            and render_path.exists()
            and -score_drop_tolerance <= delta <= 0.0
        ):
            verdict = directed_pairwise_judge(current_render_path, render_path)
            if verdict == "B":
                accept = True
                entry["human_goal_override"] = "accepted_score_drop"

        if accept:
            best_dsl = revised
            best_score = new_score
            best_metrics = new_metrics
            best_quality = new_quality
            best_glsl = cr.glsl
            if render_path.exists():
                current_render_path = render_path
            extra_feedback = []
        else:
            rollback_note = (
                f"[ROLLBACK] Your last revision dropped the score from "
                f"{best_score:.3f} to {new_score:.3f}. The system reverted to "
                f"the previous best version. Changes were: {entry['changes_summary']}. "
                f"Do NOT repeat the same approach. Try a different strategy."
            )
            extra_feedback = [rollback_note]

        entry["accepted"] = bool(accept)
        entry["best_score_after"] = round(best_score, 4)

        if delta >= min_improvement:
            no_improvement_count = 0
        elif entry.get("human_goal_override"):
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        _record(entry)

        if no_improvement_count >= no_improvement_patience:
            stop_reason = "no_improvement_patience"
            break
    else:
        stop_reason = "max_iterations"

    logger.info(
        "refinement done: stop_reason=%s best_score=%.4f iters=%d",
        stop_reason,
        float(best_score),
        len(history),
    )
    log_event(
        logger,
        "dsl_refinement_done",
        stop_reason=stop_reason,
        best_score=float(best_score),
        iterations=len(history),
    )
    return {
        "best_dsl": best_dsl,
        "best_glsl": best_glsl,
        "best_score": best_score,
        "best_metrics": best_metrics,
        "best_quality": best_quality,
        "history": history,
        "stop_reason": stop_reason,
    }


def _should_run_refinement(
    refinement_mode: str,
    selected: CandidateRecord | None,
    selected_quality: dict | None,
    *,
    threshold: float,
    high_score_stop: float,
    force_first: bool = False,
) -> tuple[bool, str]:
    """Decide whether to enter the LLM refinement loop (DSL or GLSL).

    ``force_first`` (human-in-loop directed branch) forces at least one
    iteration even when the checkpoint already scores above the early-stop
    thresholds.
    """
    if refinement_mode == "off":
        return False, "refinement_mode_off"
    if selected is None:
        return False, "no_selected_candidate"
    is_dsl = selected.dsl is not None
    is_glsl = (
        not is_dsl
        and selected.output_kind == "glsl"
        and selected.compile_success
        and bool(selected.compile_glsl)
    )
    if not is_dsl and not is_glsl:
        return False, "selected_candidate_not_refinable"
    if selected_quality is None:
        return False, "missing_quality_router"

    score = float(selected.final_score)
    if refinement_mode == "auto":
        if score < threshold:
            return True, "auto_below_threshold"
        if force_first:
            return True, "human_forced_first_iteration"
        return False, "auto_threshold_reached"

    if refinement_mode == "on":
        if score >= high_score_stop and not force_first:
            return False, "force_high_score_stop"
        return True, "force_enabled"

    return False, "invalid_refinement_mode"


def _build_revision_patch(
    dsl: dict,
    preprocess: dict,
    failure_type: str,
    *,
    protected_aspects: list[str] | None = None,
) -> RevisionPatch | None:
    if protected_aspects is None:
        protected_aspects = ["layer_count", "primitive_types", "background"]
    layers = dsl.get("layers", []) if isinstance(dsl, dict) else []
    if not layers:
        return None

    alpha_coverage = float(preprocess.get("alpha_coverage", 1.0))
    palette: list[str] = preprocess.get("palette", [])
    ops: list[PatchOp] = []
    expected: list[str] = []

    if failure_type == "color" and palette:
        top_color = palette[0]
        second_color = palette[1] if len(palette) > 1 else top_color
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            layer_id = layer.get("id")
            if not layer_id:
                continue
            current_fill = layer.get("fill") or {}
            fill_type = current_fill.get("type", "solid")
            if fill_type == "solid":
                new_fill = {"type": "solid", "color": top_color}
            elif fill_type in ("linearGradient", "radialGradient"):
                stops = current_fill.get("stops", [])
                if len(stops) >= 2:
                    new_stops = list(stops)
                    new_stops[0] = {"color": top_color, "position": float(stops[0].get("position", 0.0))}
                    new_stops[-1] = {"color": second_color, "position": float(stops[-1].get("position", 1.0))}
                else:
                    new_stops = [
                        {"color": top_color, "position": 0.0},
                        {"color": second_color, "position": 1.0},
                    ]
                new_fill = dict(current_fill)
                new_fill["stops"] = new_stops
            else:
                new_fill = {"type": "solid", "color": top_color}
            ops.append(PatchOp(
                op="update_layer_material",
                layer_id=layer_id,
                params={"fill": new_fill},
            ))
        expected.append("update layer colors to match reference palette")
    else:
        for layer in layers:
            if not isinstance(layer, dict):
                continue
            layer_id = layer.get("id")
            if not layer_id:
                continue

            primitive = layer.get("type")
            updates: dict = {"center": [0.5, 0.5]}
            if primitive in ("circle", "ring", "polygon"):
                radius = math.sqrt(max(0.01, min(1.0, alpha_coverage)) / math.pi)
                updates["radius"] = round(max(0.12, min(0.5, radius)), 4)
            elif primitive in ("box", "roundedBox"):
                size = max(0.25, min(0.9, math.sqrt(max(0.01, min(1.0, alpha_coverage)))))
                updates["size"] = [round(size, 4), round(max(0.18, min(0.9, size * 0.75)), 4)]
            else:
                continue

            ops.append(PatchOp(op="update_layer_params", layer_id=layer_id, params={"updates": updates}))
        expected.extend(["recenter primary layer", "resize primary primitive to source coverage"])

    if not ops:
        return None

    valid_failure_types = {"structure", "parameter", "color", "layer_order", "budget", "unsupported"}
    return RevisionPatch(
        revision_type="parameter",
        failure_type=failure_type if failure_type in valid_failure_types else "parameter",
        ops=ops,
        protected_aspects=list(protected_aspects),
        expected_improvement=expected,
    )
