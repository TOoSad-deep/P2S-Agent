"""LLM-driven GLSL refinement loop for PNG-to-Shader."""

from __future__ import annotations

import difflib
import logging
import time
from pathlib import Path
from typing import Callable

from app.pipeline.artifacts import save_json
from app.pipeline.refinement import build_recent_history_notes, build_semantic_notes
from app.services.shader_validator import validate_shader_static

logger = logging.getLogger(__name__)


def _short_exception(exc: Exception) -> str:
    message = str(exc).strip()
    if len(message) > 500:
        message = message[:497] + "..."
    return f"{exc.__class__.__name__}: {message}" if message else exc.__class__.__name__


def _diff_glsl_summary(old_glsl: str, new_glsl: str) -> str:
    """Return a compact summary of shader line changes."""
    diff = difflib.unified_diff(
        old_glsl.splitlines(), new_glsl.splitlines(), lineterm="", n=0
    )
    changed = [
        line for line in diff
        if line[:1] in "+-" and not line.startswith(("+++", "---"))
    ]
    if not changed:
        return "no changes"
    define_changes = [line.strip() for line in changed if "#define" in line]
    summary = f"{len(changed)} changed lines"
    if define_changes:
        summary += "; " + "; ".join(define_changes[:4])
    return summary


def _build_failure_log_note(history: list[dict], best_score: float) -> str:
    recent_scores = [
        h.get("score_after")
        for h in history[-3:]
        if h.get("score_after") is not None
    ]
    failed_directions = [
        f"iter {h.get('iteration')}: {h.get('changes_summary') or h.get('error') or 'n/a'}"
        for h in history[-3:]
    ]
    return (
        "[FRESH RESTART] Incremental revision has stalled "
        f"(recent scores: {recent_scores}, best so far: {best_score:.3f}). "
        "Discard the current implementation approach and write a NEW shader "
        "from scratch for the same reference image using a different technique. "
        "Avoid repeating these failed directions: "
        + "; ".join(failed_directions[:3])
    )


def run_glsl_refinement_loop(
    initial_glsl: str,
    initial_score: float,
    initial_metrics: dict,
    initial_quality: dict,
    reference_path: Path,
    *,
    evaluate_fn: "Callable[[str, Path], tuple[dict, dict, float, Path | None]]",
    initial_render_path: "Path | None" = None,
    max_iterations: int = 3,
    threshold: float = 0.80,
    high_score_stop: float = 0.92,
    min_improvement: float = 0.01,
    no_improvement_patience: int = 2,
    max_fresh_restarts: int = 1,
    force_first_iteration: bool = False,
    loop_dir: Path,
    strategy_reader: "Callable[[], dict] | None" = None,
    pairwise_judge: "Callable[[Path, Path], str | None] | None" = None,
    rubric_judge: "Callable[[Path], dict | None] | None" = None,
) -> dict:
    """Drive iterative LLM revisions for a raw Shadertoy GLSL candidate."""
    from app.candidates.llm_scene import generate_llm_glsl_refinement

    loop_dir.mkdir(parents=True, exist_ok=True)
    render_dir = loop_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    best_glsl = initial_glsl
    best_score = initial_score
    best_metrics = dict(initial_metrics)
    best_quality = dict(initial_quality)
    best_render_path = initial_render_path
    current_render_path = initial_render_path

    history: list[dict] = []
    stop_reason = "max_iterations"
    no_improvement_count = 0
    extra_feedback: list[str] = []
    fresh_restarts_left = max_fresh_restarts
    pending_fresh_start = False
    stagnation_anchor = 0

    def _trigger_fresh_restart() -> bool:
        nonlocal fresh_restarts_left, pending_fresh_start
        nonlocal stagnation_anchor, no_improvement_count, extra_feedback
        if fresh_restarts_left <= 0:
            return False
        fresh_restarts_left -= 1
        pending_fresh_start = True
        stagnation_anchor = len(history)
        no_improvement_count = 0
        extra_feedback = [_build_failure_log_note(history, best_score)]
        logger.info("glsl refinement fresh restart triggered")
        return True

    for i in range(max_iterations):
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
                if i >= int(live_strategy["max_refinement_iterations"]):
                    stop_reason = "user_lowered_cap"
                    break

        if best_score >= high_score_stop:
            stop_reason = "high_score_stop"
            break
        if best_score >= threshold and not (force_first_iteration and not history):
            stop_reason = "threshold_reached"
            break

        scored_entries = [
            h["score_after"]
            for h in history[stagnation_anchor:]
            if h.get("score_after") is not None
        ]
        if len(scored_entries) >= 3:
            recent = scored_entries[-3:]
            if max(recent) - min(recent) < 0.02:
                if not _trigger_fresh_restart():
                    stop_reason = "stagnation"
                    break

        was_fresh = pending_fresh_start
        pending_fresh_start = False

        entry: dict = {
            "iteration": i + 1,
            "score_before": round(best_score, 4),
            "score_after": None,
            "delta": None,
            "improved": False,
            "meaningful_improvement": False,
            "fresh_start": was_fresh,
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
            revised = generate_llm_glsl_refinement(
                current_glsl=best_glsl,
                metrics=best_metrics,
                quality_router=best_quality,
                reference_image_path=reference_path,
                current_render_path=current_render_path,
                extra_feedback=(
                    extra_feedback + history_notes + semantic_notes + region_notes
                ) or None,
                fresh_start=was_fresh,
            )
        except Exception as exc:
            entry["llm_duration_ms"] = int((time.monotonic() - llm_start) * 1000)
            entry["error_type"] = exc.__class__.__name__
            entry["error"] = f"LLM call failed: {_short_exception(exc)}"
            history.append(entry)
            save_json(loop_dir / f"iter_{i + 1}.json", entry)
            stop_reason = "llm_call_failed"
            break

        entry["llm_duration_ms"] = int((time.monotonic() - llm_start) * 1000)
        if revised and isinstance(revised, dict):
            entry["llm_io"] = revised.pop("_io", None)

        revised_glsl = revised.get("glsl") if isinstance(revised, dict) else None
        if not revised_glsl:
            entry["error_type"] = "llm_returned_none"
            entry["error"] = (
                "LLM returned no usable GLSL: response content was empty, was "
                "not valid JSON, or did not contain a mainImage entry point."
            )
            history.append(entry)
            save_json(loop_dir / f"iter_{i + 1}.json", entry)
            stop_reason = "llm_returned_none"
            break

        static = validate_shader_static(revised_glsl)
        if not static["valid"]:
            entry["error"] = f"GLSL invalid: {static['errors'][:2]}"
            history.append(entry)
            save_json(loop_dir / f"iter_{i + 1}.json", entry)
            no_improvement_count += 1
            extra_feedback = [
                "[COMPILE FEEDBACK] Your last revision failed static validation: "
                + "; ".join(static["errors"][:3])
                + ". Fix these issues and return the full corrected shader."
            ]
            if no_improvement_count >= no_improvement_patience and not _trigger_fresh_restart():
                stop_reason = "no_improvement_patience"
                break
            continue

        entry["compile_glsl"] = revised_glsl
        render_path = render_dir / f"iter_{i + 1}.png"
        new_metrics, new_quality, new_score, actual_render = evaluate_fn(
            revised_glsl, render_path
        )

        if actual_render is None and new_score <= 0.0:
            entry["error_type"] = "render_failed"
            entry["error"] = (
                "render failed: WebGL produced no screenshot "
                "(compile or runtime error)"
            )
            entry["score_after"] = 0.0
            history.append(entry)
            save_json(loop_dir / f"iter_{i + 1}.json", entry)
            no_improvement_count += 1
            extra_feedback = [
                "[RENDER FAILED] The revised shader passed static checks but "
                "failed to render in WebGL. Fix undefined symbols, int/float "
                "mismatches, loop bounds, and runtime errors while preserving "
                "the visual intent."
            ]
            if no_improvement_count >= no_improvement_patience and not _trigger_fresh_restart():
                stop_reason = "no_improvement_patience"
                break
            continue

        entry["score_after"] = round(new_score, 4)
        delta = new_score - best_score
        entry["delta"] = round(delta, 4)
        entry["improved"] = delta > 0.0
        entry["meaningful_improvement"] = delta >= min_improvement
        entry["changes_summary"] = _diff_glsl_summary(best_glsl, revised_glsl)

        if (
            pairwise_judge is not None
            and 0.0 < delta < min_improvement
            and current_render_path is not None
            and actual_render is not None
        ):
            verdict = pairwise_judge(current_render_path, actual_render)
            if verdict == "A":
                entry["vlm_override"] = "veto_small_gain"
                delta = 0.0
                entry["improved"] = False

        if delta > 0.0:
            best_glsl = revised_glsl
            best_score = new_score
            best_metrics = new_metrics
            best_quality = new_quality
            if actual_render is not None:
                current_render_path = actual_render
                best_render_path = actual_render
            extra_feedback = []
        else:
            extra_feedback = [
                f"[ROLLBACK] Your last revision dropped the score from "
                f"{best_score:.3f} to {new_score:.3f}. The system reverted to "
                f"the previous best version. Changes were: {entry['changes_summary']}. "
                f"Do NOT repeat the same approach. Try a different strategy."
            ]

        if delta >= min_improvement:
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        history.append(entry)
        save_json(loop_dir / f"iter_{i + 1}.json", entry)

        if no_improvement_count >= no_improvement_patience and not _trigger_fresh_restart():
            stop_reason = "no_improvement_patience"
            break

    logger.info(
        "glsl refinement done: stop_reason=%s best_score=%.4f iters=%d",
        stop_reason,
        float(best_score),
        len(history),
    )
    return {
        "best_glsl": best_glsl,
        "best_score": best_score,
        "best_metrics": best_metrics,
        "best_quality": best_quality,
        "best_render_path": str(best_render_path) if best_render_path else None,
        "history": history,
        "stop_reason": stop_reason,
    }
