"""P2S-Agent Pipeline Orchestrator

Uses LangGraph StateGraph for the core pipeline flow:
  preprocess -> candidates -> scoring -> selection

Optimization, revision, and refinement stages run as post-pipeline functions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from langgraph.graph import END, StateGraph

from app.candidates.llm_scene import Implementation
from app.dsl.compiler import compile_dsl
from app.pipeline.artifacts import copy_artifact, create_run_dir, save_json, write_manifest
from app.pipeline.glsl_optimizer import (
    build_glsl_optimization_artifacts,
    optimize_glsl_candidate,
)
from app.pipeline.input_spec import build_input_spec
from app.pipeline.optimizer import build_optimization_artifacts, optimize_candidate
from app.pipeline.pool import (
    CandidateRecord,
    _candidate_detail,
    build_scoreboard,
    run_candidate_pool,
    select_best_candidate,
)
from app.pipeline.preprocess import preprocess_image, save_preprocess_artifacts
from app.pipeline.refinement import (
    _build_revision_patch,
    _should_run_refinement,
    run_dsl_refinement_loop,
)
from app.pipeline.revision import apply_revision_with_rollback, build_revision_log_entry
from app.pipeline.residual_layers import add_residual_layers
from app.llm.vlm_judge import judge_pairwise, judge_rubric
from app.metrics.quality_router import compute_final_score
from app.pipeline.scoring import (
    _accept_improvement,
    _evaluate_glsl_with_webgl,
    _make_render_dsl_fn,
    _make_render_glsl_fn,
    _make_revision_scorer,
    _score_candidates,
    _sync_selected_record_for_response,
)
from app.state import P2SPipelineState
from app.strategy_config_loader import clamp as strategy_clamp, get_default

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def node_preprocess(state: P2SPipelineState) -> dict:
    """Preprocess the input image."""
    logger.info("node_preprocess: image=%s", state.get("image_path"))

    image_path = Path(state["image_path"])
    run_dir = Path(state["run_dir"])

    preprocess = preprocess_image(image_path)
    save_preprocess_artifacts(preprocess, run_dir, image_path)
    llm_image_path = run_dir / "llm_reference_input.png"
    preprocess["llm_reference_background"] = "#000000"

    logger.info(
        "preprocess done: alpha_coverage=%.3f colors=%d edge_sharpness=%.3f",
        float(preprocess.get("alpha_coverage", 0.0)),
        int(preprocess.get("color_count_estimate", 0)),
        float(preprocess.get("edge_sharpness", 0.0)),
    )

    return {
        "preprocess": preprocess,
        "llm_image_path": str(llm_image_path),
        "progress": "preprocessing",
    }


def node_candidates(state: P2SPipelineState) -> dict:
    """Run the candidate pool to generate candidates."""
    logger.info("node_candidates: generating candidates")

    preprocess = state["preprocess"]
    input_spec = state["input_spec"]
    image_path = Path(state["image_path"])
    llm_image_path = Path(state.get("llm_image_path") or image_path)

    candidates = run_candidate_pool(
        preprocess,
        input_spec,
        image_path=image_path,
        llm_image_path=llm_image_path,
        llm_enabled=state.get("llm_enabled", False),
        llm_implementation=state.get("llm_implementation", "auto"),
        cv_enabled=state.get("cv_enabled", True),
        canvas_width=state.get("canvas_width", 512),
        canvas_height=state.get("canvas_height", 512),
    )

    return {"candidates": candidates, "progress": "candidates"}


def node_scoring(state: P2SPipelineState) -> dict:
    """Score all candidates."""
    logger.info("node_scoring: scoring candidates")

    candidates = state["candidates"]
    run_dir = Path(state["run_dir"])
    reference_path = run_dir / "reference_input.png"

    _score_candidates(
        candidates,
        reference_path,
        run_dir / "candidates",
        canvas_width=state.get("canvas_width", 512),
        canvas_height=state.get("canvas_height", 512),
        max_shader_chars=state.get("max_shader_chars", 12000),
        glsl_render_enabled=state.get("glsl_render_enabled", False),
        protected_aspects=state.get("protected_aspects", ["layer_count", "primitive_types", "background"]),
    )

    return {"scored": True, "progress": "scoring"}


def node_selection(state: P2SPipelineState) -> dict:
    """Select the best candidate."""
    logger.info("node_selection: selecting best candidate")

    candidates = state["candidates"]

    # Determine preference for GLSL output
    prefer_output_kind = None
    refinement_requested = (
        state.get("refinement_mode", "auto") != "off"
        and state.get("max_refinement_iterations", 0) > 0
    )

    if state.get("glsl_render_enabled", False) and not refinement_requested:
        best_glsl_score = max(
            (
                c.final_score
                for c in candidates
                if c.source == "llm"
                and c.output_kind == "glsl"
                and c.compile_success
                and c.final_score > 0
            ),
            default=0.0,
        )
        best_dsl_score = max(
            (
                c.final_score
                for c in candidates
                if c.dsl is not None and c.compile_success
            ),
            default=0.0,
        )
        if best_glsl_score > 0 and best_glsl_score >= best_dsl_score:
            prefer_output_kind = "glsl"

    # VLM near-tie arbitration
    if state.get("vlm_judge_enabled"):
        run_dir = Path(state["run_dir"])
        reference_path = run_dir / "reference_input.png"
        ranked = sorted(
            [c for c in candidates if c.compile_success and c.render_path],
            key=lambda c: -c.final_score,
        )
        if (
            len(ranked) >= 2
            and (ranked[0].final_score - ranked[1].final_score)
            < float(state.get("vlm_tie_epsilon", 0.05))
        ):
            verdict = judge_pairwise(
                reference_path, ranked[0].render_path, ranked[1].render_path,
                work_dir=run_dir / "judge",
            )
            logger.info(
                "vlm near-tie arbitration: %s vs %s -> %s",
                ranked[0].id, ranked[1].id, verdict,
            )
            if verdict == "B":
                bump = ranked[0].final_score - ranked[1].final_score + 0.001
                ranked[1].final_score += bump
                ranked[1].reason.append(f"vlm pairwise judge won near-tie (+{bump:.4f})")
            elif verdict == "A":
                ranked[0].reason.append("vlm pairwise judge confirmed near-tie winner")

    selected = select_best_candidate(candidates, prefer_output_kind=prefer_output_kind)

    return {
        "selected_candidate_id": selected.id if selected else None,
        "selected_dsl": selected.dsl if selected else None,
        "selected_glsl": selected.compile_glsl if selected else None,
        "selected_metrics": dict(selected.objective_metrics) if selected else {},
        "selected_quality": dict(selected.quality_router) if selected and selected.quality_router else None,
        "progress": "selecting",
    }


# ---------------------------------------------------------------------------
# Post-pipeline processing (not a LangGraph node)
# ---------------------------------------------------------------------------

def _run_post_pipeline(
    state: P2SPipelineState,
    *,
    strategy_reader: Callable[[], dict] | None = None,
) -> P2SPipelineState:
    """Run optimization, revision, and refinement after selection.

    This is called after the LangGraph pipeline completes.
    """
    from app.pipeline.scoring import _accept_improvement  # avoid circular import

    selected_dsl = state.get("selected_dsl")
    selected_glsl = state.get("selected_glsl")
    selected_metrics = dict(state.get("selected_metrics", {}))
    selected_quality = dict(state.get("selected_quality", {})) if state.get("selected_quality") else None
    candidates = state.get("candidates", [])

    # Find the selected candidate record
    selected = None
    for c in candidates:
        if c.id == state.get("selected_candidate_id"):
            selected = c
            break

    if selected is None:
        return state

    run_dir = Path(state["run_dir"])
    reference_path = run_dir / "reference_input.png"
    canvas_width = state.get("canvas_width", 512)
    canvas_height = state.get("canvas_height", 512)
    max_shader_chars = state.get("max_shader_chars", 12000)
    optimizer_iterations = state.get("optimizer_iterations", 0)
    protected_aspects = state.get("protected_aspects", ["layer_count", "primitive_types", "background"])

    optimization_summary = None
    revision_summary = None
    refinement_summary = {
        "mode": state.get("refinement_mode", "auto"),
        "enabled": False,
        "decision": "not_evaluated",
        "iterations": 0,
        "initial_score": selected.final_score,
        "final_score": selected.final_score,
        "stop_reason": None,
        "threshold": state.get("refinement_threshold", 0.5),
        "high_score_stop": state.get("refinement_high_score_stop", 0.95),
        "min_improvement": state.get("refinement_min_improvement", 0.01),
        "patience": state.get("refinement_patience", 2),
    }
    refinement_history: list = []

    # Optimization and revision
    if selected.dsl and selected_quality:
        next_action = selected_quality.get("next_action")
        if next_action in {"optimize", "revise"} and optimizer_iterations > 0:
            opt_dir = run_dir / "optimization"
            render_dsl_fn = _make_render_dsl_fn(
                opt_dir / "renders",
                canvas_width=canvas_width,
                canvas_height=canvas_height,
            )
            opt_result = optimize_candidate(
                selected.dsl,
                reference_path,
                render_dsl_fn=render_dsl_fn,
                max_iterations=optimizer_iterations,
                strategy="coordinate_descent",
                seed=0,
            )
            optimization_summary = build_optimization_artifacts(opt_result)
            save_json(opt_dir / "optimizer.json", optimization_summary)

            if opt_result.improved:
                accepted = _accept_improvement(
                    selected,
                    opt_result.best_dsl,
                    reference_path,
                    opt_dir / "optimized_render.png",
                    canvas_width=canvas_width,
                    canvas_height=canvas_height,
                    max_shader_chars=max_shader_chars,
                    protected_aspects=protected_aspects,
                    reason=f"optimization improved score {selected.final_score:.4f} -> {opt_result.best_score:.4f}",
                )
                if accepted is not None:
                    selected_dsl, selected_glsl, selected_metrics, selected_quality = accepted

        next_action = selected_quality.get("next_action") if selected_quality else None
        if next_action in {"revise", "fallback"} and selected.dsl:
            effective_failure_type = selected_quality.get("failure_type", "parameter") if selected_quality else "parameter"
            force_failure_type = state.get("force_failure_type")
            if force_failure_type:
                effective_failure_type = force_failure_type
            patch = _build_revision_patch(
                selected.dsl,
                state.get("preprocess", {}),
                effective_failure_type,
                protected_aspects=protected_aspects,
            )
            if patch is not None:
                rev_dir = run_dir / "revision"
                rev_dir.mkdir(parents=True, exist_ok=True)

                _revision_score = _make_revision_scorer(
                    rev_dir,
                    canvas_width=canvas_width,
                    canvas_height=canvas_height,
                    max_shader_chars=max_shader_chars,
                    protected_aspects=protected_aspects,
                )

                def score_fn(candidate_dsl: dict) -> float:
                    return _revision_score(candidate_dsl, reference_path)

                rev_result = apply_revision_with_rollback(selected.dsl, patch, score_fn)
                revision_summary = build_revision_log_entry(patch, rev_result)
                save_json(rev_dir / "revision.json", revision_summary)

                if rev_result.success and rev_result.improved:
                    accepted = _accept_improvement(
                        selected,
                        rev_result.final_dsl,
                        reference_path,
                        rev_dir / "revised_render.png",
                        canvas_width=canvas_width,
                        canvas_height=canvas_height,
                        max_shader_chars=max_shader_chars,
                        protected_aspects=protected_aspects,
                        reason=f"revision improved score {selected.final_score:.4f} -> {rev_result.best_score:.4f}",
                    )
                    if accepted is not None:
                        selected_dsl, selected_glsl, selected_metrics, selected_quality = accepted

        # Residual-driven layer addition: construct what optimization can't fix.
        max_added_layers_val = int(state.get("max_added_layers", 0))
        if max_added_layers_val > 0 and selected.final_score < float(
            state.get("refinement_high_score_stop", 0.95)
        ):
            res_dir = run_dir / "residual_layers"
            res_render_fn = _make_render_dsl_fn(
                res_dir / "renders",
                canvas_width=canvas_width,
                canvas_height=canvas_height,
            )
            _res_score = _make_revision_scorer(
                res_dir / "scores",
                canvas_width=canvas_width,
                canvas_height=canvas_height,
                max_shader_chars=max_shader_chars,
                protected_aspects=protected_aspects,
            )
            try:
                res_result = add_residual_layers(
                    selected.dsl,
                    reference_path,
                    score_fn=lambda d: _res_score(d, reference_path),
                    render_fn=lambda d: res_render_fn(d, ""),
                    max_added=max_added_layers_val,
                )
            except Exception:
                logger.exception("residual layer addition failed")
                res_result = None

            if res_result is not None:
                save_json(res_dir / "residual.json", {
                    "initial_score": res_result.initial_score,
                    "final_score": res_result.final_score,
                    "layers_added": res_result.layers_added,
                    "log": res_result.log,
                })
                if res_result.layers_added > 0 and res_result.final_score > selected.final_score:
                    accepted = _accept_improvement(
                        selected,
                        res_result.final_dsl,
                        reference_path,
                        res_dir / "residual_render.png",
                        canvas_width=canvas_width,
                        canvas_height=canvas_height,
                        max_shader_chars=max_shader_chars,
                        protected_aspects=protected_aspects,
                        reason=(
                            f"residual layers (+{res_result.layers_added}) improved score "
                            f"{res_result.initial_score:.4f} -> {res_result.final_score:.4f}"
                        ),
                    )
                    if accepted is not None:
                        selected_dsl, selected_glsl, selected_metrics, selected_quality = accepted

    # GLSL optimizer (for GLSL candidates)
    elif selected and selected.output_kind == "glsl" and selected.compile_glsl and selected_quality:
        next_action = selected_quality.get("next_action")
        if next_action in {"optimize", "revise"} and optimizer_iterations > 0:
            opt_dir = run_dir / "glsl_optimization"
            opt_dir.mkdir(parents=True, exist_ok=True)
            render_glsl_fn = _make_render_glsl_fn(
                opt_dir / "renders",
                canvas_width=canvas_width,
                canvas_height=canvas_height,
            )
            try:
                glsl_opt_result = optimize_glsl_candidate(
                    selected.compile_glsl,
                    reference_path,
                    render_glsl_fn,
                    max_iterations=optimizer_iterations,
                    max_shader_chars=max_shader_chars,
                    seed=0,
                )
            except Exception:
                logger.exception("glsl optimizer failed")
                glsl_opt_result = None

            if glsl_opt_result is not None:
                optimization_summary = build_glsl_optimization_artifacts(glsl_opt_result)
                save_json(opt_dir / "optimizer.json", optimization_summary)

                if glsl_opt_result.improved:
                    try:
                        metrics, quality, score, render_path = _evaluate_glsl_with_webgl(
                            glsl_opt_result.best_glsl,
                            reference_path,
                            opt_dir / "optimized_render.png",
                            canvas_width=canvas_width,
                            canvas_height=canvas_height,
                            max_shader_chars=max_shader_chars,
                        )
                    except Exception:
                        logger.exception("glsl optimizer re-evaluation failed")
                    else:
                        selected.compile_glsl = glsl_opt_result.best_glsl
                        selected.objective_metrics = metrics
                        selected.quality_router = quality
                        selected.final_score = score
                        selected_glsl = selected.compile_glsl
                        selected_metrics = metrics
                        selected_quality = quality

    # LLM refinement
    effective_llm_enabled = state.get("llm_enabled", False)
    effective_refinement_mode = state.get("refinement_mode", "auto")
    max_refinement_iterations = state.get("max_refinement_iterations", 0)
    refinement_threshold = state.get("refinement_threshold", 0.5)
    refinement_high_score_stop = state.get("refinement_high_score_stop", 0.95)
    refinement_min_improvement = state.get("refinement_min_improvement", 0.01)
    refinement_patience = state.get("refinement_patience", 2)

    should_refine, refinement_decision = _should_run_refinement(
        effective_refinement_mode,
        selected,
        selected_quality,
        threshold=refinement_threshold,
        high_score_stop=refinement_high_score_stop,
    )
    if max_refinement_iterations <= 0:
        should_refine = False
        refinement_decision = "max_refinement_iterations_zero"
    elif effective_refinement_mode == "auto" and not effective_llm_enabled:
        should_refine = False
        refinement_decision = "auto_llm_disabled"
    refinement_summary["enabled"] = should_refine
    refinement_summary["decision"] = refinement_decision

    if should_refine and selected and selected.dsl:
        initial_refinement_score = selected.final_score
        ref_result = run_dsl_refinement_loop(
            preprocess=state.get("preprocess", {}),
            initial_dsl=selected.dsl,
            initial_score=selected.final_score,
            initial_metrics=dict(selected.objective_metrics),
            initial_quality=dict(selected.quality_router) if selected.quality_router else {},
            reference_path=reference_path,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            max_shader_chars=max_shader_chars,
            max_iterations=max_refinement_iterations,
            threshold=refinement_threshold,
            high_score_stop=refinement_high_score_stop,
            min_improvement=refinement_min_improvement,
            no_improvement_patience=refinement_patience,
            force_first_iteration=effective_refinement_mode == "on",
            loop_dir=run_dir / "refinement",
            strategy_reader=strategy_reader,
            protected_aspects=protected_aspects,
            pairwise_judge=(
                (lambda cur, new: judge_pairwise(
                    reference_path, cur, new, work_dir=run_dir / "judge"
                ))
                if state.get("vlm_judge_enabled") else None
            ),
        )
        refinement_history = ref_result.get("history", [])
        refinement_summary.update({
            "iterations": len(refinement_history),
            "initial_score": initial_refinement_score,
            "final_score": ref_result.get("best_score", initial_refinement_score),
            "improved": ref_result.get("best_score", 0) > initial_refinement_score,
            "stop_reason": ref_result.get("stop_reason"),
        })

        if ref_result.get("best_score", 0) > selected.final_score:
            refined_dsl = ref_result["best_dsl"]
            refined_compile = compile_dsl(refined_dsl) if isinstance(refined_dsl, dict) else None
            selected.dsl = refined_dsl
            if refined_compile is not None and refined_compile.glsl.strip():
                selected.compile_glsl = refined_compile.glsl
                selected.compile_success = refined_compile.success
                selected.compile_errors = list(refined_compile.errors)
            selected.objective_metrics = ref_result["best_metrics"]
            selected.quality_router = ref_result["best_quality"]
            selected.final_score = ref_result["best_score"]
            selected_dsl = selected.dsl
            selected_glsl = selected.compile_glsl
            selected_metrics = ref_result["best_metrics"]
            selected_quality = ref_result["best_quality"]

    # VLM final gate
    judge_summary = None
    if state.get("vlm_judge_enabled") and selected is not None and selected.render_path:
        rubric = judge_rubric(
            reference_path, selected.render_path, work_dir=run_dir / "judge"
        )
        if rubric is not None:
            blended = compute_final_score(selected_metrics, rubric["semantic_scores"])
            judge_summary = {
                **rubric,
                "objective_score": float(selected.final_score),
                "blended_score": blended,
            }
            logger.info(
                "vlm final gate: objective=%.4f blended=%.4f failure_type=%s",
                float(selected.final_score), blended, rubric["failure_type"],
            )
            if selected_quality is not None:
                selected_quality = {
                    **selected_quality,
                    "final_score": blended,
                    "semantic_scores": rubric["semantic_scores"],
                    "vlm_failure_type": rubric["failure_type"],
                }
            selected.final_score = blended
            save_json(run_dir / "judge" / "final_rubric.json", judge_summary)

    # Sync selected record
    _sync_selected_record_for_response(
        selected,
        selected_dsl=selected_dsl,
        selected_glsl=selected_glsl,
        selected_metrics=selected_metrics,
        selected_quality=selected_quality,
    )

    # Build scoreboard and save artifacts
    scoreboard = build_scoreboard(candidates)
    candidate_details = [_candidate_detail(c) for c in candidates]
    save_json(run_dir / "candidates.json", candidate_details)
    save_json(run_dir / "scoreboard.json", scoreboard)
    save_json(run_dir / "objective_metrics.json", selected_metrics)
    save_json(run_dir / "quality_router.json", selected_quality or {})
    save_json(run_dir / "refinement_summary.json", refinement_summary)
    if selected_dsl is not None:
        save_json(run_dir / "selected_dsl.json", selected_dsl)
    if selected_glsl is not None:
        (run_dir / "selected_shader.glsl").write_text(selected_glsl, encoding="utf-8")

    return {
        **state,
        "optimization": optimization_summary,
        "revision": revision_summary,
        "refinement_summary": refinement_summary,
        "refinement_history": refinement_history,
        "scoreboard": scoreboard,
        "candidate_details": candidate_details,
        "selected_dsl": selected_dsl,
        "selected_glsl": selected_glsl,
        "selected_metrics": selected_metrics,
        "selected_quality": selected_quality or {},
        "vlm_judge": judge_summary,
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_graph() -> StateGraph:
    """Build the LangGraph StateGraph for the core pipeline."""
    graph = StateGraph(P2SPipelineState)

    # Add nodes (use _step suffix to avoid conflict with state keys)
    graph.add_node("preprocess_step", node_preprocess)
    graph.add_node("candidates_step", node_candidates)
    graph.add_node("scoring_step", node_scoring)
    graph.add_node("selection_step", node_selection)

    # Define edges
    graph.set_entry_point("preprocess_step")
    graph.add_edge("preprocess_step", "candidates_step")
    graph.add_edge("candidates_step", "scoring_step")
    graph.add_edge("scoring_step", "selection_step")
    graph.add_edge("selection_step", END)

    return graph


# Compile the graph
_pipeline_graph = _build_graph().compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_png_shader_pipeline(
    image_path: str | Path,
    input_spec: dict | None = None,
    run_id: str | None = None,
    *,
    llm_enabled: bool | None = None,
    llm_implementation: Implementation | None = None,
    progress_callback: Callable[[str], None] | None = None,
    strategy_reader: Callable[[], dict] | None = None,
) -> dict:
    """Run the full PNG-to-Shader pipeline and return structured results.

    The core pipeline (preprocess -> candidates -> scoring -> selection) runs
    through LangGraph. Post-pipeline stages (optimization, revision, refinement)
    run as synchronous function calls.
    """
    image_path = Path(image_path)
    effective_run_id = run_id or ("run_" + str(uuid4())[:8])
    run_dir_obj = create_run_dir(effective_run_id, "single")

    if input_spec is None:
        input_spec = build_input_spec(image_path)

    # Extract configuration from input_spec
    target = input_spec.get("target", {})
    resolution = target.get("resolution", [512, 512])
    canvas_width = int(resolution[0]) if len(resolution) >= 1 else 512
    canvas_height = int(resolution[1]) if len(resolution) >= 2 else 512
    max_shader_chars = int(target.get("max_shader_chars", 12000))
    quality_config = input_spec.get("quality", {})
    candidate_config = input_spec.get("candidates", {})

    effective_llm_enabled = (
        bool(candidate_config.get("llm_enabled", False)) if llm_enabled is None else llm_enabled
    )
    raw_llm_implementation = (
        candidate_config.get("llm_implementation", "auto")
        if llm_implementation is None
        else llm_implementation
    )
    effective_llm_implementation: Implementation = (
        raw_llm_implementation
        if raw_llm_implementation in {"auto", "png_dsl", "shadertoy_glsl"}
        else "auto"
    )
    effective_cv_enabled = bool(candidate_config.get("cv_enabled", True))
    requested_glsl_render_enabled = bool(candidate_config.get("glsl_render_enabled", False))
    auto_glsl_render_enabled = (
        effective_llm_enabled and effective_llm_implementation in {"auto", "shadertoy_glsl"}
    )
    effective_glsl_render_enabled = requested_glsl_render_enabled or auto_glsl_render_enabled

    optimizer_iterations = int(
        strategy_clamp("max_iterations", int(quality_config.get("max_iterations", get_default("max_iterations"))))
    )
    raw_refinement_mode = str(quality_config.get("refinement_mode", "auto"))
    effective_refinement_mode = (
        raw_refinement_mode if raw_refinement_mode in {"off", "auto", "on"} else "auto"
    )
    max_refinement_iterations = int(
        strategy_clamp(
            "max_refinement_iterations",
            int(quality_config.get("max_refinement_iterations", get_default("max_refinement_iterations"))),
        )
    )
    refinement_threshold = strategy_clamp(
        "refinement_threshold",
        float(quality_config.get("refinement_threshold", get_default("refinement_threshold"))),
    )
    refinement_high_score_stop = strategy_clamp(
        "refinement_high_score_stop",
        float(quality_config.get("refinement_high_score_stop", get_default("refinement_high_score_stop"))),
    )
    refinement_min_improvement = strategy_clamp(
        "refinement_min_improvement",
        float(quality_config.get("refinement_min_improvement", get_default("refinement_min_improvement"))),
    )
    refinement_patience = int(
        strategy_clamp(
            "refinement_patience",
            int(quality_config.get("refinement_patience", get_default("refinement_patience"))),
        )
    )
    max_added_layers = int(
        strategy_clamp(
            "max_added_layers",
            int(quality_config.get("max_added_layers", get_default("max_added_layers"))),
        )
    )
    vlm_judge_enabled = (
        bool(int(strategy_clamp(
            "vlm_judge_enabled",
            int(quality_config.get("vlm_judge_enabled", get_default("vlm_judge_enabled"))),
        )))
        and effective_llm_enabled
    )
    vlm_tie_epsilon = float(strategy_clamp(
        "vlm_tie_epsilon",
        float(quality_config.get("vlm_tie_epsilon", get_default("vlm_tie_epsilon"))),
    ))
    protected_aspects = quality_config.get(
        "protected_aspects", ["layer_count", "primitive_types", "background"]
    )
    if not isinstance(protected_aspects, list):
        protected_aspects = ["layer_count", "primitive_types", "background"]

    force_failure_type = quality_config.get("force_failure_type", None)

    # Write manifest
    write_manifest(
        run_dir_obj,
        input_spec,
        config={
            "llm_enabled": effective_llm_enabled,
            "llm_implementation": effective_llm_implementation,
            "cv_enabled": effective_cv_enabled,
            "glsl_render_enabled": effective_glsl_render_enabled,
            "canvas_width": canvas_width,
            "canvas_height": canvas_height,
            "max_shader_chars": max_shader_chars,
            "optimizer_iterations": optimizer_iterations,
            "refinement_mode": effective_refinement_mode,
            "max_refinement_iterations": max_refinement_iterations,
            "refinement_threshold": refinement_threshold,
            "refinement_high_score_stop": refinement_high_score_stop,
            "refinement_min_improvement": refinement_min_improvement,
            "refinement_patience": refinement_patience,
            "protected_aspects": list(protected_aspects),
            "quality_mode": quality_config.get("mode", "balanced"),
            "force_failure_type": force_failure_type,
            "max_added_layers": max_added_layers,
            "vlm_judge_enabled": vlm_judge_enabled,
            "vlm_tie_epsilon": vlm_tie_epsilon,
        },
    )
    copy_artifact(image_path, run_dir_obj.path / "reference_input.png")
    save_json(run_dir_obj.path / "input_spec.json", input_spec)

    # Build initial state
    initial_state: P2SPipelineState = {
        "image_path": str(image_path),
        "input_spec": input_spec,
        "run_id": effective_run_id,
        "run_dir": str(run_dir_obj.path),
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        "max_shader_chars": max_shader_chars,
        "llm_enabled": effective_llm_enabled,
        "llm_implementation": effective_llm_implementation,
        "cv_enabled": effective_cv_enabled,
        "glsl_render_enabled": effective_glsl_render_enabled,
        "optimizer_iterations": optimizer_iterations,
        "refinement_mode": effective_refinement_mode,
        "max_refinement_iterations": max_refinement_iterations,
        "refinement_threshold": refinement_threshold,
        "refinement_high_score_stop": refinement_high_score_stop,
        "refinement_min_improvement": refinement_min_improvement,
        "refinement_patience": refinement_patience,
        "protected_aspects": protected_aspects,
        "quality_mode": quality_config.get("mode", "balanced"),
        "force_failure_type": force_failure_type,
        "max_added_layers": max_added_layers,
        "vlm_judge_enabled": vlm_judge_enabled,
        "vlm_tie_epsilon": vlm_tie_epsilon,
    }

    # Run the LangGraph pipeline
    logger.info("pipeline start: run_id=%s image=%s", effective_run_id, image_path.name)
    if progress_callback:
        progress_callback("preprocessing")

    state = _pipeline_graph.invoke(initial_state)

    # Run post-pipeline (optimization, revision, refinement)
    if progress_callback:
        progress_callback("optimizing")

    state = _run_post_pipeline(state, strategy_reader=strategy_reader)

    logger.info("pipeline done: run_id=%s", effective_run_id)

    return {
        "run_id": effective_run_id,
        "run_dir": str(run_dir_obj.path),
        "input_spec": input_spec,
        "preprocess": state.get("preprocess", {}),
        "scoreboard": state.get("scoreboard", {}),
        "selected_candidate_id": state.get("selected_candidate_id"),
        "selected_dsl": state.get("selected_dsl"),
        "selected_glsl": state.get("selected_glsl"),
        "objective_metrics": state.get("selected_metrics", {}),
        "quality_router": state.get("selected_quality", {}),
        "optimization": state.get("optimization"),
        "revision": state.get("revision"),
        "refinement_summary": state.get("refinement_summary", {}),
        "refinement_history": state.get("refinement_history", []),
        "candidate_details": state.get("candidate_details", []),
        "vlm_judge": state.get("vlm_judge"),
    }
