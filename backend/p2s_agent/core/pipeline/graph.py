"""P2S-Agent Pipeline Orchestrator

核心 LangGraph 流程:
  preprocess -> candidates -> scoring -> selection

优化、修订、残差补层、LLM 精修和 VLM 评审作为 post-pipeline
同步函数运行。修改下方编排逻辑时，请同步维护 CORE_PIPELINE_FLOWCHART。

Seed-GLSL 入口（run_png_shader_pipeline(seed_glsl=...)）跳过 LangGraph
核心链路，经 _run_seed_glsl_path 合成单个 GLSL 候选后直接进入 post-pipeline。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from langgraph.graph import END, StateGraph

from p2s_agent.core.candidates.llm_scene import Implementation
from p2s_agent.core.dsl.compiler import compile_dsl
from p2s_agent.core.pipeline.artifacts import copy_artifact, create_run_dir, save_json, write_manifest
from p2s_agent.core.pipeline.glsl_optimizer import (
    build_glsl_optimization_artifacts,
    optimize_glsl_candidate,
)
from p2s_agent.core.pipeline.glsl_refinement import run_glsl_refinement_loop
from p2s_agent.core.pipeline.input_spec import build_input_spec
from p2s_agent.core.pipeline.optimizer import build_optimization_artifacts, optimize_candidate
from p2s_agent.core.pipeline.pool import (
    CandidateRecord,
    _candidate_detail,
    build_scoreboard,
    run_candidate_pool,
    select_best_candidate,
)
from p2s_agent.core.pipeline.preprocess import preprocess_image, save_preprocess_artifacts
from p2s_agent.core.pipeline.refinement import (
    _build_revision_patch,
    _should_run_refinement,
    run_dsl_refinement_loop,
)
from p2s_agent.core.pipeline.revision import apply_revision_with_rollback, build_revision_log_entry
from p2s_agent.core.pipeline.residual_layers import add_residual_layers
from p2s_agent.core.llm.vlm_judge import judge_directed_pairwise, judge_pairwise, judge_rubric
from p2s_agent.core.metrics.quality_router import compute_final_score
from p2s_agent.core.pipeline.scoring import (
    _accept_improvement,
    _evaluate_glsl_with_webgl,
    _gate_quality_score,
    _make_render_dsl_fn,
    _make_render_glsl_fn,
    _make_revision_scorer,
    _score_candidates,
    _sync_selected_record_for_response,
)
from p2s_agent.core.logging_config import log_event
from p2s_agent.state import P2SPipelineState
from p2s_agent.strategy import clamp as strategy_clamp, get_default

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 核心流程图
# ---------------------------------------------------------------------------
# 这张图是 graph.py 的“开发者阅读版流程图”：
# - LangGraph 只负责四个确定性节点：预处理、候选池、评分、选择。
# - 质量驱动的优化、修订、残差补层、LLM 精修、VLM 评审都在后处理运行。
# - 更细的候选生成实现维护在 p2s_agent.core.pipeline.pool 和 p2s_agent.core.candidates.* 中。
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │ 入口：run_png_shader_pipeline(image_path, input_spec, run_id, ...) │
# └───────────────────────────────┬─────────────────────────────────────┘
#                                 ▼
# ┌─────────────────────────────────────────────────────────────────────┐
# │ 初始化运行上下文                                                    │
# │ 1. input_spec 缺省补全与校验                                         │
# │ 2. 解析 target / candidates / quality 配置                           │
# │ 3. 创建 artifacts/run_dir，保存 manifest、reference_input、input_spec │
# └───────────────────────────────┬─────────────────────────────────────┘
#                                 ▼
# ┌─────────────────────────────────────────────────────────────────────┐
# │ LangGraph 核心链路                                                   │
# │                                                                     │
# │ preprocess_step                                                     │
# │   └─ preprocess_image：提取尺寸、透明度、调色板、边缘、纹理等特征       │
# │                                                                     │
# │ candidates_step                                                     │
# │   └─ run_candidate_pool：生成 baseline / rule / decompose / CV / LLM │
# │      / fallback 候选，并完成 DSL 校验、DSL 编译或 raw GLSL 校验         │
# │                                                                     │
# │ scoring_step                                                        │
# │   └─ 渲染候选，计算 MSE / SSIM / alpha / color / edge / shader budget │
# │      等指标，再通过 quality_router 产出 final_score 和 next_action     │
# │                                                                     │
# │ selection_step                                                      │
# │   └─ 可选 VLM 近分仲裁后，按 score / priority / GLSL 长度选择最佳候选   │
# └───────────────────────────────┬─────────────────────────────────────┘
#                                 ▼
# ┌─────────────────────────────────────────────────────────────────────┐
# │ Post Pipeline：质量驱动后处理                                        │
# │                                                                     │
# │ selected 是 DSL                                                     │
# │   ├─ next_action=optimize/revise 且允许迭代：坐标下降参数优化          │
# │   ├─ next_action=revise/fallback：生成 revision patch，失败则回滚      │
# │   └─ 分数不足且允许补层：基于残差添加 layer                            │
# │                                                                     │
# │ selected 是 GLSL                                                    │
# │   └─ next_action=optimize/revise 且允许迭代：GLSL optimizer + WebGL 评分│
# │                                                                     │
# │ LLM refinement                                                      │
# │   └─ 根据 refinement_mode、阈值、LLM 开关决定是否闭环精修 DSL/GLSL      │
# │      精修期间可读取运行中策略更新，也可响应 stop_requested             │
# │                                                                     │
# │ VLM final gate                                                      │
# │   └─ 可选 judge_rubric，将语义评分融合回 final_score                  │
# └───────────────────────────────┬─────────────────────────────────────┘
#                                 ▼
# ┌─────────────────────────────────────────────────────────────────────┐
# │ 收尾输出                                                            │
# │ 同步 selected record，保存 candidates/scoreboard/metrics/quality/    │
# │ selected_dsl/selected_shader/refinement_summary，并返回结构化结果。    │
# └─────────────────────────────────────────────────────────────────────┘
CORE_PIPELINE_FLOWCHART = """
flowchart TD
    A["入口：run_png_shader_pipeline"] --> B["补全 input_spec"]
    B --> C["解析 target / candidates / quality 配置"]
    C --> D["创建 run_dir"]
    D --> E["保存 manifest / reference_input / input_spec"]

    subgraph LG["LangGraph 核心链路"]
        F["preprocess_step<br/>提取图像特征并保存预处理工件"]
        G["candidates_step<br/>运行候选池"]
        H["scoring_step<br/>渲染、计算客观指标、质量路由"]
        I["selection_step<br/>选择最佳候选"]
        F --> G --> H --> I
    end

    E --> F

    subgraph POOL["候选池内部摘要"]
        G1["baseline / rule<br/>确定性基础候选"]
        G2["decompose / CV<br/>基于图像结构的候选"]
        G3["LLM<br/>可选 DSL 或 GLSL 候选"]
        G4["fallback<br/>兜底候选"]
        G5["校验 DSL/GLSL<br/>编译 DSL 或校验 raw GLSL"]
        G1 --> G5
        G2 --> G5
        G3 --> G5
        G4 --> G5
    end

    G -.-> G1
    G5 -.-> H

    I --> J{"启用 VLM 近分仲裁？"}
    J -->|"是"| K["judge_pairwise<br/>调整近分候选分数"]
    J -->|"否"| L["保留 objective ranking"]
    K --> M["selected candidate"]
    L --> M

    subgraph POST["Post Pipeline：质量驱动后处理"]
        M --> N{"选中结果类型？"}
        N -->|"DSL"| O{"quality.next_action"}
        O -->|"optimize / revise"| P["DSL 参数优化<br/>只接受提分结果"]
        O -->|"revise / fallback"| Q["revision patch<br/>失败自动回滚"]
        O -->|"其他"| R["跳过 DSL 修正"]
        P --> S{"允许残差补层且分数不足？"}
        Q --> S
        R --> S
        S -->|"是"| T["residual_layers<br/>基于残差添加 layer"]
        S -->|"否"| U["进入精修判断"]
        T --> U

        N -->|"GLSL"| V{"quality.next_action"}
        V -->|"optimize / revise"| W["GLSL optimizer<br/>WebGL 渲染评分"]
        V -->|"其他"| U
        W --> U

        U --> X{"是否运行 LLM refinement？"}
        X -->|"是"| Y["refinement loop（DSL/GLSL）<br/>支持运行中策略更新和停止"]
        X -->|"否"| Z["跳过精修"]
        Y --> AA{"启用 VLM final gate？"}
        Z --> AA
        AA -->|"是"| AB["judge_rubric<br/>融合语义评分"]
        AA -->|"否"| AC["同步 selected record"]
        AB --> AC
    end

    AC --> AD["保存 candidates / scoreboard / metrics / quality"]
    AD --> AE["保存 selected_dsl / selected_shader / summaries"]
    AE --> AF["返回结构化结果"]
"""


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

    # Determine preference for GLSL output. GLSL candidates are refinable, so
    # an active refinement config no longer disables this preference.
    prefer_output_kind = None
    if state.get("glsl_render_enabled", False):
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

def _build_region_veto_fn(selected, protect_regions, run_dir, canvas_width, canvas_height, *, floor, ceil):
    """Build an injected veto callable for the refinement loops, or None.

    Baseline = the selected candidate's render captured HERE, before optimization /
    revision / refinement mutate the candidate. This is the user-selected checkpoint's
    look (the "constraint-set-time" anchor per the design's D3) — intentionally the
    fixed thing 'protect' guards, not a rolling per-iteration render.
    Returns None when there are no protect regions / no selected / no usable baseline,
    so the loops behave exactly as before.
    """
    protect = [r for r in (protect_regions or []) if getattr(r, "mode", None) == "protect"]
    if not protect or selected is None:
        return None
    baseline = Path(selected.render_path) if getattr(selected, "render_path", None) else None
    if baseline is None or not baseline.exists():
        # DSL candidate may not carry a render_path: render it once as the baseline.
        try:
            if getattr(selected, "dsl", None):
                from p2s_agent.core.dsl.renderer import render_dsl_to_image
                baseline = Path(run_dir) / "protect_baseline.png"
                render_dsl_to_image(selected.dsl, baseline, width=canvas_width, height=canvas_height)
            else:
                baseline = None
        except Exception:
            logger.warning("protect-veto baseline render failed", exc_info=True)
            baseline = None
    if baseline is None or not baseline.exists():
        return None
    from p2s_agent.core.pipeline.region_metrics import evaluate_protect_veto
    return (
        lambda cand, _r=protect, _b=baseline, _f=floor, _c=ceil:
        evaluate_protect_veto(_b, cand, _r, floor=_f, ceil=_c)
    )


def _run_post_pipeline(
    state: P2SPipelineState,
    *,
    strategy_reader: Callable[[], dict] | None = None,
    publish_partial: Callable[[dict], None] | None = None,
) -> P2SPipelineState:
    """Run optimization, revision, and refinement after selection.

    This is called after the LangGraph pipeline completes. When
    ``publish_partial`` is provided, a baseline snapshot is published before
    refinement and one partial per refinement iteration via ``on_iteration``,
    so a polling client can render progress live.
    """
    from p2s_agent.core.pipeline.scoring import _accept_improvement  # avoid circular import

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

    from p2s_agent.config import settings as _veto_settings
    _region_veto_fn = _build_region_veto_fn(
        selected,
        state.get("protect_regions"),
        run_dir,
        canvas_width,
        canvas_height,
        floor=float(_veto_settings.protect_veto_ssim_floor),
        ceil=float(_veto_settings.protect_veto_ssim_ceil),
    )

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

    def _publish_iteration(snapshot: dict) -> None:
        if publish_partial is None:
            return
        hist = list(snapshot.get("history") or [])
        best = snapshot.get("best_score")
        try:
            partial = {
                "refinement_history": hist,
                "refinement_summary": {
                    **refinement_summary,
                    "enabled": True,
                    "iterations": len(hist),
                    "final_score": best if best is not None
                    else refinement_summary.get("final_score"),
                },
                "objective_metrics": dict(snapshot.get("best_metrics") or {}),
                "quality_router": dict(snapshot.get("best_quality") or {}),
            }
            # Only refresh the previewed shader when this iteration has one; a
            # DSL recompile can yield empty GLSL, and publishing None would blank
            # the baseline-published preview until the next improving iteration.
            best_glsl = snapshot.get("best_glsl")
            if best_glsl:
                partial["selected_glsl"] = best_glsl
            publish_partial(partial)
        except Exception:
            logger.warning("publish_partial (iteration) failed", exc_info=True)

    # Baseline #1: surface candidates + initial selection ASAP.
    if publish_partial is not None:
        try:
            publish_partial({
                # run_dir + lineage let a human-in-loop branch endpoint locate
                # reference_input.png and trace parentage while still running.
                "run_dir": str(run_dir),
                "lineage": state.get("lineage"),
                "preprocess": state.get("preprocess", {}),
                "scoreboard": build_scoreboard(candidates),
                "selected_candidate_id": selected.id,
                "selected_glsl": selected.compile_glsl,
                "objective_metrics": dict(selected.objective_metrics),
                "quality_router": dict(selected.quality_router) if selected.quality_router else {},
            })
        except Exception:
            logger.warning("publish_partial (baseline) failed", exc_info=True)

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
                        reason=f"revision improved score {selected.final_score:.4f} -> {rev_result.final_score:.4f}",
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
    force_first_refinement = bool(state.get("force_first_refinement_iteration", False))

    # Human-in-loop directed acceptance: build a goal-aware pairwise judge from
    # the run's directed_acceptance config when enabled and VLM is available.
    # Falls back to None (metric-only acceptance) otherwise.
    directed_acceptance_cfg = state.get("directed_acceptance") or None
    directed_pairwise_judge = None
    if (
        directed_acceptance_cfg
        and directed_acceptance_cfg.get("enabled")
        and state.get("vlm_judge_enabled")
    ):
        _directed_goal = directed_acceptance_cfg.get("feedback") or ""

        def directed_pairwise_judge(cur, new, _goal=_directed_goal):
            return judge_directed_pairwise(
                reference_path, cur, new, user_feedback=_goal, work_dir=run_dir / "judge"
            )

    # Baseline #2: reflect optimizer / revision / residual gains before refinement.
    if publish_partial is not None:
        try:
            publish_partial({
                "scoreboard": build_scoreboard(candidates),
                "selected_glsl": selected_glsl,
                "objective_metrics": dict(selected_metrics),
                "quality_router": dict(selected_quality) if selected_quality else {},
            })
        except Exception:
            logger.warning("publish_partial (pre-refine) failed", exc_info=True)

    # The refinement gate is an ABSOLUTE quality check. A VLM near-tie may have
    # bumped selected.final_score purely to win SORTING in select_best_candidate;
    # re-sync it to the true objective score (quality_router['final_score']) so a
    # candidate below refinement_threshold cannot be bumped over it and silently
    # skip refinement. Legitimate optimizer/revision gains keep both in sync, so
    # this only strips the ordering-only tie-break adjustment.
    if selected is not None:
        selected.final_score = _gate_quality_score(selected)

    should_refine, refinement_decision = _should_run_refinement(
        effective_refinement_mode,
        selected,
        selected_quality,
        threshold=refinement_threshold,
        high_score_stop=refinement_high_score_stop,
        force_first=force_first_refinement,
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
        # Reflect the post-optimization baseline in iteration partials.
        refinement_summary["initial_score"] = initial_refinement_score
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
            force_first_iteration=(
                state.get("force_first_refinement_iteration", False)
                or effective_refinement_mode == "on"
            ),
            initial_extra_feedback=state.get("human_feedback_notes") or None,
            directed_acceptance=directed_acceptance_cfg,
            directed_pairwise_judge=directed_pairwise_judge,
            loop_dir=run_dir / "refinement",
            strategy_reader=strategy_reader,
            protected_aspects=protected_aspects,
            pairwise_judge=(
                (lambda cur, new: judge_pairwise(
                    reference_path, cur, new, work_dir=run_dir / "judge"
                ))
                if state.get("vlm_judge_enabled") else None
            ),
            rubric_judge=(
                (lambda render: judge_rubric(
                    reference_path, render, work_dir=run_dir / "judge"
                ))
                if state.get("vlm_judge_enabled") else None
            ),
            on_iteration=_publish_iteration,
            region_veto_fn=_region_veto_fn,
        )
        refinement_history = ref_result.get("history", [])
        refinement_summary.update({
            "iterations": len(refinement_history),
            "initial_score": initial_refinement_score,
            "final_score": ref_result.get("best_score", initial_refinement_score),
            "improved": ref_result.get("best_score", 0) > initial_refinement_score,
            "stop_reason": ref_result.get("stop_reason"),
        })
        _veto_iters = [h for h in refinement_history if h.get("rejected_reason") == "protect_region_veto"]
        _protect = [r for r in (state.get("protect_regions") or []) if getattr(r, "mode", None) == "protect"]
        if _protect:
            refinement_summary["protect_regions"] = {
                "count": len(_protect),
                "veto_count": len(_veto_iters),
                "last_constraint_score": (_veto_iters[-1].get("constraint_score") if _veto_iters else None),
            }

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

    elif should_refine and selected and selected.output_kind == "glsl" and selected.compile_glsl:
        if not state.get("glsl_render_enabled", False):
            refinement_summary["enabled"] = False
            refinement_summary["decision"] = "glsl_render_disabled"
        else:
            initial_refinement_score = selected.final_score
            # Reflect the post-optimization baseline in iteration partials.
            refinement_summary["initial_score"] = initial_refinement_score

            def _glsl_refinement_evaluate(
                glsl: str, render_path: Path
            ) -> tuple[dict, dict, float, "Path | None"]:
                try:
                    return _evaluate_glsl_with_webgl(
                        glsl,
                        reference_path,
                        render_path,
                        canvas_width=canvas_width,
                        canvas_height=canvas_height,
                        max_shader_chars=max_shader_chars,
                    )
                except Exception:
                    logger.exception("glsl refinement evaluation failed")
                    return {}, {}, 0.0, None

            ref_result = run_glsl_refinement_loop(
                selected.compile_glsl,
                selected.final_score,
                dict(selected.objective_metrics),
                dict(selected.quality_router) if selected.quality_router else {},
                reference_path,
                evaluate_fn=_glsl_refinement_evaluate,
                initial_render_path=Path(selected.render_path) if selected.render_path else None,
                max_iterations=max_refinement_iterations,
                threshold=refinement_threshold,
                high_score_stop=refinement_high_score_stop,
                min_improvement=refinement_min_improvement,
                no_improvement_patience=refinement_patience,
                force_first_iteration=(
                    state.get("force_first_refinement_iteration", False)
                    or effective_refinement_mode == "on"
                ),
                initial_extra_feedback=state.get("human_feedback_notes") or None,
                directed_acceptance=directed_acceptance_cfg,
                directed_pairwise_judge=directed_pairwise_judge,
                loop_dir=run_dir / "glsl_refinement",
                strategy_reader=strategy_reader,
                pairwise_judge=(
                    (lambda cur, new: judge_pairwise(
                        reference_path, cur, new, work_dir=run_dir / "judge"
                    ))
                    if state.get("vlm_judge_enabled") else None
                ),
                rubric_judge=(
                    (lambda render: judge_rubric(
                        reference_path, render, work_dir=run_dir / "judge"
                    ))
                    if state.get("vlm_judge_enabled") else None
                ),
                on_iteration=_publish_iteration,
                region_veto_fn=_region_veto_fn,
            )
            refinement_history = ref_result.get("history", [])
            refinement_summary.update({
                "kind": "glsl",
                "iterations": len(refinement_history),
                "initial_score": initial_refinement_score,
                "final_score": ref_result.get("best_score", initial_refinement_score),
                "improved": ref_result.get("best_score", 0) > initial_refinement_score,
                "stop_reason": ref_result.get("stop_reason"),
            })
            _veto_iters = [h for h in refinement_history if h.get("rejected_reason") == "protect_region_veto"]
            _protect = [r for r in (state.get("protect_regions") or []) if getattr(r, "mode", None) == "protect"]
            if _protect:
                refinement_summary["protect_regions"] = {
                    "count": len(_protect),
                    "veto_count": len(_veto_iters),
                    "last_constraint_score": (_veto_iters[-1].get("constraint_score") if _veto_iters else None),
                }

            if ref_result.get("best_score", 0) > selected.final_score:
                selected.compile_glsl = ref_result["best_glsl"]
                selected.objective_metrics = ref_result["best_metrics"]
                selected.quality_router = ref_result["best_quality"]
                selected.final_score = ref_result["best_score"]
                if ref_result.get("best_render_path"):
                    selected.render_path = str(ref_result["best_render_path"])
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

    # Reconcile the refinement summary with the authoritative selected score.
    # The VLM final gate above may have blended a semantic score into
    # selected.final_score AFTER refinement_summary recorded the objective
    # refinement trajectory — leaving refinement_summary.final_score (objective
    # best) disagreeing with the selected/quality score the UI shows (blended).
    # Re-point final_score at the selected score so the branch's reported "final"
    # agrees everywhere, and recompute `improved` against the summary's own
    # initial_score. (BUG-010)
    if judge_summary is not None and selected is not None:
        refinement_summary["final_score"] = selected.final_score
        refinement_summary["improved"] = (
            selected.final_score > refinement_summary.get("initial_score", selected.final_score)
        )

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


def _run_seed_glsl_path(
    seed_glsl: str,
    state: P2SPipelineState,
    *,
    strategy_reader: Callable[[], dict] | None = None,
    publish_partial: Callable[[dict], None] | None = None,
) -> P2SPipelineState:
    """Seed entry: refine an externally-supplied GLSL instead of generating
    candidates. Runs preprocess + adapt + score to synthesize a single selected
    GLSL candidate, then hands off to the unchanged ``_run_post_pipeline``.

    Raises ValueError when the seed cannot be adapted to renderable Shadertoy
    GLSL, so the caller/worker marks the run failed.  Raises RuntimeError when
    the WebGL renderer is unavailable during initial scoring of the seed shader.
    """
    from p2s_agent.core.pipeline.seed_glsl import adapt_seed_glsl, build_seed_candidate

    run_dir = Path(state["run_dir"])
    image_path = Path(state["image_path"])

    # Preprocess: target features + canvas context (artifacts mirror node_preprocess).
    preprocess = preprocess_image(image_path)
    save_preprocess_artifacts(preprocess, run_dir, image_path)
    preprocess["llm_reference_background"] = "#000000"
    state["preprocess"] = preprocess
    state["llm_image_path"] = str(run_dir / "llm_reference_input.png")

    # Adapt the seed to renderable Shadertoy GLSL.
    adapted = adapt_seed_glsl(seed_glsl)
    (run_dir / "seed_input.glsl").write_text(seed_glsl, encoding="utf-8")
    if not adapted.valid:
        raise ValueError(
            "seed GLSL could not be adapted to renderable Shadertoy GLSL: "
            f"{adapted.errors[:3]}"
        )
    (run_dir / "seed_adapted.glsl").write_text(adapted.glsl, encoding="utf-8")

    candidate = build_seed_candidate(
        adapted.glsl, adapted_by=adapted.adapted_by, warnings=adapted.warnings
    )

    # Score once to seed the loop (real initial score/metrics/quality).
    reference_path = run_dir / "reference_input.png"
    candidate_dir = run_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    try:
        metrics, quality, score, render_path = _evaluate_glsl_with_webgl(
            adapted.glsl,
            reference_path,
            candidate_dir / f"{candidate.id}_webgl.png",
            canvas_width=state.get("canvas_width", 512),
            canvas_height=state.get("canvas_height", 512),
            max_shader_chars=state.get("max_shader_chars", 12000),
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"seed GLSL initial scoring failed (WebGL renderer unavailable?): {exc}"
        ) from exc
    candidate.objective_metrics = metrics
    candidate.quality_router = quality
    candidate.final_score = score
    candidate.render_path = str(render_path) if render_path else None

    state["candidates"] = [candidate]
    state["selected_candidate_id"] = candidate.id
    state["selected_dsl"] = None
    state["selected_glsl"] = candidate.compile_glsl
    state["selected_metrics"] = dict(metrics)
    state["selected_quality"] = dict(quality)

    return _run_post_pipeline(
        state, strategy_reader=strategy_reader, publish_partial=publish_partial
    )


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
    seed_glsl: str | None = None,
    llm_enabled: bool | None = None,
    llm_implementation: Implementation | None = None,
    progress_callback: Callable[[str], None] | None = None,
    strategy_reader: Callable[[], dict] | None = None,
    publish_partial: Callable[[dict], None] | None = None,
    human_feedback_notes: list[str] | None = None,
    directed_acceptance: dict | None = None,
    force_first_refinement_iteration: bool = False,
    lineage: dict | None = None,
    extra_artifacts: dict | None = None,
    protect_regions: list | None = None,
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

    # Seed-GLSL mode: refine an externally-supplied shader. Force the GLSL
    # closed loop on (WebGL render scoring + LLM available); refinement_mode
    # defaults to "on" upstream in the router.
    if seed_glsl is not None:
        effective_glsl_render_enabled = True
        effective_llm_enabled = True
        input_spec = {**input_spec, "seed_glsl": seed_glsl}

    # Human-in-loop branch lineage travels with the run for artifacts + audit.
    if lineage is not None:
        input_spec = {**input_spec, "lineage": lineage}

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

    # Caller-supplied artifacts (e.g. human-in-loop branch_request / lineage /
    # source checkpoint). dicts/lists are saved as JSON; everything else as text.
    if extra_artifacts:
        for name, content in extra_artifacts.items():
            dest = run_dir_obj.path / name
            if isinstance(content, (dict, list)):
                save_json(dest, content)
            else:
                dest.write_text(str(content), encoding="utf-8")

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
        # Human-in-loop branch refinement (V1).
        "human_feedback_notes": list(human_feedback_notes or []),
        "directed_acceptance": directed_acceptance,
        "force_first_refinement_iteration": bool(force_first_refinement_iteration),
        "lineage": lineage,
        "protect_regions": list(protect_regions or []),
    }

    # Run the LangGraph pipeline
    log_event(
        logger,
        "pipeline_start",
        run_id=effective_run_id,
        image=image_path.name,
        run_dir=str(run_dir_obj.path),
        llm_enabled=effective_llm_enabled,
        refinement_mode=effective_refinement_mode,
        max_refinement_iterations=max_refinement_iterations,
    )
    if progress_callback:
        progress_callback("preprocessing")

    if seed_glsl is not None:
        if progress_callback:
            progress_callback("optimizing")
        state = _run_seed_glsl_path(
            seed_glsl, initial_state, strategy_reader=strategy_reader,
            publish_partial=publish_partial,
        )
    else:
        state = _pipeline_graph.invoke(initial_state)

        # Run post-pipeline (optimization, revision, refinement)
        if progress_callback:
            progress_callback("optimizing")

        state = _run_post_pipeline(
            state, strategy_reader=strategy_reader, publish_partial=publish_partial
        )

    log_event(
        logger,
        "pipeline_done",
        run_id=effective_run_id,
        selected_id=state.get("scoreboard", {}).get("selected_id"),
        score=state.get("selected_quality", {}).get("final_score"),
        refinement=state.get("refinement_summary", {}),
    )

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
        "lineage": state.get("lineage"),
    }
