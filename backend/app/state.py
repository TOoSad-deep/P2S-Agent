"""Pipeline State Definition for P2S-Agent

Defines the TypedDict for the LangGraph StateGraph pipeline state.
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict


class P2SPipelineState(TypedDict, total=False):
    """State for the PNG-to-Shader pipeline.

    The state is divided into input, per-node outputs, and final outputs.
    Each node reads from and writes to specific fields.
    """

    # === Input fields ===
    image_path: str
    input_spec: dict
    run_id: str
    run_dir: str

    # === Preprocess node output ===
    preprocess: dict

    # === Candidates node output ===
    candidates: list[Any]  # list[CandidateRecord]

    # === Scoring node output ===
    scored: bool

    # === Selection node output ===
    selected_candidate_id: Optional[str]
    selected_dsl: Optional[dict]
    selected_glsl: Optional[str]
    selected_metrics: dict
    selected_quality: dict

    # === Post-pipeline results (filled by optimization/revision/refinement) ===
    optimization: Optional[dict]
    revision: Optional[dict]
    refinement_summary: dict
    refinement_history: list

    # === Final output ===
    scoreboard: dict
    candidate_details: list

    # === Control ===
    progress: str
    error: Optional[str]

    # === Configuration (passed through from input_spec) ===
    canvas_width: int
    canvas_height: int
    max_shader_chars: int
    llm_enabled: bool
    llm_implementation: str
    cv_enabled: bool
    glsl_render_enabled: bool
    optimizer_iterations: int
    refinement_mode: str
    max_refinement_iterations: int
    refinement_threshold: float
    refinement_high_score_stop: float
    refinement_min_improvement: float
    refinement_patience: int
    protected_aspects: list[str]
    quality_mode: str
    force_failure_type: Optional[str]
    max_added_layers: int
    vlm_judge_enabled: bool
    vlm_tie_epsilon: float
