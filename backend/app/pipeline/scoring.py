"""Evaluation and render callbacks for PNG-to-Shader candidates: DSL/GLSL
rendering, objective metrics, quality routing, per-candidate scoring, and the
render-callback factories used by the optimizer/revision stages. Split out of
graph.py (2026-06-11); behavior is unchanged."""

from __future__ import annotations

import logging
import shutil
from dataclasses import asdict
from pathlib import Path

logger = logging.getLogger(__name__)

from app.dsl.compiler import compile_dsl
from app.dsl.renderer import render_dsl_to_image
from app.metrics.compute import compute_objective_metrics, hex_to_rgb01
from app.pipeline.pool import CandidateRecord, _candidate_detail
from app.metrics.quality_router import route
from app.pipeline.artifacts import save_json

try:
    from app.services.browser_render import render_multiple_frames
except ImportError:
    render_multiple_frames = None


def _sync_selected_record_for_response(
    selected: CandidateRecord | None,
    *,
    selected_dsl: dict | None,
    selected_glsl: str | None,
    selected_metrics: dict,
    selected_quality: dict | None,
) -> None:
    """Keep the selected candidate row aligned with final response fields.

    Optimizer/revision/refinement stages mutate the selected candidate after
    initial scoring. The frontend previews rows from ``scoreboard.candidates``,
    so the row must carry the same final GLSL as ``selected_glsl``.
    """
    if selected is None:
        return

    selected.selected = True
    if selected_dsl is not None and selected.output_kind == "dsl":
        selected.dsl = selected_dsl
    # Only overwrite the row's compile_glsl when the new value is actually
    # usable. An empty selected_glsl can sneak in from a refinement loop that
    # bailed before producing GLSL — in that case keep the previously good GLSL
    # so the scoreboard row is still previewable.
    if selected_glsl is not None and selected_glsl.strip():
        selected.compile_glsl = selected_glsl
        selected.compile_success = True
        selected.compile_errors = []
    selected.objective_metrics = dict(selected_metrics)
    if selected_quality is not None:
        selected.quality_router = dict(selected_quality)
        if "final_score" in selected_quality:
            selected.final_score = float(selected_quality["final_score"])


def _accept_improvement(
    selected: CandidateRecord,
    new_dsl: dict,
    reference_path: Path,
    out_render_path: Path,
    *,
    canvas_width: int,
    canvas_height: int,
    max_shader_chars: int,
    protected_aspects: list[str],
    reason: str,
) -> "tuple[dict, str, dict, dict] | None":
    """Compile + re-evaluate an improved DSL and update *selected* in place.

    Returns (dsl, glsl, metrics, quality) on success so callers can refresh
    their local response variables, or None when compilation fails
    (*selected* is left untouched).
    """
    compile_result = compile_dsl(new_dsl)
    if not compile_result.success:
        return None
    metrics, quality, score, render_path = _evaluate_dsl(
        new_dsl,
        compile_result.glsl,
        reference_path,
        out_render_path,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        max_shader_chars=max_shader_chars,
        protected_aspects=protected_aspects,
    )
    selected.dsl = new_dsl
    selected.compile_glsl = compile_result.glsl
    selected.compile_success = True
    selected.compile_errors = []
    selected.objective_metrics = metrics
    selected.quality_router = quality
    selected.final_score = score
    selected.render_path = str(render_path) if render_path else None
    selected.reason.append(reason)
    return new_dsl, compile_result.glsl, metrics, quality


def _metric_render_size(canvas_width: int, canvas_height: int) -> tuple[int, int]:
    """Use a bounded render size for fast objective metrics."""
    max_dim = 192
    largest = max(canvas_width, canvas_height, 1)
    if largest <= max_dim:
        return canvas_width, canvas_height
    scale = max_dim / largest
    return max(1, int(canvas_width * scale)), max(1, int(canvas_height * scale))


def _evaluate_dsl(
    dsl: dict,
    glsl: str,
    ref_path: Path,
    output_path: Path,
    *,
    canvas_width: int,
    canvas_height: int,
    max_shader_chars: int,
    protected_aspects: list[str] | None = None,
) -> tuple[dict, dict, float, Path | None]:
    """Render and route a single DSL candidate."""
    if protected_aspects is None:
        protected_aspects = ["layer_count", "primitive_types", "background"]
    render_width, render_height = _metric_render_size(canvas_width, canvas_height)
    try:
        render_path = render_dsl_to_image(
            dsl,
            output_path,
            width=render_width,
            height=render_height,
        )
        bg_hex = "#000000"
        if isinstance(dsl, dict):
            bg_hex = (dsl.get("canvas") or {}).get("background") or "#000000"
        metrics = compute_objective_metrics(
            ref_path,
            render_path,
            shader_chars=len(glsl),
            max_shader_chars=max_shader_chars,
            background_rgb=hex_to_rgb01(bg_hex),
        )
        quality = route(
            {"compiled": True, "rendered": True},
            metrics,
            protected_aspects=protected_aspects,
        )
        quality_dict = asdict(quality)
        return metrics, quality_dict, quality.final_score, render_path
    except Exception as exc:
        metrics = {
            "mse": 1.0,
            "simple_ssim": 0.0,
            "alpha_coverage_diff": 1.0,
            "color_histogram_score": 0.0,
            "edge_density_diff": 1.0,
            "nonblank_render": False,
            "within_shader_budget": len(glsl) <= max_shader_chars,
            "render_error": str(exc),
        }
        quality = route(
            {"compiled": True, "rendered": False},
            metrics,
            protected_aspects=protected_aspects,
        )
        return metrics, asdict(quality), 0.0, None


def _evaluate_glsl_with_webgl(
    glsl: str,
    ref_path: Path,
    output_path: Path,
    *,
    canvas_width: int,
    canvas_height: int,
    max_shader_chars: int,
) -> tuple[dict, dict, float, Path | None]:
    """Render raw GLSL in the browser preview and score it against reference."""
    if render_multiple_frames is None:
        raise RuntimeError("browser renderer is unavailable")

    render_width, render_height = _metric_render_size(canvas_width, canvas_height)
    screenshots = render_multiple_frames(
        glsl,
        times=[0.0],
        width=render_width,
        height=render_height,
    )
    if not screenshots:
        raise RuntimeError("browser renderer returned no screenshots")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(screenshots[0], output_path)
    metrics = compute_objective_metrics(
        ref_path,
        output_path,
        shader_chars=len(glsl),
        max_shader_chars=max_shader_chars,
    )
    metrics["backend_rasterized"] = True
    metrics["render_backend"] = "webgl"
    quality = route(
        {"compiled": True, "rendered": True},
        metrics,
        protected_aspects=["visual_causality", "technique_plan", "tunable_parameters"],
    )
    return metrics, asdict(quality), quality.final_score, output_path


def _score_candidates(
    candidates: list[CandidateRecord],
    ref_path: Path,
    candidate_dir: Path,
    *,
    canvas_width: int,
    canvas_height: int,
    max_shader_chars: int,
    glsl_render_enabled: bool = False,
    protected_aspects: list[str] | None = None,
) -> None:
    """Populate render artifacts, metrics, router output, and score per candidate."""
    if protected_aspects is None:
        protected_aspects = ["layer_count", "primitive_types", "background"]
    candidate_dir.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        if not candidate.compile_success or not candidate.dsl:
            if candidate.output_kind == "glsl" and candidate.compile_success:
                if glsl_render_enabled:
                    try:
                        render_path = candidate_dir / f"{candidate.id}_webgl.png"
                        metrics, quality, score, actual_render_path = _evaluate_glsl_with_webgl(
                            candidate.compile_glsl,
                            ref_path,
                            render_path,
                            canvas_width=canvas_width,
                            canvas_height=canvas_height,
                            max_shader_chars=max_shader_chars,
                        )
                        candidate.objective_metrics = metrics
                        candidate.quality_router = quality
                        candidate.final_score = score
                        candidate.render_path = str(actual_render_path) if actual_render_path else None
                        save_json(candidate_dir / f"{candidate.id}.json", _candidate_detail(candidate))
                        logger.info(
                            "candidate scored: id=%s score=%.4f band=%s next=%s backend=webgl",
                            candidate.id,
                            float(candidate.final_score),
                            (candidate.quality_router or {}).get("quality_band"),
                            (candidate.quality_router or {}).get("next_action"),
                        )
                        continue
                    except Exception as exc:
                        logger.warning("WebGL GLSL scoring failed", exc_info=True)
                        candidate.reason.append(f"webgl scoring failed: {exc}")

                candidate.objective_metrics = {
                    "backend_rasterized": False,
                    "reason": "GLSL LLM candidate is preview-compatible but not DSL-rasterizable",
                    "within_shader_budget": len(candidate.compile_glsl) <= max_shader_chars,
                }
                quality = route(
                    {"compiled": True, "rendered": False},
                    candidate.objective_metrics,
                    protected_aspects=protected_aspects,
                )
                candidate.quality_router = asdict(quality)
                candidate.final_score = 0.0
                save_json(candidate_dir / f"{candidate.id}.json", _candidate_detail(candidate))
                continue
            quality = route(
                {"compiled": False, "rendered": False},
                {},
                protected_aspects=protected_aspects,
            )
            candidate.quality_router = asdict(quality)
            candidate.final_score = 0.0
            continue

        render_path = candidate_dir / f"{candidate.id}_render.png"
        metrics, quality, score, actual_render_path = _evaluate_dsl(
            candidate.dsl,
            candidate.compile_glsl,
            ref_path,
            render_path,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            max_shader_chars=max_shader_chars,
            protected_aspects=protected_aspects,
        )
        candidate.objective_metrics = metrics
        candidate.quality_router = quality
        candidate.final_score = score
        candidate.render_path = str(actual_render_path) if actual_render_path else None
        save_json(candidate_dir / f"{candidate.id}.json", _candidate_detail(candidate))
        logger.info(
            "candidate scored: id=%s score=%.4f band=%s next=%s",
            candidate.id,
            float(candidate.final_score),
            (candidate.quality_router or {}).get("quality_band"),
            (candidate.quality_router or {}).get("next_action"),
        )


def _make_render_dsl_fn(
    render_dir: Path,
    *,
    canvas_width: int,
    canvas_height: int,
):
    counter = {"value": 0}
    render_dir.mkdir(parents=True, exist_ok=True)

    def render_dsl_fn(dsl: dict, glsl: str) -> Path | None:
        counter["value"] += 1
        path = render_dir / f"render_{counter['value']:03d}.png"
        render_width, render_height = _metric_render_size(canvas_width, canvas_height)
        return render_dsl_to_image(dsl, path, width=render_width, height=render_height)

    return render_dsl_fn


def _make_render_glsl_fn(
    render_dir: Path,
    *,
    canvas_width: int,
    canvas_height: int,
):
    """Build a callable rendering raw GLSL via the WebGL backend.

    Used by the GLSL `#define` optimizer to evaluate perturbed shaders.
    Returns None when the renderer is unavailable or fails so the optimizer
    can treat the step as a 0-score and reject it.
    """
    counter = {"value": 0}
    render_dir.mkdir(parents=True, exist_ok=True)
    render_width, render_height = _metric_render_size(canvas_width, canvas_height)

    def render_glsl_fn(glsl: str) -> Path | None:
        if render_multiple_frames is None:
            return None
        counter["value"] += 1
        try:
            screenshots = render_multiple_frames(
                glsl,
                times=[0.0],
                width=render_width,
                height=render_height,
            )
        except Exception:
            logger.warning("glsl render failed during optimization", exc_info=True)
            return None
        if not screenshots:
            return None
        path = render_dir / f"render_{counter['value']:03d}.png"
        try:
            shutil.copyfile(screenshots[0], path)
        except Exception:
            logger.warning("glsl render copy failed", exc_info=True)
            return None
        return path

    return render_glsl_fn


def _make_revision_scorer(
    render_dir: Path,
    *,
    canvas_width: int,
    canvas_height: int,
    max_shader_chars: int,
    protected_aspects: list[str] | None = None,
):
    if protected_aspects is None:
        protected_aspects = ["layer_count", "primitive_types", "background"]
    counter = {"value": 0}
    render_dir.mkdir(parents=True, exist_ok=True)

    def score_dsl_value(
        dsl: dict,
        ref_path: Path,
    ) -> float:
        compile_result = compile_dsl(dsl)
        if not compile_result.success:
            return 0.0
        counter["value"] += 1
        path = render_dir / f"revision_score_{counter['value']:03d}.png"
        _metrics, _quality, score, _render_path = _evaluate_dsl(
            dsl,
            compile_result.glsl,
            ref_path,
            path,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            max_shader_chars=max_shader_chars,
            protected_aspects=protected_aspects,
        )
        return score

    return score_dsl_value
