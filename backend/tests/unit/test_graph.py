"""Unit tests for the PNG-to-Shader candidate pool orchestrator (graph.py).

No LLM, no browser. Uses synthetic PNGs created with Pillow.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.pipeline.graph import (
    CandidateRecord,
    _should_run_refinement,
    build_scoreboard,
    run_candidate_pool,
    run_dsl_refinement_loop,
    run_png_shader_pipeline,
    select_best_candidate,
)
from app.pipeline.preprocess import preprocess_image


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def make_solid_png(
    tmp_path: Path,
    name: str = "test.png",
    color: tuple = (200, 100, 50, 255),
    size: tuple = (64, 64),
) -> Path:
    """Create a solid-color RGBA PNG for testing."""
    from PIL import Image

    img = Image.new("RGBA", size, color)
    path = tmp_path / name
    img.save(path)
    return path


def _preprocess_from_png(tmp_path: Path) -> dict:
    """Return a real preprocess dict from a synthetic PNG."""
    png_path = make_solid_png(tmp_path)
    return preprocess_image(png_path)


def _make_minimal_input_spec() -> dict:
    """Return a minimal valid input spec."""
    return {
        "input_image": "test.png",
        "target": {
            "backend": "glsl",
            "shader_env": "webgl2",
            "resolution": [512, 512],
            "allow_texture": False,
            "allow_sdf_texture": False,
            "max_shader_chars": 12000,
            "max_layers": 24,
            "max_render_time_ms": 8,
        },
        "quality": {
            "mode": "balanced",
            "max_iterations": 5,
            "optimization_budget": 300,
            "refinement_mode": "auto",
            "max_refinement_iterations": 3,
            "refinement_threshold": 0.80,
        },
    }


def _make_refinement_candidate(*, score: float = 0.5, has_dsl: bool = True) -> CandidateRecord:
    return CandidateRecord(
        id="test_0",
        source="rule",
        enabled=True,
        priority=1,
        dsl={"layers": []} if has_dsl else None,
        output_kind="dsl" if has_dsl else "glsl",
        validation_valid=True,
        validation_errors=[],
        compile_success=True,
        compile_glsl="void mainImage(out vec4 fragColor, in vec2 fragCoord) { fragColor = vec4(1.0); }",
        compile_errors=[],
        final_score=score,
        selected=True,
    )


# ---------------------------------------------------------------------------
# run_candidate_pool tests
# ---------------------------------------------------------------------------


def test_should_run_refinement_off_never_runs():
    candidate = _make_refinement_candidate(score=0.1)
    should_run, reason = _should_run_refinement(
        "off",
        candidate,
        {"final_score": 0.1},
        threshold=0.8,
        high_score_stop=0.92,
    )

    assert should_run is False
    assert reason == "refinement_mode_off"


def test_should_run_refinement_auto_only_below_threshold():
    low_candidate = _make_refinement_candidate(score=0.7)
    high_candidate = _make_refinement_candidate(score=0.85)

    should_run_low, low_reason = _should_run_refinement(
        "auto",
        low_candidate,
        {"final_score": 0.7},
        threshold=0.8,
        high_score_stop=0.92,
    )
    should_run_high, high_reason = _should_run_refinement(
        "auto",
        high_candidate,
        {"final_score": 0.85},
        threshold=0.8,
        high_score_stop=0.92,
    )

    assert should_run_low is True
    assert low_reason == "auto_below_threshold"
    assert should_run_high is False
    assert high_reason == "auto_threshold_reached"


def test_should_run_refinement_on_forces_until_high_score_stop():
    mid_candidate = _make_refinement_candidate(score=0.85)
    excellent_candidate = _make_refinement_candidate(score=0.94)

    should_run_mid, mid_reason = _should_run_refinement(
        "on",
        mid_candidate,
        {"final_score": 0.85},
        threshold=0.8,
        high_score_stop=0.92,
    )
    should_run_excellent, excellent_reason = _should_run_refinement(
        "on",
        excellent_candidate,
        {"final_score": 0.94},
        threshold=0.8,
        high_score_stop=0.92,
    )

    assert should_run_mid is True
    assert mid_reason == "force_enabled"
    assert should_run_excellent is False
    assert excellent_reason == "force_high_score_stop"


def test_should_run_refinement_skips_non_dsl_candidate():
    candidate = _make_refinement_candidate(score=0.3, has_dsl=False)

    should_run, reason = _should_run_refinement(
        "on",
        candidate,
        {"final_score": 0.3},
        threshold=0.8,
        high_score_stop=0.92,
    )

    assert should_run is False
    assert reason == "selected_candidate_is_not_dsl"


def test_run_candidate_pool_returns_records(tmp_path):
    preprocess = _preprocess_from_png(tmp_path)
    input_spec = _make_minimal_input_spec()
    records = run_candidate_pool(preprocess, input_spec)
    assert isinstance(records, list)
    assert len(records) >= 2


def test_run_candidate_pool_always_has_baseline(tmp_path):
    preprocess = _preprocess_from_png(tmp_path)
    input_spec = _make_minimal_input_spec()
    records = run_candidate_pool(preprocess, input_spec)
    sources = [r.source for r in records]
    assert "baseline" in sources


def test_run_candidate_pool_always_has_fallback(tmp_path):
    preprocess = _preprocess_from_png(tmp_path)
    input_spec = _make_minimal_input_spec()
    records = run_candidate_pool(preprocess, input_spec)
    sources = [r.source for r in records]
    assert "fallback" in sources


def test_run_candidate_pool_baseline_compiles(tmp_path):
    preprocess = _preprocess_from_png(tmp_path)
    input_spec = _make_minimal_input_spec()
    records = run_candidate_pool(preprocess, input_spec)
    baseline = next((r for r in records if r.source == "baseline"), None)
    assert baseline is not None
    assert baseline.compile_success is True


def test_run_candidate_pool_fallback_compiles(tmp_path):
    preprocess = _preprocess_from_png(tmp_path)
    input_spec = _make_minimal_input_spec()
    records = run_candidate_pool(preprocess, input_spec)
    fallback = next((r for r in records if r.source == "fallback"), None)
    assert fallback is not None
    assert fallback.compile_success is True


# ---------------------------------------------------------------------------
# select_best_candidate tests
# ---------------------------------------------------------------------------

def test_select_best_candidate_returns_record(tmp_path):
    preprocess = _preprocess_from_png(tmp_path)
    input_spec = _make_minimal_input_spec()
    records = run_candidate_pool(preprocess, input_spec)
    selected = select_best_candidate(records)
    assert selected is not None
    assert isinstance(selected, CandidateRecord)


def test_select_best_candidate_is_compilable(tmp_path):
    preprocess = _preprocess_from_png(tmp_path)
    input_spec = _make_minimal_input_spec()
    records = run_candidate_pool(preprocess, input_spec)
    selected = select_best_candidate(records)
    assert selected is not None
    assert selected.compile_success is True


def test_select_best_candidate_marked_selected(tmp_path):
    preprocess = _preprocess_from_png(tmp_path)
    input_spec = _make_minimal_input_spec()
    records = run_candidate_pool(preprocess, input_spec)
    selected = select_best_candidate(records)
    assert selected is not None
    assert selected.selected is True


def test_select_best_candidate_lowest_priority_wins(tmp_path):
    """Baseline (priority=0) should beat fallback (priority=99)."""
    preprocess = _preprocess_from_png(tmp_path)
    input_spec = _make_minimal_input_spec()
    records = run_candidate_pool(preprocess, input_spec)
    selected = select_best_candidate(records)
    assert selected is not None
    # Baseline has priority 0, which is the lowest (best)
    assert selected.priority < 99


# ---------------------------------------------------------------------------
# build_scoreboard tests
# ---------------------------------------------------------------------------

def test_build_scoreboard_has_required_keys(tmp_path):
    preprocess = _preprocess_from_png(tmp_path)
    input_spec = _make_minimal_input_spec()
    records = run_candidate_pool(preprocess, input_spec)
    select_best_candidate(records)
    scoreboard = build_scoreboard(records)
    for key in ("total", "enabled", "compiled", "selected_id", "candidates"):
        assert key in scoreboard, f"Missing key: {key}"


def test_build_scoreboard_candidates_list_length(tmp_path):
    preprocess = _preprocess_from_png(tmp_path)
    input_spec = _make_minimal_input_spec()
    records = run_candidate_pool(preprocess, input_spec)
    select_best_candidate(records)
    scoreboard = build_scoreboard(records)
    assert len(scoreboard["candidates"]) == scoreboard["total"]


def test_build_scoreboard_selected_id_matches(tmp_path):
    preprocess = _preprocess_from_png(tmp_path)
    input_spec = _make_minimal_input_spec()
    records = run_candidate_pool(preprocess, input_spec)
    selected = select_best_candidate(records)
    scoreboard = build_scoreboard(records)
    assert scoreboard["selected_id"] == (selected.id if selected else None)


def test_build_scoreboard_compiled_count(tmp_path):
    preprocess = _preprocess_from_png(tmp_path)
    input_spec = _make_minimal_input_spec()
    records = run_candidate_pool(preprocess, input_spec)
    scoreboard = build_scoreboard(records)
    expected = sum(1 for r in records if r.compile_success)
    assert scoreboard["compiled"] == expected


def test_build_scoreboard_marks_empty_glsl_not_previewable():
    record = CandidateRecord(
        id="llm_0",
        source="llm",
        enabled=True,
        priority=3,
        dsl=None,
        output_kind="glsl",
        validation_valid=True,
        validation_errors=[],
        compile_success=True,
        compile_glsl="",
        compile_errors=[],
        final_score=0.0,
        selected=True,
    )

    scoreboard = build_scoreboard([record])

    entry = scoreboard["candidates"][0]
    assert entry["compile_success"] is True
    assert entry["previewable"] is False


# ---------------------------------------------------------------------------
# run_png_shader_pipeline tests
# ---------------------------------------------------------------------------

def test_run_pipeline_returns_run_id(tmp_path):
    png_path = make_solid_png(tmp_path)
    result = run_png_shader_pipeline(png_path)
    assert "run_id" in result
    assert isinstance(result["run_id"], str)
    assert result["run_id"].startswith("run_")


def test_run_pipeline_has_scoreboard(tmp_path):
    png_path = make_solid_png(tmp_path)
    result = run_png_shader_pipeline(png_path)
    assert "scoreboard" in result
    assert "selected_id" in result["scoreboard"]
    assert result["scoreboard"]["compiled"] >= 1


def test_run_pipeline_has_selected_glsl(tmp_path):
    png_path = make_solid_png(tmp_path)
    result = run_png_shader_pipeline(png_path)
    assert "selected_glsl" in result
    assert isinstance(result["selected_glsl"], str)
    assert len(result["selected_glsl"]) > 0


def test_run_pipeline_has_preprocess(tmp_path):
    png_path = make_solid_png(tmp_path)
    result = run_png_shader_pipeline(png_path)
    assert "preprocess" in result
    assert "palette" in result["preprocess"]


def test_run_pipeline_has_quality_router_and_metrics(tmp_path):
    png_path = make_solid_png(tmp_path)
    result = run_png_shader_pipeline(png_path)
    assert result["quality_router"] is not None
    assert 0.0 <= result["quality_router"]["final_score"] <= 1.0
    assert "mse" in result["objective_metrics"]


def test_run_pipeline_writes_artifacts(tmp_path):
    png_path = make_solid_png(tmp_path)
    result = run_png_shader_pipeline(png_path)
    run_dir = Path(result["run_dir"])
    assert run_dir.is_dir()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "scoreboard.json").exists()
    assert (run_dir / "selected_shader.glsl").exists()


def test_run_pipeline_selected_candidate_id_in_scoreboard(tmp_path):
    png_path = make_solid_png(tmp_path)
    result = run_png_shader_pipeline(png_path)
    selected_id = result["selected_candidate_id"]
    scoreboard_ids = [c["id"] for c in result["scoreboard"]["candidates"]]
    assert selected_id in scoreboard_ids


def test_llm_candidate_not_called_by_default(tmp_path):
    """With llm_enabled=False (default), no 'llm' source should appear in the pool."""
    preprocess = _preprocess_from_png(tmp_path)
    input_spec = _make_minimal_input_spec()
    records = run_candidate_pool(preprocess, input_spec, llm_enabled=False)
    llm_records = [r for r in records if r.source == "llm"]
    assert len(llm_records) == 0


def test_run_candidate_pool_accepts_glsl_llm_candidate(tmp_path, monkeypatch):
    preprocess = _preprocess_from_png(tmp_path)
    input_spec = _make_minimal_input_spec()

    def fake_llm_candidate(*args, **kwargs):
        return {
            "_meta": {
                "source": "llm",
                "priority": 3,
                "output_kind": "glsl",
                "implementation": "shadertoy_glsl",
            },
            "glsl": "void mainImage(out vec4 fragColor, in vec2 fragCoord) { fragColor = vec4(1.0); }",
        }

    monkeypatch.setattr(
        "app.pipeline.pool.generate_llm_scene_candidate",
        fake_llm_candidate,
    )

    records = run_candidate_pool(
        preprocess,
        input_spec,
        llm_enabled=True,
        llm_implementation="shadertoy_glsl",
        cv_enabled=False,
    )
    llm = next((r for r in records if r.source == "llm"), None)

    assert llm is not None
    assert llm.output_kind == "glsl"
    assert llm.dsl is None
    assert llm.compile_success is True
    selected = select_best_candidate(records, prefer_output_kind="glsl")
    assert selected is llm


def test_run_pipeline_auto_scores_glsl_llm_candidate_from_input_spec(tmp_path, monkeypatch):
    png_path = make_solid_png(tmp_path, color=(255, 255, 255, 255))
    input_spec = _make_minimal_input_spec()
    input_spec["candidates"] = {
        "llm_enabled": True,
        "llm_implementation": "shadertoy_glsl",
        "cv_enabled": False,
    }

    def fake_llm_candidate(*args, **kwargs):
        return {
            "_meta": {
                "source": "llm",
                "priority": 3,
                "output_kind": "glsl",
                "implementation": "shadertoy_glsl",
            },
            "glsl": "void mainImage(out vec4 fragColor, in vec2 fragCoord) { fragColor = vec4(1.0); }",
        }

    def fake_render_multiple_frames(*args, **kwargs):
        from PIL import Image

        render_path = tmp_path / "fake_auto_webgl_render.png"
        Image.new("RGBA", (64, 64), (255, 255, 255, 255)).save(render_path)
        return [str(render_path)]

    monkeypatch.setattr(
        "app.pipeline.pool.generate_llm_scene_candidate",
        fake_llm_candidate,
    )
    monkeypatch.setattr("app.pipeline.scoring.render_multiple_frames", fake_render_multiple_frames)

    result = run_png_shader_pipeline(png_path, input_spec=input_spec)

    llm_detail = next(
        c for c in result["candidate_details"] if c["id"] == "llm_0"
    )
    assert llm_detail["output_kind"] == "glsl"
    assert llm_detail["objective_metrics"]["backend_rasterized"] is True
    assert llm_detail["objective_metrics"]["render_backend"] == "webgl"
    assert llm_detail["final_score"] > 0.0

    scoreboard_detail = next(
        c for c in result["scoreboard"]["candidates"] if c["id"] == "llm_0"
    )
    assert scoreboard_detail["quality_router"]["final_score"] == llm_detail["quality_router"]["final_score"]


def test_run_pipeline_can_webgl_score_glsl_llm_candidate(tmp_path, monkeypatch):
    png_path = make_solid_png(tmp_path, color=(255, 255, 255, 255))
    input_spec = _make_minimal_input_spec()
    input_spec["candidates"] = {
        "llm_enabled": True,
        "llm_implementation": "shadertoy_glsl",
        "cv_enabled": False,
        "glsl_render_enabled": True,
    }

    def fake_llm_candidate(*args, **kwargs):
        return {
            "_meta": {
                "source": "llm",
                "priority": 3,
                "output_kind": "glsl",
                "implementation": "shadertoy_glsl",
            },
            "glsl": """
                #define core_radius 0.3
                void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                    fragColor = vec4(vec3(1.0), 1.0);
                }
            """,
            "glsl_metadata": {"tunable_parameters": [{"name": "core_radius", "value": 0.3}]},
        }

    def fake_render_multiple_frames(*args, **kwargs):
        from PIL import Image

        render_path = tmp_path / "fake_webgl_render.png"
        Image.new("RGBA", (64, 64), (255, 255, 255, 255)).save(render_path)
        return [str(render_path)]

    monkeypatch.setattr("app.pipeline.pool.generate_llm_scene_candidate", fake_llm_candidate)
    monkeypatch.setattr("app.pipeline.scoring.render_multiple_frames", fake_render_multiple_frames)

    result = run_png_shader_pipeline(png_path, input_spec=input_spec)

    selected_detail = next(
        c for c in result["candidate_details"] if c["id"] == result["selected_candidate_id"]
    )
    assert selected_detail["id"] == "llm_0"
    assert selected_detail["objective_metrics"]["backend_rasterized"] is True
    assert selected_detail["objective_metrics"]["render_backend"] == "webgl"
    assert selected_detail["final_score"] > 0.0


def test_refinement_loop_records_llm_call_exception(tmp_path, monkeypatch):
    png_path = make_solid_png(tmp_path)
    preprocess = _preprocess_from_png(tmp_path)
    initial_dsl = {
        "schema_version": 1,
        "canvas": {"width": 64, "height": 64, "background": "#000000"},
        "layers": [
            {
                "id": "circle_01",
                "type": "circle",
                "fill": {"type": "solid", "color": "#ffffff"},
                "params": {"center": [0.5, 0.5], "radius": 0.3},
                "opacity": 1.0,
            }
        ],
    }

    def fake_refinement(**kwargs):
        raise TimeoutError("upstream gateway timed out after 50s")

    monkeypatch.setattr(
        "app.candidates.llm_scene.generate_llm_refinement",
        fake_refinement,
    )

    loop_dir = tmp_path / "refinement"
    result = run_dsl_refinement_loop(
        preprocess=preprocess,
        initial_dsl=initial_dsl,
        initial_score=0.2,
        initial_metrics={"mse": 0.2},
        initial_quality={"final_score": 0.2, "quality_band": "poor"},
        reference_path=png_path,
        canvas_width=64,
        canvas_height=64,
        max_shader_chars=12000,
        max_iterations=1,
        threshold=0.8,
        high_score_stop=0.92,
        loop_dir=loop_dir,
    )

    assert result["stop_reason"] == "llm_call_failed"
    entry = result["history"][0]
    assert entry["error_type"] == "TimeoutError"
    assert "upstream gateway timed out after 50s" in entry["error"]
    assert isinstance(entry["llm_duration_ms"], int)
    assert (loop_dir / "iter_1.json").exists()


def test_run_pipeline_syncs_refined_llm_candidate_glsl_into_scoreboard(tmp_path, monkeypatch):
    png_path = make_solid_png(tmp_path)
    input_spec = _make_minimal_input_spec()
    input_spec["candidates"] = {
        "llm_enabled": True,
        "llm_implementation": "png_dsl",
        "cv_enabled": False,
    }
    input_spec["quality"] = {
        **input_spec["quality"],
        "max_iterations": 0,
        "refinement_mode": "on",
        "max_refinement_iterations": 1,
    }

    def fake_llm_candidate(*args, **kwargs):
        return {
            "_meta": {"source": "llm", "priority": 0, "output_kind": "dsl"},
            "schema_version": 1,
            "canvas": {"width": 64, "height": 64, "background": "#000000"},
            "layers": [
                {
                    "id": "ai_circle",
                    "type": "circle",
                    "fill": {"type": "solid", "color": "#ffffff"},
                    "params": {"center": [0.5, 0.5], "radius": 0.25},
                    "opacity": 1.0,
                }
            ],
        }

    def fake_score_candidates(candidates, *args, **kwargs):
        for candidate in candidates:
            if not candidate.compile_success:
                continue
            candidate.objective_metrics = {"mse": 0.4, "simple_ssim": 0.2}
            candidate.quality_router = {
                "status": "failed",
                "quality_band": "poor",
                "next_action": "manual_review",
                "final_score": 0.2 if candidate.source == "llm" else 0.1,
                "failure_type": "color",
                "reason": ["needs improvement"],
                "protected_aspects": [],
            }
            candidate.final_score = candidate.quality_router["final_score"]

    refined_glsl = "void mainImage(out vec4 fragColor, in vec2 fragCoord) { fragColor = vec4(0.2, 0.4, 1.0, 1.0); }"

    def fake_refinement_loop(**kwargs):
        return {
            "best_dsl": kwargs["initial_dsl"],
            "best_glsl": refined_glsl,
            "best_score": 0.65,
            "best_metrics": {"mse": 0.1, "simple_ssim": 0.8},
            "best_quality": {
                "status": "preview",
                "quality_band": "acceptable",
                "next_action": "accept",
                "final_score": 0.65,
                "failure_type": "none",
                "reason": [],
                "protected_aspects": [],
            },
            "history": [{"iteration": 1, "score_before": 0.2, "score_after": 0.65}],
            "stop_reason": "max_iterations",
        }

    monkeypatch.setattr("app.pipeline.pool.generate_llm_scene_candidate", fake_llm_candidate)
    monkeypatch.setattr("app.pipeline.graph._score_candidates", fake_score_candidates)
    monkeypatch.setattr("app.pipeline.graph.run_dsl_refinement_loop", fake_refinement_loop)

    result = run_png_shader_pipeline(png_path, input_spec=input_spec)

    assert result["selected_candidate_id"] == "llm_0"
    # After refinement is accepted, the pipeline re-compiles best_dsl so
    # selected_glsl is guaranteed in-sync with selected_dsl. The exact string
    # comes from the compiler, not from refinement's reported best_glsl.
    selected_glsl = result["selected_glsl"]
    assert isinstance(selected_glsl, str) and selected_glsl.strip()
    assert "void main" in selected_glsl
    selected_entry = next(c for c in result["scoreboard"]["candidates"] if c["id"] == "llm_0")
    assert selected_entry["compile_glsl"] == selected_glsl
    assert selected_entry["previewable"] is True


def test_run_pipeline_accepts_custom_run_id(tmp_path):
    png_path = make_solid_png(tmp_path)
    result = run_png_shader_pipeline(png_path, run_id="my_custom_run")
    assert result["run_id"] == "my_custom_run"


def test_run_pipeline_different_image_colors(tmp_path):
    """Pipeline should work for various image colors."""
    for color in [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255), (128, 128, 128, 255)]:
        png_path = make_solid_png(tmp_path, name=f"test_{color[0]}_{color[1]}_{color[2]}.png", color=color)
        result = run_png_shader_pipeline(png_path)
        assert result["selected_glsl"] is not None
        assert len(result["selected_glsl"]) > 0


def test_pipeline_threads_protected_aspects_to_quality_router(tmp_path):
    from PIL import Image
    from app.pipeline.input_spec import build_input_spec

    img_path = tmp_path / "in.png"
    Image.new("RGBA", (32, 32), (200, 50, 20, 255)).save(img_path)

    spec = build_input_spec(
        img_path,
        quality={"protected_aspects": ["layer_count", "visual_causality"]},
    )
    result = run_png_shader_pipeline(img_path, spec, run_id="prot_aspects_run")
    qr = result["quality_router"]
    assert qr is not None
    assert qr["protected_aspects"] == ["layer_count", "visual_causality"]


def test_pipeline_force_failure_type_overrides_router(tmp_path, monkeypatch):
    from dataclasses import replace as _dataclass_replace
    from PIL import Image
    from app.pipeline import graph as graph_module
    from app.pipeline import scoring as scoring_module
    from app.pipeline import quality_router as qr_module
    from app.pipeline.input_spec import build_input_spec

    img_path = tmp_path / "in.png"
    Image.new("RGBA", (32, 32), (200, 50, 20, 255)).save(img_path)

    # Force the router to always return "revise" so the override path is exercised.
    real_route = qr_module.route

    def forced_route(observed, metrics, *, protected_aspects=None):
        result = real_route(observed, metrics, protected_aspects=protected_aspects)
        return _dataclass_replace(result, next_action="revise", failure_type="parameter")

    monkeypatch.setattr(scoring_module, "route", forced_route)

    captured = {}
    real_build_revision_patch = graph_module._build_revision_patch

    def spy_build_revision_patch(dsl, preprocess, failure_type, **kwargs):
        captured["failure_type"] = failure_type
        return real_build_revision_patch(dsl, preprocess, failure_type, **kwargs)

    monkeypatch.setattr(graph_module, "_build_revision_patch", spy_build_revision_patch)

    spec = build_input_spec(
        img_path,
        quality={"force_failure_type": "color"},
    )
    run_png_shader_pipeline(img_path, spec, run_id="force_fft_run")

    assert "failure_type" in captured, "revision branch was not reached"
    assert captured["failure_type"] == "color"


def test_pipeline_accepts_strategy_reader_kwarg(tmp_path):
    """Smoke test: pipeline should accept strategy_reader kwarg without crashing."""
    from PIL import Image
    from app.pipeline.input_spec import build_input_spec

    img_path = tmp_path / "in.png"
    Image.new("RGBA", (32, 32), (200, 50, 20, 255)).save(img_path)

    def reader():
        return {"strategy": {}, "stop_requested": False}

    spec = build_input_spec(img_path)
    result = run_png_shader_pipeline(
        img_path, spec, run_id="strategy_reader_smoke", strategy_reader=reader
    )
    assert result["run_id"] == "strategy_reader_smoke"
    assert result["selected_glsl"]


def test_pipeline_manifest_includes_strategy_fields(tmp_path):
    import json as _json
    from PIL import Image
    from app.pipeline.input_spec import build_input_spec

    img_path = tmp_path / "in.png"
    Image.new("RGBA", (32, 32), (200, 50, 20, 255)).save(img_path)

    spec = build_input_spec(
        img_path,
        quality={
            "mode": "aggressive",
            "force_failure_type": "color",
            "protected_aspects": ["layer_count", "visual_causality"],
        },
    )
    result = run_png_shader_pipeline(img_path, spec, run_id="manifest_strategy_run")
    manifest_path = Path(result["run_dir"]) / "manifest.json"
    assert manifest_path.exists()
    manifest = _json.loads(manifest_path.read_text())
    cfg = manifest.get("config", {})
    assert cfg.get("force_failure_type") == "color"
    assert cfg.get("protected_aspects") == ["layer_count", "visual_causality"]
    assert cfg.get("quality_mode") == "aggressive"
