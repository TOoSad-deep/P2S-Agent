"""Unit tests for the PNG-to-Shader candidate pool orchestrator (graph.py).

No LLM, no browser. Uses synthetic PNGs created with Pillow.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.pipeline.graph import (
    CandidateRecord,
    _run_post_pipeline,
    _should_run_refinement,
    build_scoreboard,
    node_selection,
    run_candidate_pool,
    run_dsl_refinement_loop,
    run_png_shader_pipeline,
    select_best_candidate,
)
from app.pipeline.input_spec import build_input_spec
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


def test_should_run_refinement_allows_glsl_candidate():
    """GLSL candidates with compiled shader text are now refinable."""
    candidate = _make_refinement_candidate(score=0.3, has_dsl=False)

    should_run, reason = _should_run_refinement(
        "on",
        candidate,
        {"final_score": 0.3},
        threshold=0.8,
        high_score_stop=0.92,
    )

    assert should_run is True
    assert reason == "force_enabled"


def test_should_run_refinement_skips_unrefinable_candidate():
    candidate = _make_refinement_candidate(score=0.3, has_dsl=False)
    candidate.compile_glsl = ""
    candidate.compile_success = False

    should_run, reason = _should_run_refinement(
        "on",
        candidate,
        {"final_score": 0.3},
        threshold=0.8,
        high_score_stop=0.92,
    )

    assert should_run is False
    assert reason == "selected_candidate_not_refinable"


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


def test_node_selection_prefers_glsl_even_when_refinement_requested():
    glsl_cand = CandidateRecord(
        id="llm_0",
        source="llm",
        enabled=True,
        priority=5,
        dsl=None,
        output_kind="glsl",
        validation_valid=True,
        validation_errors=[],
        compile_success=True,
        compile_glsl="void mainImage(out vec4 fragColor, in vec2 fragCoord){fragColor=vec4(1.0);}",
        compile_errors=[],
        final_score=0.7,
        selected=False,
    )
    dsl_cand = CandidateRecord(
        id="rule_0",
        source="rule",
        enabled=True,
        priority=1,
        dsl={"layers": []},
        output_kind="dsl",
        validation_valid=True,
        validation_errors=[],
        compile_success=True,
        compile_glsl="void mainImage(out vec4 fragColor, in vec2 fragCoord){fragColor=vec4(0.0);}",
        compile_errors=[],
        final_score=0.7,
        selected=False,
    )

    result = node_selection({
        "candidates": [dsl_cand, glsl_cand],
        "glsl_render_enabled": True,
        "refinement_mode": "on",
        "max_refinement_iterations": 3,
    })

    assert result["selected_candidate_id"] == "llm_0"


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
    assert entry["score_status"] == "pending"


def test_run_candidate_pool_passes_model_reference_to_llm(tmp_path, monkeypatch):
    preprocess = _preprocess_from_png(tmp_path)
    input_spec = _make_minimal_input_spec()
    source_path = tmp_path / "source.png"
    llm_path = tmp_path / "llm_reference_input.png"
    source_path.write_bytes(b"source")
    llm_path.write_bytes(b"llm")
    captured = {}

    def fake_llm_candidate(*args, **kwargs):
        captured["image_path"] = kwargs.get("image_path")
        return None

    monkeypatch.setattr("app.pipeline.pool.generate_llm_scene_candidate", fake_llm_candidate)

    records = run_candidate_pool(
        preprocess,
        input_spec,
        image_path=source_path,
        llm_image_path=llm_path,
        llm_enabled=True,
        llm_implementation="shadertoy_glsl",
        cv_enabled=False,
    )

    assert captured["image_path"] == llm_path
    llm = next(record for record in records if record.id == "llm_0")
    assert llm.enabled is False
    assert llm.output_kind == "glsl"


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


def test_run_post_pipeline_runs_glsl_refinement(tmp_path, monkeypatch):
    glsl = "void mainImage(out vec4 fragColor, in vec2 fragCoord){fragColor=vec4(1.0);}"
    cand = CandidateRecord(
        id="llm_0",
        source="llm",
        enabled=True,
        priority=5,
        dsl=None,
        output_kind="glsl",
        validation_valid=True,
        validation_errors=[],
        compile_success=True,
        compile_glsl=glsl,
        compile_errors=[],
        final_score=0.4,
        selected=True,
        objective_metrics={"mse": 0.5},
        quality_router={"final_score": 0.4, "next_action": "accept"},
    )
    improved_glsl = "#define R 0.9\n" + glsl
    improved_render = tmp_path / "glsl_refinement" / "renders" / "iter_1.png"

    def fake_loop(*args, **kwargs):
        return {
            "best_glsl": improved_glsl,
            "best_score": 0.8,
            "best_metrics": {"mse": 0.1},
            "best_quality": {"final_score": 0.8, "next_action": "accept"},
            "best_render_path": str(improved_render),
            "history": [{"iteration": 1, "score_after": 0.8, "improved": True}],
            "stop_reason": "threshold_reached",
        }

    monkeypatch.setattr("app.pipeline.graph.run_glsl_refinement_loop", fake_loop)

    result = _run_post_pipeline({
        "selected_candidate_id": "llm_0",
        "candidates": [cand],
        "run_dir": str(tmp_path),
        "selected_dsl": None,
        "selected_glsl": glsl,
        "selected_metrics": {"mse": 0.5},
        "selected_quality": {"final_score": 0.4, "next_action": "accept"},
        "refinement_mode": "on",
        "max_refinement_iterations": 2,
        "llm_enabled": False,
        "glsl_render_enabled": True,
        "vlm_judge_enabled": False,
    })

    assert result["selected_glsl"] == improved_glsl
    assert result["refinement_summary"]["enabled"] is True
    assert result["refinement_summary"]["decision"] == "force_enabled"
    assert result["refinement_summary"]["improved"] is True
    assert result["refinement_summary"]["stop_reason"] == "threshold_reached"
    assert cand.final_score == 0.8
    assert cand.compile_glsl == improved_glsl
    assert cand.render_path == str(improved_render)
    assert (tmp_path / "selected_shader.glsl").read_text(encoding="utf-8") == improved_glsl


def test_run_post_pipeline_skips_glsl_refinement_when_render_disabled(tmp_path, monkeypatch):
    glsl = "void mainImage(out vec4 fragColor, in vec2 fragCoord){fragColor=vec4(1.0);}"
    cand = CandidateRecord(
        id="llm_0",
        source="llm",
        enabled=True,
        priority=5,
        dsl=None,
        output_kind="glsl",
        validation_valid=True,
        validation_errors=[],
        compile_success=True,
        compile_glsl=glsl,
        compile_errors=[],
        final_score=0.4,
        selected=True,
        quality_router={"final_score": 0.4, "next_action": "accept"},
    )

    def fail_loop(*args, **kwargs):
        raise AssertionError("refinement loop must not run without the renderer")

    monkeypatch.setattr("app.pipeline.graph.run_glsl_refinement_loop", fail_loop)

    result = _run_post_pipeline({
        "selected_candidate_id": "llm_0",
        "candidates": [cand],
        "run_dir": str(tmp_path),
        "selected_dsl": None,
        "selected_glsl": glsl,
        "selected_metrics": {},
        "selected_quality": {"final_score": 0.4, "next_action": "accept"},
        "refinement_mode": "on",
        "max_refinement_iterations": 2,
        "llm_enabled": False,
        "glsl_render_enabled": False,
        "vlm_judge_enabled": False,
    })

    assert result["selected_glsl"] == glsl
    assert result["refinement_summary"]["decision"] == "glsl_render_disabled"
    assert result["refinement_summary"]["enabled"] is False


def test_dsl_refinement_feeds_history_and_semantic_notes(tmp_path, monkeypatch):
    from types import SimpleNamespace

    captured: list[dict] = []

    def fake_refinement(**kwargs):
        captured.append(kwargs)
        return {"layers": [], "_io": {}}

    monkeypatch.setattr(
        "app.candidates.llm_scene.generate_llm_refinement",
        fake_refinement,
    )
    monkeypatch.setattr(
        "app.pipeline.refinement.render_dsl_to_image",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "app.dsl.validator.validate_dsl",
        lambda d: SimpleNamespace(valid=True, errors=[]),
    )
    monkeypatch.setattr(
        "app.dsl.compiler.compile_dsl",
        lambda d: SimpleNamespace(success=True, glsl="void mainImage(){}", errors=[]),
    )
    monkeypatch.setattr(
        "app.pipeline.refinement._evaluate_dsl",
        lambda *a, **k: ({}, {"final_score": 0.2}, 0.2, None),
    )

    run_dsl_refinement_loop(
        preprocess={},
        initial_dsl={"layers": [{"id": "a", "type": "circle", "params": {}}]},
        initial_score=0.5,
        initial_metrics={},
        initial_quality={"final_score": 0.5},
        reference_path=tmp_path / "ref.png",
        canvas_width=512,
        canvas_height=512,
        max_shader_chars=12000,
        max_iterations=2,
        threshold=0.9,
        high_score_stop=0.95,
        no_improvement_patience=3,
        loop_dir=tmp_path / "loop",
        rubric_judge=lambda render: {
            "differences": ["edges too sharp"],
            "revision_hints": ["soften the edges"],
        },
    )

    assert len(captured) == 2
    second_feedback = captured[1]["extra_feedback"]
    assert any("[HISTORY iter 1]" in n for n in second_feedback)
    assert any("[VISUAL ISSUE] edges too sharp" in n for n in second_feedback)
    assert any("[ROLLBACK]" in n for n in second_feedback)


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
    from app.metrics import quality_router as qr_module
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


def test_pipeline_threads_strategy_reader_to_refinement_loop(tmp_path, monkeypatch):
    from PIL import Image
    from app.pipeline.input_spec import build_input_spec

    img_path = tmp_path / "in.png"
    Image.new("RGBA", (32, 32), (200, 50, 20, 255)).save(img_path)

    spec = build_input_spec(
        img_path,
        candidates={
            "llm_enabled": True,
            "llm_implementation": "png_dsl",
            "cv_enabled": False,
        },
        quality={
            "max_iterations": 0,
            "refinement_mode": "on",
            "max_refinement_iterations": 1,
            "refinement_threshold": 0.9,
        },
    )

    def fake_llm_candidate(*args, **kwargs):
        return {
            "_meta": {"source": "llm", "priority": 0, "output_kind": "dsl"},
            "schema_version": 1,
            "canvas": {"width": 32, "height": 32, "background": "#000000"},
            "layers": [],
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

    captured = {}

    def fake_refinement_loop(**kwargs):
        captured["strategy_reader"] = kwargs.get("strategy_reader")
        return {
            "best_dsl": kwargs["initial_dsl"],
            "best_glsl": "",
            "best_score": kwargs["initial_score"],
            "best_metrics": kwargs["initial_metrics"],
            "best_quality": kwargs["initial_quality"],
            "history": [],
            "stop_reason": "user_stop",
        }

    def reader():
        return {"strategy": {}, "stop_requested": True}

    monkeypatch.setattr("app.pipeline.pool.generate_llm_scene_candidate", fake_llm_candidate)
    monkeypatch.setattr("app.pipeline.graph._score_candidates", fake_score_candidates)
    monkeypatch.setattr("app.pipeline.graph.run_dsl_refinement_loop", fake_refinement_loop)

    run_png_shader_pipeline(img_path, spec, run_id="strategy_reader_refinement", strategy_reader=reader)

    assert captured["strategy_reader"] is reader


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


def test_seed_glsl_pipeline_skips_pool_and_refines(tmp_path, monkeypatch):
    """A seed_glsl run must build one 'seed' candidate (no pool generation)
    and drive it through run_glsl_refinement_loop."""
    import app.pipeline.graph as graph_mod

    png = make_solid_png(tmp_path, color=(120, 60, 30, 255))

    seed = (
        "#define R 0.30\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) { fragColor = vec4(R); }"
    )
    improved = seed.replace("0.30", "0.50")

    def fake_eval(glsl, ref_path, output_path, *, canvas_width, canvas_height, max_shader_chars):
        score = 0.30
        for line in glsl.splitlines():
            if line.startswith("#define R"):
                score = float(line.split()[-1])
        return ({"mse": 1.0 - score}, {"final_score": score, "next_action": "refine"}, score, output_path)

    def fake_refine(**kwargs):
        return {"glsl": improved, "_io": {"mode": "glsl_refinement"}}

    monkeypatch.setattr(graph_mod, "_evaluate_glsl_with_webgl", fake_eval)
    monkeypatch.setattr(
        "app.candidates.llm_scene.generate_llm_glsl_refinement", fake_refine
    )

    spec = build_input_spec(
        png,
        quality={"refinement_mode": "on", "max_refinement_iterations": 2, "refinement_patience": 1},
        candidates={"glsl_render_enabled": True},
    )

    result = run_png_shader_pipeline(png, spec, run_id="seedtest", seed_glsl=seed)

    details = result["candidate_details"]
    assert len(details) == 1
    assert details[0]["source"] == "seed"
    assert result["refinement_summary"]["enabled"] is True
    assert "0.50" in (result["selected_glsl"] or "")


def test_seed_glsl_invalid_raises(tmp_path, monkeypatch):
    """A seed that cannot be adapted (and no LLM client) must raise so the
    background worker marks the run failed."""
    png = make_solid_png(tmp_path)
    monkeypatch.setattr(
        "app.candidates.llm_scene.generate_llm_glsl_refinement",
        lambda **kwargs: None,
    )
    spec = build_input_spec(png, candidates={"glsl_render_enabled": True})

    with pytest.raises(ValueError, match="seed GLSL"):
        run_png_shader_pipeline(
            png, spec, run_id="seedbad", seed_glsl="float helper(){ return 1.0; }"
        )


def test_dsl_loop_invokes_on_iteration_each_iteration(tmp_path, monkeypatch):
    from types import SimpleNamespace

    def fake_refinement(**kwargs):
        return {"layers": [], "_io": {}}

    monkeypatch.setattr(
        "app.candidates.llm_scene.generate_llm_refinement", fake_refinement
    )
    monkeypatch.setattr(
        "app.pipeline.refinement.render_dsl_to_image", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "app.dsl.validator.validate_dsl",
        lambda d: SimpleNamespace(valid=True, errors=[]),
    )
    monkeypatch.setattr(
        "app.dsl.compiler.compile_dsl",
        lambda d: SimpleNamespace(success=True, glsl="void mainImage(){}", errors=[]),
    )
    scores = iter([0.6, 0.7])
    monkeypatch.setattr(
        "app.pipeline.refinement._evaluate_dsl",
        lambda *a, **k: ({}, {"final_score": 0.6}, next(scores), None),
    )

    snaps: list[int] = []
    run_dsl_refinement_loop(
        preprocess={},
        initial_dsl={"layers": [{"id": "a", "type": "circle", "params": {}}]},
        initial_score=0.5,
        initial_metrics={},
        initial_quality={"final_score": 0.5},
        reference_path=tmp_path / "ref.png",
        canvas_width=64,
        canvas_height=64,
        max_shader_chars=12000,
        max_iterations=2,
        threshold=0.99,
        high_score_stop=0.999,
        min_improvement=0.001,
        no_improvement_patience=5,
        loop_dir=tmp_path / "loop",
        on_iteration=lambda s: snaps.append(len(s["history"])),
    )

    assert snaps == [1, 2]


def test_run_post_pipeline_publishes_baseline_and_iterations(tmp_path, monkeypatch):
    glsl = "void mainImage(out vec4 fragColor, in vec2 fragCoord){fragColor=vec4(1.0);}"
    cand = CandidateRecord(
        id="llm_0",
        source="llm",
        enabled=True,
        priority=5,
        dsl=None,
        output_kind="glsl",
        validation_valid=True,
        validation_errors=[],
        compile_success=True,
        compile_glsl=glsl,
        compile_errors=[],
        final_score=0.4,
        selected=True,
        objective_metrics={"mse": 0.5},
        quality_router={"final_score": 0.4, "next_action": "accept"},
    )
    improved_glsl = "#define R 0.9\n" + glsl

    def fake_loop(*args, **kwargs):
        on_iteration = kwargs.get("on_iteration")
        if on_iteration is not None:
            on_iteration({
                "best_glsl": improved_glsl,
                "best_score": 0.8,
                "best_metrics": {"mse": 0.1},
                "best_quality": {"final_score": 0.8, "next_action": "accept"},
                "history": [{"iteration": 1, "score_after": 0.8, "improved": True}],
            })
        return {
            "best_glsl": improved_glsl,
            "best_score": 0.8,
            "best_metrics": {"mse": 0.1},
            "best_quality": {"final_score": 0.8, "next_action": "accept"},
            "best_render_path": None,
            "history": [{"iteration": 1, "score_after": 0.8, "improved": True}],
            "stop_reason": "threshold_reached",
        }

    monkeypatch.setattr("app.pipeline.graph.run_glsl_refinement_loop", fake_loop)

    published: list[dict] = []
    result = _run_post_pipeline(
        {
            "selected_candidate_id": "llm_0",
            "candidates": [cand],
            "run_dir": str(tmp_path),
            "preprocess": {"palette": ["#ffffff"]},
            "selected_dsl": None,
            "selected_glsl": glsl,
            "selected_metrics": {"mse": 0.5},
            "selected_quality": {"final_score": 0.4, "next_action": "accept"},
            "refinement_mode": "on",
            "max_refinement_iterations": 2,
            "llm_enabled": False,
            "glsl_render_enabled": True,
            "vlm_judge_enabled": False,
        },
        publish_partial=lambda p: published.append(p),
    )

    assert any("scoreboard" in p for p in published)
    iter_partials = [p for p in published if "refinement_history" in p]
    assert iter_partials, "expected at least one per-iteration partial"
    last = iter_partials[-1]
    assert last["refinement_history"][0]["iteration"] == 1
    assert last["selected_glsl"] == improved_glsl
    assert last["refinement_summary"]["iterations"] == 1
    assert last["refinement_summary"]["enabled"] is True
    assert result["selected_glsl"] == improved_glsl


def test_pipeline_threads_publish_partial_baseline(tmp_path):
    png = make_solid_png(tmp_path)
    seen: list[dict] = []
    run_png_shader_pipeline(
        png, run_id="pub_smoke", publish_partial=lambda p: seen.append(p)
    )
    assert any("scoreboard" in p for p in seen)
    # Both baseline publishes fire on a normal run; guards the threading too.
    assert len(seen) >= 2


def test_seed_pipeline_publishes_iterations(tmp_path, monkeypatch):
    import app.pipeline.graph as graph_mod

    png = make_solid_png(tmp_path, color=(120, 60, 30, 255))
    seed = (
        "#define R 0.30\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) { fragColor = vec4(R); }"
    )
    improved = seed.replace("0.30", "0.50")

    def fake_eval(glsl, ref_path, output_path, *, canvas_width, canvas_height, max_shader_chars):
        score = 0.30
        for line in glsl.splitlines():
            if line.startswith("#define R"):
                score = float(line.split()[-1])
        return ({"mse": 1.0 - score}, {"final_score": score, "next_action": "refine"}, score, output_path)

    def fake_refine(**kwargs):
        return {"glsl": improved, "_io": {"mode": "glsl_refinement"}}

    monkeypatch.setattr(graph_mod, "_evaluate_glsl_with_webgl", fake_eval)
    monkeypatch.setattr(
        "app.candidates.llm_scene.generate_llm_glsl_refinement", fake_refine
    )

    spec = build_input_spec(
        png,
        quality={"refinement_mode": "on", "max_refinement_iterations": 2, "refinement_patience": 1},
        candidates={"glsl_render_enabled": True},
    )
    seen: list[dict] = []
    result = run_png_shader_pipeline(
        png, spec, run_id="seedpub", seed_glsl=seed, publish_partial=lambda p: seen.append(p)
    )

    assert any("refinement_history" in p for p in seen)
    # Final result converged to the improved shader (not reverted).
    assert "0.50" in (result["selected_glsl"] or "")
