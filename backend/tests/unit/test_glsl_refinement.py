"""Tests for the GLSL LLM refinement loop."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.pipeline.glsl_refinement import _diff_glsl_summary, run_glsl_refinement_loop
from app.pipeline.refinement import build_recent_history_notes, build_semantic_notes

VALID_GLSL_A = (
    "#define R 0.30\n"
    "void mainImage(out vec4 fragColor, in vec2 fragCoord) { fragColor = vec4(R); }"
)
VALID_GLSL_B = (
    "#define R 0.50\n"
    "void mainImage(out vec4 fragColor, in vec2 fragCoord) { fragColor = vec4(R); }"
)


def _evaluate_by_r(glsl: str, render_path: Path):
    for line in glsl.splitlines():
        if line.startswith("#define R"):
            score = float(line.split()[-1])
            return {"mse": 1.0 - score}, {"final_score": score}, score, None
    return {}, {}, 0.0, None


def test_loop_accepts_improvement_and_stops_at_threshold(tmp_path, monkeypatch):
    calls: list[dict] = []

    def fake_refine(**kwargs):
        calls.append(kwargs)
        return {"glsl": VALID_GLSL_B, "_io": {"mode": "glsl_refinement"}}

    monkeypatch.setattr(
        "app.candidates.llm_scene.generate_llm_glsl_refinement", fake_refine
    )

    result = run_glsl_refinement_loop(
        VALID_GLSL_A,
        0.30,
        {"mse": 0.7},
        {"final_score": 0.30},
        tmp_path / "ref.png",
        evaluate_fn=_evaluate_by_r,
        max_iterations=3,
        threshold=0.45,
        high_score_stop=0.92,
        loop_dir=tmp_path / "loop",
    )

    assert result["best_glsl"] == VALID_GLSL_B
    assert result["best_score"] == pytest.approx(0.50)
    assert result["stop_reason"] == "threshold_reached"
    assert len(result["history"]) == 1
    assert result["history"][0]["improved"] is True
    assert calls[0]["fresh_start"] is False
    assert (tmp_path / "loop" / "iter_1.json").exists()


def test_loop_rolls_back_and_feeds_back(tmp_path, monkeypatch):
    worse = VALID_GLSL_A.replace("0.30", "0.10")
    calls: list[dict] = []

    def fake_refine(**kwargs):
        calls.append(kwargs)
        return {"glsl": worse, "_io": {}}

    monkeypatch.setattr(
        "app.candidates.llm_scene.generate_llm_glsl_refinement", fake_refine
    )

    result = run_glsl_refinement_loop(
        VALID_GLSL_A,
        0.30,
        {},
        {"final_score": 0.30},
        tmp_path / "ref.png",
        evaluate_fn=_evaluate_by_r,
        max_iterations=5,
        threshold=0.80,
        high_score_stop=0.92,
        no_improvement_patience=2,
        max_fresh_restarts=0,
        loop_dir=tmp_path / "loop",
    )

    assert result["best_glsl"] == VALID_GLSL_A
    assert result["stop_reason"] == "no_improvement_patience"
    assert len(calls) == 2
    assert any("[ROLLBACK]" in n for n in calls[1]["extra_feedback"])


def test_loop_fresh_restart_after_patience(tmp_path, monkeypatch):
    worse = VALID_GLSL_A.replace("0.30", "0.10")
    calls: list[dict] = []

    def fake_refine(**kwargs):
        calls.append(kwargs)
        return {"glsl": worse, "_io": {}}

    monkeypatch.setattr(
        "app.candidates.llm_scene.generate_llm_glsl_refinement", fake_refine
    )

    result = run_glsl_refinement_loop(
        VALID_GLSL_A,
        0.30,
        {},
        {"final_score": 0.30},
        tmp_path / "ref.png",
        evaluate_fn=_evaluate_by_r,
        max_iterations=6,
        threshold=0.80,
        high_score_stop=0.92,
        no_improvement_patience=2,
        max_fresh_restarts=1,
        loop_dir=tmp_path / "loop",
    )

    assert result["stop_reason"] == "no_improvement_patience"
    assert len(calls) == 4
    assert calls[2]["fresh_start"] is True
    assert any("[FRESH RESTART]" in n for n in calls[2]["extra_feedback"])
    assert calls[3]["fresh_start"] is False


def test_loop_feeds_render_failure_back(tmp_path, monkeypatch):
    calls: list[dict] = []

    def fake_refine(**kwargs):
        calls.append(kwargs)
        return {"glsl": VALID_GLSL_B, "_io": {}}

    monkeypatch.setattr(
        "app.candidates.llm_scene.generate_llm_glsl_refinement", fake_refine
    )

    result = run_glsl_refinement_loop(
        VALID_GLSL_A,
        0.30,
        {},
        {"final_score": 0.30},
        tmp_path / "ref.png",
        evaluate_fn=lambda _g, _p: ({}, {}, 0.0, None),
        max_iterations=4,
        threshold=0.80,
        high_score_stop=0.92,
        no_improvement_patience=2,
        max_fresh_restarts=0,
        loop_dir=tmp_path / "loop",
    )

    assert result["best_glsl"] == VALID_GLSL_A
    assert result["history"][0]["error_type"] == "render_failed"
    assert len(calls) == 2
    assert any("[RENDER FAILED]" in n for n in calls[1]["extra_feedback"])


def test_loop_injects_semantic_feedback_from_rubric(tmp_path, monkeypatch):
    calls: list[dict] = []

    def fake_refine(**kwargs):
        calls.append(kwargs)
        return {"glsl": VALID_GLSL_B, "_io": {}}

    monkeypatch.setattr(
        "app.candidates.llm_scene.generate_llm_glsl_refinement", fake_refine
    )

    run_glsl_refinement_loop(
        VALID_GLSL_A,
        0.30,
        {},
        {"final_score": 0.30},
        tmp_path / "ref.png",
        evaluate_fn=_evaluate_by_r,
        initial_render_path=tmp_path / "current.png",
        rubric_judge=lambda render: {
            "differences": ["edges too sharp"],
            "revision_hints": ["soften the edges"],
        },
        max_iterations=1,
        threshold=0.80,
        high_score_stop=0.92,
        loop_dir=tmp_path / "loop",
    )

    feedback = calls[0]["extra_feedback"]
    assert any("[VISUAL ISSUE] edges too sharp" in n for n in feedback)
    assert any("[VISUAL GOAL] soften the edges" in n for n in feedback)


def test_loop_includes_recent_history_notes(tmp_path, monkeypatch):
    worse = VALID_GLSL_A.replace("0.30", "0.10")
    calls: list[dict] = []

    def fake_refine(**kwargs):
        calls.append(kwargs)
        return {"glsl": worse, "_io": {}}

    monkeypatch.setattr(
        "app.candidates.llm_scene.generate_llm_glsl_refinement", fake_refine
    )

    run_glsl_refinement_loop(
        VALID_GLSL_A,
        0.30,
        {},
        {"final_score": 0.30},
        tmp_path / "ref.png",
        evaluate_fn=_evaluate_by_r,
        max_iterations=2,
        threshold=0.80,
        high_score_stop=0.92,
        no_improvement_patience=3,
        max_fresh_restarts=0,
        loop_dir=tmp_path / "loop",
    )

    assert len(calls) == 2
    assert any("[HISTORY iter 1]" in n for n in calls[1]["extra_feedback"])


def test_loop_skips_invalid_glsl_with_compile_feedback(tmp_path, monkeypatch):
    calls: list[dict] = []

    def fake_refine(**kwargs):
        calls.append(kwargs)
        return {"glsl": "void broken() {", "_io": {}}

    monkeypatch.setattr(
        "app.candidates.llm_scene.generate_llm_glsl_refinement", fake_refine
    )

    result = run_glsl_refinement_loop(
        VALID_GLSL_A,
        0.30,
        {},
        {"final_score": 0.30},
        tmp_path / "ref.png",
        evaluate_fn=_evaluate_by_r,
        max_iterations=5,
        threshold=0.80,
        high_score_stop=0.92,
        no_improvement_patience=2,
        max_fresh_restarts=0,
        loop_dir=tmp_path / "loop",
    )

    assert result["best_glsl"] == VALID_GLSL_A
    assert result["stop_reason"] == "no_improvement_patience"
    assert result["history"][0]["error"].startswith("GLSL invalid")
    assert len(calls) == 2
    assert any("[COMPILE FEEDBACK]" in n for n in calls[1]["extra_feedback"])


def test_loop_stops_when_llm_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.candidates.llm_scene.generate_llm_glsl_refinement",
        lambda **kwargs: None,
    )

    result = run_glsl_refinement_loop(
        VALID_GLSL_A,
        0.30,
        {},
        {"final_score": 0.30},
        tmp_path / "ref.png",
        evaluate_fn=_evaluate_by_r,
        max_iterations=3,
        threshold=0.80,
        high_score_stop=0.92,
        loop_dir=tmp_path / "loop",
    )

    assert result["stop_reason"] == "llm_returned_none"
    assert result["history"][0]["error_type"] == "llm_returned_none"


def test_loop_high_score_stops_without_llm_call(tmp_path, monkeypatch):
    def fail_refine(**kwargs):
        raise AssertionError("LLM must not be called above high_score_stop")

    monkeypatch.setattr(
        "app.candidates.llm_scene.generate_llm_glsl_refinement", fail_refine
    )

    result = run_glsl_refinement_loop(
        VALID_GLSL_A,
        0.95,
        {},
        {"final_score": 0.95},
        tmp_path / "ref.png",
        evaluate_fn=_evaluate_by_r,
        max_iterations=3,
        threshold=0.80,
        high_score_stop=0.92,
        loop_dir=tmp_path / "loop",
    )

    assert result["stop_reason"] == "high_score_stop"
    assert result["history"] == []


def test_diff_glsl_summary_reports_define_changes():
    summary = _diff_glsl_summary(VALID_GLSL_A, VALID_GLSL_B)
    assert "changed lines" in summary
    assert "#define R" in summary
    assert _diff_glsl_summary(VALID_GLSL_A, VALID_GLSL_A) == "no changes"


def test_build_semantic_notes_maps_rubric_fields():
    notes = build_semantic_notes(
        {"differences": ["bg mismatch"], "revision_hints": ["make bg white"]}
    )
    assert notes == ["[VISUAL ISSUE] bg mismatch", "[VISUAL GOAL] make bg white"]


def test_build_recent_history_notes_excludes_code_and_caps_entries():
    history = [
        {
            "iteration": i,
            "score_before": 0.3,
            "score_after": 0.3,
            "improved": False,
            "changes_summary": f"change {i}",
            "error": None,
        }
        for i in range(1, 6)
    ]
    notes = build_recent_history_notes(history, max_entries=3)
    assert len(notes) == 3
    assert "[HISTORY iter 3]" in notes[0]
    assert "[HISTORY iter 5]" in notes[2]
