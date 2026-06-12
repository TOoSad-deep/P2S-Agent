"""Tests for the GLSL #define parameter optimizer."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.pipeline.glsl_optimizer import (
    GlslOptimizeResult,
    build_glsl_optimization_artifacts,
    optimize_glsl_candidate,
    parse_glsl_defines,
    update_glsl_define,
)


# ---------------------------------------------------------------------------
# parse_glsl_defines
# ---------------------------------------------------------------------------


def test_parse_scalar_float_defines():
    glsl = """
#define RADIUS 0.35
#define GLOW_INTENSITY 8.0
#define EDGE_SOFTNESS 0.02
void mainImage(out vec4 fragColor, in vec2 fragCoord) {}
""".strip()
    params = parse_glsl_defines(glsl)
    by_name = {p.name: p for p in params}
    assert set(by_name) == {"RADIUS", "GLOW_INTENSITY", "EDGE_SOFTNESS"}
    assert all(p.glsl_type == "float" for p in params)
    assert by_name["RADIUS"].value == pytest.approx(0.35)
    assert by_name["GLOW_INTENSITY"].value == pytest.approx(8.0)


def test_parse_skips_bare_integers():
    """Bare ints are loop counts per the generator's prompt rule — leave alone."""
    glsl = """
#define ITERATIONS 8
#define POLYGON_SIDES 6
#define RADIUS 0.35
""".strip()
    params = parse_glsl_defines(glsl)
    assert {p.name for p in params} == {"RADIUS"}


def test_parse_skips_denylisted_names():
    glsl = """
#define ANIM_SPEED 1.5
#define SAMPLES 16
#define LOOP_TIME 2.0
#define ITER_COUNT 5
#define FREQ_BASE 4.0
#define COLOR_INTENSITY 1.2
""".strip()
    params = parse_glsl_defines(glsl)
    # ANIM_SPEED (speed), LOOP_TIME (loop/time), ITER_COUNT (iter/count),
    # FREQ_BASE (freq), SAMPLES — all denylisted. COLOR_INTENSITY survives
    # despite containing "intensity" (no denylist match) and benefits from
    # color clamping in the optimizer.
    assert {p.name for p in params} == {"COLOR_INTENSITY"}


def test_parse_vec_defines():
    glsl = """
#define CORE_COLOR vec3(0.9, 0.4, 0.6)
#define CENTER vec2(0.5, 0.5)
#define TINT vec4(0.1, 0.2, 0.3, 1.0)
""".strip()
    params = parse_glsl_defines(glsl)
    by_name = {p.name: p for p in params}
    assert by_name["CORE_COLOR"].glsl_type == "vec3"
    assert by_name["CORE_COLOR"].value == pytest.approx([0.9, 0.4, 0.6])
    assert by_name["CENTER"].glsl_type == "vec2"
    assert by_name["TINT"].glsl_type == "vec4"


def test_parse_skips_malformed_vec():
    glsl = """
#define BROKEN vec3(0.5, 0.5)
#define ALSO_BROKEN vec3(not, a, number)
#define VALID vec2(0.1, 0.2)
""".strip()
    params = parse_glsl_defines(glsl)
    assert {p.name for p in params} == {"VALID"}


def test_parse_empty_glsl_returns_empty():
    assert parse_glsl_defines("") == []
    assert parse_glsl_defines("void main() {}") == []


# ---------------------------------------------------------------------------
# update_glsl_define
# ---------------------------------------------------------------------------


def test_update_scalar_define():
    glsl = "#define RADIUS 0.35\nvoid main() {}"
    updated = update_glsl_define(glsl, "RADIUS", 0.40, "float")
    assert "#define RADIUS 0.4" in updated
    assert "void main()" in updated


def test_update_vec_define_preserves_type():
    glsl = "#define CORE vec3(0.5, 0.5, 0.5)\nvoid main() {}"
    updated = update_glsl_define(glsl, "CORE", [0.1, 0.2, 0.3], "vec3")
    assert "#define CORE vec3(0.1, 0.2, 0.3)" in updated


def test_update_unknown_name_is_noop():
    glsl = "#define RADIUS 0.35\nvoid main() {}"
    updated = update_glsl_define(glsl, "MISSING", 0.5, "float")
    assert updated == glsl


def test_update_float_always_emits_decimal_point():
    glsl = "#define POWER 2.0\nvoid main() {}"
    updated = update_glsl_define(glsl, "POWER", 3.0, "float")
    # GLSL `int * float` is a compile error — the integer-looking write must
    # still have a decimal point so the shader keeps compiling.
    assert "#define POWER 3.0" in updated


# ---------------------------------------------------------------------------
# optimize_glsl_candidate
# ---------------------------------------------------------------------------


def _make_score_driver(monkeypatch, score_for_glsl):
    """Patch score_glsl so the test controls the score returned per shader."""
    calls: list[str] = []

    def fake_score(glsl, ref_path, render_glsl_fn, *, max_shader_chars=12000):
        calls.append(glsl)
        return score_for_glsl(glsl)

    monkeypatch.setattr("app.png_shader.glsl_optimizer.score_glsl", fake_score)
    return calls


def test_optimize_returns_unchanged_when_no_defines(monkeypatch, tmp_path):
    glsl = "void main() { gl_FragColor = vec4(1.0); }"

    def render_fn(_glsl: str) -> Path | None:
        return tmp_path / "render.png"

    monkeypatch.setattr(
        "app.png_shader.glsl_optimizer.score_glsl",
        lambda *a, **k: 0.5,
    )

    result = optimize_glsl_candidate(
        glsl,
        tmp_path / "ref.png",
        render_fn,
        max_iterations=5,
    )
    assert isinstance(result, GlslOptimizeResult)
    assert result.best_glsl == glsl
    assert result.iterations_run == 0
    assert result.improved is False


def test_optimize_accepts_improvement(monkeypatch, tmp_path):
    """Optimizer keeps a perturbation when the mock scorer rewards it."""
    glsl = "#define RADIUS 0.35\nvoid main() {}"

    def score_for_glsl(g: str) -> float:
        # Reward larger RADIUS up to 0.5.
        for line in g.splitlines():
            if line.startswith("#define RADIUS"):
                val = float(line.split()[-1])
                return 1.0 - abs(0.5 - val)
        return 0.0

    _make_score_driver(monkeypatch, score_for_glsl)

    result = optimize_glsl_candidate(
        glsl,
        tmp_path / "ref.png",
        render_glsl_fn=lambda _g: tmp_path / "render.png",
        max_iterations=10,
        scale=0.05,
    )
    assert result.improved is True
    assert result.best_score > result.initial_score
    # Optimizer should have nudged toward 0.5 by at least one accepted step.
    assert any(s.accepted for s in result.optimizer_log)


def test_optimize_rejects_when_no_improvement(monkeypatch, tmp_path):
    glsl = "#define RADIUS 0.35\nvoid main() {}"

    # Constant score: every perturbation is rejected.
    _make_score_driver(monkeypatch, lambda _g: 0.5)

    result = optimize_glsl_candidate(
        glsl,
        tmp_path / "ref.png",
        render_glsl_fn=lambda _g: tmp_path / "render.png",
        max_iterations=4,
    )
    assert result.improved is False
    assert result.best_score == result.initial_score
    assert all(not s.accepted for s in result.optimizer_log)


def test_optimize_respects_max_iterations(monkeypatch, tmp_path):
    glsl = """
#define A 0.2
#define B 0.3
#define C 0.4
""".strip()

    _make_score_driver(monkeypatch, lambda _g: 0.5)

    result = optimize_glsl_candidate(
        glsl,
        tmp_path / "ref.png",
        render_glsl_fn=lambda _g: tmp_path / "render.png",
        max_iterations=2,
    )
    # Each render counts as one step; cap must be honored.
    assert result.iterations_run <= 2


def test_optimize_handles_vec_perturbation(monkeypatch, tmp_path):
    glsl = "#define CENTER vec2(0.5, 0.5)\nvoid main() {}"

    def score_for_glsl(g: str) -> float:
        # Reward CENTER closer to (0.7, 0.7).
        for line in g.splitlines():
            if line.startswith("#define CENTER"):
                inside = line[line.index("(") + 1 : line.index(")")]
                xs = [float(x.strip()) for x in inside.split(",")]
                return 1.0 - (abs(0.7 - xs[0]) + abs(0.7 - xs[1])) / 2.0
        return 0.0

    _make_score_driver(monkeypatch, score_for_glsl)

    result = optimize_glsl_candidate(
        glsl,
        tmp_path / "ref.png",
        render_glsl_fn=lambda _g: tmp_path / "render.png",
        max_iterations=10,
        scale=0.05,
    )
    assert result.improved is True
    assert "#define CENTER vec2(" in result.best_glsl


# ---------------------------------------------------------------------------
# Artifact builder
# ---------------------------------------------------------------------------


def test_build_artifacts_shape(monkeypatch, tmp_path):
    glsl = "#define RADIUS 0.35\nvoid main() {}"

    def score_for_glsl(g: str) -> float:
        for line in g.splitlines():
            if line.startswith("#define RADIUS"):
                return float(line.split()[-1])
        return 0.0

    _make_score_driver(monkeypatch, score_for_glsl)

    result = optimize_glsl_candidate(
        glsl,
        tmp_path / "ref.png",
        render_glsl_fn=lambda _g: tmp_path / "render.png",
        max_iterations=4,
    )
    summary = build_glsl_optimization_artifacts(result)
    assert summary["mode"] == "glsl_defines"
    assert summary["initial_score"] == result.initial_score
    assert summary["best_score"] == result.best_score
    assert summary["improved"] == result.improved
    assert isinstance(summary["steps"], list)
    assert all(
        {"iteration", "param_name", "old_value", "new_value", "accepted"} <= step.keys()
        for step in summary["steps"]
    )
