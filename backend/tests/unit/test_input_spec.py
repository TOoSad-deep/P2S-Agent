"""Unit tests for p2s_agent.core.pipeline.input_spec (Phase 2)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from p2s_agent.core.pipeline.input_spec import (
    DEFAULT_INPUT_SPEC,
    build_input_spec,
    validate_input_spec,
)
from p2s_agent.strategy import get_strategy_config


# ---------------------------------------------------------------------------
# build_input_spec
# ---------------------------------------------------------------------------


def test_build_input_spec_sets_image_path():
    spec = build_input_spec("/tmp/icon.png")
    assert spec["input_image"] == "/tmp/icon.png"


def test_build_input_spec_accepts_path_object():
    spec = build_input_spec(Path("/tmp/icon.png"))
    assert spec["input_image"] == "/tmp/icon.png"


def test_build_input_spec_defaults_are_complete():
    spec = build_input_spec("img.png")
    # Top-level keys
    assert "input_image" in spec
    assert "target" in spec
    assert "quality" in spec
    assert "candidates" in spec

    # Target sub-keys
    target = spec["target"]
    assert target["backend"] == "glsl"
    assert target["shader_env"] == "webgl2"
    assert target["resolution"] == [512, 512]
    assert target["allow_texture"] is False
    assert target["allow_sdf_texture"] is False
    assert target["max_shader_chars"] == 12000
    assert target["max_layers"] == 24
    assert target["max_render_time_ms"] == 8

    # Quality sub-keys
    quality = spec["quality"]
    assert quality["mode"] == "balanced"
    assert quality["max_iterations"] == 5
    assert quality["optimization_budget"] == 300
    assert quality["refinement_mode"] == "auto"
    assert quality["max_refinement_iterations"] == 3
    assert quality["refinement_threshold"] == 0.80

    # Candidate sub-keys
    candidates = spec["candidates"]
    assert candidates["llm_enabled"] is False
    assert candidates["llm_implementation"] == "auto"
    assert candidates["cv_enabled"] is True
    assert candidates["glsl_render_enabled"] is False


def test_build_input_spec_target_override():
    spec = build_input_spec(
        "img.png",
        target={"shader_env": "webgl1", "max_layers": 16},
    )
    assert spec["target"]["shader_env"] == "webgl1"
    assert spec["target"]["max_layers"] == 16
    # Other target keys must survive
    assert spec["target"]["backend"] == "glsl"
    assert spec["target"]["resolution"] == [512, 512]


def test_build_input_spec_quality_override():
    spec = build_input_spec(
        "img.png",
        quality={"mode": "fast", "max_iterations": 2, "refinement_mode": "off"},
    )
    assert spec["quality"]["mode"] == "fast"
    assert spec["quality"]["max_iterations"] == 2
    assert spec["quality"]["optimization_budget"] == 300
    assert spec["quality"]["refinement_mode"] == "off"


def test_build_input_spec_candidates_override():
    spec = build_input_spec(
        "img.png",
        candidates={
            "llm_enabled": True,
            "llm_implementation": "shadertoy_glsl",
            "cv_enabled": False,
            "glsl_render_enabled": True,
        },
    )
    assert spec["candidates"]["llm_enabled"] is True
    assert spec["candidates"]["llm_implementation"] == "shadertoy_glsl"
    assert spec["candidates"]["cv_enabled"] is False
    assert spec["candidates"]["glsl_render_enabled"] is True


def test_build_does_not_mutate_default():
    original_target = copy.deepcopy(DEFAULT_INPUT_SPEC["target"])
    build_input_spec("img.png", target={"max_layers": 99})
    assert DEFAULT_INPUT_SPEC["target"]["max_layers"] == original_target["max_layers"]


# ---------------------------------------------------------------------------
# validate_input_spec
# ---------------------------------------------------------------------------


def test_validate_input_spec_valid():
    spec = build_input_spec("img.png")
    errors = validate_input_spec(spec)
    assert errors == []


def test_validate_input_spec_missing_image():
    spec = build_input_spec("")
    errors = validate_input_spec(spec)
    assert any("input_image" in e for e in errors)


def test_validate_input_spec_whitespace_image():
    spec = build_input_spec("   ")
    errors = validate_input_spec(spec)
    assert any("input_image" in e for e in errors)


def test_validate_input_spec_bad_backend():
    spec = build_input_spec("img.png", target={"backend": "hlsl"})
    errors = validate_input_spec(spec)
    assert any("backend" in e for e in errors)


def test_validate_input_spec_bad_resolution():
    spec = build_input_spec("img.png", target={"resolution": [512]})
    errors = validate_input_spec(spec)
    assert any("resolution" in e for e in errors)


def test_validate_input_spec_resolution_non_positive():
    spec = build_input_spec("img.png", target={"resolution": [0, 512]})
    errors = validate_input_spec(spec)
    assert any("resolution" in e for e in errors)


def test_validate_input_spec_resolution_floats():
    spec = build_input_spec("img.png", target={"resolution": [512.0, 512.0]})
    errors = validate_input_spec(spec)
    assert any("resolution" in e for e in errors)


def test_validate_input_spec_bad_mode():
    spec = build_input_spec("img.png", quality={"mode": "ultra"})
    errors = validate_input_spec(spec)
    assert any("mode" in e for e in errors)


def test_validate_input_spec_all_valid_modes():
    for mode in ("fast", "balanced", "quality", "aggressive"):
        spec = build_input_spec("img.png", quality={"mode": mode})
        errors = validate_input_spec(spec)
        assert errors == [], f"mode={mode!r} should be valid"


def test_validate_input_spec_all_valid_refinement_modes():
    for mode in ("off", "auto", "on"):
        spec = build_input_spec("img.png", quality={"refinement_mode": mode})
        errors = validate_input_spec(spec)
        assert errors == [], f"refinement_mode={mode!r} should be valid"


def test_validate_input_spec_bad_candidate_flags():
    spec = build_input_spec(
        "img.png",
        candidates={"llm_enabled": "yes", "cv_enabled": 1, "glsl_render_enabled": "true"},
    )
    errors = validate_input_spec(spec)
    assert any("llm_enabled" in e for e in errors)
    assert any("cv_enabled" in e for e in errors)
    assert any("glsl_render_enabled" in e for e in errors)


def test_validate_input_spec_bad_llm_implementation():
    spec = build_input_spec("img.png", candidates={"llm_implementation": "native"})
    errors = validate_input_spec(spec)
    assert any("llm_implementation" in e for e in errors)


def test_validate_input_spec_bad_refinement_mode():
    spec = build_input_spec("img.png", quality={"refinement_mode": "always"})
    errors = validate_input_spec(spec)
    assert any("refinement_mode" in e for e in errors)


def test_default_quality_has_new_strategy_fields():
    spec = build_input_spec("img.png")
    q = spec["quality"]
    assert q["refinement_high_score_stop"] == 0.92
    assert q["refinement_min_improvement"] == 0.01
    assert q["refinement_patience"] == 2
    assert q["force_failure_type"] is None
    assert q["protected_aspects"] == ["layer_count", "primitive_types", "background"]


def test_validate_aggressive_mode_is_allowed():
    spec = build_input_spec("img.png", quality={"mode": "aggressive"})
    assert validate_input_spec(spec) == []


def test_validate_max_refinement_iterations_bounds():
    cfg = get_strategy_config()
    meta = cfg.params["max_refinement_iterations"]
    for bad in (meta.min - 1, meta.max + 1, 1.5, "x"):
        spec = build_input_spec("img.png", quality={"max_refinement_iterations": bad})
        errors = validate_input_spec(spec)
        assert any("max_refinement_iterations" in e for e in errors), f"bad={bad!r}"


def test_validate_refinement_threshold_bounds():
    cfg = get_strategy_config()
    meta = cfg.params["refinement_threshold"]
    for bad in (meta.min - 0.01, meta.max + 0.01, "x"):
        spec = build_input_spec("img.png", quality={"refinement_threshold": bad})
        errors = validate_input_spec(spec)
        assert any("refinement_threshold" in e for e in errors), f"bad={bad!r}"


def test_validate_high_score_stop_bounds_and_relation():
    cfg = get_strategy_config()
    meta = cfg.params["refinement_high_score_stop"]
    spec = build_input_spec("img.png", quality={"refinement_high_score_stop": meta.min - 0.01})
    assert any("refinement_high_score_stop" in e for e in validate_input_spec(spec))
    spec = build_input_spec(
        "img.png",
        quality={"refinement_threshold": 0.95, "refinement_high_score_stop": 0.90},
    )
    assert any(
        "refinement_high_score_stop" in e and "refinement_threshold" in e
        for e in validate_input_spec(spec)
    )


def test_validate_min_improvement_bounds():
    cfg = get_strategy_config()
    meta = cfg.params["refinement_min_improvement"]
    for bad in (meta.min - 0.001, meta.max + 0.01, "x"):
        spec = build_input_spec("img.png", quality={"refinement_min_improvement": bad})
        errors = validate_input_spec(spec)
        assert any("refinement_min_improvement" in e for e in errors), f"bad={bad!r}"


def test_validate_patience_bounds():
    cfg = get_strategy_config()
    meta = cfg.params["refinement_patience"]
    for bad in (meta.min - 1, meta.max + 1, 1.5, "x"):
        spec = build_input_spec("img.png", quality={"refinement_patience": bad})
        errors = validate_input_spec(spec)
        assert any("refinement_patience" in e for e in errors), f"bad={bad!r}"


def test_validate_force_failure_type():
    spec = build_input_spec("img.png", quality={"force_failure_type": "color"})
    assert validate_input_spec(spec) == []
    spec = build_input_spec("img.png", quality={"force_failure_type": None})
    assert validate_input_spec(spec) == []
    spec = build_input_spec("img.png", quality={"force_failure_type": "rainbow"})
    assert any("force_failure_type" in e for e in validate_input_spec(spec))


def test_validate_protected_aspects():
    spec = build_input_spec("img.png", quality={"protected_aspects": []})
    assert validate_input_spec(spec) == []
    spec = build_input_spec(
        "img.png", quality={"protected_aspects": ["layer_count", "visual_causality"]}
    )
    assert validate_input_spec(spec) == []
    spec = build_input_spec("img.png", quality={"protected_aspects": ["bogus"]})
    assert any("protected_aspects" in e for e in validate_input_spec(spec))
    spec = build_input_spec("img.png", quality={"protected_aspects": "layer_count"})
    assert any("protected_aspects" in e for e in validate_input_spec(spec))
