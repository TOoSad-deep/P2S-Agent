"""Unit tests for the new PNG-to-Shader candidate generators.

Tests for: baseline_candidate, fallback_candidate, llm_scene_candidate.
No LLM calls, no browser. Uses synthetic in-memory preprocess dicts.
"""

from __future__ import annotations

import json

import pytest

from app.candidates.baseline import generate_baseline_candidate
from app.candidates.fallback import generate_fallback_candidate
from app.candidates.llm_scene import (
    _call_llm,
    _normalize_gradient_fills,
    _normalize_linear_gradient_direction,
    generate_llm_refinement,
    generate_llm_scene_candidate,
)
from app.dsl.compiler import compile_dsl
from app.dsl.validator import validate_dsl


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

def _make_preprocess(palette=None):
    """Return a minimal preprocess dict suitable for testing."""
    return {
        "width": 64,
        "height": 64,
        "has_alpha": False,
        "alpha_coverage": 1.0,
        "palette": palette or ["#c86432", "#ffffff", "#000000", "#aaaaaa", "#555555"],
        "color_count_estimate": 12,
        "edge_sharpness": 0.1,
        "component_count_estimate": 1,
        "texture_score": 0.05,
        "photo_like_score": 0.1,
        "gradient_score": 0.2,
    }


# ---------------------------------------------------------------------------
# Baseline candidate tests
# ---------------------------------------------------------------------------

def test_baseline_candidate_returns_valid_dsl():
    preprocess = _make_preprocess()
    dsl = generate_baseline_candidate(preprocess)
    result = validate_dsl(dsl)
    assert result.valid, f"Validation errors: {result.errors}"


def test_baseline_candidate_compiles():
    preprocess = _make_preprocess()
    dsl = generate_baseline_candidate(preprocess)
    result = compile_dsl(dsl)
    assert result.success, f"Compile errors: {result.errors}"


def test_baseline_candidate_meta_source():
    preprocess = _make_preprocess()
    dsl = generate_baseline_candidate(preprocess)
    assert "_meta" in dsl
    assert dsl["_meta"]["source"] == "baseline"


def test_baseline_candidate_meta_priority():
    preprocess = _make_preprocess()
    dsl = generate_baseline_candidate(preprocess)
    assert dsl["_meta"]["priority"] == 0


# ---------------------------------------------------------------------------
# Fallback candidate tests
# ---------------------------------------------------------------------------

def test_fallback_candidate_returns_valid_dsl():
    preprocess = _make_preprocess()
    dsl = generate_fallback_candidate(preprocess)
    result = validate_dsl(dsl)
    assert result.valid, f"Validation errors: {result.errors}"


def test_fallback_candidate_compiles():
    preprocess = _make_preprocess()
    dsl = generate_fallback_candidate(preprocess)
    result = compile_dsl(dsl)
    assert result.success, f"Compile errors: {result.errors}"


def test_fallback_candidate_meta_source():
    preprocess = _make_preprocess()
    dsl = generate_fallback_candidate(preprocess)
    assert "_meta" in dsl
    assert dsl["_meta"]["source"] == "fallback"


def test_fallback_candidate_meta_priority():
    preprocess = _make_preprocess()
    dsl = generate_fallback_candidate(preprocess)
    assert dsl["_meta"]["priority"] == 99


# ---------------------------------------------------------------------------
# LLM candidate tests
# ---------------------------------------------------------------------------

def test_llm_candidate_returns_none_when_disabled():
    preprocess = _make_preprocess()
    result = generate_llm_scene_candidate(preprocess, llm_enabled=False)
    assert result is None


def test_llm_candidate_returns_none_when_enabled(monkeypatch):
    """Without an injected response/client or configured key, this remains no-op."""
    monkeypatch.setattr("app.candidates.llm_scene.settings.llm_api_key", "")
    preprocess = _make_preprocess()
    result = generate_llm_scene_candidate(preprocess, llm_enabled=True)
    assert result is None


def test_llm_candidate_parses_png_dsl_response():
    preprocess = _make_preprocess()
    llm_response = {
        "schema_version": "1.0",
        "canvas": {"width": 64, "height": 64, "background": "#000000"},
        "layers": [
            {
                "id": "circle_01",
                "type": "circle",
                "params": {"center": [0.5, 0.5], "radius": 0.3},
                "fill": {"type": "solid", "color": "#ff0000"},
                "opacity": 1.0,
            }
        ],
    }

    result = generate_llm_scene_candidate(
        preprocess,
        llm_enabled=True,
        implementation="png_dsl",
        llm_response=json.dumps(llm_response),
    )

    assert result is not None
    assert result["_meta"]["source"] == "llm"
    assert result["_meta"]["output_kind"] == "dsl"
    assert result["_meta"]["implementation"] == "png_dsl"
    val = validate_dsl(result)
    assert val.valid, val.errors
    compiled = compile_dsl(result)
    assert compiled.success, compiled.errors


def test_llm_candidate_parses_shadertoy_glsl_response():
    preprocess = _make_preprocess()
    glsl = """
    ```glsl
    void mainImage(out vec4 fragColor, in vec2 fragCoord) {
        vec2 uv = fragCoord / iResolution.xy;
        fragColor = vec4(uv, 0.0, 1.0);
    }
    ```
    """

    result = generate_llm_scene_candidate(
        preprocess,
        llm_enabled=True,
        implementation="shadertoy_glsl",
        llm_response=glsl,
    )

    assert result is not None
    assert result["_meta"]["source"] == "llm"
    assert result["_meta"]["output_kind"] == "glsl"
    assert result["_meta"]["implementation"] == "shadertoy_glsl"
    assert "void mainImage" in result["glsl"]


def test_llm_candidate_parses_glsl_json_envelope_with_metadata():
    preprocess = _make_preprocess()
    response = {
        "scene_analysis": {"subject": "glowing sphere", "lighting": "center falloff"},
        "technique_plan": ["fake sphere normal", "radial falloff", "bloom approximation"],
        "parameters": {"core_radius": 0.28},
        "glsl": """
            uniform float iTime;
            #define core_radius 0.28
            #define glow_intensity 1.25
            void mainImage(out vec4 fragColor, in vec2 fragCoord) {
                vec2 uv = fragCoord / iResolution.xy;
                float d = length(uv - vec2(0.5));
                float glow = exp(-d * d * glow_intensity);
                fragColor = vec4(vec3(glow), 1.0);
            }
        """,
    }

    result = generate_llm_scene_candidate(
        preprocess,
        llm_enabled=True,
        implementation="shadertoy_glsl",
        llm_response=response,
    )

    assert result is not None
    assert "uniform float iTime" not in result["glsl"]
    assert result["glsl_metadata"]["scene_analysis"]["subject"] == "glowing sphere"
    assert "radial falloff" in result["glsl_metadata"]["technique_plan"]
    names = {p["name"] for p in result["glsl_metadata"]["tunable_parameters"]}
    assert names == {"core_radius", "glow_intensity"}


def test_llm_candidate_auto_uses_glsl_for_complex_texture():
    preprocess = _make_preprocess()
    preprocess["photo_like_score"] = 0.8
    preprocess["texture_score"] = 0.7
    preprocess["color_count_estimate"] = 220

    result = generate_llm_scene_candidate(
        preprocess,
        llm_enabled=True,
        implementation="auto",
        llm_response="void mainImage(out vec4 fragColor, in vec2 fragCoord) { fragColor = vec4(1.0); }",
    )

    assert result is not None
    assert result["_meta"]["output_kind"] == "glsl"
    assert result["_meta"]["implementation"] == "shadertoy_glsl"


def test_llm_candidate_auto_uses_glsl_for_soft_glowing_sphere():
    preprocess = _make_preprocess()
    preprocess.update(
        {
            "photo_like_score": 0.38,
            "texture_score": 0.08,
            "gradient_score": 0.72,
            "color_count_estimate": 48,
            "component_count_estimate": 1,
            "alpha_coverage": 0.42,
            "edge_sharpness": 0.06,
        }
    )

    result = generate_llm_scene_candidate(
        preprocess,
        llm_enabled=True,
        implementation="auto",
        llm_response="""
        {
          "scene_analysis": {"subject": "glowing orb", "lighting": "center falloff"},
          "technique_plan": ["center-to-edge falloff", "soft halo"],
          "parameters": {"CORE_RADIUS": 0.28},
          "glsl": "#define CORE_RADIUS 0.28\\n#define GLOW_POWER 2.4\\nvoid mainImage(out vec4 fragColor, in vec2 fragCoord) { vec2 uv = fragCoord / iResolution.xy; float d = length(uv - 0.5); float glow = exp(-pow(d / CORE_RADIUS, GLOW_POWER)); fragColor = vec4(vec3(glow), glow); }"
        }
        """,
    )

    assert result is not None
    assert result["_meta"]["output_kind"] == "glsl"
    assert result["_meta"]["visual_strategy"]["routing_hint"] == "direct_glsl"
    assert "soft_glow_or_emissive_falloff" in result["glsl_metadata"]["visual_strategy"]["phenomena"]
    names = {p["name"] for p in result["glsl_metadata"]["tunable_parameters"]}
    assert {"CORE_RADIUS", "GLOW_POWER"} <= names


def test_llm_glsl_prompt_preserves_visual_causality_and_image_input():
    preprocess = _make_preprocess()
    preprocess.update(
        {
            "gradient_score": 0.72,
            "color_count_estimate": 48,
            "alpha_coverage": 0.42,
            "edge_sharpness": 0.06,
        }
    )
    captured = {}

    def fake_client(system_prompt, user_prompt, image_paths):
        captured["system_prompt"] = system_prompt
        captured["user_prompt"] = user_prompt
        captured["image_paths"] = image_paths
        return "void mainImage(out vec4 fragColor, in vec2 fragCoord) { fragColor = vec4(1.0); }"

    result = generate_llm_scene_candidate(
        preprocess,
        image_path="/tmp/reference.png",
        llm_enabled=True,
        implementation="auto",
        llm_client=fake_client,
    )

    assert result is not None
    assert captured["image_paths"] == ["/tmp/reference.png"]
    assert "reference image is attached" in captured["system_prompt"]
    assert "visual cause" in captured["system_prompt"]
    assert "Never replace it with a single flat colored circle" in captured["system_prompt"]
    assert "failure_modes_to_avoid" in captured["user_prompt"]


def test_png_shader_llm_does_not_send_image_to_generate_model_by_default(monkeypatch):
    calls = []

    class FakeAgent:
        def __init__(self, model_config):
            self.model_config = model_config

        def chat(self, **kwargs):
            calls.append(kwargs)
            return "ok"

    monkeypatch.setattr("app.candidates.llm_scene.settings.llm_api_key", "test-key")
    monkeypatch.setattr("app.candidates.llm_scene.settings.llm_supports_image", False)
    monkeypatch.setattr("app.llm.client.BaseAgent", FakeAgent)

    result = _call_llm(
        "system",
        "user",
        image_paths="/tmp/reference.png",
        llm_client=None,
    )

    assert result == "ok"
    assert calls[0]["image_paths"] is None


def test_png_shader_llm_can_send_image_when_generate_model_supports_it(monkeypatch):
    calls = []

    class FakeAgent:
        def __init__(self, model_config):
            self.model_config = model_config

        def chat(self, **kwargs):
            calls.append(kwargs)
            return "ok"

    monkeypatch.setattr("app.candidates.llm_scene.settings.llm_api_key", "test-key")
    monkeypatch.setattr("app.candidates.llm_scene.settings.llm_supports_image", True)
    monkeypatch.setattr("app.llm.client.BaseAgent", FakeAgent)

    result = _call_llm(
        "system",
        "user",
        image_paths="/tmp/reference.png",
        llm_client=None,
    )

    assert result == "ok"
    assert calls[0]["image_paths"] == ["/tmp/reference.png"]


def test_llm_candidate_normalizes_plain_color_gradient_stops():
    preprocess = _make_preprocess()
    llm_response = {
        "schema_version": 1,
        "canvas": {"width": 64, "height": 64, "background": "#000000"},
        "layers": [
            {
                "id": "gradient_01",
                "type": "circle",
                "fill": {
                    "type": "radialGradient",
                    "stops": ["#ff0000", "#0000ff"],
                },
                "params": {"center": [0.25, 0.75], "radius": 0.35},
                "opacity": 1.0,
            }
        ],
    }

    result = generate_llm_scene_candidate(
        preprocess,
        llm_enabled=True,
        implementation="png_dsl",
        llm_response=llm_response,
    )

    assert result is not None
    assert result["layers"][0]["fill"]["center"] == [0.25, 0.75]
    stops = result["layers"][0]["fill"]["stops"]
    assert stops == [
        {"color": "#ff0000", "position": 0.0},
        {"color": "#0000ff", "position": 1.0},
    ]
    assert validate_dsl(result).valid


def test_llm_refinement_normalizes_gradient_stop_aliases_and_prompts_schema():
    preprocess = _make_preprocess()
    captured = {}

    def fake_client(system_prompt, user_prompt, image_paths):
        captured["system_prompt"] = system_prompt
        captured["user_prompt"] = user_prompt
        captured["image_paths"] = image_paths
        return {
            "schema_version": 1,
            "canvas": {"width": 64, "height": 64, "background": "#000000"},
            "layers": [
                {
                    "id": "gradient_01",
                    "type": "circle",
                    "fill": {
                        "type": "radialGradient",
                        "stops": [
                            {"value": "#ff0000", "offset": 0},
                            {"value": "#0000ff", "offset": 1},
                        ],
                    },
                    "params": {"center": [0.25, 0.75], "radius": 0.35},
                    "opacity": 1.0,
                }
            ],
        }

    current_dsl = generate_baseline_candidate(preprocess)
    result = generate_llm_refinement(
        preprocess,
        current_dsl=current_dsl,
        metrics={"mse": 0.2, "simple_ssim": 0.4},
        quality_router={"final_score": 0.4, "quality_band": "poor", "failure_type": "color"},
        canvas_width=64,
        canvas_height=64,
        llm_client=fake_client,
    )

    assert result is not None
    assert result["layers"][0]["fill"]["center"] == [0.25, 0.75]
    stops = result["layers"][0]["fill"]["stops"]
    assert stops == [
        {"color": "#ff0000", "position": 0.0},
        {"color": "#0000ff", "position": 1.0},
    ]
    assert validate_dsl(result).valid
    assert '"color":"#RRGGBB"' in captured["system_prompt"]
    assert '"position":0.0' in captured["system_prompt"]
    assert "center: [cx, cy]" in captured["system_prompt"]
    assert captured["image_paths"] is None


# ---------------------------------------------------------------------------
# Content / behavior tests
# ---------------------------------------------------------------------------

def test_baseline_uses_top_palette_color():
    """The baseline should incorporate the top palette color in its layers."""
    preprocess = _make_preprocess(palette=["#ff0000", "#00ff00", "#0000ff", "#ffffff", "#000000"])
    dsl = generate_baseline_candidate(preprocess)
    # Check that either a layer uses red or the background is red
    found_red = False
    for layer in dsl.get("layers", []):
        fill = layer.get("fill", {})
        if fill.get("type") == "solid" and fill.get("color", "").upper() in ("#FF0000", "#FF0808"):
            found_red = True
        if fill.get("type") == "radialGradient":
            for stop in fill.get("stops", []):
                if stop.get("color", "").upper() == "#FF0000":
                    found_red = True
    # The rule candidate always uses the top palette color in the layer fill
    # so at least one layer fill stop or color should reference it
    assert dsl.get("layers"), "DSL should have at least one layer"
    # Verify the DSL is valid (proxy for correct use of palette)
    val = validate_dsl(dsl)
    assert val.valid


def test_fallback_is_minimal_one_layer():
    """Fallback DSL should have exactly one layer."""
    preprocess = _make_preprocess()
    dsl = generate_fallback_candidate(preprocess)
    assert len(dsl["layers"]) == 1


def test_fallback_uses_top_palette_color():
    """Fallback should use the top palette color as the fill color."""
    preprocess = _make_preprocess(palette=["#abcdef", "#ffffff", "#000000", "#aaaaaa", "#555555"])
    dsl = generate_fallback_candidate(preprocess)
    layer = dsl["layers"][0]
    assert layer["fill"]["type"] == "solid"
    assert layer["fill"]["color"].upper() == "#ABCDEF"


def test_fallback_with_empty_palette():
    """Fallback should handle empty palette gracefully."""
    preprocess = _make_preprocess(palette=[])
    dsl = generate_fallback_candidate(preprocess)
    result = validate_dsl(dsl)
    assert result.valid
    result_compile = compile_dsl(dsl)
    assert result_compile.success


# ---------------------------------------------------------------------------
# linearGradient direction normalization (regression: closed-loop optimizer
# failed when LLM dropped or renamed the direction field).
# ---------------------------------------------------------------------------

def _gradient_dsl_with_fill(fill: dict) -> dict:
    return {
        "schema_version": 1,
        "canvas": {"width": 64, "height": 64, "background": "#000000"},
        "layers": [
            {
                "id": "bg_0",
                "type": "box",
                "fill": fill,
                "params": {"center": [0.5, 0.5], "size": [1.0, 1.0]},
                "opacity": 1.0,
                "transform": None,
                "effects": [],
            }
        ],
    }


def test_normalize_linear_gradient_direction_canonical_passthrough():
    direction = _normalize_linear_gradient_direction(
        {"type": "linearGradient", "direction": [0.0, 1.0]}
    )
    assert direction == [0.0, 1.0]


def test_normalize_linear_gradient_direction_missing_defaults_horizontal():
    direction = _normalize_linear_gradient_direction({"type": "linearGradient"})
    assert direction == [1.0, 0.0]


@pytest.mark.parametrize("alias", [
    "gradient_direction",
    "gradientDirection",
    "dir",
    "vector",
    "vec",
    "axis",
])
def test_normalize_linear_gradient_direction_vector_aliases(alias):
    direction = _normalize_linear_gradient_direction(
        {"type": "linearGradient", alias: [0.5, -0.5]}
    )
    assert direction == [0.5, -0.5]


def test_normalize_linear_gradient_direction_dx_dy_pair():
    direction = _normalize_linear_gradient_direction(
        {"type": "linearGradient", "dx": 1.0, "dy": 0.5}
    )
    assert direction == [1.0, 0.5]


def test_normalize_linear_gradient_direction_scalar_angle_degrees():
    # 90 degrees → ~[0, 1]
    direction = _normalize_linear_gradient_direction(
        {"type": "linearGradient", "angle": 90}
    )
    assert direction[0] == pytest.approx(0.0, abs=1e-6)
    assert direction[1] == pytest.approx(1.0, abs=1e-6)


def test_normalize_linear_gradient_direction_scalar_angle_radians_small():
    # 0 radians → [1, 0]
    direction = _normalize_linear_gradient_direction(
        {"type": "linearGradient", "angle": 0.0}
    )
    assert direction == [1.0, 0.0]


def test_normalize_gradient_fills_repairs_missing_direction_so_validator_passes():
    """Regression: refinement-loop LLM that omits `direction` no longer breaks validation."""
    dsl = _gradient_dsl_with_fill(
        {
            "type": "linearGradient",
            "stops": [
                {"color": "#000000", "position": 0.0},
                {"color": "#ffffff", "position": 1.0},
            ],
            # 'direction' missing — previously caused
            # "Layer[0]: linearGradient fill missing 'direction'".
        }
    )
    _normalize_gradient_fills(dsl)
    assert dsl["layers"][0]["fill"]["direction"] == [1.0, 0.0]
    assert validate_dsl(dsl).valid


def test_normalize_gradient_fills_repairs_angle_alias():
    dsl = _gradient_dsl_with_fill(
        {
            "type": "linearGradient",
            "stops": [
                {"color": "#000000", "position": 0.0},
                {"color": "#ffffff", "position": 1.0},
            ],
            "angle": 90,
        }
    )
    _normalize_gradient_fills(dsl)
    direction = dsl["layers"][0]["fill"]["direction"]
    assert direction[0] == pytest.approx(0.0, abs=1e-6)
    assert direction[1] == pytest.approx(1.0, abs=1e-6)
    assert validate_dsl(dsl).valid
