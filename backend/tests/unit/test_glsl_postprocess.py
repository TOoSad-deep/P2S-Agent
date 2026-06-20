"""Unit tests for GLSL LLM post-processing helpers."""

from __future__ import annotations

from p2s_agent.core.utils.glsl_postprocess import (
    build_visual_strategy,
    normalize_shadertoy_glsl,
    parse_glsl_response_payload,
    scan_undeclared_parameters,
)


def test_normalize_shadertoy_glsl_removes_conflicting_uniforms_and_extracts_defines():
    source = """
    ```glsl
    #version 300 es
    uniform float iTime;
    #define core_radius 0.28
    #define glow_intensity 1.4
    void mainImage(out vec4 fragColor, in vec2 fragCoord) {
        fragColor = vec4(core_radius + glow_intensity);
    }
    ```
    """

    result = normalize_shadertoy_glsl(source)

    assert "#version" not in result.glsl
    assert "uniform float iTime" not in result.glsl
    assert "void mainImage" in result.glsl
    assert {p["name"] for p in result.tunable_parameters} == {"core_radius", "glow_intensity"}
    assert any("removed_conflicting_uniforms" in w for w in result.warnings)


def test_parse_glsl_response_payload_accepts_json_envelope():
    text = """
    {
      "scene_analysis": {"subject": "glowing sphere"},
      "technique_plan": ["radial falloff", "bloom approximation"],
      "parameters": {"core_radius": 0.3},
      "glsl": "void mainImage(out vec4 fragColor, in vec2 fragCoord) { fragColor = vec4(1.0); }"
    }
    """

    payload = parse_glsl_response_payload(text)

    assert payload["scene_analysis"]["subject"] == "glowing sphere"
    assert "radial falloff" in payload["technique_plan"]
    assert "void mainImage" in payload["glsl"]


def test_build_visual_strategy_recommends_glsl_for_complex_material():
    strategy = build_visual_strategy(
        {
            "photo_like_score": 0.7,
            "texture_score": 0.5,
            "gradient_score": 0.4,
            "color_count_estimate": 160,
            "component_count_estimate": 1,
            "alpha_coverage": 0.8,
            "edge_sharpness": 0.1,
        }
    )

    assert strategy["routing_hint"] == "direct_glsl"
    assert "fake_normal" in strategy["recommended_techniques"]
    assert "exponential_glow" in strategy["recommended_techniques"]


def test_build_visual_strategy_routes_soft_glowing_blob_to_direct_glsl():
    strategy = build_visual_strategy(
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

    assert strategy["routing_hint"] == "direct_glsl"
    assert "soft_glow_or_emissive_falloff" in strategy["phenomena"]
    assert "center_to_edge_falloff" in strategy["recommended_techniques"]
    assert "flat_color_proxy_for_glow_or_material" in strategy["failure_modes_to_avoid"]


def test_normalize_shadertoy_glsl_injects_missing_defines():
    """LLM sometimes omits the #define block entirely. Postprocess must
    auto-inject sensible defaults so the shader still compiles."""
    source = """
    void mainImage(out vec4 fragColor, in vec2 fragCoord) {
        vec2 uv = fragCoord / iResolution.xy;
        vec2 center = vec2(CENTER_X, CENTER_Y);
        vec2 p = uv - center;
        float dist = length(p);
        float core = 1.0 - smoothstep(0.0, RADIUS_CORE, dist);
        core = pow(core, FALLOFF_POWER);
        float glow = exp(-pow(dist / RADIUS_GLOW, 2.0)) * GLOW_INTENSITY;
        vec3 colCenter = vec3(COLOR_CENTER_R, COLOR_CENTER_G, COLOR_CENTER_B);
        vec3 colEdge = vec3(COLOR_EDGE_R, COLOR_EDGE_G, COLOR_EDGE_B);
        vec3 color = mix(colEdge, colCenter, core);
        fragColor = vec4(color, core + glow);
    }
    """

    result = normalize_shadertoy_glsl(source)

    expected = {
        "CENTER_X", "CENTER_Y", "RADIUS_CORE", "RADIUS_GLOW",
        "FALLOFF_POWER", "GLOW_INTENSITY",
        "COLOR_CENTER_R", "COLOR_CENTER_G", "COLOR_CENTER_B",
        "COLOR_EDGE_R", "COLOR_EDGE_G", "COLOR_EDGE_B",
    }
    for name in expected:
        assert f"#define {name}" in result.glsl, f"missing #define for {name}"

    injected_warning = next(
        (w for w in result.warnings if w.startswith("auto_injected_defines:")),
        None,
    )
    assert injected_warning is not None
    injected = set(injected_warning.split(":", 1)[1].split(","))
    assert injected == expected

    # After repair the body has no undeclared parameter references.
    assert scan_undeclared_parameters(result.glsl) == []


def test_normalize_shadertoy_glsl_only_injects_what_is_missing():
    """Pre-declared identifiers (#define, const, uniform) are not duplicated."""
    source = """
    #define CENTER_X 0.4
    const float CENTER_Y = 0.5;
    uniform float RADIUS_CORE;
    void mainImage(out vec4 fragColor, in vec2 fragCoord) {
        vec2 p = vec2(CENTER_X, CENTER_Y);
        float r = RADIUS_CORE;
        float g = GLOW_INTENSITY;
        fragColor = vec4(r + g);
    }
    """

    result = normalize_shadertoy_glsl(source)

    # Only GLOW_INTENSITY should be auto-injected; the others were declared.
    assert result.glsl.count("#define CENTER_X") == 1
    assert result.glsl.count("#define CENTER_Y") == 0
    assert result.glsl.count("#define RADIUS_CORE") == 0
    assert "#define GLOW_INTENSITY" in result.glsl

    injected_warning = next(
        (w for w in result.warnings if w.startswith("auto_injected_defines:")),
        None,
    )
    assert injected_warning == "auto_injected_defines:GLOW_INTENSITY"


def test_normalize_shadertoy_glsl_coerces_int_define_to_float():
    """LLM sometimes writes `#define FALLOFF_POWER 2` (int). GLSL doesn't
    auto-promote int to float, so `float * int` fails. Postprocess must
    rewrite the literal to keep the shader compiling."""
    source = """
    #define FALLOFF_POWER 2
    #define GLOW_INTENSITY 1
    #define RADIUS_CORE 0.15
    void mainImage(out vec4 fragColor, in vec2 fragCoord) {
        float dist = length(fragCoord);
        float glow = exp(-dist * dist * FALLOFF_POWER) * GLOW_INTENSITY;
        fragColor = vec4(glow * RADIUS_CORE);
    }
    """

    result = normalize_shadertoy_glsl(source)

    assert "#define FALLOFF_POWER 2.0" in result.glsl
    assert "#define GLOW_INTENSITY 1.0" in result.glsl
    # Already-float defines are not touched (no double `.0`).
    assert "#define RADIUS_CORE 0.15" in result.glsl
    assert "0.15.0" not in result.glsl

    coerced_warning = next(
        (w for w in result.warnings if w.startswith("coerced_float_defines:")),
        None,
    )
    assert coerced_warning is not None
    coerced = set(coerced_warning.split(":", 1)[1].split(","))
    assert coerced == {"FALLOFF_POWER", "GLOW_INTENSITY"}


def test_normalize_shadertoy_glsl_keeps_genuine_int_counts():
    """SIDES/COUNT/ITERATIONS-style names must stay integers — they index
    discrete things and GLSL functions like sdPolygon take int."""
    source = """
    #define POLYGON_SIDES 6
    #define LOOP_ITERATIONS 8
    #define BRIGHTNESS 3
    void mainImage(out vec4 fragColor, in vec2 fragCoord) {
        fragColor = vec4(float(POLYGON_SIDES) / float(LOOP_ITERATIONS) * float(BRIGHTNESS));
    }
    """

    result = normalize_shadertoy_glsl(source)

    assert "#define POLYGON_SIDES 6" in result.glsl
    assert "#define POLYGON_SIDES 6.0" not in result.glsl
    assert "#define LOOP_ITERATIONS 8" in result.glsl
    assert "#define LOOP_ITERATIONS 8.0" not in result.glsl
    # BRIGHTNESS is not in the count-token list -> coerced to float.
    assert "#define BRIGHTNESS 3.0" in result.glsl


def test_scan_undeclared_parameters_ignores_glsl_keywords_and_system_names():
    source = """
    #define CENTER_X 0.5
    void mainImage(out vec4 fragColor, in vec2 fragCoord) {
        vec2 uv = fragCoord / iResolution.xy;
        float t = iTime;
        vec2 p = uv - vec2(CENTER_X, 0.5);
        fragColor = vec4(p, 0.0, 1.0);
    }
    """

    # iResolution/iTime are system, fragColor/mainImage are reserved,
    # CENTER_X is declared, lowercase names are skipped by the regex itself.
    assert scan_undeclared_parameters(source) == []


def test_scan_undeclared_parameters_preserves_order_and_dedupes():
    source = """
    void mainImage(out vec4 fragColor, in vec2 fragCoord) {
        float a = ALPHA_ONE + ALPHA_TWO + ALPHA_ONE + BETA;
        fragColor = vec4(a);
    }
    """

    assert scan_undeclared_parameters(source) == ["ALPHA_ONE", "ALPHA_TWO", "BETA"]


def test_build_visual_strategy_keeps_simple_icon_dsl_compatible():
    strategy = build_visual_strategy(
        {
            "photo_like_score": 0.05,
            "texture_score": 0.01,
            "gradient_score": 0.10,
            "color_count_estimate": 4,
            "component_count_estimate": 1,
            "alpha_coverage": 0.30,
            "edge_sharpness": 0.55,
        }
    )

    assert strategy["routing_hint"] == "dsl_or_glsl"
    assert "simple_2d_shape" in strategy["phenomena"]
