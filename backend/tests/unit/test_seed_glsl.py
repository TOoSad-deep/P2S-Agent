"""Tests for seed-GLSL adaptation and candidate construction."""

from __future__ import annotations

import json

from app.pipeline.seed_glsl import (
    SeedAdaptResult,
    adapt_seed_glsl,
    build_seed_candidate,
)

VALID_SHADERTOY = (
    "#define R 0.30\n"
    "void mainImage(out vec4 fragColor, in vec2 fragCoord) { fragColor = vec4(R); }"
)
LEGACY_MAIN = "void main() { gl_FragColor = vec4(0.3, 0.4, 0.5, 1.0); }"
UNWRAPPABLE = "float helper(float x) { return x * 2.0; }"


def test_valid_shadertoy_passes_through_as_normalized():
    result = adapt_seed_glsl(VALID_SHADERTOY)
    assert isinstance(result, SeedAdaptResult)
    assert result.valid is True
    assert result.adapted_by == "normalized"
    assert "void mainImage" in result.glsl


def test_legacy_main_is_wrapped_into_mainimage():
    result = adapt_seed_glsl(LEGACY_MAIN)
    assert result.valid is True
    assert result.adapted_by == "wrapped"
    assert "void mainImage(out vec4 fragColor, in vec2 fragCoord)" in result.glsl
    assert "fragColor = vec4(0.3, 0.4, 0.5, 1.0)" in result.glsl
    assert "gl_FragColor" not in result.glsl


def test_unwrappable_falls_back_to_llm_port():
    def fake_client(system_prompt, user_prompt, image_paths=None):
        return json.dumps({"glsl": VALID_SHADERTOY})

    result = adapt_seed_glsl(UNWRAPPABLE, llm_client=fake_client)
    assert result.valid is True
    assert result.adapted_by == "llm_ported"
    assert "void mainImage" in result.glsl


def test_invalid_when_all_stages_fail():
    def empty_client(system_prompt, user_prompt, image_paths=None):
        return ""

    result = adapt_seed_glsl(UNWRAPPABLE, llm_client=empty_client)
    assert result.valid is False
    assert result.adapted_by == "failed"
    assert result.errors


def test_empty_source_is_invalid():
    result = adapt_seed_glsl("   ")
    assert result.valid is False
    assert result.adapted_by == "failed"
    assert result.errors


def test_build_seed_candidate_fields():
    candidate = build_seed_candidate(
        VALID_SHADERTOY, adapted_by="normalized", warnings=["w1"]
    )
    assert candidate.id == "seed_0"
    assert candidate.source == "seed"
    assert candidate.output_kind == "glsl"
    assert candidate.dsl is None
    assert candidate.compile_success is True
    assert candidate.compile_glsl == VALID_SHADERTOY
    assert candidate.selected is True
    assert candidate.glsl_metadata["adapted_by"] == "normalized"
    assert candidate.glsl_metadata["warnings"] == ["w1"]
