"""Tests for generate_llm_glsl_refinement."""

from __future__ import annotations

import json

from app.candidates.llm_scene import generate_llm_glsl_refinement

VALID_GLSL = (
    "#define R 0.5\n"
    "void mainImage(out vec4 fragColor, in vec2 fragCoord) { fragColor = vec4(R); }"
)


def test_returns_normalized_glsl_with_io():
    def fake_client(system_prompt, user_prompt, image_paths):
        return json.dumps({"glsl": "```glsl\n" + VALID_GLSL + "\n```"})

    result = generate_llm_glsl_refinement(
        current_glsl=VALID_GLSL,
        metrics={"mse": 0.4, "ssim": 0.5},
        quality_router={"final_score": 0.4, "quality_band": "low"},
        llm_client=fake_client,
    )

    assert result is not None
    assert "void mainImage" in result["glsl"]
    assert "```" not in result["glsl"]
    assert result["_io"]["mode"] == "glsl_refinement"
    assert "current_glsl" in result["_io"]["user_prompt"]


def test_fresh_start_omits_current_glsl_and_requests_rewrite():
    captured = {}

    def fake_client(system_prompt, user_prompt, image_paths):
        captured["system"] = system_prompt
        captured["user"] = user_prompt
        return json.dumps({"glsl": VALID_GLSL})

    result = generate_llm_glsl_refinement(
        current_glsl=VALID_GLSL,
        metrics={},
        quality_router={"final_score": 0.3},
        fresh_start=True,
        llm_client=fake_client,
    )

    assert result is not None
    assert "current_glsl" not in captured["user"]
    assert "from scratch" in captured["system"]
    assert result["_io"]["fresh_start"] is True


def test_returns_none_on_empty_response():
    result = generate_llm_glsl_refinement(
        current_glsl=VALID_GLSL,
        metrics={},
        quality_router={},
        llm_client=lambda *args: "",
    )
    assert result is None


def test_returns_none_when_no_mainimage():
    def fake_client(system_prompt, user_prompt, image_paths):
        return json.dumps({"glsl": "float x = 1.0;"})

    result = generate_llm_glsl_refinement(
        current_glsl=VALID_GLSL,
        metrics={},
        quality_router={},
        llm_client=fake_client,
    )
    assert result is None
