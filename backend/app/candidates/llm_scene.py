"""LLM-based scene candidates for PNG-to-Shader.

Two implementations are supported:

- ``png_dsl``: returns a schema-shaped DSL candidate that can enter the
  deterministic DSL -> GLSL compiler, metrics, optimizer, and revision flow.
- ``shadertoy_glsl``: returns a GLSL candidate envelope for legacy/full shader
  preview use. This is compatible with the frontend renderer, but is not a DSL
  and cannot be optimized by the PNG DSL pipeline.
"""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, Optional, Union

logger = logging.getLogger(__name__)

from app.config import settings
from app.dsl.schema import DSL_SCHEMA_VERSION
from app.pipeline.preprocess import feature_num
from app.services.logging_config import log_event
from app.utils.glsl_postprocess import (
    build_visual_strategy,
    extract_tunable_parameters,
    normalize_shadertoy_glsl,
    parse_glsl_response_payload,
)
from app.utils.color import normalize_color

Implementation = Literal["auto", "png_dsl", "shadertoy_glsl"]
LlmClient = Callable[[str, str, Optional[list[str]]], Union[str, dict, list, None]]


class LlmCallError(RuntimeError):
    """Raised when an LLM API call fails before returning usable content."""


NO_USABLE_DSL_MESSAGE = (
    "LLM returned no usable DSL: response content was empty, was not valid JSON, "
    "or did not contain a 'layers' field. Check provider support for "
    "response_format=json_object and the raw_response in this iteration's llm_io."
)


def generate_llm_scene_candidate(
    preprocess: dict,
    canvas_width: int = 512,
    canvas_height: int = 512,
    *,
    image_path: "str | Path | None" = None,
    llm_enabled: bool = False,
    implementation: Implementation = "auto",
    llm_client: LlmClient | None = None,
    llm_response: str | dict | None = None,
) -> dict | None:
    """Generate an LLM candidate as PNG DSL or Shadertoy GLSL.

    Args:
        preprocess: Dict from preprocess_image().
        canvas_width: Output canvas width in pixels.
        canvas_height: Output canvas height in pixels.
        image_path: Optional source image path for multimodal LLM calls.
        llm_enabled: If False and no injected response/client is supplied,
            return None without calling an API.
        implementation: ``png_dsl``, ``shadertoy_glsl``, or ``auto``.
            ``auto`` chooses DSL for icon/shape-like images and GLSL for
            photo-like/complex texture inputs that exceed the DSL's scope.
        llm_client: Optional test/production injection. Called as
            ``llm_client(system_prompt, user_prompt, image_paths)``.
        llm_response: Optional raw response injection for tests.

    Returns:
        A DSL dict with ``_meta.output_kind == "dsl"``, a GLSL envelope with
        ``_meta.output_kind == "glsl"``, or None on disabled/failed generation.
    """
    if not llm_enabled and llm_client is None and llm_response is None:
        return None

    visual_strategy = build_visual_strategy(preprocess)
    mode = _choose_implementation(preprocess, implementation, visual_strategy=visual_strategy)
    system_prompt, user_prompt = _build_prompts(
        preprocess,
        canvas_width,
        canvas_height,
        mode,
        visual_strategy=visual_strategy,
    )
    image_paths_for_io = _normalize_image_paths(image_path)

    response = llm_response
    if response is None:
        response = _call_llm(
            system_prompt,
            user_prompt,
            image_paths=image_path,
            llm_client=llm_client,
            response_format={"type": "json_object"},
        )
    content = _response_content(response)
    if not content:
        return None

    io_record = {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "raw_response": content,
        "mode": mode,
        "image_paths": (
            image_paths_for_io
            if image_paths_for_io and (llm_client is not None or settings.llm.supports_image)
            else []
        ),
    }

    if mode == "png_dsl":
        dsl = _parse_dsl_response(content, canvas_width, canvas_height)
        if dsl is not None:
            dsl.setdefault("_meta", {})
            dsl["_meta"]["visual_strategy"] = visual_strategy
            dsl["_io"] = io_record
            return dsl
        glsl = _extract_glsl(content)
        if glsl:
            env = _make_glsl_envelope(glsl, mode="png_dsl_fallback", visual_strategy=visual_strategy)
            env["_io"] = io_record
            return env
        return None

    payload = parse_glsl_response_payload(content)
    glsl = _extract_glsl(str(payload.get("glsl") or ""))
    if glsl:
        env = _make_glsl_envelope(
            glsl,
            mode=mode,
            scene_analysis=payload.get("scene_analysis"),
            technique_plan=payload.get("technique_plan"),
            parameter_hints=payload.get("parameter_hints"),
            visual_strategy=visual_strategy,
        )
        env, repair_io = _maybe_repair_shadertoy_envelope(
            env,
            mode=mode,
            scene_analysis=payload.get("scene_analysis"),
            technique_plan=payload.get("technique_plan"),
            parameter_hints=payload.get("parameter_hints"),
            visual_strategy=visual_strategy,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_path=image_path,
            llm_client=llm_client,
            allow_retry=llm_response is None,
        )
        if repair_io is not None:
            io_record["repair"] = repair_io
        env["_io"] = io_record
        return env

    result = _parse_dsl_response(content, canvas_width, canvas_height)
    if result is not None:
        result.setdefault("_meta", {})
        result["_meta"]["visual_strategy"] = visual_strategy
        result["_io"] = io_record
    return result


def _choose_implementation(
    preprocess: dict,
    implementation: Implementation,
    *,
    visual_strategy: dict | None = None,
) -> Literal["png_dsl", "shadertoy_glsl"]:
    if implementation == "png_dsl":
        return "png_dsl"
    if implementation == "shadertoy_glsl":
        return "shadertoy_glsl"

    strategy = visual_strategy or build_visual_strategy(preprocess)
    if strategy.get("routing_hint") == "direct_glsl":
        return "shadertoy_glsl"

    photo_like = feature_num(preprocess, "photo_like_score", 0.0)
    texture = feature_num(preprocess, "texture_score", 0.0)
    colors = int(feature_num(preprocess, "color_count_estimate", 0))
    components = int(feature_num(preprocess, "component_count_estimate", 1))

    if photo_like >= 0.65 or texture >= 0.55 or colors >= 180 or components >= 16:
        return "shadertoy_glsl"
    return "png_dsl"


def _build_prompts(
    preprocess: dict,
    canvas_width: int,
    canvas_height: int,
    mode: Literal["png_dsl", "shadertoy_glsl"],
    *,
    visual_strategy: dict | None = None,
) -> tuple[str, str]:
    alpha_note = ""
    if bool(preprocess.get("has_alpha")) and feature_num(preprocess, "alpha_coverage", 1.0) < 0.999:
        alpha_note = (
            " The attached reference is an opaque RGB preview composited over "
            f"{preprocess.get('llm_reference_background', '#000000')}; match the visible colors and "
            "lighting, and do not interpret transparency as a white alpha mask."
        )

    common = (
        "You generate 2D/2.5D UI visual effects from image analysis features. "
        "Avoid 3D raymarching, volume rendering, and external textures. "
        "If a reference image is attached, inspect it directly and use the JSON features only as support."
        + alpha_note
    )
    strategy = visual_strategy or build_visual_strategy(preprocess)
    payload = json.dumps(
        {
            "preprocess": preprocess,
            "visual_strategy": strategy,
            "canvas": {"width": canvas_width, "height": canvas_height},
        },
        ensure_ascii=False,
        indent=2,
    )

    if mode == "png_dsl":
        system_prompt = (
            common
            + "\nReturn ONLY JSON. The JSON must be a PNG Shader DSL object with "
            "schema_version, canvas, and layers. Supported layer types: circle, "
            "ellipse, box, roundedBox, ring, polygon. Supported fills: solid, "
            "linearGradient, radialGradient. Supported effects: glow, vignette, grain."
            + "\nFor linearGradient/radialGradient, every stop MUST be an object: "
            '{"color":"#RRGGBB","position":0.0}. '
            "Use position values in [0.0, 1.0]. Do not use offset, value, arrays, "
            "or plain color strings for stops."
            + "\nFor radialGradient, fill MUST include center: [cx, cy] in normalized "
            "UV coordinates. Use the layer params.center when the gradient is centered "
            "on the same shape."
            + "\nFor linearGradient, fill MUST include direction: [dx, dy] as a "
            "2-element array (e.g. [1.0, 0.0] horizontal, [0.0, 1.0] vertical, "
            "[1.0, 1.0] diagonal). Do not use 'angle', 'gradient_direction', or "
            "scalar values for direction."
            + '\n\nExamples:'
            + '\n1) Transparent icon with single color → {"schema_version":1,"canvas":{"width":512,"height":512,"background":"#000000"},'
            + '"layers":[{"id":"icon_0","type":"circle","fill":{"type":"solid","color":"#ff4444"},'
            + '"params":{"center":[0.5,0.5],"radius":0.3},"opacity":1.0,"transform":null,"effects":[]}]}'
            + '\n2) Gradient background → {"schema_version":1,"canvas":{"width":512,"height":512,"background":"#111111"},'
            + '"layers":[{"id":"bg_0","type":"box","fill":{"type":"linearGradient",'
            + '"stops":[{"color":"#1a2a6c","position":0.0},{"color":"#b21f1f","position":0.5},{"color":"#fdbb2d","position":1.0}],'
            + '"direction":[1.0,1.0]},"params":{"center":[0.5,0.5],"size":[1.0,1.0]},"opacity":1.0,"transform":null,"effects":[]}]}'
            + '\n3) Glowing ring → {"schema_version":1,"canvas":{"width":512,"height":512,"background":"#000000"},'
            + '"layers":[{"id":"ring_0","type":"ring","fill":{"type":"solid","color":"#00ffff"},'
            + '"params":{"center":[0.5,0.5],"radius":0.35,"thickness":0.02},"opacity":1.0,"transform":null,'
            + '"effects":[{"type":"glow","intensity":8.0,"color":"#00ffff"}]}]}'
        )
        user_prompt = (
            "Create one concise, schema-valid DSL candidate for these features. "
            "Do not include markdown.\n\n"
            + payload
        )
    else:
        system_prompt = (
            common
            + "\nYou are not doing pixel tracing. First infer the visual cause: material, lighting, falloff, "
            + "coordinate transform, and procedural technique. Then write a compact Shadertoy shader."
            + "\nReturn ONLY JSON with keys: scene_analysis, technique_plan, parameters, glsl."
            + "\nscene_analysis must name the subject, material/lighting cause, edge behavior, color falloff, "
            + "and any symmetry/repetition. technique_plan must map each visible phenomenon to GLSL code."
            + "\nThe glsl value must be Shadertoy-compatible and contain "
            + "void mainImage(out vec4 fragColor, in vec2 fragCoord)."
            + "\nFor complex PNGs prefer analytic GLSL techniques over flat geometry: radial/exponential falloff, "
            + "fake sphere normals, specular highlights, Fresnel rim, bloom approximation, fbm/domain warp, "
            + "cell/hash particles, inverse coordinate transforms."
            + "\nFor a glowing sphere/bubble/orb, preserve center-to-edge brightness decay, soft outer halo, "
            + "rim light, and highlight placement. Never replace it with a single flat colored circle."
            + "\nUse #define for ALL tunable visual parameters so users can adjust them later. "
            + "Prefer #define COLOR_NAME vec3(r, g, b) for colors with float 0-1 components; "
            + "if you split colors into COLOR_NAME_R/G/B, keep all three channels adjacent. "
            + "Use #define for positions, radii, sizes, falloff powers, "
            + "glow intensity, highlight position, noise scale, speed, and thresholds. "
            + "Reference these names in the shader body instead of hardcoding visual constants."
            + "\nSTRICT RULE: every ALL_CAPS identifier you reference in the shader body (e.g. CENTER_X, "
            + "RADIUS_GLOW, COLOR_EDGE_R) MUST have a matching `#define NAME value` line at the top of "
            + "the shader BEFORE void mainImage. Do not emit code that uses any uppercase parameter "
            + "name without first declaring it; an undeclared identifier is treated as a compile failure."
            + "\nFLOAT LITERAL RULE: every #define value used as a float in the body (radii, intensities, "
            + "powers, colors, offsets, etc.) MUST be written with a decimal point, e.g. `#define POWER 2.0` "
            + "not `#define POWER 2`. GLSL does not auto-promote int to float and `float * int` is a compile "
            + "error. The only #defines that may stay as bare integers are true counts (polygon SIDES, "
            + "loop ITERATIONS, etc.)."
            + "\nDo not declare Shadertoy system uniforms such as iTime, iResolution, iMouse, iChannel0, or iChannel1."
            + "\nDo not include markdown fences. Keep loops bounded with small constant counts for mobile WebGL."
        )
        user_prompt = (
            "Generate a shader by following this sequence: "
            "1) identify visible phenomena, 2) map each phenomenon to shader techniques, "
            "3) set named #define parameters, 4) implement GLSL. "
            "If the image looks like a glowing sphere, bubble, glass, fluid, plasma, particle field, "
            "or soft gradient, model the lighting/falloff instead of drawing a flat color shape. "
            "Respect visual_strategy.prompt_constraints and avoid visual_strategy.failure_modes_to_avoid.\n\n"
            + payload
        )
    return system_prompt, user_prompt


# Output-token ceiling for all LLM generation/refinement calls. Large enough
# that a full revised DSL or Shadertoy shader is never truncated mid-JSON
# (finish_reason="length"), which otherwise surfaces as "LLM returned no usable
# DSL/GLSL". Only generated tokens are billed, so a high cap costs nothing on
# short responses.
LLM_MAX_OUTPUT_TOKENS = 32000


def _call_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    image_paths: "list[str | Path] | str | Path | None" = None,
    llm_client: LlmClient | None,
    max_tokens: int = LLM_MAX_OUTPUT_TOKENS,
    response_format: dict | None = None,
) -> str | dict | list | None:
    normalized_paths = _normalize_image_paths(image_paths)

    if llm_client is not None:
        log_event(
            logger,
            "llm_call_injected_client",
            prompt_len=len(system_prompt) + len(user_prompt),
            image_count=len(normalized_paths or []),
            response_format=response_format,
            max_tokens=max_tokens,
        )
        return llm_client(system_prompt, user_prompt, normalized_paths)

    if not settings.llm.api_key:
        log_event(
            logger,
            "llm_call_skipped",
            level=logging.WARNING,
            reason="missing_api_key",
        )
        return None

    try:
        from app.llm.client import BaseAgent
        agent = BaseAgent(settings.llm)
    except Exception as exc:
        raise LlmCallError(f"LLM client init failed: {_format_exception(exc)}") from exc

    paths = normalized_paths if normalized_paths and settings.llm.supports_image else None
    log_event(
        logger,
        "llm_call_start",
        prompt_len=len(system_prompt) + len(user_prompt),
        system_prompt_len=len(system_prompt),
        user_prompt_len=len(user_prompt),
        image_count=len(paths) if paths else 0,
        model=settings.llm.model,
        max_tokens=max_tokens,
        response_format=response_format,
    )
    try:
        result = agent.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            image_paths=paths,
            temperature=0.2,
            max_tokens=max_tokens,
            return_raw=True,
            response_format=response_format,
        )
        if response_format is not None and not _response_content(result):
            log_event(
                logger,
                "llm_call_empty_json_mode_retry",
                level=logging.WARNING,
                model=settings.llm.model,
                response_format=response_format,
            )
            result = agent.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_paths=paths,
                temperature=0.2,
                max_tokens=max_tokens,
                return_raw=True,
                response_format=None,
            )
    except Exception as first_exc:
        # Model may not support image input — retry text-only.
        # In normal production config this branch should be rare because
        # image input is gated by GENERATE_SUPPORTS_IMAGE.
        if paths:
            try:
                log_event(
                    logger,
                    "llm_call_text_only_retry",
                    level=logging.WARNING,
                    reason="image_call_failed",
                    error=_format_exception(first_exc),
                )
                return agent.chat(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    image_paths=None,
                    temperature=0.2,
                    max_tokens=max_tokens,
                    return_raw=True,
                    response_format=response_format,
                )
            except Exception as retry_exc:
                log_event(
                    logger,
                    "llm_call_failed",
                    level=logging.ERROR,
                    image_error=_format_exception(first_exc),
                    retry_error=_format_exception(retry_exc),
                )
                raise LlmCallError(
                    "LLM image call failed: "
                    f"{_format_exception(first_exc)}; text-only retry failed: "
                    f"{_format_exception(retry_exc)}"
                ) from retry_exc
        log_event(
            logger,
            "llm_call_failed",
            level=logging.ERROR,
            error=_format_exception(first_exc),
        )
        raise LlmCallError(f"LLM call failed: {_format_exception(first_exc)}") from first_exc

    content = _response_content(result)
    content_preview = content[:120] if content else "None"
    log_event(
        logger,
        "llm_call_done",
        response_type=type(result).__name__,
        content_len=len(content),
        preview=content_preview,
        finish_reason=result.get("finish_reason") if isinstance(result, dict) else None,
        usage=result.get("usage") if isinstance(result, dict) else None,
    )
    return result


def _normalize_image_paths(
    image_paths: "list[str | Path] | str | Path | None",
) -> list[str] | None:
    if image_paths is None:
        return None
    if isinstance(image_paths, (str, Path)):
        return [str(image_paths)]
    paths = [str(p) for p in image_paths if p is not None]
    return paths or None


def _format_exception(exc: Exception) -> str:
    message = str(exc).strip()
    if len(message) > 500:
        message = message[:497] + "..."
    return f"{exc.__class__.__name__}: {message}" if message else exc.__class__.__name__


def _response_content(response: str | dict | list | None) -> str:
    if response is None:
        return ""
    if isinstance(response, list):
        parts: list[str] = []
        for block in response:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, list):
                    text = _response_content(text)
                if text:
                    parts.append(str(text))
            else:
                text = getattr(block, "text", None) or getattr(block, "content", None)
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    if isinstance(response, dict):
        if "layers" in response or "schema_version" in response:
            return json.dumps(response, ensure_ascii=False)
        for key in (
            "dsl",
            "revised_dsl",
            "revisedDSL",
            "png_dsl",
            "pngDsl",
            "shader_dsl",
            "shaderDsl",
            "candidate",
            "result",
            "output",
            "scene",
        ):
            value = response.get(key)
            if isinstance(value, dict):
                return json.dumps(response, ensure_ascii=False)
        if "glsl" in response or "shader" in response or "scene_analysis" in response:
            return json.dumps(response, ensure_ascii=False)
        value = (
            response.get("content")
            or response.get("text")
            or response.get("message")
            or response.get("response")
            or response.get("output_text")
            or response.get("answer")
            or response.get("output")
            or response.get("shader")
            or response.get("glsl")
            or response.get("dsl")
        )
        if isinstance(value, (dict, list)):
            return _response_content(value)
        return str(value or "")
    return str(response)


def _parse_dsl_response(text: str, canvas_width: int, canvas_height: int) -> dict | None:
    candidates = _extract_json_candidates(text)
    if not candidates:
        return None

    data: dict | None = None
    for candidate in candidates:
        if isinstance(candidate, list):
            if all(isinstance(item, dict) for item in candidate):
                data = {"layers": candidate}
            continue
        if not isinstance(candidate, dict):
            continue
        unwrapped = _unwrap_dsl_payload(candidate)
        if isinstance(unwrapped, dict) and "layers" in unwrapped:
            # Prefer the last DSL-shaped object; models often put examples or
            # the previous DSL before the final revised DSL.
            data = unwrapped

    if data is None or "layers" not in data:
        return None

    _normalize_gradient_fills(data)

    if not isinstance(data.get("schema_version"), int):
        data["schema_version"] = DSL_SCHEMA_VERSION
    canvas = data.setdefault("canvas", {})
    if isinstance(canvas, dict):
        canvas.setdefault("width", canvas_width)
        canvas.setdefault("height", canvas_height)
        canvas.setdefault("background", "#000000")
    existing_meta = data.get("_meta") if isinstance(data.get("_meta"), dict) else {}
    try:
        priority = int(existing_meta.get("priority", 3))
    except (TypeError, ValueError):
        priority = 3
    data["_meta"] = {
        **existing_meta,
        "source": "llm",
        "priority": priority,
        "output_kind": "dsl",
        "implementation": "png_dsl",
    }
    return data


def _unwrap_dsl_payload(data: dict) -> dict | None:
    """Find the DSL object inside common LLM envelope shapes."""
    if "layers" in data:
        return data

    wrapper_keys = (
        "dsl",
        "revised_dsl",
        "revisedDSL",
        "png_dsl",
        "pngDsl",
        "shader_dsl",
        "shaderDsl",
        "candidate",
        "result",
        "output",
        "scene",
    )
    for key in wrapper_keys:
        value = data.get(key)
        if isinstance(value, dict):
            unwrapped = _unwrap_dsl_payload(value)
            if unwrapped is not None:
                return unwrapped

    for value in data.values():
        if isinstance(value, dict) and "layers" in value:
            return value
    return None


def _normalize_gradient_fills(dsl: dict) -> None:
    """Repair common LLM gradient aliases in-place.

    Models often return stops as strings, ``{"value": color, "offset": t}``,
    or two-item tuples. They also often omit ``radialGradient.center`` or
    ``linearGradient.direction`` because the layer already has ``params.center``
    or the model treats the gradient as implicit. The DSL validator requires
    these fields explicitly inside the fill object.
    """
    layers = dsl.get("layers")
    if not isinstance(layers, list):
        return

    for layer in layers:
        if not isinstance(layer, dict):
            continue
        fill = layer.get("fill")
        if not isinstance(fill, dict):
            continue
        fill_type = fill.get("type")
        if fill_type not in ("linearGradient", "radialGradient"):
            continue

        if fill_type == "radialGradient":
            fill["center"] = _normalize_radial_gradient_center(fill, layer)
        elif fill_type == "linearGradient":
            fill["direction"] = _normalize_linear_gradient_direction(fill)

        stops = fill.get("stops")
        if not isinstance(stops, list):
            continue

        normalized = [
            _normalize_gradient_stop(stop, idx, len(stops))
            for idx, stop in enumerate(stops)
        ]
        survived = [stop for stop in normalized if stop is not None]
        # The validator requires >=2 stops. Dropping unparseable stops can leave
        # 0 or 1, turning a malformed-but-rescuable gradient into an invalid DSL.
        # Guarantee a 2-stop minimum so normalization always rescues rather than
        # invalidates.
        fill["stops"] = _ensure_two_stops(survived)


_GRADIENT_FALLBACK_STOPS = [
    {"color": "#000000", "position": 0.0},
    {"color": "#ffffff", "position": 1.0},
]


def _ensure_two_stops(stops: list[dict]) -> list[dict]:
    """Return a list of at least two gradient stops.

    - 0 survivors → a default black→white gradient.
    - 1 survivor → duplicate it at positions 0.0 and 1.0.
    - 2+ survivors → returned unchanged.
    """
    if len(stops) >= 2:
        return stops
    if len(stops) == 1:
        color = stops[0].get("color", "#000000")
        return [
            {"color": color, "position": 0.0},
            {"color": color, "position": 1.0},
        ]
    return [dict(stop) for stop in _GRADIENT_FALLBACK_STOPS]


def _normalize_gradient_stop(stop: Any, idx: int, total: int) -> dict | None:
    fallback_position = idx / max(1, total - 1)
    color_value: Any = None
    position_value: Any = None

    if isinstance(stop, dict):
        color_value = (
            stop.get("color")
            or stop.get("colour")
            or stop.get("value")
            or stop.get("rgb")
            or stop.get("rgba")
        )
        position_value = (
            stop.get("position")
            if "position" in stop
            else stop.get("offset", stop.get("pos", stop.get("stop")))
        )
    elif isinstance(stop, (list, tuple)) and len(stop) == 2:
        first, second = stop
        if isinstance(first, (int, float)):
            position_value = first
            color_value = second
        else:
            color_value = first
            position_value = second
    else:
        color_value = stop

    color = normalize_color(color_value)
    if color is None:
        return None

    try:
        position = float(position_value) if position_value is not None else fallback_position
    except (TypeError, ValueError):
        position = fallback_position

    position = max(0.0, min(1.0, position))
    return {"color": color, "position": position}


def _normalize_linear_gradient_direction(fill: dict) -> list[float]:
    """Resolve a linearGradient ``direction`` from common LLM aliases.

    Accepts canonical ``direction: [dx, dy]`` as-is, plus aliases like
    ``gradient_direction`` / ``gradientDirection`` / ``dir`` / ``vector`` /
    ``axis``, scalar ``angle`` / ``angle_deg`` / ``angle_rad`` (converted via
    cos/sin), and per-component ``{dx, dy}`` / ``{x, y}`` pairs. Falls back
    to ``[1.0, 0.0]`` (horizontal) so the validator never sees a missing
    field.
    """
    vec_candidates: list[Any] = [
        fill.get("direction"),
        fill.get("gradient_direction"),
        fill.get("gradientDirection"),
        fill.get("dir"),
        fill.get("vector"),
        fill.get("vec"),
        fill.get("axis"),
    ]
    for candidate in vec_candidates:
        vec = _normalize_direction_vec2(candidate)
        if vec is not None:
            return vec

    for x_key, y_key in (("dx", "dy"), ("x", "y"), ("direction_x", "direction_y")):
        if x_key in fill and y_key in fill:
            vec = _normalize_direction_vec2([fill[x_key], fill[y_key]])
            if vec is not None:
                return vec

    angle_value = fill.get("angle")
    angle_unit = "deg" if "angle_deg" in fill or "degrees" in fill else None
    if angle_value is None:
        angle_value = fill.get("angle_deg", fill.get("degrees"))
        if angle_value is not None:
            angle_unit = "deg"
    if angle_value is None:
        angle_value = fill.get("angle_rad", fill.get("radians"))
        if angle_value is not None:
            angle_unit = "rad"
    if isinstance(angle_value, (int, float)):
        try:
            angle = float(angle_value)
        except (TypeError, ValueError):
            angle = None
        if angle is not None:
            if angle_unit is None:
                # No explicit unit: heuristically treat large magnitudes as
                # degrees (|angle| > 2*pi ~ 6.28 is almost certainly degrees).
                angle_unit = "deg" if abs(angle) > 6.5 else "rad"
            if angle_unit == "deg":
                angle = math.radians(angle)
            return [math.cos(angle), math.sin(angle)]

    return [1.0, 0.0]


def _normalize_direction_vec2(value: Any) -> list[float] | None:
    """Coerce *value* to a 2-element direction vector ``[dx, dy]``.

    Unlike ``_normalize_uv_vec2`` (which clamps to [0, 1] for UV coords),
    direction components may be negative. Returns None if the input cannot
    be coerced to two finite floats.
    """
    if isinstance(value, dict):
        for x_key, y_key in (("dx", "dy"), ("x", "y")):
            if x_key in value and y_key in value:
                return _normalize_direction_vec2([value[x_key], value[y_key]])
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        dx = float(value[0])
        dy = float(value[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(dx) and math.isfinite(dy)):
        return None
    return [dx, dy]


def _normalize_radial_gradient_center(fill: dict, layer: dict) -> list[float]:
    params = layer.get("params") if isinstance(layer.get("params"), dict) else {}
    candidates = [
        fill.get("center"),
        fill.get("centre"),
        fill.get("gradient_center"),
        fill.get("gradientCenter"),
        params.get("center"),
    ]

    for source in (fill, params):
        xy = _center_from_xy_aliases(source)
        if xy is not None:
            candidates.append(xy)

    for candidate in candidates:
        center = _normalize_uv_vec2(candidate)
        if center is not None:
            return center
    return [0.5, 0.5]


def _center_from_xy_aliases(source: Any) -> list[Any] | None:
    if not isinstance(source, dict):
        return None

    pairs = [
        ("center_x", "center_y"),
        ("centerX", "centerY"),
        ("cx", "cy"),
        ("gradient_cx", "gradient_cy"),
        ("gradientCenterX", "gradientCenterY"),
        ("x", "y"),
    ]
    for x_key, y_key in pairs:
        if x_key in source and y_key in source:
            return [source[x_key], source[y_key]]
    return None


def _normalize_uv_vec2(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        if "uv" in value:
            return _normalize_uv_vec2(value["uv"])
        alias = _center_from_xy_aliases(value)
        if alias is not None:
            return _normalize_uv_vec2(alias)
        return None

    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None

    try:
        x = float(value[0])
        y = float(value[1])
    except (TypeError, ValueError):
        return None

    return [max(0.0, min(1.0, x)), max(0.0, min(1.0, y))]


def _extract_json(text: str) -> Any:
    candidates = _extract_json_candidates(text)
    return candidates[0] if candidates else None


def _extract_json_candidates(text: str) -> list[Any]:
    stripped = text.strip()
    text_candidates: list[str] = []
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL | re.IGNORECASE)
    if fenced:
        text_candidates.append(fenced.group(1).strip())
    text_candidates.append(stripped)

    parsed: list[Any] = []
    seen: set[str] = set()
    for candidate in text_candidates:
        try:
            parsed.append(json.loads(candidate))
            continue
        except json.JSONDecodeError:
            pass

        for snippet in _iter_json_object_snippets(candidate):
            if snippet in seen:
                continue
            seen.add(snippet)
            try:
                parsed.append(json.loads(snippet))
            except json.JSONDecodeError:
                continue
    return parsed


def _iter_json_object_snippets(text: str):
    start: int | None = None
    depth = 0
    in_string = False
    escaped = False

    for idx, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
            continue
        if ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start:idx + 1]
                start = None


def _extract_glsl(text: str) -> str:
    stripped = text.strip()
    match = re.search(r"```(?:glsl|shader|c)?\s*(.*?)```", stripped, re.DOTALL)
    if match:
        stripped = match.group(1).strip()

    self_check_idx = stripped.find("[Self-check]")
    if self_check_idx > 0:
        stripped = stripped[:self_check_idx].strip()

    if "void mainImage" in stripped:
        # Anchor start-token detection to start-of-line declarations so that a
        # token appearing *inside* a leading line comment (e.g.
        # "// helper returns vec3 color value") is not mistaken for the start
        # of the shader body. Without the anchor, ``find`` would slice into the
        # comment text and the extracted GLSL would begin with prose.
        start_tokens = (
            "#version",
            "#define",
            "precision ",
            "const ",
            "float ",
            "vec2 ",
            "vec3 ",
            "vec4 ",
            "void mainImage",
        )
        start_candidates: list[int] = []
        for token in start_tokens:
            m = re.search(rf"^[ \t]*{re.escape(token)}", stripped, re.MULTILINE)
            if m is not None:
                start_candidates.append(m.start())
        start = min(start_candidates) if start_candidates else stripped.find("void mainImage")
        return stripped[start:].strip()
    if "void main()" in stripped or "#version" in stripped:
        return stripped
    return ""


def _maybe_repair_shadertoy_envelope(
    envelope: dict,
    *,
    mode: str,
    scene_analysis: Any,
    technique_plan: Any,
    parameter_hints: Any,
    visual_strategy: dict | None,
    system_prompt: str,
    user_prompt: str,
    image_path: "str | Path | list[str | Path] | None",
    llm_client: LlmClient | None,
    allow_retry: bool,
) -> tuple[dict, dict | None]:
    """Re-prompt the LLM once if postprocess had to fabricate missing #defines.

    Returns (envelope, repair_io). When no retry is performed, repair_io is None
    and the original envelope is returned unchanged. When a retry happens,
    repair_io carries the prompts/response for transparency, and the returned
    envelope is whichever variant carries fewer auto-injected names.
    """
    if not allow_retry:
        return envelope, None

    warnings = envelope.get("glsl_metadata", {}).get("postprocess_warnings", [])
    injected_names = _extract_injected_names(warnings)
    if not injected_names:
        return envelope, None

    # No LLM access in this context -> can't retry, keep auto-injected version.
    if llm_client is None and not settings.llm.api_key:
        return envelope, None

    repair_system = (
        system_prompt
        + "\n\nThe previous response referenced these ALL_CAPS identifiers without "
        + "declaring them as #define: "
        + ", ".join(injected_names)
        + ". Return the SAME shader with corrected #define declarations for every "
        + "uppercase parameter at the top, before void mainImage. Do not change the "
        + "visual intent — only add the missing declarations."
    )
    repair_user = (
        user_prompt
        + "\n\n(Repair pass: ensure every ALL_CAPS identifier used in the body is "
        + "declared with #define above mainImage. Keep the JSON response format.)"
    )

    try:
        repair_response = _call_llm(
            repair_system,
            repair_user,
            image_paths=image_path,
            llm_client=llm_client,
            response_format={"type": "json_object"},
        )
    except LlmCallError:
        # Repair attempt failed — fall back to the auto-injected version.
        return envelope, None

    repair_content = _response_content(repair_response)
    if not repair_content:
        return envelope, None

    repair_payload = parse_glsl_response_payload(repair_content)
    repair_glsl = _extract_glsl(str(repair_payload.get("glsl") or ""))
    if not repair_glsl:
        return envelope, {"raw_response": repair_content, "outcome": "no_glsl"}

    repaired = _make_glsl_envelope(
        repair_glsl,
        mode=mode,
        scene_analysis=scene_analysis or repair_payload.get("scene_analysis"),
        technique_plan=technique_plan or repair_payload.get("technique_plan"),
        parameter_hints=parameter_hints or repair_payload.get("parameter_hints"),
        visual_strategy=visual_strategy,
    )
    repaired_injected = _extract_injected_names(
        repaired.get("glsl_metadata", {}).get("postprocess_warnings", [])
    )

    repair_io = {
        "system_prompt": repair_system,
        "user_prompt": repair_user,
        "raw_response": repair_content,
        "original_injected": injected_names,
        "remaining_injected": repaired_injected,
    }

    # Keep whichever version needed less fabrication.
    if len(repaired_injected) < len(injected_names):
        repair_io["outcome"] = "applied"
        return repaired, repair_io
    repair_io["outcome"] = "no_improvement"
    return envelope, repair_io


def _extract_injected_names(warnings: list) -> list[str]:
    if not isinstance(warnings, list):
        return []
    for warning in warnings:
        if isinstance(warning, str) and warning.startswith("auto_injected_defines:"):
            tail = warning.split(":", 1)[1]
            return [name for name in tail.split(",") if name]
    return []


def _make_glsl_envelope(
    glsl: str,
    *,
    mode: str,
    scene_analysis: Any = None,
    technique_plan: Any = None,
    parameter_hints: Any = None,
    visual_strategy: dict | None = None,
) -> dict:
    normalized = normalize_shadertoy_glsl(glsl)
    return {
        "_meta": {
            "source": "llm",
            "priority": 3,
            "output_kind": "glsl",
            "implementation": mode,
            "visual_strategy": visual_strategy or {},
        },
        "glsl": normalized.glsl,
        "glsl_metadata": {
            "scene_analysis": scene_analysis,
            "technique_plan": technique_plan or [],
            "parameter_hints": parameter_hints or {},
            "visual_strategy": visual_strategy or {},
            "tunable_parameters": normalized.tunable_parameters,
            "postprocess_warnings": normalized.warnings,
        },
    }


# ---------------------------------------------------------------------------
# LLM refinement (closed-loop DSL revision)
# ---------------------------------------------------------------------------

def _build_feedback_issues(metrics: dict, quality_router: dict) -> list[str]:
    """Map objective metrics to semantic, prioritized issue strings for LLM feedback.

    Each issue describes the *visual cause* rather than raw metric names,
    so the LLM can reason about what to change in the DSL.
    """
    issues: list[str] = []
    color_hist = float(metrics.get("color_histogram_score", 1.0))
    alpha_diff = float(metrics.get("alpha_coverage_diff", 0.0))
    ssim = float(metrics.get("simple_ssim", 1.0))
    edge_diff = float(metrics.get("edge_density_diff", 0.0))
    mse = float(metrics.get("mse", 0.0))

    if color_hist < 0.70:
        issues.append(
            f"COLOR MISMATCH (priority HIGH): The rendered colors don't match the "
            f"reference palette (histogram={color_hist:.2f}, target >0.80). "
            f"Adjust fill colors, gradient stop colors, glow colors, and background "
            f"to match the dominant colors of the source image."
        )
    if alpha_diff > 0.10:
        issues.append(
            f"SHAPE COVERAGE (priority HIGH): The shape's visible area doesn't match "
            f"the reference (alpha_diff={alpha_diff:.2f}, target <0.05). "
            f"Adjust radius/size parameters so the foreground coverage matches "
            f"the source image's non-transparent area."
        )
    if ssim < 0.60:
        issues.append(
            f"STRUCTURAL SIMILARITY (priority MEDIUM): Overall structure differs "
            f"from reference (ssim={ssim:.2f}, target >0.75). Check layer "
            f"positioning (center), primitive type choice, gradient direction, "
            f"and whether the correct number of layers is used."
        )
    if edge_diff > 0.15:
        issues.append(
            f"EDGE/DETAIL (priority LOW): Edge density differs from reference "
            f"(edge_diff={edge_diff:.2f}, target <0.10). Consider adding glow "
            f"effects for soft halos, adjusting edge softness, or adding vignette "
            f"for darkened borders."
        )
    if mse > 0.15 and not issues:
        issues.append(
            f"OVERALL DIVERGENCE (priority MEDIUM): The render is noticeably "
            f"different from the reference (mse={mse:.2f}). Review all layer "
            f"parameters holistically."
        )

    if not issues:
        reason = quality_router.get("reason", [])
        issues = list(reason[:2]) if isinstance(reason, list) else [
            "Minor fine-tuning needed: adjust opacity, glow intensity, or gradient positions."
        ]

    return issues


def _format_attempt_raw_responses(attempts: list[dict]) -> str:
    parts: list[str] = []
    for attempt in attempts:
        mode = attempt.get("mode", "unknown")
        parse_success = attempt.get("parse_success")
        raw = str(attempt.get("raw_response") or "")
        parts.append(
            f"[attempt={mode} parse_success={parse_success}]\n"
            + (raw if raw else "<empty response>")
        )
    return "\n\n".join(parts)


def _dsl_parse_diagnostics(text: str, dsl: dict | None) -> dict:
    candidates = _extract_json_candidates(text) if text else []
    candidate_shapes: list[str] = []
    for candidate in candidates[:5]:
        if isinstance(candidate, dict):
            keys = ",".join(sorted(str(k) for k in candidate.keys())[:6])
            candidate_shapes.append(f"dict:{keys}")
        else:
            candidate_shapes.append(type(candidate).__name__)

    return {
        "raw_response_len": len(text or ""),
        "json_candidate_count": len(candidates),
        "candidate_shapes": candidate_shapes,
        "parsed_layer_count": len(dsl.get("layers", [])) if isinstance(dsl, dict) else None,
    }


def generate_llm_refinement(
    preprocess: dict,
    current_dsl: dict,
    metrics: dict,
    quality_router: dict,
    canvas_width: int,
    canvas_height: int,
    *,
    reference_image_path: "str | Path | None" = None,
    current_render_path: "str | Path | None" = None,
    extra_feedback: "list[str] | None" = None,
    llm_client: LlmClient | None = None,
) -> dict | None:
    """Ask LLM to revise a DSL candidate based on objective-metric feedback.

    When ``reference_image_path`` and ``current_render_path`` are provided AND
    the configured Generate model supports image input, both are sent so the
    model can visually compare target vs. current output. The metric ``issues``
    list is still included as a textual hint.

    ``extra_feedback`` is an optional list of additional instruction strings
    (e.g. rollback notes, iteration history) injected into the feedback block.

    Returns a revised DSL dict with ``_io`` embedded, or None on failure.
    The caller must pop ``_io`` before passing the DSL to validate/compile.
    """
    issues = _build_feedback_issues(metrics, quality_router)
    if extra_feedback:
        issues = list(extra_feedback) + issues
    protected = quality_router.get("protected_aspects", [])
    image_paths = _normalize_image_paths([reference_image_path, current_render_path])
    has_images = bool(image_paths) and settings.llm.supports_image

    system_prompt = (
        "You are a 2D shader DSL expert. "
        + (
            "Two images are attached: image 1 is the TARGET (reference PNG), "
            "image 2 is the CURRENT rendered output of the DSL below. "
            "Diff them visually and revise the DSL so the next render matches "
            "image 1 more closely. "
            if has_images
            else "Given a reference image analysis, the current PNG Shader DSL, and quality feedback, "
                 "revise the DSL to better match the reference. "
        )
        + "Supported layer types: circle, ellipse, box, roundedBox, ring, polygon. "
        "Supported fills: solid, linearGradient, radialGradient. "
        "Supported effects: glow, vignette, grain. "
        "For linearGradient/radialGradient, every stop MUST be an object with "
        "both fields: {\"color\":\"#RRGGBB\",\"position\":0.0}. "
        "Use position values in [0.0, 1.0]. Do not use offset, value, arrays, "
        "or plain color strings for stops. "
        "For radialGradient, fill MUST include center: [cx, cy] in normalized "
        "UV coordinates. Use the layer params.center when the gradient is centered "
        "on the same shape. "
        "For linearGradient, fill MUST include direction: [dx, dy] as a 2-element "
        "array (e.g. [1.0, 0.0] horizontal, [0.0, 1.0] vertical, [1.0, 1.0] diagonal). "
        "Do not use 'angle', 'gradient_direction', or scalar values for direction. "
        "Return ONLY a single JSON object: the full revised DSL with schema_version, "
        "canvas, and layers. Do not include markdown, prose, or any text outside the JSON."
    )

    user_prompt = json.dumps(
        {
            "canvas": {"width": canvas_width, "height": canvas_height},
            "current_dsl": current_dsl,
            "feedback": {
                "current_score": round(float(quality_router.get("final_score", 0.0)), 4),
                "quality_band": quality_router.get("quality_band", "unknown"),
                "failure_type": quality_router.get("failure_type", "unknown"),
                "issues": issues,
                "protected_aspects": protected,
                "instruction": (
                    f"Fix: {'; '.join(issues[:3])}."
                    + (f" Do NOT change: {', '.join(protected)}." if protected else "")
                ),
            },
            # preprocess features are only included when no images are attached;
            # otherwise the model should rely on visual diff instead.
            **({} if has_images else {"preprocess": preprocess}),
        },
        ensure_ascii=False,
        indent=2,
    )

    attempts: list[dict] = []

    response = _call_llm(
        system_prompt,
        user_prompt,
        image_paths=image_paths if has_images else None,
        llm_client=llm_client,
        max_tokens=LLM_MAX_OUTPUT_TOKENS,
        response_format={"type": "json_object"},
    )
    content = _response_content(response)
    dsl = _parse_dsl_response(content, canvas_width, canvas_height) if content else None
    diagnostics = _dsl_parse_diagnostics(content, dsl)
    attempts.append({
        "mode": "json_object",
        "raw_response": content,
        "parse_success": dsl is not None,
        **diagnostics,
    })
    log_event(
        logger,
        "llm_refinement_parse_attempt",
        mode="json_object",
        parse_success=dsl is not None,
        **diagnostics,
    )

    if dsl is None:
        retry_system_prompt = (
            system_prompt
            + "\n\nJSON-mode retry: return plain text containing exactly one JSON "
            "object. The object itself must be the full revised DSL with a top-level "
            "`layers` array; do not wrap it in prose."
        )
        try:
            retry_response = _call_llm(
                retry_system_prompt,
                user_prompt,
                image_paths=image_paths if has_images else None,
                llm_client=llm_client,
                max_tokens=LLM_MAX_OUTPUT_TOKENS,
                response_format=None,
            )
        except Exception:
            logger.warning("refinement retry without response_format failed", exc_info=True)
            retry_response = None
        retry_content = _response_content(retry_response)
        retry_dsl = _parse_dsl_response(retry_content, canvas_width, canvas_height) if retry_content else None
        retry_diagnostics = _dsl_parse_diagnostics(retry_content, retry_dsl)
        attempts.append({
            "mode": "plain_json_retry",
            "raw_response": retry_content,
            "parse_success": retry_dsl is not None,
            **retry_diagnostics,
        })
        log_event(
            logger,
            "llm_refinement_parse_attempt",
            mode="plain_json_retry",
            parse_success=retry_dsl is not None,
            **retry_diagnostics,
        )
        if retry_dsl is not None:
            content = retry_content
            dsl = retry_dsl

    if dsl is None:
        log_event(
            logger,
            "llm_refinement_parse_failed",
            level=logging.WARNING,
            attempts=[
                {
                    "mode": attempt.get("mode"),
                    "parse_success": attempt.get("parse_success"),
                    "raw_response_len": attempt.get("raw_response_len"),
                    "json_candidate_count": attempt.get("json_candidate_count"),
                    "candidate_shapes": attempt.get("candidate_shapes"),
                }
                for attempt in attempts
            ],
        )
        return {
            "_parse_error": NO_USABLE_DSL_MESSAGE,
            "_io": {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "raw_response": _format_attempt_raw_responses(attempts),
                "mode": "refinement",
                "image_paths": image_paths if has_images else [],
                "attempts": attempts,
            },
        }

    dsl["_io"] = {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "raw_response": content,
        "mode": "refinement",
        "image_paths": image_paths if has_images else [],
    }
    if len(attempts) > 1:
        dsl["_io"]["attempts"] = attempts
    log_event(
        logger,
        "llm_refinement_parse_done",
        layer_count=len(dsl.get("layers", [])),
        attempts=len(attempts),
    )
    return dsl


def generate_llm_glsl_refinement(
    current_glsl: str,
    metrics: dict,
    quality_router: dict,
    *,
    reference_image_path: "str | Path | None" = None,
    current_render_path: "str | Path | None" = None,
    extra_feedback: "list[str] | None" = None,
    fresh_start: bool = False,
    llm_client: LlmClient | None = None,
) -> dict | None:
    """Ask the LLM to revise a Shadertoy GLSL candidate from feedback."""
    issues = _build_feedback_issues(metrics, quality_router)
    if extra_feedback:
        issues = list(extra_feedback) + issues
    image_paths = _normalize_image_paths([reference_image_path, current_render_path])
    has_images = bool(image_paths) and settings.llm.supports_image

    restart_clause = (
        "The previous incremental approach has stalled. Write a completely new "
        "shader from scratch with a different technique; do not anchor on the "
        "old implementation. "
        if fresh_start
        else ""
    )

    system_prompt = (
        "You are a Shadertoy GLSL expert. "
        + (
            "Two images are attached: image 1 is the TARGET (reference PNG), "
            "image 2 is the CURRENT rendered output. Diff them visually and "
            "produce GLSL whose render matches image 1 more closely. "
            if has_images
            else "Given quality feedback about the current render, produce GLSL "
                 "that better matches the reference image. "
        )
        + restart_clause
        + "Rules: keep a valid `void mainImage(out vec4 fragColor, in vec2 fragCoord)` "
        "entry point. Do NOT declare iTime/iResolution/iMouse/iFrame uniforms "
        "(they are auto-injected). Do NOT use discard. Every float literal must "
        "contain a decimal point (write 2.0, never 2). Keep all visually tunable "
        "constants as `#define NAME value` lines at the top of the shader. "
        "The target is a static image: do not animate with iTime. Return ONLY "
        "a JSON object {\"glsl\": \"<the full shader>\"} with no markdown or "
        "prose outside the JSON."
    )

    user_prompt = json.dumps(
        {
            **({} if fresh_start else {"current_glsl": current_glsl}),
            "feedback": {
                "current_score": round(float(quality_router.get("final_score", 0.0)), 4),
                "quality_band": quality_router.get("quality_band", "unknown"),
                "failure_type": quality_router.get("failure_type", "unknown"),
                "issues": issues,
                "instruction": f"Fix: {'; '.join(issues[:3])}.",
            },
        },
        ensure_ascii=False,
        indent=2,
    )

    response = _call_llm(
        system_prompt,
        user_prompt,
        image_paths=image_paths if has_images else None,
        llm_client=llm_client,
        max_tokens=LLM_MAX_OUTPUT_TOKENS,
        response_format={"type": "json_object"},
    )
    content = _response_content(response)
    if not content:
        return None

    payload = parse_glsl_response_payload(content)
    glsl = _extract_glsl(str(payload.get("glsl") or ""))
    if not glsl:
        glsl = _extract_glsl(content)
    if not glsl:
        return None

    normalized = normalize_shadertoy_glsl(glsl)
    if "void mainImage" not in normalized.glsl:
        return None

    return {
        "glsl": normalized.glsl,
        "postprocess_warnings": list(normalized.warnings),
        "_io": {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": content,
            "mode": "glsl_refinement",
            "fresh_start": fresh_start,
            "image_paths": image_paths if has_images else [],
        },
    }


def parameterize_glsl(
    current_glsl: str,
    *,
    llm_client: LlmClient | None = None,
) -> dict | None:
    """Lift hardcoded visual constants in a Shadertoy shader to ``#define`` tunables.

    Pure code transform (no image): the LLM finds inline literals in the shader
    body that control the look (colors, positions, radii, intensities, falloff,
    counts, angles, thresholds, speeds) and hoists each into a ``#define NAME
    value`` at the top — keeping the rendered output numerically identical — so
    the frontend param panel can detect and tune them.

    Returns a dict with the parameterized GLSL plus before/after tunable counts,
    or ``None`` when the LLM produced no usable shader.
    """
    system_prompt = (
        "You are a Shadertoy GLSL expert performing a refactor, NOT a redesign. "
        "Examine the shader and find numeric/color literals in the BODY that "
        "control the visual look (colors, center/position, radius/size, "
        "intensity/brightness, falloff/power, counts, angles, thresholds, "
        "speeds). Hoist each into a descriptive ALL_CAPS `#define NAME value` "
        "line placed at the TOP, before `void mainImage`, and replace the inline "
        "literal in the body with that NAME. "
        "CRITICAL: keep the rendered output numerically identical — reuse the "
        "exact same values, do not tweak them. "
        "Rules: keep a valid `void mainImage(out vec4 fragColor, in vec2 "
        "fragCoord)` entry point. Prefer `#define COLOR_NAME vec3(r, g, b)` for "
        "colors (or keep COLOR_NAME_R/_G/_B adjacent). Every float literal must "
        "contain a decimal point (write 2.0, never 2). Loop bounds that must stay "
        "integer counters may remain int. Do NOT declare iTime/iResolution/"
        "iMouse/iFrame uniforms (they are auto-injected). Do NOT use discard. "
        "Return ONLY a JSON object {\"glsl\": \"<the full shader>\"} with no "
        "markdown or prose outside the JSON."
    )

    user_prompt = json.dumps(
        {
            "current_glsl": current_glsl,
            "instruction": (
                "Expose all hardcoded visual constants as #define parameters at "
                "the top; keep the visual output identical."
            ),
        },
        ensure_ascii=False,
        indent=2,
    )

    response = _call_llm(
        system_prompt,
        user_prompt,
        image_paths=None,
        llm_client=llm_client,
        max_tokens=LLM_MAX_OUTPUT_TOKENS,
        response_format={"type": "json_object"},
    )
    content = _response_content(response)
    if not content:
        return None

    payload = parse_glsl_response_payload(content)
    glsl = _extract_glsl(str(payload.get("glsl") or ""))
    if not glsl:
        glsl = _extract_glsl(content)
    if not glsl:
        return None

    normalized = normalize_shadertoy_glsl(glsl)
    if "void mainImage" not in normalized.glsl:
        return None

    return {
        "glsl": normalized.glsl,
        "tunable_parameters": normalized.tunable_parameters,
        "postprocess_warnings": list(normalized.warnings),
        "param_count_before": len(extract_tunable_parameters(current_glsl)),
        "param_count_after": len(normalized.tunable_parameters),
        "_io": {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": content,
            "mode": "glsl_parameterize",
            "image_paths": [],
        },
    }
