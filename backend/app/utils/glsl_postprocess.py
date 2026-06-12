"""Post-processing helpers for LLM-generated Shadertoy GLSL.

The LLM path is intentionally more expressive than the PNG DSL path, but its
output still needs a stable contract before entering preview/scoring. These
helpers keep the shader Shadertoy-shaped, remove common wrapper conflicts, and
extract tunable parameters for later UI/manual optimization.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


SYSTEM_NAMES = {
    "iTime",
    "iResolution",
    "iMouse",
    "iFrame",
    "iChannel0",
    "iChannel1",
    "u_time",
    "u_resolution",
    "u_mouse",
}


# GLSL reserved keywords / built-in types that look like identifiers but must
# never be treated as undeclared user constants. Names here are matched against
# the case-sensitive identifier scan below.
_GLSL_RESERVED = {
    # types
    "void", "bool", "int", "uint", "float", "double",
    "vec2", "vec3", "vec4", "bvec2", "bvec3", "bvec4",
    "ivec2", "ivec3", "ivec4", "uvec2", "uvec3", "uvec4",
    "mat2", "mat3", "mat4",
    "sampler2D", "samplerCube",
    # qualifiers / control flow
    "true", "false", "if", "else", "for", "while", "do", "return", "break", "continue",
    "in", "out", "inout", "const", "uniform", "varying", "attribute",
    "precision", "highp", "mediump", "lowp", "discard",
    "struct", "layout",
    # common built-in functions / vars (the body uses them by name)
    "length", "normalize", "dot", "cross", "mix", "clamp", "smoothstep",
    "step", "abs", "sign", "min", "max", "pow", "exp", "log", "sqrt",
    "sin", "cos", "tan", "atan", "asin", "acos", "floor", "ceil",
    "fract", "mod", "round", "texture", "texture2D",
    "gl_FragCoord", "gl_FragColor", "gl_Position", "gl_PointCoord",
    "fragColor", "fragCoord", "mainImage", "main",
}


# Identifier shape considered a "named parameter": at least two characters,
# leading uppercase letter, all-caps with digits/underscores. This matches
# patterns like CENTER_X, RADIUS_GLOW, COLOR_EDGE_R, HIGHLIGHT_SIZE while
# excluding ordinary GLSL identifiers like p, uv, dist, colCenter.
_PARAM_NAME_RE = re.compile(r"\b([A-Z][A-Z0-9_]{1,})\b")

_DECL_DEFINE_RE = re.compile(r"^\s*#define\s+([A-Za-z_]\w*)\b", re.MULTILINE)
_DECL_CONST_RE = re.compile(
    r"^\s*const\s+\w+\s+([A-Za-z_]\w*)\s*(?:=|;)", re.MULTILINE
)
_DECL_UNIFORM_RE = re.compile(
    r"^\s*uniform\s+\w+\s+([A-Za-z_]\w*)\s*;", re.MULTILINE
)
_DECL_VAR_RE = re.compile(
    r"^\s*(?:in|out|attribute|varying)\s+\w+\s+([A-Za-z_]\w*)\s*;", re.MULTILINE
)


@dataclass
class GlslPostprocessResult:
    glsl: str
    tunable_parameters: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def normalize_shadertoy_glsl(source: str) -> GlslPostprocessResult:
    """Return cleaned Shadertoy GLSL and extracted ``#define`` parameters."""
    warnings: list[str] = []
    glsl = _strip_markdown(source)
    glsl = _strip_self_check(glsl)
    glsl, removed = _remove_conflicting_uniforms(glsl)
    if removed:
        warnings.append(f"removed_conflicting_uniforms:{','.join(removed)}")

    if "#version" in glsl:
        glsl = "\n".join(line for line in glsl.splitlines() if not line.strip().startswith("#version"))
        warnings.append("removed_version_directive")

    main_idx = glsl.find("void mainImage")
    if main_idx > 0:
        prefix = glsl[:main_idx]
        if prefix.strip():
            glsl = prefix.strip() + "\n\n" + glsl[main_idx:].strip()
        else:
            glsl = glsl[main_idx:].strip()

    if "void mainImage" not in glsl:
        warnings.append("missing_mainImage")

    glsl, coerced = _coerce_int_defines_to_float(glsl)
    if coerced:
        warnings.append(f"coerced_float_defines:{','.join(coerced)}")

    glsl, injected = _inject_missing_defines(glsl)
    if injected:
        warnings.append(f"auto_injected_defines:{','.join(injected)}")

    return GlslPostprocessResult(
        glsl=glsl.strip(),
        tunable_parameters=extract_tunable_parameters(glsl),
        warnings=warnings,
    )


def scan_undeclared_parameters(glsl: str) -> list[str]:
    """Return ALL_CAPS-style identifiers referenced in *glsl* but never declared.

    Used both by the auto-repair injector and by the static shader validator.
    The result is deduplicated and preserves first-seen order. Comments are
    stripped before scanning so notes like ``// TODO`` or
    ``// auto-injected (LLM omitted ...)`` are not mistaken for identifiers.
    """
    declared = _collect_declared_symbols(glsl)
    scrubbed = _strip_comments(glsl)
    seen: list[str] = []
    seen_set: set[str] = set()
    for match in _PARAM_NAME_RE.finditer(scrubbed):
        name = match.group(1)
        if name in seen_set:
            continue
        if name in declared or name in SYSTEM_NAMES or name in _GLSL_RESERVED:
            continue
        seen_set.add(name)
        seen.append(name)
    return seen


_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT_RE = re.compile(r"//[^\n]*")


def _strip_comments(glsl: str) -> str:
    """Remove ``//`` and ``/* */`` comments. Preserves line breaks for stability."""
    without_block = _BLOCK_COMMENT_RE.sub("", glsl)
    return _LINE_COMMENT_RE.sub("", without_block)


def _collect_declared_symbols(glsl: str) -> set[str]:
    declared: set[str] = set()
    for regex in (_DECL_DEFINE_RE, _DECL_CONST_RE, _DECL_UNIFORM_RE, _DECL_VAR_RE):
        declared.update(regex.findall(glsl))
    return declared


# Names whose value SHOULD stay an integer literal even though they look like
# tunable params. Matched as substrings against the uppercased identifier.
_INT_PARAM_TOKENS = (
    "SIDES", "COUNT", "INDEX", "STEP", "STEPS",
    "ITERATIONS", "ITERATION", "SAMPLES", "OCTAVE", "OCTAVES",
    "LAYERS_N", "N_LAYERS", "RING_COUNT",
)

_INT_DEFINE_RE = re.compile(
    r"^(\s*#define\s+([A-Za-z_]\w*)\s+)(-?\d+)(\s*)(//.*)?$",
    re.MULTILINE,
)


def _coerce_int_defines_to_float(glsl: str) -> tuple[str, list[str]]:
    """Rewrite ``#define NAME 2`` to ``#define NAME 2.0`` for visual tunables.

    GLSL does not implicitly promote int to float, so an LLM-written
    ``#define FALLOFF_POWER 2`` causes ``float * int`` compile errors deeper in
    the body. We only coerce when the name does not look like a true integer
    counter (sides, count, steps, iterations, etc.).
    """
    coerced: list[str] = []

    def _replace(match: "re.Match[str]") -> str:
        prefix = match.group(1)
        name = match.group(2)
        value = match.group(3)
        trailing_ws = match.group(4) or ""
        trailing_comment = match.group(5) or ""

        if _looks_like_int_param(name):
            return match.group(0)

        coerced.append(name)
        return f"{prefix}{value}.0{trailing_ws}{trailing_comment}"

    return _INT_DEFINE_RE.sub(_replace, glsl), coerced


def _looks_like_int_param(name: str) -> bool:
    upper = name.upper()
    return any(token in upper for token in _INT_PARAM_TOKENS)


def _inject_missing_defines(glsl: str) -> tuple[str, list[str]]:
    """Prepend ``#define`` defaults for any parameters referenced but undeclared.

    Returns the (possibly modified) shader and the list of injected names.
    Defaults are heuristic — meant to keep the shader compiling and visible
    rather than to exactly reproduce the artist's intent. Postprocess emits a
    warning so the downstream pipeline / UI can surface that auto-repair ran.
    """
    missing = scan_undeclared_parameters(glsl)
    if not missing:
        return glsl, []

    lines = ["// auto-injected defaults (LLM omitted these declarations)"]
    for name in missing:
        lines.append(f"#define {name} {_default_value_for(name)}")
    block = "\n".join(lines) + "\n\n"

    main_idx = glsl.find("void mainImage")
    if main_idx < 0:
        main_idx = glsl.find("void main(")
    if main_idx < 0:
        return block + glsl, missing
    return glsl[:main_idx] + block + glsl[main_idx:], missing


# Default values used by _inject_missing_defines. The mapping is intentionally
# coarse: keep the shader visible, do not try to recover the exact look. Names
# are compared with the uppercased identifier for matching.
def _default_value_for(name: str) -> str:
    upper = name.upper()

    # Color components: R / G / B / A suffix on a *_R / *_G / *_B name.
    if upper.endswith("_R"):
        return "1.0"
    if upper.endswith("_G"):
        return "0.6"
    if upper.endswith("_B"):
        return "0.8"
    if upper.endswith("_A") or upper.endswith("_ALPHA"):
        return "1.0"

    if "CENTER_X" in upper or upper.endswith("_X"):
        return "0.5" if "CENTER" in upper else "0.0"
    if "CENTER_Y" in upper or upper.endswith("_Y"):
        return "0.5" if "CENTER" in upper else "0.0"

    if "RADIUS_CORE" in upper or "CORE_RADIUS" in upper:
        return "0.15"
    if "RADIUS_GLOW" in upper or "GLOW_RADIUS" in upper or "HALO" in upper:
        return "0.45"
    # Highlight radii/sizes are tiny — check before the generic SIZE/RADIUS rule.
    if "HIGHLIGHT" in upper and ("SIZE" in upper or "RADIUS" in upper):
        return "0.05"
    if "RADIUS" in upper or "SIZE" in upper or "WIDTH" in upper or "HEIGHT" in upper:
        return "0.25"

    if "FALLOFF" in upper or "POWER" in upper:
        return "2.0"
    if "GLOW" in upper and "INTENSITY" in upper:
        return "1.2"
    if "INTENSITY" in upper or "BRIGHTNESS" in upper:
        return "1.0"
    if "OFFSET" in upper:
        return "0.0"
    if "SPEED" in upper or "TIME" in upper or "PHASE" in upper:
        return "1.0"
    if "SCALE" in upper:
        return "1.0"
    if "THRESHOLD" in upper:
        return "0.5"

    # Generic floating-point fallback.
    return "1.0"


def extract_tunable_parameters(glsl: str) -> list[dict]:
    """Extract user-facing tunables from ``#define`` statements."""
    params: list[dict] = []
    pattern = re.compile(r"^\s*#define\s+([A-Za-z_]\w*)\s+(.+?)\s*(?://.*)?$", re.MULTILINE)
    system_value_prefixes = ("iTime", "iResolution", "iMouse", "u_time", "u_resolution", "u_mouse")
    for match in pattern.finditer(glsl):
        name = match.group(1)
        raw_value = match.group(2).strip()
        if name in SYSTEM_NAMES or raw_value.startswith(system_value_prefixes):
            continue
        parsed = _parse_define_value(raw_value)
        params.append(
            {
                "name": name,
                "value": parsed,
                "raw": raw_value,
                "role": _classify_parameter(name),
            }
        )
    return params


def parse_glsl_response_payload(text: str) -> dict:
    """Parse an optional JSON envelope around LLM GLSL output.

    Supported response shapes:
    - plain GLSL text
    - fenced GLSL
    - JSON with ``glsl``/``shader`` plus optional scene analysis fields
    """
    data = _extract_json_object(text)
    if isinstance(data, dict):
        glsl = data.get("glsl") or data.get("shader") or data.get("code") or ""
        if isinstance(glsl, str) and glsl.strip():
            return {
                "glsl": glsl,
                "scene_analysis": data.get("scene_analysis") or data.get("visual_causality"),
                "technique_plan": data.get("technique_plan") or data.get("shader_techniques") or [],
                "parameter_hints": data.get("parameters") or data.get("parameter_hints") or {},
            }

    return {"glsl": text, "scene_analysis": None, "technique_plan": [], "parameter_hints": {}}


def build_visual_strategy(preprocess: dict) -> dict:
    """Infer a compact visual-causality strategy from preprocess features.

    This is intentionally not a pixel-to-shader reverse engineering step. It
    classifies what kind of visual *cause* the generator should preserve. That
    distinction matters for PNGs like glowing spheres: the image may have one
    main component, but the important signal is emissive falloff and lighting,
    not a flat circle.
    """
    photo_like = float(preprocess.get("photo_like_score", 0.0))
    texture = float(preprocess.get("texture_score", 0.0))
    gradient = float(preprocess.get("gradient_score", 0.0))
    colors = int(preprocess.get("color_count_estimate", 0))
    components = int(preprocess.get("component_count_estimate", 1))
    alpha_coverage = float(preprocess.get("alpha_coverage", 1.0))
    edge = float(preprocess.get("edge_sharpness", 0.0))

    phenomena: list[str] = []
    techniques: list[str] = []
    routing_reasons: list[str] = []
    prompt_constraints: list[str] = []

    soft_alpha = 0.03 < alpha_coverage < 0.95
    soft_edge = edge <= 0.22
    continuous_falloff = gradient >= 0.45 and soft_edge
    likely_emissive_blob = continuous_falloff and (soft_alpha or photo_like >= 0.35 or colors >= 32)
    likely_particle_field = components >= 8 or (components >= 5 and texture >= 0.30)
    likely_material = photo_like >= 0.45 and (gradient >= 0.30 or texture >= 0.20)
    likely_texture = texture >= 0.35 or colors >= 96

    if gradient >= 0.25 or colors >= 24:
        phenomena.append("continuous_color_or_brightness_falloff")
        techniques.extend(["color_ramp", "smoothstep_falloff", "radial_or_linear_gradient"])
        prompt_constraints.append("preserve continuous gradients; do not collapse them into flat fills")
    if likely_emissive_blob:
        phenomena.append("soft_glow_or_emissive_falloff")
        techniques.extend(["exponential_glow", "center_to_edge_falloff", "bloom_approximation"])
        routing_reasons.append("soft continuous falloff/glow needs direct GLSL lighting model")
        prompt_constraints.append("model glow as additive falloff and halo, not as a solid primitive")
    if texture >= 0.35:
        phenomena.append("procedural_texture_or_soft_material")
        techniques.extend(["fbm_noise", "domain_warp_lite", "soft_color_mixing"])
        routing_reasons.append("texture/material detail exceeds simple PNG DSL primitives")
    if photo_like >= 0.5:
        phenomena.append("material_lighting_not_exact_geometry")
        techniques.extend(["fake_normal", "specular_highlight", "rim_or_fresnel"])
        routing_reasons.append("material lighting should be synthesized procedurally")
    elif likely_material:
        phenomena.append("simple_material_lighting")
        techniques.extend(["fake_normal", "soft_specular", "rim_light"])
        routing_reasons.append("material-like gradient needs lighting cues")
    if components >= 6:
        phenomena.append("multi_component_or_particle_structure")
        techniques.extend(["hash_points", "cell_space", "additive_blending"])
        if likely_particle_field:
            routing_reasons.append("many components are better represented as procedural particles")
    if edge <= 0.2 and alpha_coverage < 0.9:
        phenomena.append("soft_edges_or_glow")
        techniques.extend(["exponential_glow", "alpha_falloff", "bloom_approximation"])
        prompt_constraints.append("match alpha/edge softness using smooth falloff")
    if not phenomena:
        phenomena.append("simple_2d_shape")
        techniques.extend(["sdf_shape", "solid_or_gradient_fill", "anti_aliasing"])

    direct_glsl = (
        likely_emissive_blob
        or likely_texture
        or likely_material
        or likely_particle_field
        or photo_like >= 0.60
        or colors >= 160
    )

    if not direct_glsl and gradient >= 0.55 and colors >= 32 and edge <= 0.12:
        direct_glsl = True
        routing_reasons.append("smooth multi-stop gradient benefits from analytic GLSL ramps")

    return {
        "phenomena": _dedupe(phenomena),
        "recommended_techniques": _dedupe(techniques),
        "routing_hint": "direct_glsl" if direct_glsl else "dsl_or_glsl",
        "routing_reasons": _dedupe(routing_reasons),
        "prompt_constraints": _dedupe(prompt_constraints),
        "scores": {
            "photo_like": round(photo_like, 4),
            "texture": round(texture, 4),
            "gradient": round(gradient, 4),
            "edge_sharpness": round(edge, 4),
            "alpha_coverage": round(alpha_coverage, 4),
            "color_count_estimate": colors,
            "component_count_estimate": components,
        },
        "failure_modes_to_avoid": [
            "flat_color_proxy_for_glow_or_material",
            "single_solid_shape_for_center_to_edge_brightness_falloff",
            "pure_rectangle_baseline_for_soft_complex_png",
        ],
    }


def _strip_markdown(text: str) -> str:
    stripped = text.strip()
    match = re.search(r"```(?:glsl|shader|c)?\s*(.*?)```", stripped, re.DOTALL)
    return match.group(1).strip() if match else stripped


def _strip_self_check(text: str) -> str:
    idx = text.find("[Self-check]")
    return text[:idx].strip() if idx > 0 else text.strip()


def _remove_conflicting_uniforms(glsl: str) -> tuple[str, list[str]]:
    removed: list[str] = []
    kept_lines: list[str] = []
    pattern = re.compile(r"^\s*uniform\s+\w+\s+([A-Za-z_]\w*)\s*;\s*$")
    for line in glsl.splitlines():
        match = pattern.match(line)
        if match and match.group(1) in SYSTEM_NAMES:
            removed.append(match.group(1))
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines), removed


def _parse_define_value(raw: str) -> Any:
    if raw.startswith("vec"):
        return raw
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw.strip('"')


def _classify_parameter(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("color", "col", "hue", "sat", "rgb")):
        return "color"
    if any(token in lowered for token in ("center", "pos", "x", "y", "offset")):
        return "position"
    if any(token in lowered for token in ("radius", "size", "width", "height", "scale")):
        return "geometry"
    if any(token in lowered for token in ("glow", "intensity", "power", "brightness", "alpha")):
        return "lighting"
    if any(token in lowered for token in ("speed", "time", "phase")):
        return "animation"
    return "numeric"


def _extract_json_object(text: str) -> Any:
    stripped = text.strip()
    fenced = re.search(r"```json\s*(.*?)```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(stripped[start:end + 1])
    except json.JSONDecodeError:
        return None


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
