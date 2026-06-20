"""Deterministic DSL-to-GLSL compiler for PNG-to-Shader.

Principle: same DSL always produces identical GLSL. No randomness.
Schema-valid input that fails to compile is a compiler bug.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class CompileResult:
    glsl: str
    success: bool
    errors: list[str] = field(default_factory=list)
    layer_errors: dict[str, str] = field(default_factory=dict)  # layer_id -> error


# ---------------------------------------------------------------------------
# GLSL header / SDF library (fixed, deterministic)
# ---------------------------------------------------------------------------

_GLSL_HEADER = """\
#version 300 es
precision highp float;
out vec4 fragColor;
uniform vec2 iResolution;
uniform float iTime;
"""

_SDF_LIB = """\
// --- SDF functions ---
float sdCircle(vec2 p, float r) { return length(p) - r; }
float sdBox(vec2 p, vec2 b) { vec2 d = abs(p) - b; return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0); }
float sdRoundedBox(vec2 p, vec2 b, float r) { vec2 q = abs(p) - b + r; return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r; }
float sdEllipse(vec2 p, vec2 ab) { p = abs(p); if (p.x > p.y) { p = p.yx; ab = ab.yx; } float l = ab.y * ab.y - ab.x * ab.x; float m = ab.x * p.x / l; float n = ab.y * p.y / l; float m2 = m * m; float n2 = n * n; float c = (m2 + n2 - 1.0) / 3.0; float c3 = c * c * c; float q = c3 + m2 * n2 * 2.0; float d = c3 + m2 * n2; float g = m + m * n2; float co; if (d < 0.0) { float h = acos(q / c3) / 3.0; float s = cos(h); float t = sin(h) * sqrt(3.0); float rx = sqrt(-c * (s + t + 2.0) + m2); float ry = sqrt(-c * (s - t + 2.0) + m2); co = (ry + sign(l) * rx + abs(g) / (rx * ry) - m) / 2.0; } else { float h = 2.0 * m * n * sqrt(d); float s = sign(q + h) * pow(abs(q + h), 1.0/3.0); float t = sign(q - h) * pow(abs(q - h), 1.0/3.0); float rx = -(s + t) - c * 4.0 + 2.0 * m2; float ry = (s - t) * sqrt(3.0); float rm = sqrt(rx * rx + ry * ry); co = (ry / sqrt(rm - rx) + 2.0 * g / rm - m) / 2.0; } vec2 r2 = ab * vec2(co, sqrt(1.0 - co * co)); return length(r2 - p) * sign(p.y - r2.y); }
float sdRing(vec2 p, float r, float thickness) { return abs(length(p) - r) - thickness; }
float sdPolygon(vec2 p, float r, int n) { float a = atan(p.x, p.y) + 3.14159265; float s = 6.28318530 / float(n); float f = s * floor(a / s + 0.5); float d = length(p) * cos(a - f) - r * cos(3.14159265 / float(n)); return d; }
"""


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """Convert a CSS hex color string to (r, g, b) floats in [0, 1].

    Accepts #RGB, #RRGGBB, and #RRGGBBAA. For 8-digit RGBA the alpha is
    dropped and the leading RGB is used (collapsing to white was a bug that
    also diverged from the renderer). Unparseable colors fall back to white.
    """
    h = str(hex_color).lstrip("#")
    if len(h) == 3:
        h = h[0] * 2 + h[1] * 2 + h[2] * 2
    elif len(h) == 8:
        h = h[0:6]
    if len(h) != 6:
        h = "ffffff"
    try:
        return int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0
    except ValueError:
        return 1.0, 1.0, 1.0


def _hex_to_vec3(hex_color: str) -> str:
    """Convert a CSS hex color string to a GLSL vec3 literal."""
    r, g, b = _hex_to_rgb(hex_color)
    return f"vec3({r:.6f}, {g:.6f}, {b:.6f})"


def _float(v) -> str:
    """Format a number as a GLSL float literal.

    Non-finite values (inf/nan) have no valid GLSL float literal — emitting
    'inf'/'nan' produces a shader that fails to link. Clamp them to a finite
    sentinel (0.0) so the compiler always emits parseable GLSL. Callers that
    care surface a warning/error separately; value-level validation
    (validate_dsl) rejects non-finite numerics up front.
    """
    f = float(v)
    if not math.isfinite(f):
        f = 0.0
    return f"{f:.6f}"


def _vec2(lst) -> str:
    """Format a 2-element list as a GLSL vec2 literal."""
    return f"vec2({_float(lst[0])}, {_float(lst[1])})"


def _define_name(layer_idx: int, param: str) -> str:
    """Generate a #define parameter name for a layer.

    This is the single source of truth for define name formatting.
    Both _generate_param_defines and code generation functions
    (_sdf_for_layer, _fill_code, etc.) must use this function.
    """
    return f"L{layer_idx}_{param}"


# ---------------------------------------------------------------------------
# Parameter #define generation
# ---------------------------------------------------------------------------

def _generate_param_defines(dsl: dict) -> str:
    """Generate #define declarations for all tunable DSL parameters.

    These defines are referenced in the generated GLSL body, allowing
    frontend tools to adjust values by rewriting #define lines and
    recompiling the shader.
    """
    lines: list[str] = []
    canvas = dsl.get("canvas", {})
    bg_hex = canvas.get("background", "#000000") if isinstance(canvas, dict) else "#000000"
    bg_r, bg_g, bg_b = _hex_to_rgb(bg_hex)
    lines.append(f"#define bg_r {bg_r:.6f}")
    lines.append(f"#define bg_g {bg_g:.6f}")
    lines.append(f"#define bg_b {bg_b:.6f}")

    layers = dsl.get("layers", [])
    if not isinstance(layers, list):
        layers = []

    for idx, layer in enumerate(layers):
        if not isinstance(layer, dict):
            continue
        p = layer.get("params") or {}
        center = p.get("center", [0.5, 0.5])
        cx, cy = float(center[0]), float(center[1])
        ptype = layer.get("type", "box")
        layer_id = layer.get("id", f"layer_{idx}")
        opacity = float(layer.get("opacity", 1.0))

        lines.append(f"// layer {idx}: {layer_id} ({ptype})")
        lines.append(f"#define {_define_name(idx, 'center_x')} {cx:.6f}")
        lines.append(f"#define {_define_name(idx, 'center_y')} {cy:.6f}")

        if ptype == "circle":
            lines.append(f"#define {_define_name(idx, 'radius')} {_float(p.get('radius', 0.3))}")
        elif ptype == "ellipse":
            ab = p.get("ab", [0.3, 0.2])
            lines.append(f"#define {_define_name(idx, 'ab_x')} {_float(ab[0])}")
            lines.append(f"#define {_define_name(idx, 'ab_y')} {_float(ab[1])}")
        elif ptype in ("box", "roundedBox"):
            size = p.get("size", [0.3, 0.2])
            lines.append(f"#define {_define_name(idx, 'size_x')} {_float(size[0])}")
            lines.append(f"#define {_define_name(idx, 'size_y')} {_float(size[1])}")
            if ptype == "roundedBox":
                lines.append(f"#define {_define_name(idx, 'corner_radius')} {_float(p.get('radius', 0.05))}")
        elif ptype == "ring":
            lines.append(f"#define {_define_name(idx, 'radius')} {_float(p.get('radius', 0.3))}")
            lines.append(f"#define {_define_name(idx, 'thickness')} {_float(p.get('thickness', 0.02))}")
        elif ptype == "polygon":
            lines.append(f"#define {_define_name(idx, 'radius')} {_float(p.get('radius', 0.3))}")

        lines.append(f"#define {_define_name(idx, 'opacity')} {opacity:.6f}")

        fill = layer.get("fill", {"type": "solid", "color": "#ffffff"})
        ftype = fill.get("type", "solid")
        if ftype == "solid":
            r, g, b = _hex_to_rgb(fill.get("color", "#ffffff"))
            lines.append(f"#define {_define_name(idx, 'fill_r')} {r:.6f}")
            lines.append(f"#define {_define_name(idx, 'fill_g')} {g:.6f}")
            lines.append(f"#define {_define_name(idx, 'fill_b')} {b:.6f}")
        elif ftype in ("linearGradient", "radialGradient"):
            stops = fill.get("stops", [])
            for si, stop in enumerate(stops):
                r, g, b = _hex_to_rgb(stop.get("color", "#ffffff"))
                pos = float(stop.get("position", si / max(1, len(stops) - 1)))
                lines.append(f"#define {_define_name(idx, f'stop_{si}_r')} {r:.6f}")
                lines.append(f"#define {_define_name(idx, f'stop_{si}_g')} {g:.6f}")
                lines.append(f"#define {_define_name(idx, f'stop_{si}_b')} {b:.6f}")
                lines.append(f"#define {_define_name(idx, f'stop_{si}_pos')} {pos:.6f}")
            if ftype == "linearGradient":
                direction = fill.get("direction", [1.0, 0.0])
                lines.append(f"#define {_define_name(idx, 'dir_x')} {_float(direction[0])}")
                lines.append(f"#define {_define_name(idx, 'dir_y')} {_float(direction[1])}")
            else:
                gc = fill.get("center", [0.5, 0.5])
                lines.append(f"#define {_define_name(idx, 'grad_cx')} {_float(gc[0])}")
                lines.append(f"#define {_define_name(idx, 'grad_cy')} {_float(gc[1])}")

        transform = layer.get("transform")
        if isinstance(transform, dict):
            ttype = transform.get("type", "")
            if ttype == "translate":
                lines.append(f"#define {_define_name(idx, 'translate_x')} {_float(transform.get('x', 0.0))}")
                lines.append(f"#define {_define_name(idx, 'translate_y')} {_float(transform.get('y', 0.0))}")
            elif ttype == "rotate":
                lines.append(f"#define {_define_name(idx, 'rotate_angle')} {_float(transform.get('angle', 0.0))}")
            elif ttype == "scale":
                # Mirror the renderer's `float(...) or 1.0`: a zero scale would
                # emit `p /= vec2(0.0, ...)` (div-by-zero); the renderer guards
                # it by falling back to 1.0, so the compiler must too.
                sx = float(transform.get("x", 1.0)) or 1.0
                sy = float(transform.get("y", 1.0)) or 1.0
                lines.append(f"#define {_define_name(idx, 'scale_x')} {_float(sx)}")
                lines.append(f"#define {_define_name(idx, 'scale_y')} {_float(sy)}")

        for eidx, effect in enumerate(layer.get("effects", []) or []):
            if not isinstance(effect, dict):
                continue
            etype = effect.get("type", "")
            if etype == "glow":
                lines.append(f"#define {_define_name(idx, f'glow_{eidx}_intensity')} {_float(effect.get('intensity', 5.0))}")
                r, g, b = _hex_to_rgb(effect.get("color", "#ffffff"))
                lines.append(f"#define {_define_name(idx, f'glow_{eidx}_r')} {r:.6f}")
                lines.append(f"#define {_define_name(idx, f'glow_{eidx}_g')} {g:.6f}")
                lines.append(f"#define {_define_name(idx, f'glow_{eidx}_b')} {b:.6f}")
            elif etype == "vignette":
                lines.append(f"#define {_define_name(idx, f'vignette_{eidx}_strength')} {_float(effect.get('strength', 0.8))}")
            elif etype == "grain":
                lines.append(f"#define {_define_name(idx, f'grain_{eidx}_amount')} {_float(effect.get('amount', 0.05))}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# SDF code generation per primitive
# ---------------------------------------------------------------------------

def _sdf_for_layer(layer: dict, var: str, idx: int = 0) -> str:
    """Return a GLSL statement computing the signed distance for a layer.

    References ``#define`` parameters (e.g. ``L0_radius``) rather than
    inlining literal values, so the frontend can tune them at runtime.
    """
    ptype = layer.get("type", "box")

    if ptype == "circle":
        return f"float {var} = sdCircle(p, {_define_name(idx, 'radius')});"

    elif ptype == "ellipse":
        return f"float {var} = sdEllipse(p, vec2({_define_name(idx, 'ab_x')}, {_define_name(idx, 'ab_y')}));"

    elif ptype == "box":
        return f"float {var} = sdBox(p, vec2({_define_name(idx, 'size_x')} * 0.5, {_define_name(idx, 'size_y')} * 0.5));"

    elif ptype == "roundedBox":
        return f"float {var} = sdRoundedBox(p, vec2({_define_name(idx, 'size_x')} * 0.5, {_define_name(idx, 'size_y')} * 0.5), {_define_name(idx, 'corner_radius')});"

    elif ptype == "ring":
        return f"float {var} = sdRing(p, {_define_name(idx, 'radius')}, {_define_name(idx, 'thickness')});"

    elif ptype == "polygon":
        # Mirror the renderer's max(3, int(sides)): fewer than 3 sides makes
        # sdPolygon divide by float(n)==0 in GLSL (success=True, broken shader).
        n = max(3, int((layer.get("params") or {}).get("sides", 6)))
        return f"float {var} = sdPolygon(p, {_define_name(idx, 'radius')}, {n});"

    else:
        return f"float {var} = sdBox(p, vec2(0.200000, 0.150000));"


# ---------------------------------------------------------------------------
# Transform code generation
# ---------------------------------------------------------------------------

def _transform_code(transform: dict | None, uv_var: str, center: list, idx: int = 0) -> str:
    """Return GLSL that computes ``p`` (the position in primitive space).

    References ``#define`` parameters (e.g. ``L0_center_x``) for center
    and transform values.
    """
    lines = [f"vec2 p = {uv_var} - vec2({_define_name(idx, 'center_x')}, {_define_name(idx, 'center_y')});"]

    if transform is None:
        return "\n".join(lines)

    ttype = transform.get("type", "")

    if ttype == "translate":
        lines.append(f"p -= vec2({_define_name(idx, 'translate_x')}, {_define_name(idx, 'translate_y')});")

    elif ttype == "rotate":
        ca = f"_ca_{idx}"
        sa = f"_sa_{idx}"
        lines.append(f"float {ca} = cos({_define_name(idx, 'rotate_angle')}); float {sa} = sin({_define_name(idx, 'rotate_angle')});")
        lines.append(f"p = vec2({ca} * p.x + {sa} * p.y, -{sa} * p.x + {ca} * p.y);")

    elif ttype == "scale":
        lines.append(f"p /= vec2({_define_name(idx, 'scale_x')}, {_define_name(idx, 'scale_y')});")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fill code generation
# ---------------------------------------------------------------------------

def _interpolate_stops_glsl(stops: list[dict], t_var: str, idx: int = 0) -> str:
    """Build a nested mix() expression for multi-stop gradient interpolation.

    References ``#define`` parameters (e.g. ``L0_stop_0_r``) for colors
    and positions rather than inlining literal values.
    """
    if len(stops) < 2:
        if stops:
            return f"vec3({_define_name(idx, 'stop_0_r')}, {_define_name(idx, 'stop_0_g')}, {_define_name(idx, 'stop_0_b')})"
        return "vec3(1.0, 1.0, 1.0)"

    if len(stops) == 2:
        c0 = f"vec3({_define_name(idx, 'stop_0_r')}, {_define_name(idx, 'stop_0_g')}, {_define_name(idx, 'stop_0_b')})"
        c1 = f"vec3({_define_name(idx, 'stop_1_r')}, {_define_name(idx, 'stop_1_g')}, {_define_name(idx, 'stop_1_b')})"
        return (
            f"mix({c0}, {c1}, "
            f"smoothstep({_define_name(idx, 'stop_0_pos')}, {_define_name(idx, 'stop_1_pos')}, {t_var}))"
        )

    last = len(stops) - 1
    result = f"vec3({_define_name(idx, f'stop_{last}_r')}, {_define_name(idx, f'stop_{last}_g')}, {_define_name(idx, f'stop_{last}_b')})"
    for i in range(last - 1, -1, -1):
        ci = f"vec3({_define_name(idx, f'stop_{i}_r')}, {_define_name(idx, f'stop_{i}_g')}, {_define_name(idx, f'stop_{i}_b')})"
        result = f"mix({ci}, {result}, smoothstep({_define_name(idx, f'stop_{i}_pos')}, {_define_name(idx, f'stop_{i + 1}_pos')}, {t_var}))"
    return result


def _fill_code(fill: dict, sdf_var: str, fill_color_var: str, idx: int = 0) -> str:
    """Return GLSL computing the fill color (vec3) into fill_color_var.

    References ``#define`` parameters for colors, gradient stops, and
    gradient direction/center.
    """
    ftype = fill.get("type", "solid")

    if ftype == "solid":
        return f"vec3 {fill_color_var} = vec3({_define_name(idx, 'fill_r')}, {_define_name(idx, 'fill_g')}, {_define_name(idx, 'fill_b')});"

    elif ftype == "linearGradient":
        stops = fill.get("stops", [
            {"color": "#000000", "position": 0.0},
            {"color": "#ffffff", "position": 1.0},
        ])
        t_var = f"_grad_t_{idx}"
        color_expr = _interpolate_stops_glsl(stops, t_var, idx)
        return (
            f"float {t_var} = clamp(dot(uv, vec2({_define_name(idx, 'dir_x')}, {_define_name(idx, 'dir_y')})), 0.0, 1.0);\n"
            f"vec3 {fill_color_var} = {color_expr};"
        )

    elif ftype == "radialGradient":
        stops = fill.get("stops", [
            {"color": "#ffffff", "position": 0.0},
            {"color": "#000000", "position": 1.0},
        ])
        t_var = f"_rad_t_{idx}"
        color_expr = _interpolate_stops_glsl(stops, t_var, idx)
        return (
            f"float {t_var} = clamp(length(uv - vec2({_define_name(idx, 'grad_cx')}, {_define_name(idx, 'grad_cy')})) * 2.0, 0.0, 1.0);\n"
            f"vec3 {fill_color_var} = {color_expr};"
        )

    else:
        return f"vec3 {fill_color_var} = vec3(1.0, 1.0, 1.0);"


# ---------------------------------------------------------------------------
# Effect code generation
# ---------------------------------------------------------------------------

def _effect_code(effect: dict, sdf_var: str, color_var: str, uv_var: str, idx: int = 0, effect_idx: int = 0) -> str:
    """Return GLSL applying a post-effect in-place on color_var.

    References ``#define`` parameters for effect values.
    """
    etype = effect.get("type", "")
    suffix = f"{idx}_{effect_idx}"

    if etype == "glow":
        glow_var = f"_glow_{suffix}"
        intensity_name = _define_name(idx, f"glow_{effect_idx}_intensity")
        r_name = _define_name(idx, f"glow_{effect_idx}_r")
        g_name = _define_name(idx, f"glow_{effect_idx}_g")
        b_name = _define_name(idx, f"glow_{effect_idx}_b")
        return (
            f"{{\n"
            f"  float {glow_var} = exp(-max(0.0, {sdf_var}) * {intensity_name});\n"
            f"  {color_var}.rgb += vec3({r_name}, {g_name}, {b_name}) * {glow_var};\n"
            f"}}"
        )

    elif etype == "vignette":
        v_var = f"_v_{suffix}"
        strength_name = _define_name(idx, f"vignette_{effect_idx}_strength")
        return (
            f"{{\n"
            f"  float {v_var} = 1.0 - smoothstep(0.5, 0.8, length({uv_var} - 0.5));\n"
            f"  {color_var}.rgb *= mix(1.0, {v_var}, {strength_name});\n"
            f"}}"
        )

    elif etype == "grain":
        grain_var = f"_grain_{suffix}"
        amount_name = _define_name(idx, f"grain_{effect_idx}_amount")
        return (
            f"{{\n"
            f"  float {grain_var} = fract(sin(dot({uv_var}, vec2(12.9898, 78.233))) * 43758.5453);\n"
            f"  {color_var}.rgb += ({grain_var} - 0.5) * {amount_name};\n"
            f"}}"
        )

    return ""


# ---------------------------------------------------------------------------
# Per-layer code generation
# ---------------------------------------------------------------------------

def _layer_code(layer: dict, idx: int) -> str:
    """Return the GLSL block for a single DSL layer (wrapped in braces)."""
    params = layer.get("params") or {}
    center = params.get("center", [0.5, 0.5])
    transform = layer.get("transform")
    fill = layer.get("fill", {"type": "solid", "color": "#ffffff"})
    effects = layer.get("effects") or []

    sdf_var = f"sdf_{idx}"
    fill_color_var = f"fill_col_{idx}"
    blend_alpha_var = f"alpha_{idx}"
    layer_color_var = f"layer_col_{idx}"

    transform_lines = _transform_code(transform, "uv", center, idx)
    sdf_line = _sdf_for_layer(layer, sdf_var, idx)
    fill_lines = _fill_code(fill, sdf_var, fill_color_var, idx)

    alpha_line = (
        f"float {blend_alpha_var} = clamp(1.0 - smoothstep(-0.005, 0.005, {sdf_var}), 0.0, 1.0) * {_define_name(idx, 'opacity')};"
    )

    layer_color_line = f"vec4 {layer_color_var} = vec4({fill_color_var}, {blend_alpha_var});"

    effect_lines = []
    for eidx, effect in enumerate(effects):
        ec = _effect_code(effect, sdf_var, layer_color_var, "uv", idx, eidx)
        if ec:
            effect_lines.append(ec)

    # Alpha-blend layer onto accumulated color
    blend_line = (
        f"acc_color = mix(acc_color, {layer_color_var}.rgb, {layer_color_var}.a);"
    )

    parts = [
        "  {",
        f"  // layer {idx}: {layer.get('id', '')} ({layer.get('type', '')})",
        *["  " + ln for ln in transform_lines.splitlines()],
        f"  {sdf_line}",
        *["  " + ln for ln in fill_lines.splitlines()],
        f"  {alpha_line}",
        f"  {layer_color_line}",
    ]
    for el in effect_lines:
        parts.extend(["  " + ln for ln in el.splitlines()])
    parts.append(f"  {blend_line}")
    parts.append("  }")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main compile function
# ---------------------------------------------------------------------------

def compile_dsl(dsl: dict) -> CompileResult:
    """Compile a DSL dict into a WebGL2 GLSL fragment shader.

    Always returns a CompileResult. If critical errors occur the glsl
    field will still contain a fallback background-only shader.
    """
    errors: list[str] = []
    layer_errors: dict[str, str] = {}

    if not isinstance(dsl, dict):
        errors.append("DSL must be a dict")
        return CompileResult(glsl=_background_only_shader("#000000"), success=False, errors=errors)

    canvas = dsl.get("canvas", {})
    bg_color = canvas.get("background", "#000000") if isinstance(canvas, dict) else "#000000"

    layers = dsl.get("layers", [])
    if not isinstance(layers, list):
        layers = []

    layer_blocks: list[str] = []
    for idx, layer in enumerate(layers):
        layer_id = layer.get("id", f"layer_{idx}") if isinstance(layer, dict) else f"layer_{idx}"
        try:
            block = _layer_code(layer, idx)
            layer_blocks.append(block)
        except Exception as exc:
            msg = f"layer compile error: {exc}"
            layer_errors[layer_id] = msg
            errors.append(f"[{layer_id}] {msg}")

    # _generate_param_defines coerces param values with float()/int() and can
    # raise TypeError/ValueError on schema-valid-but-bad input (e.g. radius=
    # 'big', center=0.5). It runs outside the per-layer try/except above, so an
    # unguarded call would propagate and crash the scoring/refinement worker
    # thread. compile_dsl must ALWAYS return a CompileResult.
    try:
        param_defines = _generate_param_defines(dsl)
    except Exception as exc:
        errors.append(f"param define generation error: {exc}")
        return CompileResult(
            glsl=_background_only_shader(bg_color),
            success=False,
            errors=errors,
            layer_errors=layer_errors,
        )

    main_lines = [
        "void main() {",
        "  vec2 uv = gl_FragCoord.xy / iResolution.xy;",
        "  uv.y = 1.0 - uv.y;",
        "  vec3 acc_color = vec3(bg_r, bg_g, bg_b);",
    ]

    for block in layer_blocks:
        main_lines.append(block)

    main_lines += [
        "  fragColor = vec4(acc_color, 1.0);",
        "}",
    ]

    glsl = _GLSL_HEADER + "\n" + param_defines + "\n\n" + _SDF_LIB + "\n" + "\n".join(main_lines) + "\n"

    success = len(errors) == 0
    return CompileResult(glsl=glsl, success=success, errors=errors, layer_errors=layer_errors)


def _background_only_shader(bg_color: str) -> str:
    """Return a minimal valid shader that just outputs the background color."""
    main_lines = [
        "void main() {",
        f"  fragColor = vec4({_hex_to_vec3(bg_color)}, 1.0);",
        "}",
    ]
    return _GLSL_HEADER + "\n" + _SDF_LIB + "\n" + "\n".join(main_lines) + "\n"
