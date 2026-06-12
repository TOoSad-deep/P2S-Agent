"""Vectorized NumPy/Pillow raster renderer for PNG-to-Shader DSL.

Deterministic backend render used for objective metrics and artifacts.
Mirrors the DSL compiler's GLSL semantics (same SDFs, smoothstep gradient
interpolation, effects) without a browser. All per-pixel math runs as
NumPy array operations over a uv grid — ~100x faster than the previous
per-pixel Python loops, which matters because every optimizer step and
refinement iteration renders.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

EDGE_SOFTNESS = 0.005  # smoothstep half-width for shape edges, in uv units


def render_dsl_to_image(
    dsl: dict,
    output_path: "str | Path",
    *,
    width: int | None = None,
    height: int | None = None,
) -> Path:
    """Rasterize a DSL scene to an RGBA PNG and return the output path."""
    canvas = dsl.get("canvas", {}) if isinstance(dsl, dict) else {}
    w = int(width or canvas.get("width", 512))
    h = int(height or canvas.get("height", 512))
    bg = _hex_to_rgb(canvas.get("background", "#000000"))

    xs = np.arange(w, dtype=np.float32) / max(1, w - 1)
    ys = np.arange(h, dtype=np.float32) / max(1, h - 1)
    u, v = np.meshgrid(xs, ys)  # each (h, w)

    acc = np.empty((h, w, 3), dtype=np.float32)
    acc[..., 0] = bg[0]
    acc[..., 1] = bg[1]
    acc[..., 2] = bg[2]

    layers = dsl.get("layers", []) if isinstance(dsl, dict) else []
    if not isinstance(layers, list):
        layers = []

    for layer in layers:
        if not isinstance(layer, dict):
            continue
        sdf = _sdf_for_layer(layer, u, v)
        layer_rgb = _fill_rgb(layer.get("fill", {}), u, v)
        opacity = min(1.0, max(0.0, float(layer.get("opacity", 1.0))))
        alpha = (1.0 - _smoothstep(-EDGE_SOFTNESS, EDGE_SOFTNESS, sdf)) * opacity

        for effect in layer.get("effects", []) or []:
            if isinstance(effect, dict):
                _apply_effect(effect, sdf, u, v, layer_rgb)

        acc = acc * (1.0 - alpha)[..., None] + layer_rgb * alpha[..., None]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rgba = np.empty((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = (np.clip(acc, 0.0, 255.0) + 0.5).astype(np.uint8)
    rgba[..., 3] = 255
    Image.fromarray(rgba, "RGBA").save(out)
    return out


# ---------------------------------------------------------------------------
# SDF evaluation (grid-valued)
# ---------------------------------------------------------------------------


def _sdf_for_layer(layer: dict, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    px, py = _transform_point(layer, u, v)
    ptype = layer.get("type", "box")
    params = layer.get("params") or {}

    if ptype == "circle":
        return np.hypot(px, py) - float(params.get("radius", 0.3))
    if ptype == "ellipse":
        ab = params.get("ab", [0.3, 0.2])
        return _sd_ellipse(px, py, max(float(ab[0]), 1e-6), max(float(ab[1]), 1e-6))
    if ptype == "roundedBox":
        size = params.get("size", [0.3, 0.2])
        return _sd_rounded_box(
            px, py, float(size[0]) * 0.5, float(size[1]) * 0.5,
            float(params.get("radius", 0.05)),
        )
    if ptype == "ring":
        radius = float(params.get("radius", 0.3))
        thickness = float(params.get("thickness", 0.02))
        return np.abs(np.hypot(px, py) - radius) - thickness
    if ptype == "polygon":
        radius = float(params.get("radius", 0.3))
        sides = max(3, int(params.get("sides", 6)))
        angle = np.arctan2(px, py) + np.pi
        step = 2.0 * math.pi / sides
        facet = step * np.floor(angle / step + 0.5)
        return np.hypot(px, py) * np.cos(angle - facet) - radius * math.cos(math.pi / sides)

    size = params.get("size", [0.3, 0.2])
    return _sd_box(px, py, float(size[0]) * 0.5, float(size[1]) * 0.5)


def _transform_point(layer: dict, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    params = layer.get("params") or {}
    center = params.get("center", [0.5, 0.5])
    px = u - float(center[0])
    py = v - float(center[1])
    transform = layer.get("transform")
    if not isinstance(transform, dict):
        return px, py

    ttype = transform.get("type")
    if ttype == "translate":
        px = px - float(transform.get("x", 0.0))
        py = py - float(transform.get("y", 0.0))
    elif ttype == "rotate":
        angle = float(transform.get("angle", 0.0))
        ca, sa = math.cos(angle), math.sin(angle)
        px, py = ca * px + sa * py, -sa * px + ca * py
    elif ttype == "scale":
        sx = float(transform.get("x", 1.0)) or 1.0
        sy = float(transform.get("y", 1.0)) or 1.0
        px = px / sx
        py = py / sy
    return px, py


def _sd_box(px: np.ndarray, py: np.ndarray, hx: float, hy: float) -> np.ndarray:
    dx = np.abs(px) - hx
    dy = np.abs(py) - hy
    outside = np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0))
    inside = np.minimum(np.maximum(dx, dy), 0.0)
    return outside + inside


def _sd_rounded_box(px: np.ndarray, py: np.ndarray, hx: float, hy: float, radius: float) -> np.ndarray:
    qx = np.abs(px) - hx + radius
    qy = np.abs(py) - hy + radius
    outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
    inside = np.minimum(np.maximum(qx, qy), 0.0)
    return outside + inside - radius


def _sd_ellipse(px: np.ndarray, py: np.ndarray, a: float, b: float) -> np.ndarray:
    """Exact ellipse SDF (iq), vectorized mirror of the previous scalar code.

    Both branches of the closed-form solve are computed on the full grid and
    selected with np.where; intermediate values in the unused branch are
    clamped so they never produce NaN/inf.
    """
    apx = np.abs(px)
    apy = np.abs(py)
    swap = apx > apy
    sx = np.where(swap, apy, apx)
    sy = np.where(swap, apx, apy)
    sa = np.where(swap, b, a)
    sb = np.where(swap, a, b)

    l = sb * sb - sa * sa
    circle_like = np.abs(l) < 1e-12
    l_safe = np.where(circle_like, 1.0, l)

    m = sa * sx / l_safe
    n = sb * sy / l_safe
    m2 = m * m
    n2 = n * n
    c = (m2 + n2 - 1.0) / 3.0
    c3 = c * c * c
    q = c3 + m2 * n2 * 2.0
    d = c3 + m2 * n2
    g = m + m * n2

    # branch 1: d < 0
    # sign-preserving guard so the discarded d>=0 branch never divides by zero
    c3_safe = np.where(c3 >= 0, 1e-30, -1e-30)
    c3_div = np.where(np.abs(c3) < 1e-30, c3_safe, c3)
    h1 = np.arccos(np.clip(q / c3_div, -1.0, 1.0)) / 3.0
    s1 = np.cos(h1)
    t1 = np.sin(h1) * math.sqrt(3.0)
    rx1 = np.sqrt(np.maximum(0.0, -c * (s1 + t1 + 2.0) + m2))
    ry1 = np.sqrt(np.maximum(0.0, -c * (s1 - t1 + 2.0) + m2))
    co1 = (ry1 + np.where(l > 0, 1.0, -1.0) * rx1 + np.abs(g) / np.maximum(rx1 * ry1, 1e-12) - m) / 2.0

    # branch 2: d >= 0
    h2 = 2.0 * m * n * np.sqrt(np.maximum(0.0, d))
    s2 = np.cbrt(q + h2)
    t2 = np.cbrt(q - h2)
    rx2 = -(s2 + t2) - c * 4.0 + 2.0 * m2
    ry2 = (s2 - t2) * math.sqrt(3.0)
    rm = np.sqrt(rx2 * rx2 + ry2 * ry2)
    co2 = (ry2 / np.sqrt(np.maximum(rm - rx2, 1e-12)) + 2.0 * g / np.maximum(rm, 1e-12) - m) / 2.0

    co = np.where(d < 0.0, co1, co2)
    r2x = sa * co
    r2y = sb * np.sqrt(np.maximum(0.0, 1.0 - co * co))
    dist = np.hypot(r2x - sx, r2y - sy)
    sdf = dist * np.where(sy >= r2y, 1.0, -1.0)
    return np.where(circle_like, np.hypot(apx, apy) - sa, sdf)


# ---------------------------------------------------------------------------
# Fills (grid-valued)
# ---------------------------------------------------------------------------


def _fill_rgb(fill: Any, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = u.shape
    if not isinstance(fill, dict):
        return np.full((h, w, 3), 255.0, dtype=np.float32)
    ftype = fill.get("type", "solid")
    if ftype == "solid":
        r, g, b = _hex_to_rgb(fill.get("color", "#ffffff"))
        out = np.empty((h, w, 3), dtype=np.float32)
        out[..., 0] = r
        out[..., 1] = g
        out[..., 2] = b
        return out

    stops = fill.get("stops") or [
        {"color": "#000000", "position": 0.0},
        {"color": "#ffffff", "position": 1.0},
    ]
    if not isinstance(stops, list) or len(stops) < 2:
        return np.full((h, w, 3), 255.0, dtype=np.float32)

    if ftype == "linearGradient":
        direction = fill.get("direction", [1.0, 0.0])
        t = np.clip(u * float(direction[0]) + v * float(direction[1]), 0.0, 1.0)
    elif ftype == "radialGradient":
        center = fill.get("center", [0.5, 0.5])
        t = np.clip(np.hypot(u - float(center[0]), v - float(center[1])) * 2.0, 0.0, 1.0)
    else:
        t = np.zeros_like(u)

    return _interpolate_stops_color(stops, t)


def _interpolate_stops_color(stops: list[dict], t: np.ndarray) -> np.ndarray:
    """Piecewise smoothstep interpolation, mirroring the GLSL compiler."""
    colors = np.array(
        [_hex_to_rgb(s.get("color", "#ffffff")) for s in stops], dtype=np.float32
    )
    positions = [
        float(s.get("position", i / max(1, len(stops) - 1))) for i, s in enumerate(stops)
    ]

    out = np.empty(t.shape + (3,), dtype=np.float32)
    out[...] = colors[0]
    for i in range(len(stops) - 1):
        lo, hi = positions[i], positions[i + 1]
        seg = (t >= lo) & (t <= hi)
        rng = hi - lo
        if rng < 1e-6:
            seg_t = np.zeros_like(t)
        else:
            seg_t = np.clip((t - lo) / rng, 0.0, 1.0)
        seg_t = seg_t * seg_t * (3.0 - 2.0 * seg_t)
        blend = colors[i] * (1.0 - seg_t[..., None]) + colors[i + 1] * seg_t[..., None]
        out[seg] = blend[seg]
    out[t >= positions[-1]] = colors[-1]
    return out


# ---------------------------------------------------------------------------
# Effects (in-place on the layer RGB grid)
# ---------------------------------------------------------------------------


def _apply_effect(effect: dict, sdf: np.ndarray, u: np.ndarray, v: np.ndarray, rgb: np.ndarray) -> None:
    etype = effect.get("type")
    if etype == "glow":
        intensity = float(effect.get("intensity", 5.0))
        glow = np.exp(-np.maximum(0.0, sdf) * intensity)
        r, g, b = _hex_to_rgb(effect.get("color", "#ffffff"))
        rgb[..., 0] = np.minimum(255.0, rgb[..., 0] + r * glow)
        rgb[..., 1] = np.minimum(255.0, rgb[..., 1] + g * glow)
        rgb[..., 2] = np.minimum(255.0, rgb[..., 2] + b * glow)
    elif etype == "vignette":
        strength = min(1.0, max(0.0, float(effect.get("strength", 0.8))))
        dist = np.hypot(u - 0.5, v - 0.5)
        vignette = 1.0 - _smoothstep(0.5, 0.8, dist)
        factor = (1.0 - strength) + vignette * strength
        rgb *= factor[..., None]
    elif etype == "grain":
        amount = float(effect.get("amount", 0.05))
        grain = np.mod(np.sin(u * 12.9898 + v * 78.233) * 43758.5453, 1.0)
        offset = (grain - 0.5) * amount
        rgb[...] = np.clip(rgb + offset[..., None], 0.0, 255.0)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    if edge0 == edge1:
        return (x >= edge1).astype(np.float32)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = str(hex_color).lstrip("#")
    if len(h) == 3:
        h = h[0] * 2 + h[1] * 2 + h[2] * 2
    if len(h) != 6:
        h = "ffffff"
    try:
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except ValueError:
        return 255, 255, 255
