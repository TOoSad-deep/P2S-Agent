"""Rule-based DSL candidate generator for PNG-to-Shader.

Generates a simple DSL based on preprocess features without LLM.
Used to verify the compiler works end-to-end.
"""

from __future__ import annotations

from p2s_agent.core.dsl.schema import DSL_SCHEMA_VERSION
from p2s_agent.core.pipeline.preprocess import feature_num


def generate_rule_candidate(
    preprocess: dict,
    canvas_width: int = 512,
    canvas_height: int = 512,
) -> dict:
    """Generate a simple DSL dict from preprocessing feature values.

    Decision rules (in priority order):
      1. gradient_score > 0.5  → radialGradient circle layer
      2. has_alpha=True and alpha_coverage > 0.1  → solid circle layer
      3. otherwise  → solid box layer

    An additional vignette effect is appended when photo_like_score > 0.3.

    Args:
        preprocess: Dict with optional keys:
            gradient_score (float 0-1)
            alpha_coverage (float 0-1)
            has_alpha (bool)
            photo_like_score (float 0-1)
            palette (list of hex strings, most dominant first)
        canvas_width: Output canvas width in pixels.
        canvas_height: Output canvas height in pixels.

    Returns:
        A fully valid DSL dict ready for dsl_validator.validate_dsl and
        compiler.compile_dsl.
    """
    gradient_score = feature_num(preprocess, "gradient_score", 0.0)
    alpha_coverage = feature_num(preprocess, "alpha_coverage", 0.0)
    edge_sharpness = feature_num(preprocess, "edge_sharpness", 0.0)
    has_alpha = bool(preprocess.get("has_alpha", False))
    photo_like_score = feature_num(preprocess, "photo_like_score", 0.0)
    color_count = int(feature_num(preprocess, "color_count_estimate", 10))
    palette: list[str] = preprocess.get("palette", ["#ffffff"])
    if not palette:
        palette = ["#ffffff"]
    top_color = palette[0]
    second_color = palette[1] if len(palette) > 1 else "#000000"

    effects: list[dict] = []
    if photo_like_score > 0.3:
        effects.append({"type": "vignette", "strength": 0.7})

    # Decision tree — ordered from most specific to most generic.
    # Goal: avoid defaulting to circle for every alpha image.
    if gradient_score > 0.5 and color_count <= 30:
        # Smooth gradient background — radial gradient fills the canvas
        layer = {
            "id": "rule_layer_0",
            "type": "circle",
            "fill": {
                "type": "radialGradient",
                "stops": [
                    {"color": top_color, "position": 0.0},
                    {"color": second_color, "position": 1.0},
                ],
                "center": [0.5, 0.5],
            },
            "params": {"center": [0.5, 0.5], "radius": 0.48},
            "opacity": 1.0,
            "transform": None,
            "effects": effects,
        }
    elif has_alpha and alpha_coverage < 0.5 and edge_sharpness > 0.08:
        # Sparse foreground with clear edges — likely a compact icon/shape.
        # Use a small circle scaled to the actual coverage area.
        radius = min(0.48, max(0.15, (alpha_coverage ** 0.5) * 0.55))
        layer = {
            "id": "rule_layer_0",
            "type": "circle",
            "fill": {"type": "solid", "color": top_color},
            "params": {"center": [0.5, 0.5], "radius": round(radius, 3)},
            "opacity": 1.0,
            "transform": None,
            "effects": effects,
        }
    elif has_alpha and alpha_coverage >= 0.5:
        # Large foreground area — more likely a wide shape (card, banner, box).
        w = min(0.9, max(0.4, alpha_coverage * 0.85))
        h = min(0.9, max(0.3, alpha_coverage * 0.7))
        layer = {
            "id": "rule_layer_0",
            "type": "roundedBox",
            "fill": {"type": "solid", "color": top_color},
            "params": {"center": [0.5, 0.5], "size": [round(w, 3), round(h, 3)], "radius": 0.05},
            "opacity": 1.0,
            "transform": None,
            "effects": effects,
        }
    elif not has_alpha and gradient_score > 0.3:
        # Opaque image with mild gradient — linear gradient box
        layer = {
            "id": "rule_layer_0",
            "type": "box",
            "fill": {
                "type": "linearGradient",
                "stops": [
                    {"color": top_color, "position": 0.0},
                    {"color": second_color, "position": 1.0},
                ],
                "direction": [1.0, 0.0],
            },
            "params": {"center": [0.5, 0.5], "size": [0.9, 0.9]},
            "opacity": 1.0,
            "transform": None,
            "effects": effects,
        }
    else:
        # Generic fallback — solid box using dominant palette color
        layer = {
            "id": "rule_layer_0",
            "type": "box",
            "fill": {"type": "solid", "color": top_color},
            "params": {"center": [0.5, 0.5], "size": [0.7, 0.5]},
            "opacity": 1.0,
            "transform": None,
            "effects": effects,
        }

    return {
        "schema_version": DSL_SCHEMA_VERSION,
        "canvas": {
            "width": canvas_width,
            "height": canvas_height,
            "background": "#000000",
        },
        "layers": [layer],
    }
