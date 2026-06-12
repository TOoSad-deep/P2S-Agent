"""Fallback candidate: always-valid minimal DSL used when all other candidates fail.

Uses the top palette color on a centered circle. Never fails to compile.
"""

from __future__ import annotations

from app.dsl.schema import DSL_SCHEMA_VERSION


def generate_fallback_candidate(
    preprocess: dict,
    canvas_width: int = 512,
    canvas_height: int = 512,
) -> dict:
    """Generate a minimal fallback DSL that always validates and compiles.

    Uses the top palette color as a solid fill on a centered circle.
    This is the candidate of last resort when all other candidates fail.

    Args:
        preprocess: Dict of preprocessed image features.
        canvas_width: Output canvas width in pixels.
        canvas_height: Output canvas height in pixels.

    Returns:
        A minimal valid DSL dict with ``_meta`` indicating source="fallback".
    """
    palette: list[str] = preprocess.get("palette", ["#ffffff"])
    if not palette:
        palette = ["#ffffff"]
    top_color = palette[0]
    second_color = palette[1] if len(palette) > 1 else "#000000"

    # Use gradient_score to pick between a gradient fill and a solid fill,
    # and alpha_coverage to size the shape — even the fallback should try to
    # reflect basic image characteristics rather than hardcoding a circle.
    gradient_score = float(preprocess.get("gradient_score", 0.0))
    alpha_coverage = float(preprocess.get("alpha_coverage", 1.0))

    if gradient_score > 0.4:
        fill: dict = {
            "type": "radialGradient",
            "stops": [
                {"color": top_color, "position": 0.0},
                {"color": second_color, "position": 1.0},
            ],
            "center": [0.5, 0.5],
        }
    else:
        fill = {"type": "solid", "color": top_color}

    # Scale the shape to the apparent foreground coverage
    size = min(0.88, max(0.3, alpha_coverage * 0.85))

    dsl = {
        "schema_version": DSL_SCHEMA_VERSION,
        "canvas": {
            "width": canvas_width,
            "height": canvas_height,
            "background": "#000000",
        },
        "layers": [
            {
                "id": "fallback_layer_0",
                "type": "roundedBox",
                "fill": fill,
                "params": {
                    "center": [0.5, 0.5],
                    "size": [round(size, 3), round(size * 0.75, 3)],
                    "radius": 0.04,
                },
                "opacity": 1.0,
                "transform": None,
                "effects": [],
            }
        ],
        "_meta": {"source": "fallback", "priority": 99},
    }
    return dsl
