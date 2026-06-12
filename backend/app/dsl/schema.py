"""DSL schema definitions for PNG-to-Shader Phase 3.

The DSL is a JSON-serializable dict that describes a 2D scene
as an ordered list of layers. The compiler converts it to GLSL.

MVP primitives: circle, ellipse, box, roundedBox, ring, polygon
MVP fills: solid, linearGradient, radialGradient
MVP effects: glow, vignette, grain (grain is very simple)
MVP transforms: translate, rotate, scale
"""

from __future__ import annotations

PRIMITIVE_TYPES = ["circle", "ellipse", "box", "roundedBox", "ring", "polygon"]
FILL_TYPES = ["solid", "linearGradient", "radialGradient"]
EFFECT_TYPES = ["glow", "vignette", "grain"]
TRANSFORM_TYPES = ["translate", "rotate", "scale"]

DSL_SCHEMA_VERSION = 1

DSL_LAYER_SCHEMA = {
    "required": ["id", "type", "fill"],
    "optional": ["transform", "effects", "opacity", "params"],
    "type_values": PRIMITIVE_TYPES,
    "fill_required": ["type"],
    "fill_type_values": FILL_TYPES,
}

DSL_SCHEMA = {
    "required": ["schema_version", "canvas", "layers"],
    "canvas_required": ["width", "height"],
    "layer_schema": DSL_LAYER_SCHEMA,
}

# ---------------------------------------------------------------------------
# Fixtures (used in tests and as reference examples)
# ---------------------------------------------------------------------------

FIXTURE_CIRCLE_SOLID = {
    "schema_version": 1,
    "canvas": {"width": 512, "height": 512, "background": "#000000"},
    "layers": [
        {
            "id": "circle_01",
            "type": "circle",
            "fill": {"type": "solid", "color": "#ffffff"},
            "params": {"center": [0.5, 0.5], "radius": 0.3},
            "opacity": 1.0,
            "transform": None,
            "effects": [],
        }
    ],
}

FIXTURE_BOX_GRADIENT = {
    "schema_version": 1,
    "canvas": {"width": 512, "height": 512, "background": "#111111"},
    "layers": [
        {
            "id": "box_01",
            "type": "box",
            "fill": {
                "type": "linearGradient",
                "stops": [
                    {"color": "#ff0000", "position": 0.0},
                    {"color": "#0000ff", "position": 1.0},
                ],
                "direction": [1.0, 0.0],
            },
            "params": {"center": [0.5, 0.5], "size": [0.4, 0.3]},
            "opacity": 1.0,
            "transform": None,
            "effects": [],
        }
    ],
}

FIXTURE_GLOW_RING = {
    "schema_version": 1,
    "canvas": {"width": 512, "height": 512, "background": "#000000"},
    "layers": [
        {
            "id": "ring_01",
            "type": "ring",
            "fill": {"type": "solid", "color": "#00ffff"},
            "params": {"center": [0.5, 0.5], "radius": 0.35, "thickness": 0.02},
            "opacity": 1.0,
            "transform": None,
            "effects": [
                {"type": "glow", "intensity": 8.0, "color": "#00ffff"}
            ],
        }
    ],
}

FIXTURE_ROUNDEDBOX_VIGNETTE = {
    "schema_version": 1,
    "canvas": {"width": 512, "height": 512, "background": "#222222"},
    "layers": [
        {
            "id": "rbox_01",
            "type": "roundedBox",
            "fill": {"type": "solid", "color": "#ffcc00"},
            "params": {"center": [0.5, 0.5], "size": [0.5, 0.3], "radius": 0.05},
            "opacity": 0.9,
            "transform": None,
            "effects": [
                {"type": "vignette", "strength": 0.8}
            ],
        }
    ],
}
