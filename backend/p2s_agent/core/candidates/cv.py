"""CV-based DSL candidate generator for PNG-to-Shader.

Generates a DSL using image analysis (Pillow only, no opencv).
Gated by cv_applicability_score — returns None if score < threshold.
"""

from __future__ import annotations

from pathlib import Path

from p2s_agent.core.utils.cv_features import get_cv_applicability_report
from p2s_agent.core.dsl.schema import DSL_SCHEMA_VERSION
from p2s_agent.core.pipeline.preprocess import feature_num


def generate_cv_candidate(
    preprocess: dict,
    image_path: "str | Path | None" = None,
    canvas_width: int = 512,
    canvas_height: int = 512,
) -> "dict | None":
    """Generate a DSL dict using CV-based image analysis.

    Gated by cv_applicability_score: returns None if score < threshold.

    Args:
        preprocess: Dict from preprocess_image().
        image_path: Optional path to the source PNG.  When provided, Pillow is
            used to detect the dominant shape (circle vs box) from the alpha
            bounding box.  When None, falls back to preprocess-based heuristics.
        canvas_width: Output canvas width in pixels.
        canvas_height: Output canvas height in pixels.

    Returns:
        A valid DSL dict, or None if CV is disabled for this image.
    """
    report = get_cv_applicability_report(preprocess)
    if not report["enabled"]:
        return None

    palette: list[str] = preprocess.get("palette", ["#ffffff"])
    if not palette:
        palette = ["#ffffff"]
    top_color = palette[0]

    edge_sharpness = feature_num(preprocess, "edge_sharpness", 0.0)
    add_glow = 0.05 <= edge_sharpness <= 0.40

    shape_detected: str | None = None

    if image_path is not None:
        shape_detected, layer = _cv_layer_from_image(
            image_path, top_color, add_glow, canvas_width, canvas_height
        )
    else:
        shape_detected, layer = _cv_layer_from_preprocess(
            preprocess, top_color, add_glow
        )

    dsl = {
        "schema_version": DSL_SCHEMA_VERSION,
        "canvas": {
            "width": canvas_width,
            "height": canvas_height,
            "background": "#000000",
        },
        "layers": [layer],
        "_meta": {
            "source": "cv",
            "priority": report["priority"],
            "shape_detected": shape_detected,
        },
    }
    return dsl


def _cv_layer_from_image(
    image_path: "str | Path",
    top_color: str,
    add_glow: bool,
    canvas_width: int,
    canvas_height: int,
) -> "tuple[str, dict]":
    """Detect shape from the image file and build a layer dict.

    Returns (shape_detected, layer_dict).
    """
    from PIL import Image

    try:
        img = Image.open(image_path).convert("RGBA")
        # Get bounding box of non-transparent pixels
        bbox = img.getbbox()
        if bbox is None:
            # Fully transparent — fall back
            return _cv_layer_from_preprocess_inner(top_color, add_glow, "box")

        left, top, right, bottom = bbox
        bbox_w = right - left
        bbox_h = bottom - top

        if bbox_w == 0 or bbox_h == 0:
            return _cv_layer_from_preprocess_inner(top_color, add_glow, "box")

        aspect_ratio = bbox_w / bbox_h

        # Centroid in normalised [0,1] coordinates
        img_w, img_h = img.size
        cx = (left + right) / 2.0 / img_w
        cy = (top + bottom) / 2.0 / img_h

        # Normalised half-extents
        half_w = (bbox_w / img_w) * 0.5
        half_h = (bbox_h / img_h) * 0.5
        radius = (half_w + half_h) / 2.0  # mean radius for circle

        # Choose primitive
        if 0.8 <= aspect_ratio <= 1.2:
            shape_detected = "circle"
            layer = _make_layer(
                "cv_layer_0",
                "circle",
                top_color,
                {"center": [round(cx, 4), round(cy, 4)], "radius": round(radius, 4)},
                add_glow,
                top_color,
            )
        else:
            shape_detected = "box"
            layer = _make_layer(
                "cv_layer_0",
                "box",
                top_color,
                {
                    "center": [round(cx, 4), round(cy, 4)],
                    "size": [round(half_w * 2, 4), round(half_h * 2, 4)],
                },
                add_glow,
                top_color,
            )

        return shape_detected, layer

    except Exception:
        # If anything goes wrong with image loading, fall back to preprocess
        return _cv_layer_from_preprocess_inner(top_color, add_glow, "box")


def _cv_layer_from_preprocess(
    preprocess: dict,
    top_color: str,
    add_glow: bool,
) -> "tuple[str, dict]":
    """Build a layer based purely on preprocess features (no image file).

    Returns (shape_detected, layer_dict).
    """
    has_alpha = bool(preprocess.get("has_alpha", False))
    alpha_coverage = feature_num(preprocess, "alpha_coverage", 0.0)

    if has_alpha and alpha_coverage > 0.1:
        primitive = "circle"
    else:
        primitive = "box"

    return _cv_layer_from_preprocess_inner(top_color, add_glow, primitive)


def _cv_layer_from_preprocess_inner(
    top_color: str,
    add_glow: bool,
    primitive: str,
) -> "tuple[str, dict]":
    """Create a layer dict for the given primitive type."""
    if primitive == "circle":
        params = {"center": [0.5, 0.5], "radius": 0.35}
    else:
        params = {"center": [0.5, 0.5], "size": [0.6, 0.4]}

    layer = _make_layer("cv_layer_0", primitive, top_color, params, add_glow, top_color)
    return primitive, layer


def _make_layer(
    layer_id: str,
    primitive: str,
    color: str,
    params: dict,
    add_glow: bool,
    glow_color: str,
) -> dict:
    """Build a DSL layer dict."""
    effects: list[dict] = []
    if add_glow:
        effects.append({"type": "glow", "intensity": 5.0, "color": glow_color})

    return {
        "id": layer_id,
        "type": primitive,
        "fill": {"type": "solid", "color": color},
        "params": params,
        "opacity": 1.0,
        "transform": None,
        "effects": effects,
    }


def get_cv_fit_report(preprocess: dict, dsl: "dict | None") -> dict:
    """Return a structured report of CV fit for a given preprocess + DSL.

    Args:
        preprocess: Dict from preprocess_image().
        dsl: The DSL dict returned by generate_cv_candidate(), or None.

    Returns:
        Dict with keys: cv_enabled, cv_priority, cv_applicability_score,
        shape_detected, dsl_generated, reason.
    """
    report = get_cv_applicability_report(preprocess)

    cv_enabled = report["enabled"]
    cv_priority = report["priority"]
    cv_score = report["score"]
    dsl_generated = dsl is not None

    # Detect shape from DSL _meta if present, else from layers
    shape_detected: str | None = None
    if dsl is not None:
        meta = dsl.get("_meta", {})
        if isinstance(meta, dict):
            shape_detected = meta.get("shape_detected")
        if shape_detected is None:
            layers = dsl.get("layers", [])
            if layers and isinstance(layers[0], dict):
                shape_detected = layers[0].get("type")

    reason_parts: list[str] = [report["reason"]]
    if dsl_generated:
        reason_parts.append(f"DSL generated with shape '{shape_detected}'.")
    else:
        reason_parts.append("DSL not generated (CV disabled or fallback).")

    return {
        "cv_enabled": cv_enabled,
        "cv_priority": cv_priority,
        "cv_applicability_score": cv_score,
        "shape_detected": shape_detected,
        "dsl_generated": dsl_generated,
        "reason": " ".join(reason_parts),
    }
