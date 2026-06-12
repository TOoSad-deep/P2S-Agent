"""Unit tests for cv_candidate.py — CV-based DSL candidate generator."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.candidates.cv import generate_cv_candidate, get_cv_fit_report
from app.dsl.compiler import compile_dsl
from app.dsl.validator import validate_dsl


# ---------------------------------------------------------------------------
# Preprocess fixtures
# ---------------------------------------------------------------------------

PREPROCESS_ICON = {
    "has_alpha": True,
    "alpha_coverage": 0.4,
    "edge_sharpness": 0.2,
    "color_count_estimate": 8,
    "component_count_estimate": 2,
    "texture_score": 0.1,
    "palette": ["#ffffff", "#aaaaaa"],
}

PREPROCESS_PHOTO = {
    "has_alpha": False,
    "alpha_coverage": 1.0,
    "edge_sharpness": 0.3,
    "color_count_estimate": 200,
    "component_count_estimate": 1,
    "texture_score": 0.9,
    "palette": ["#334455"],
}


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def make_circle_png(tmp_path: Path, name: str, radius: int = 40, size: tuple = (100, 100)) -> Path:
    """Create a PNG with a white circle on a transparent background."""
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = size[0] // 2, size[1] // 2
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(255, 255, 255, 255))
    path = tmp_path / name
    img.save(path)
    return path


def make_wide_box_png(tmp_path: Path, name: str) -> Path:
    """Create a PNG with a wide white rectangle on a transparent background."""
    from PIL import Image, ImageDraw
    size = (200, 100)
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Draw a wide box — width >> height → aspect ratio ~4:1
    draw.rectangle([10, 35, 190, 65], fill=(255, 255, 255, 255))
    path = tmp_path / name
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# Gating tests
# ---------------------------------------------------------------------------

def test_cv_candidate_returns_none_for_photo_like():
    """Photo-like preprocess → cv_applicability is low → returns None."""
    result = generate_cv_candidate(PREPROCESS_PHOTO)
    assert result is None


def test_cv_candidate_returns_dsl_for_icon():
    """High cv_applicability preprocess → returns a non-None dict."""
    result = generate_cv_candidate(PREPROCESS_ICON)
    assert result is not None
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# DSL validity and compilability
# ---------------------------------------------------------------------------

def test_cv_candidate_dsl_is_valid():
    """Generated DSL must pass validate_dsl."""
    dsl = generate_cv_candidate(PREPROCESS_ICON)
    assert dsl is not None
    result = validate_dsl(dsl)
    assert result.valid is True, f"DSL invalid: {result.errors}"


def test_cv_candidate_dsl_is_compilable():
    """Generated DSL must compile successfully."""
    dsl = generate_cv_candidate(PREPROCESS_ICON)
    assert dsl is not None
    result = compile_dsl(dsl)
    assert result.success is True, f"Compile failed: {result.errors}"
    assert len(result.glsl) > 0


# ---------------------------------------------------------------------------
# Image-based shape detection
# ---------------------------------------------------------------------------

def test_cv_candidate_with_circle_image(tmp_path: Path):
    """Centered circle PNG → DSL uses 'circle' primitive."""
    image_path = make_circle_png(tmp_path, "circle.png", radius=40, size=(100, 100))
    dsl = generate_cv_candidate(PREPROCESS_ICON, image_path=image_path)
    assert dsl is not None

    layers = dsl.get("layers", [])
    assert len(layers) >= 1
    # The shape detected should be circle (near-square bounding box)
    layer_type = layers[0].get("type")
    meta = dsl.get("_meta", {})
    shape_detected = meta.get("shape_detected")
    assert shape_detected == "circle", (
        f"Expected shape_detected='circle', got '{shape_detected}' (layer type='{layer_type}')"
    )
    assert layer_type == "circle", (
        f"Expected layer type='circle', got '{layer_type}'"
    )


def test_cv_candidate_with_wide_box_image(tmp_path: Path):
    """Wide rectangle PNG → DSL may use 'box' or 'circle' primitive (not None)."""
    image_path = make_wide_box_png(tmp_path, "wide_box.png")
    dsl = generate_cv_candidate(PREPROCESS_ICON, image_path=image_path)
    assert dsl is not None

    layers = dsl.get("layers", [])
    assert len(layers) >= 1
    layer_type = layers[0].get("type")
    assert layer_type in ("box", "circle", "roundedBox"), (
        f"Unexpected layer type: '{layer_type}'"
    )


# ---------------------------------------------------------------------------
# Fit report tests
# ---------------------------------------------------------------------------

def test_cv_fit_report_enabled_true_for_icon():
    """get_cv_fit_report with high-score preprocess → cv_enabled=True."""
    dsl = generate_cv_candidate(PREPROCESS_ICON)
    report = get_cv_fit_report(PREPROCESS_ICON, dsl)
    assert report["cv_enabled"] is True


def test_cv_fit_report_disabled_for_photo():
    """get_cv_fit_report with low-score preprocess → cv_enabled=False."""
    dsl = generate_cv_candidate(PREPROCESS_PHOTO)
    assert dsl is None
    report = get_cv_fit_report(PREPROCESS_PHOTO, dsl)
    assert report["cv_enabled"] is False
    assert report["dsl_generated"] is False
