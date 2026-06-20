"""Unit tests for p2s_agent.core.pipeline.preprocess (Phase 2).

All tests create synthetic PNG images in tmp_path using Pillow — no real
images, no network, no LLM.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from PIL import Image

from p2s_agent.core.pipeline.preprocess import (
    MAX_WORKING_DIM,
    _bounded_working_image,
    feature_num,
    preprocess_image,
    save_preprocess_artifacts,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def make_png(
    tmp_path: Path,
    name: str,
    mode: str,
    size: tuple[int, int],
    data_fn,
) -> Path:
    """Create a synthetic PNG image.

    Args:
        tmp_path: Pytest tmp_path fixture directory.
        name: Filename (without extension — .png appended automatically).
        mode: Pillow image mode, e.g. "RGB", "RGBA", "L".
        size: ``(width, height)`` tuple.
        data_fn: Callable ``(x, y) -> colour_tuple`` that returns the pixel
            colour for each position.

    Returns:
        Path to the written PNG file.
    """
    w, h = size
    img = Image.new(mode, size)
    for y in range(h):
        for x in range(w):
            img.putpixel((x, y), data_fn(x, y))
    path = tmp_path / f"{name}.png"
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# Basic structural tests
# ---------------------------------------------------------------------------


def test_preprocess_returns_all_keys(tmp_path):
    path = make_png(tmp_path, "solid", "RGB", (32, 32), lambda x, y: (128, 64, 32))
    result = preprocess_image(path)
    expected_keys = {
        "width", "height", "has_alpha", "alpha_coverage", "palette",
        "color_count_estimate", "edge_sharpness", "component_count_estimate",
        "texture_score", "photo_like_score", "gradient_score",
    }
    assert set(result.keys()) >= expected_keys


def test_preprocess_solid_opaque_image(tmp_path):
    path = make_png(tmp_path, "solid_rgb", "RGB", (32, 32), lambda x, y: (200, 100, 50))
    result = preprocess_image(path)
    assert result["has_alpha"] is False
    assert result["alpha_coverage"] == pytest.approx(1.0)
    assert result["width"] == 32
    assert result["height"] == 32


def test_preprocess_transparent_image(tmp_path):
    """Fully transparent RGBA image → alpha_coverage == 0.0."""
    path = make_png(tmp_path, "transparent", "RGBA", (32, 32), lambda x, y: (0, 0, 0, 0))
    result = preprocess_image(path)
    assert result["has_alpha"] is True
    assert result["alpha_coverage"] == pytest.approx(0.0)


def test_preprocess_half_transparent(tmp_path):
    """Left half opaque, right half transparent → alpha_coverage ~0.5."""
    def data_fn(x, y):
        return (255, 255, 255, 255) if x < 16 else (0, 0, 0, 0)

    path = make_png(tmp_path, "half", "RGBA", (32, 32), data_fn)
    result = preprocess_image(path)
    assert result["has_alpha"] is True
    assert 0.45 <= result["alpha_coverage"] <= 0.55


def test_preprocess_alpha_coverage_fully_opaque_rgba(tmp_path):
    """RGBA image with all alpha=255 → coverage=1.0."""
    path = make_png(tmp_path, "opaque_rgba", "RGBA", (16, 16), lambda x, y: (10, 20, 30, 255))
    result = preprocess_image(path)
    assert result["alpha_coverage"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------


def test_preprocess_palette_returns_5_colors(tmp_path):
    """Image with 6+ distinct colours → palette has exactly 5 entries."""
    colours = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 255),
    ]

    def data_fn(x, y):
        # Each 8-pixel column gets a different colour
        idx = (x // 8) % len(colours)
        return colours[idx]

    path = make_png(tmp_path, "palette6", "RGB", (48, 48), data_fn)
    result = preprocess_image(path)
    assert len(result["palette"]) == 5
    for entry in result["palette"]:
        assert entry.startswith("#")
        assert len(entry) == 7  # "#RRGGBB"


def test_preprocess_palette_format(tmp_path):
    path = make_png(tmp_path, "fmt", "RGB", (16, 16), lambda x, y: (17, 34, 51))
    result = preprocess_image(path)
    for color in result["palette"]:
        assert color.startswith("#")
        assert len(color) == 7


# ---------------------------------------------------------------------------
# Color count
# ---------------------------------------------------------------------------


def test_preprocess_color_count_reasonable(tmp_path):
    """Solid colour → color_count_estimate should be small (1 or very few)."""
    path = make_png(tmp_path, "mono", "RGB", (32, 32), lambda x, y: (200, 100, 50))
    result = preprocess_image(path)
    # A single colour means 1 unique 32-bin cell
    assert result["color_count_estimate"] <= 4


def test_preprocess_color_count_high_for_gradient(tmp_path):
    """Smooth gradient → many distinct 32-bin colours."""
    def data_fn(x, y):
        v = int(x * 255 / 63)
        return (v, v, v)

    path = make_png(tmp_path, "gradient_count", "RGB", (64, 32), data_fn)
    result = preprocess_image(path)
    assert result["color_count_estimate"] > 5


# ---------------------------------------------------------------------------
# Edge sharpness
# ---------------------------------------------------------------------------


def test_preprocess_edge_sharpness_high_for_checkerboard(tmp_path):
    """Checkerboard → many edge pixels → edge_sharpness > 0.3."""
    def data_fn(x, y):
        # Alternating black and white
        return (255, 255, 255) if (x + y) % 2 == 0 else (0, 0, 0)

    path = make_png(tmp_path, "checker", "RGB", (32, 32), data_fn)
    result = preprocess_image(path)
    assert result["edge_sharpness"] > 0.3


def test_preprocess_edge_sharpness_low_for_solid(tmp_path):
    """Solid colour → no edges → edge_sharpness == 0.0."""
    path = make_png(tmp_path, "solid_edge", "RGB", (32, 32), lambda x, y: (128, 128, 128))
    result = preprocess_image(path)
    assert result["edge_sharpness"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Gradient score
# ---------------------------------------------------------------------------


def test_preprocess_gradient_score_high_for_smooth(tmp_path):
    """Smooth horizontal gradient → gradient_score > 0.4."""
    def data_fn(x, y):
        v = int(x * 255 / 63)
        return (v, v, v)

    path = make_png(tmp_path, "smooth_gradient", "RGB", (64, 64), data_fn)
    result = preprocess_image(path)
    assert result["gradient_score"] > 0.4


def test_preprocess_gradient_score_low_for_checkerboard(tmp_path):
    """Checkerboard → high edge_sharpness → gradient_score < 0.5."""
    def data_fn(x, y):
        return (255, 255, 255) if (x + y) % 2 == 0 else (0, 0, 0)

    path = make_png(tmp_path, "checker_grad", "RGB", (32, 32), data_fn)
    result = preprocess_image(path)
    assert result["gradient_score"] < 0.5


# ---------------------------------------------------------------------------
# Component count
# ---------------------------------------------------------------------------


def test_preprocess_component_count_single_blob(tmp_path):
    """Single connected blob → component_count_estimate >= 1."""
    def data_fn(x, y):
        # White circle in centre on black background
        cx, cy = 15, 15
        return (255, 255, 255, 255) if (x - cx) ** 2 + (y - cy) ** 2 < 64 else (0, 0, 0, 0)

    path = make_png(tmp_path, "single_blob", "RGBA", (32, 32), data_fn)
    result = preprocess_image(path)
    assert result["component_count_estimate"] >= 1


# ---------------------------------------------------------------------------
# Scores are in [0, 1]
# ---------------------------------------------------------------------------


def test_preprocess_scores_in_range(tmp_path):
    """All float scores should be in [0.0, 1.0]."""
    def data_fn(x, y):
        return (x * 4 % 256, y * 4 % 256, (x + y) * 2 % 256)

    path = make_png(tmp_path, "noisy", "RGB", (32, 32), data_fn)
    result = preprocess_image(path)
    for key in ("edge_sharpness", "texture_score", "photo_like_score", "gradient_score", "alpha_coverage"):
        val = result[key]
        assert 0.0 <= val <= 1.0, f"{key}={val} out of [0, 1]"


# ---------------------------------------------------------------------------
# save_preprocess_artifacts
# ---------------------------------------------------------------------------


def test_save_preprocess_artifacts_creates_files(tmp_path):
    img_path = make_png(tmp_path, "src", "RGBA", (64, 64), lambda x, y: (100, 150, 200, 255))
    data = preprocess_image(img_path)

    run_dir = tmp_path / "run"
    save_preprocess_artifacts(data, run_dir, img_path)

    assert (run_dir / "preprocess.json").exists()
    assert (run_dir / "normalized_input.png").exists()
    assert (run_dir / "llm_reference_input.png").exists()
    assert (run_dir / "alpha_mask.png").exists()


def test_save_preprocess_artifacts_json_valid(tmp_path):
    import json

    img_path = make_png(tmp_path, "src2", "RGB", (32, 32), lambda x, y: (10, 20, 30))
    data = preprocess_image(img_path)

    run_dir = tmp_path / "run2"
    save_preprocess_artifacts(data, run_dir, img_path)

    body = json.loads((run_dir / "preprocess.json").read_text(encoding="utf-8"))
    assert "width" in body
    assert body["width"] == 32


def test_save_preprocess_artifacts_normalized_size(tmp_path):
    """normalized_input.png must be 128×128 regardless of source size."""
    img_path = make_png(tmp_path, "big", "RGB", (256, 128), lambda x, y: (50, 50, 50))
    data = preprocess_image(img_path)

    run_dir = tmp_path / "run3"
    save_preprocess_artifacts(data, run_dir, img_path)

    norm_img = Image.open(run_dir / "normalized_input.png")
    assert norm_img.size == (128, 128)

    alpha_img = Image.open(run_dir / "alpha_mask.png")
    assert alpha_img.size == (128, 128)


def test_save_preprocess_artifacts_composites_model_reference_over_black(tmp_path):
    img_path = make_png(
        tmp_path,
        "transparent_red",
        "RGBA",
        (8, 8),
        lambda x, y: (255, 0, 0, 255) if 2 <= x < 6 and 2 <= y < 6 else (255, 255, 255, 0),
    )
    data = preprocess_image(img_path)

    run_dir = tmp_path / "run4"
    save_preprocess_artifacts(data, run_dir, img_path)

    model_img = Image.open(run_dir / "llm_reference_input.png")
    assert model_img.mode == "RGB"
    assert model_img.getpixel((0, 0)) == (0, 0, 0)
    assert model_img.getpixel((3, 3)) == (255, 0, 0)


# ---------------------------------------------------------------------------
# feature_num — coalescing numeric accessor (Bug 4)
# ---------------------------------------------------------------------------


def test_feature_num_returns_value_when_present_and_finite():
    features = {"gradient_score": 0.42, "color_count_estimate": 12}
    assert feature_num(features, "gradient_score", 0.0) == pytest.approx(0.42)
    assert feature_num(features, "color_count_estimate", 5) == pytest.approx(12.0)


def test_feature_num_returns_default_for_none():
    features = {"gradient_score": None}
    assert feature_num(features, "gradient_score", 0.3) == pytest.approx(0.3)


def test_feature_num_returns_default_for_missing_key():
    assert feature_num({}, "edge_sharpness", 0.0) == pytest.approx(0.0)


def test_feature_num_returns_default_for_non_finite_and_unparseable():
    assert feature_num({"x": float("nan")}, "x", 1.0) == pytest.approx(1.0)
    assert feature_num({"x": float("inf")}, "x", 2.0) == pytest.approx(2.0)
    assert feature_num({"x": "not-a-number"}, "x", 7.0) == pytest.approx(7.0)


def test_feature_num_coerces_numeric_strings():
    assert feature_num({"x": "0.75"}, "x", 0.0) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Bounded working image — DoS guard against full-res pixel materialization
# (Bug 5)
# ---------------------------------------------------------------------------


def test_bounded_working_image_downscales_oversized():
    """An image larger than the working cap is downscaled so its pixel count
    is bounded before any per-pixel iteration."""
    oversized = Image.new("RGB", (MAX_WORKING_DIM * 3, MAX_WORKING_DIM * 2), (10, 20, 30))
    work = _bounded_working_image(oversized)
    w, h = work.size
    assert max(w, h) <= MAX_WORKING_DIM
    assert w * h <= MAX_WORKING_DIM * MAX_WORKING_DIM
    # Aspect ratio preserved (3:2).
    assert abs((w / h) - 1.5) < 0.05


def test_bounded_working_image_leaves_normal_size_unchanged():
    """A normal-size image (within the cap) is returned with identical
    dimensions so existing feature values do not regress."""
    normal = Image.new("RGB", (64, 48), (200, 100, 50))
    work = _bounded_working_image(normal)
    assert work.size == (64, 48)


def test_preprocess_oversized_image_reports_original_dims_and_bounds_work(tmp_path):
    """Preprocessing a huge image must still report the ORIGINAL dimensions in
    the output, but must not materialize the full-resolution pixel grid."""
    side = MAX_WORKING_DIM * 2
    img = Image.new("RGB", (side, side), (0, 0, 0))
    # A bright square so there is some structure (non-degenerate features).
    for yy in range(side // 4, side // 2):
        for xx in range(side // 4, side // 2):
            img.putpixel((xx, yy), (255, 0, 0))
    path = tmp_path / "huge.png"
    img.save(path)

    result = preprocess_image(path)

    # Original dimensions are preserved in the reported features.
    assert result["width"] == side
    assert result["height"] == side
    # Sanity: produces the full feature set with finite scores.
    for key in ("edge_sharpness", "gradient_score", "photo_like_score"):
        assert 0.0 <= result[key] <= 1.0
