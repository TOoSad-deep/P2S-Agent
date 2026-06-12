"""Phase 1 unit tests: objective metrics for PNG-to-Shader.

All tests use only Pillow to create synthetic PNG images in tmp_path.
No LLM or browser calls are made.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from app.metrics import compute as metrics
from app.metrics.compute import compute_objective_metrics, grid_color_report


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def make_solid_png(tmp_path: Path, name: str, color_rgba: tuple, size: tuple = (32, 32)) -> Path:
    """Create a solid-colour RGBA PNG and return its path."""
    img = Image.new("RGBA", size, color_rgba)
    path = tmp_path / name
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# compute_mse
# ---------------------------------------------------------------------------


def test_mse_identical_images_is_zero(tmp_path):
    img = make_solid_png(tmp_path, "ref.png", (128, 64, 200, 255))
    result = metrics.compute_mse(img, img)
    assert result == pytest.approx(0.0, abs=1e-9)


def test_mse_opposite_colors(tmp_path):
    black = make_solid_png(tmp_path, "black.png", (0, 0, 0, 255))
    white = make_solid_png(tmp_path, "white.png", (255, 255, 255, 255))
    result = metrics.compute_mse(black, white)
    assert result == pytest.approx(1.0, abs=1e-6)


def test_mse_partial_difference_is_between_zero_and_one(tmp_path):
    ref = make_solid_png(tmp_path, "ref.png", (0, 0, 0, 255))
    rnd = make_solid_png(tmp_path, "rnd.png", (128, 128, 128, 255))
    result = metrics.compute_mse(ref, rnd)
    assert 0.0 < result < 1.0


# ---------------------------------------------------------------------------
# compute_simple_ssim
# ---------------------------------------------------------------------------


def test_simple_ssim_identical_is_one(tmp_path):
    img = make_solid_png(tmp_path, "ref.png", (100, 150, 200, 255))
    result = metrics.compute_simple_ssim(img, img)
    assert result == pytest.approx(1.0, abs=1e-3)


def test_simple_ssim_different_images_less_than_one(tmp_path):
    ref = make_solid_png(tmp_path, "ref.png", (0, 0, 0, 255))
    rnd = make_solid_png(tmp_path, "rnd.png", (255, 255, 255, 255))
    result = metrics.compute_simple_ssim(ref, rnd)
    assert result < 1.0


# ---------------------------------------------------------------------------
# compute_alpha_coverage_diff
# ---------------------------------------------------------------------------


def test_alpha_coverage_diff_fully_transparent_vs_opaque(tmp_path):
    transparent = make_solid_png(tmp_path, "transparent.png", (255, 0, 0, 0))
    opaque = make_solid_png(tmp_path, "opaque.png", (255, 0, 0, 255))
    result = metrics.compute_alpha_coverage_diff(transparent, opaque)
    assert result == pytest.approx(1.0, abs=1e-6)


def test_alpha_coverage_diff_identical(tmp_path):
    img = make_solid_png(tmp_path, "img.png", (100, 200, 50, 200))
    result = metrics.compute_alpha_coverage_diff(img, img)
    assert result == pytest.approx(0.0, abs=1e-9)


def test_alpha_coverage_diff_partial(tmp_path):
    """Half-transparent vs fully opaque should give a diff of 0.5."""
    # Create a 2×2 image with 2 transparent + 2 opaque pixels.
    img_half = Image.new("RGBA", (2, 2), (255, 255, 255, 0))
    img_half.putpixel((0, 0), (255, 255, 255, 255))
    img_half.putpixel((0, 1), (255, 255, 255, 255))
    half_path = tmp_path / "half.png"
    img_half.save(half_path)

    full_path = make_solid_png(tmp_path, "full.png", (255, 255, 255, 255), size=(2, 2))

    result = metrics.compute_alpha_coverage_diff(half_path, full_path)
    assert result == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# compute_color_histogram_score
# ---------------------------------------------------------------------------


def test_color_histogram_score_identical(tmp_path):
    img = make_solid_png(tmp_path, "img.png", (200, 100, 50, 255))
    result = metrics.compute_color_histogram_score(img, img)
    assert result == pytest.approx(1.0, abs=1e-6)


def test_color_histogram_score_different_colors(tmp_path):
    red = make_solid_png(tmp_path, "red.png", (255, 0, 0, 255))
    blue = make_solid_png(tmp_path, "blue.png", (0, 0, 255, 255))
    result = metrics.compute_color_histogram_score(red, blue)
    # Red and blue have completely disjoint R and B channels; G is both 0.
    # G channel intersection = 1.0, R and B = 0.0, average = 1/3 ≈ 0.333.
    assert result < 0.5


# ---------------------------------------------------------------------------
# check_nonblank_render
# ---------------------------------------------------------------------------


def test_nonblank_render_with_content(tmp_path):
    img = make_solid_png(tmp_path, "colored.png", (0, 128, 255, 255))
    assert metrics.check_nonblank_render(img) is True


def test_nonblank_render_blank_transparent(tmp_path):
    img = make_solid_png(tmp_path, "transparent.png", (0, 0, 0, 0))
    assert metrics.check_nonblank_render(img) is False


def test_nonblank_render_blank_black_opaque(tmp_path):
    img = make_solid_png(tmp_path, "black.png", (0, 0, 0, 255))
    # All pixels are black (r=0, g=0, b=0) even though alpha=255.
    assert metrics.check_nonblank_render(img) is False


def test_nonblank_render_partially_transparent_with_color(tmp_path):
    # Some pixels are coloured and semi-transparent — should count.
    img_mix = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    img_mix.putpixel((2, 2), (255, 0, 0, 128))
    path = tmp_path / "mix.png"
    img_mix.save(path)
    assert metrics.check_nonblank_render(path) is True


# ---------------------------------------------------------------------------
# compute_objective_metrics
# ---------------------------------------------------------------------------

EXPECTED_KEYS = {
    "mse",
    "simple_ssim",
    "alpha_coverage_diff",
    "color_histogram_score",
    "edge_density_diff",
    "nonblank_render",
    "within_shader_budget",
}


def test_compute_objective_metrics_returns_all_keys(tmp_path):
    img = make_solid_png(tmp_path, "img.png", (128, 128, 128, 255))
    result = metrics.compute_objective_metrics(img, img)
    # v1 keys must still be present
    assert EXPECTED_KEYS.issubset(set(result.keys()))
    # v2 keys must also be present
    assert {"rmse", "mask_iou", "edge_iou", "grid_color_sim"}.issubset(set(result.keys()))


def test_compute_objective_metrics_values_in_range(tmp_path):
    ref = make_solid_png(tmp_path, "ref.png", (100, 150, 200, 255))
    rnd = make_solid_png(tmp_path, "rnd.png", (50, 100, 150, 255))
    result = metrics.compute_objective_metrics(ref, rnd)
    for key in ("mse", "simple_ssim", "alpha_coverage_diff", "color_histogram_score", "edge_density_diff"):
        assert 0.0 <= result[key] <= 1.0, f"{key} out of range: {result[key]}"
    assert isinstance(result["nonblank_render"], bool)
    assert isinstance(result["within_shader_budget"], bool)


def test_within_shader_budget_true_when_under_limit(tmp_path):
    img = make_solid_png(tmp_path, "img.png", (255, 255, 255, 255))
    result = metrics.compute_objective_metrics(img, img, shader_chars=5000, max_shader_chars=12000)
    assert result["within_shader_budget"] is True


def test_within_shader_budget_false_when_exceeded(tmp_path):
    img = make_solid_png(tmp_path, "img.png", (255, 255, 255, 255))
    result = metrics.compute_objective_metrics(img, img, shader_chars=15000, max_shader_chars=12000)
    assert result["within_shader_budget"] is False


def test_within_shader_budget_false_at_exact_limit_plus_one(tmp_path):
    img = make_solid_png(tmp_path, "img.png", (0, 128, 0, 255))
    result = metrics.compute_objective_metrics(img, img, shader_chars=12001, max_shader_chars=12000)
    assert result["within_shader_budget"] is False


def test_within_shader_budget_true_at_exact_limit(tmp_path):
    img = make_solid_png(tmp_path, "img.png", (0, 128, 0, 255))
    result = metrics.compute_objective_metrics(img, img, shader_chars=12000, max_shader_chars=12000)
    assert result["within_shader_budget"] is True


def test_compute_objective_metrics_resize_mismatched_sizes(tmp_path):
    """Render with different size should be resized and not raise."""
    ref = make_solid_png(tmp_path, "ref.png", (200, 200, 200, 255), size=(64, 64))
    rnd = make_solid_png(tmp_path, "rnd.png", (200, 200, 200, 255), size=(32, 32))
    result = metrics.compute_objective_metrics(ref, rnd)
    assert EXPECTED_KEYS.issubset(set(result.keys()))


# ---------------------------------------------------------------------------
# v2 metrics: position-aware, alpha-correct, vectorized
# ---------------------------------------------------------------------------


def _save_rgba(tmp_path, name, draw_fn, bg=(0, 0, 0, 0), size=(64, 64)):
    img = Image.new("RGBA", size, bg)
    draw_fn(ImageDraw.Draw(img))
    path = tmp_path / name
    img.save(path)
    return path


def test_v2_identical_images_score_perfect(tmp_path):
    p = _save_rgba(tmp_path, "a.png", lambda d: d.ellipse((16, 16, 48, 48), fill=(255, 0, 0, 255)))
    m = compute_objective_metrics(p, p)
    assert m["rmse"] < 1e-6
    assert m["mask_iou"] == 1.0
    assert m["edge_iou"] == 1.0
    assert m["grid_color_sim"] > 0.99


def test_v2_shifted_shape_punished_by_mask_iou(tmp_path):
    a = _save_rgba(tmp_path, "a.png", lambda d: d.ellipse((8, 8, 32, 32), fill=(255, 0, 0, 255)))
    b = _save_rgba(tmp_path, "b.png", lambda d: d.ellipse((32, 32, 56, 56), fill=(255, 0, 0, 255)))
    m = compute_objective_metrics(a, b)
    # v1 的 alpha_coverage_diff 看不出区别，v2 的 mask_iou 必须看出
    assert m["alpha_coverage_diff"] < 0.02
    assert m["mask_iou"] < 0.2


def test_v2_grid_color_sim_is_position_aware(tmp_path):
    a = _save_rgba(tmp_path, "a.png", lambda d: (d.rectangle((0, 0, 31, 63), fill=(255, 0, 0, 255)),
                                                 d.rectangle((32, 0, 63, 63), fill=(0, 0, 255, 255))))
    b = _save_rgba(tmp_path, "b.png", lambda d: (d.rectangle((0, 0, 31, 63), fill=(0, 0, 255, 255)),
                                                 d.rectangle((32, 0, 63, 63), fill=(255, 0, 0, 255))))
    m = compute_objective_metrics(a, b)
    assert m["color_histogram_score"] > 0.99   # v1 全局直方图被骗
    assert m["grid_color_sim"] < 0.6           # v2 分块指标不被骗


def test_v2_transparent_rgb_garbage_ignored(tmp_path):
    a = _save_rgba(tmp_path, "a.png", lambda d: d.ellipse((16, 16, 48, 48), fill=(255, 0, 0, 255)),
                   bg=(0, 0, 0, 0))
    b = _save_rgba(tmp_path, "b.png", lambda d: d.ellipse((16, 16, 48, 48), fill=(255, 0, 0, 255)),
                   bg=(255, 0, 255, 0))   # alpha=0 区域藏着洋红垃圾值
    m = compute_objective_metrics(a, b)
    assert m["rmse"] < 0.01
    assert m["grid_color_sim"] > 0.95


def test_grid_color_report_names_bad_region(tmp_path):
    a = _save_rgba(tmp_path, "a.png", lambda d: d.rectangle((0, 0, 63, 63), fill=(200, 200, 200, 255)))
    b = _save_rgba(tmp_path, "b.png", lambda d: (d.rectangle((0, 0, 63, 63), fill=(200, 200, 200, 255)),
                                                 d.rectangle((48, 0, 63, 15), fill=(0, 0, 0, 255))))
    notes = grid_color_report(a, b)
    assert notes, "should report at least one bad region"
    assert any("top-right" in n for n in notes)


def test_v2_tiny_image_no_nan(tmp_path):
    import numpy as np
    a = _save_rgba(tmp_path, "a.png", lambda d: d.point((0, 0), fill=(255, 0, 0, 255)), size=(3, 3))
    b = _save_rgba(tmp_path, "b.png", lambda d: d.point((2, 2), fill=(0, 0, 255, 255)), size=(3, 3))
    m = compute_objective_metrics(a, b)
    assert not np.isnan(m["grid_color_sim"])
    assert 0.0 <= m["grid_color_sim"] <= 1.0
