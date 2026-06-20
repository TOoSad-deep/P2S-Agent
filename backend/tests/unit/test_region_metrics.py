"""Unit tests for region_metrics.py (V4.2 per-region crop metrics).

All tests use only Pillow to create synthetic PNG images in tmp_path.
No LLM, browser, or filesystem side-effects beyond tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from p2s_agent.core.pipeline.region_metrics import compute_region_metrics
from p2s_agent.orchestration.human_constraints import RegionConstraint


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_solid_rgba(tmp_path: Path, name: str, color_rgba: tuple, size: tuple = (64, 64)) -> Path:
    """Create a solid-colour RGBA PNG."""
    img = Image.new("RGBA", size, color_rgba)
    path = tmp_path / name
    img.save(path)
    return path


def make_two_region_images(tmp_path: Path, size: tuple = (64, 64)):
    """Return (ref_path, render_path) where ONLY the top-left 32x32 differs.

    ref  = all black
    render = top-left 32x32 is white, rest is black
    Region A covers top-left 32x32 (x=0, y=0, w=0.5, h=0.5).
    Region B covers bottom-right 32x32 (x=0.5, y=0.5, w=0.5, h=0.5).
    """
    w, h = size
    ref_arr = np.zeros((h, w, 4), dtype=np.uint8)
    ref_arr[..., 3] = 255  # fully opaque black
    ref_img = Image.fromarray(ref_arr, "RGBA")
    ref_path = tmp_path / "ref.png"
    ref_img.save(ref_path)

    rnd_arr = np.zeros((h, w, 4), dtype=np.uint8)
    rnd_arr[..., 3] = 255
    # top-left half is white
    rnd_arr[: h // 2, : w // 2, :3] = 255
    rnd_img = Image.fromarray(rnd_arr, "RGBA")
    rnd_path = tmp_path / "render.png"
    rnd_img.save(rnd_path)

    return ref_path, rnd_path


def make_region(rid: str, x: float, y: float, w: float, h: float, mode: str = "modify") -> RegionConstraint:
    return RegionConstraint(
        id=rid,
        label=rid,
        mode=mode,
        instruction="",
        geometry_type="rect",
        geometry={"x": x, "y": y, "w": w, "h": h},
    )


# ---------------------------------------------------------------------------
# Empty regions list
# ---------------------------------------------------------------------------


def test_empty_regions_returns_empty_dict(tmp_path):
    ref = make_solid_rgba(tmp_path, "ref.png", (128, 0, 0, 255))
    rnd = make_solid_rgba(tmp_path, "rnd.png", (0, 0, 128, 255))
    result = compute_region_metrics(ref, rnd, [])
    assert result == {"regions": {}, "constraint_score": 1.0}


# ---------------------------------------------------------------------------
# Region-only pixel isolation — key acceptance criterion
# ---------------------------------------------------------------------------


def test_region_a_differs_region_b_identical(tmp_path):
    """Metrics are restricted to the rect: region A (where pixels differ) has
    nonzero mse; region B (where pixels are identical) has mse ≈ 0."""
    ref_path, rnd_path = make_two_region_images(tmp_path)

    region_a = make_region("A", 0.0, 0.0, 0.5, 0.5, mode="modify")
    region_b = make_region("B", 0.5, 0.5, 0.5, 0.5, mode="modify")

    result = compute_region_metrics(ref_path, rnd_path, [region_a, region_b])

    assert result["regions"]["A"]["mse"] > 0.5, "region A should have high MSE (white vs black)"
    assert result["regions"]["B"]["mse"] == pytest.approx(0.0, abs=1e-6), "region B unchanged → MSE ≈ 0"


def test_region_b_ssim_near_one_when_identical(tmp_path):
    """SSIM for an identical region should be close to 1."""
    ref_path, rnd_path = make_two_region_images(tmp_path)
    region_b = make_region("B", 0.5, 0.5, 0.5, 0.5, mode="protect")
    result = compute_region_metrics(ref_path, rnd_path, [region_b])
    assert result["regions"]["B"]["ssim"] > 0.99


# ---------------------------------------------------------------------------
# constraint_score — protect regions
# ---------------------------------------------------------------------------


def test_constraint_score_all_modify_no_protect(tmp_path):
    """No protect regions → constraint_score == 1.0."""
    ref_path, rnd_path = make_two_region_images(tmp_path)
    region_a = make_region("A", 0.0, 0.0, 0.5, 0.5, mode="modify")
    result = compute_region_metrics(ref_path, rnd_path, [region_a])
    assert result["constraint_score"] == 1.0


def test_constraint_score_protect_region_unchanged(tmp_path):
    """Protect region that is identical → ssim≈1 → constraint_score≈1."""
    ref_path, rnd_path = make_two_region_images(tmp_path)
    # region B is the bottom-right quadrant which is identical
    region_b = make_region("B", 0.5, 0.5, 0.5, 0.5, mode="protect")
    result = compute_region_metrics(ref_path, rnd_path, [region_b])
    assert result["constraint_score"] > 0.99


def test_constraint_score_protect_region_changed(tmp_path):
    """Protect region whose pixels changed a lot → constraint_score is lower
    than when protect region is unchanged."""
    ref_path, rnd_path = make_two_region_images(tmp_path)

    # protect on A (changed) → low ssim → low constraint_score
    protect_changed = make_region("A", 0.0, 0.0, 0.5, 0.5, mode="protect")
    result_changed = compute_region_metrics(ref_path, rnd_path, [protect_changed])

    # protect on B (unchanged) → ssim≈1 → high constraint_score
    protect_same = make_region("B", 0.5, 0.5, 0.5, 0.5, mode="protect")
    result_same = compute_region_metrics(ref_path, rnd_path, [protect_same])

    assert result_changed["constraint_score"] < result_same["constraint_score"]


def test_constraint_score_mixed_protect_modify(tmp_path):
    """Only protect regions contribute to constraint_score; modify regions don't."""
    ref_path, rnd_path = make_two_region_images(tmp_path)
    # A is modify (changed), B is protect (unchanged)
    region_a = make_region("A", 0.0, 0.0, 0.5, 0.5, mode="modify")
    region_b = make_region("B", 0.5, 0.5, 0.5, 0.5, mode="protect")
    result = compute_region_metrics(ref_path, rnd_path, [region_a, region_b])
    # protect region B is unchanged → constraint_score near 1
    assert result["constraint_score"] > 0.99


# ---------------------------------------------------------------------------
# mean_delta
# ---------------------------------------------------------------------------


def test_mean_delta_is_length_3_list_of_floats(tmp_path):
    ref_path, rnd_path = make_two_region_images(tmp_path)
    region_a = make_region("A", 0.0, 0.0, 0.5, 0.5, mode="modify")
    result = compute_region_metrics(ref_path, rnd_path, [region_a])
    md = result["regions"]["A"]["mean_delta"]
    assert isinstance(md, list)
    assert len(md) == 3
    for v in md:
        assert isinstance(v, float)


def test_mean_delta_positive_when_render_brighter(tmp_path):
    """render is white in region A, ref is black → mean_delta should be positive."""
    ref_path, rnd_path = make_two_region_images(tmp_path)
    region_a = make_region("A", 0.0, 0.0, 0.5, 0.5, mode="modify")
    result = compute_region_metrics(ref_path, rnd_path, [region_a])
    md = result["regions"]["A"]["mean_delta"]
    assert all(v > 0.9 for v in md), f"Expected near +1 for all channels, got {md}"


# ---------------------------------------------------------------------------
# edge_delta
# ---------------------------------------------------------------------------


def test_edge_delta_is_float(tmp_path):
    ref_path, rnd_path = make_two_region_images(tmp_path)
    region_a = make_region("A", 0.0, 0.0, 0.5, 0.5, mode="modify")
    result = compute_region_metrics(ref_path, rnd_path, [region_a])
    assert isinstance(result["regions"]["A"]["edge_delta"], float)


# ---------------------------------------------------------------------------
# JSON serialisability
# ---------------------------------------------------------------------------


def test_output_is_json_serialisable(tmp_path):
    ref_path, rnd_path = make_two_region_images(tmp_path)
    region_a = make_region("A", 0.0, 0.0, 0.5, 0.5, mode="modify")
    region_b = make_region("B", 0.5, 0.5, 0.5, 0.5, mode="protect")
    result = compute_region_metrics(ref_path, rnd_path, [region_a, region_b])
    # must not raise
    serialised = json.dumps(result)
    parsed = json.loads(serialised)
    assert "regions" in parsed
    assert "constraint_score" in parsed


# ---------------------------------------------------------------------------
# Non-rect geometry_type → unsupported_geometry
# ---------------------------------------------------------------------------


def test_polygon_geometry_type_returns_unsupported(tmp_path):
    ref = make_solid_rgba(tmp_path, "ref.png", (0, 0, 0, 255))
    rnd = make_solid_rgba(tmp_path, "rnd.png", (255, 0, 0, 255))
    region = RegionConstraint(
        id="poly1",
        label="polygon region",
        mode="modify",
        instruction="",
        geometry_type="polygon",
        geometry={"points": [[0, 0], [1, 0], [0.5, 1]]},
    )
    result = compute_region_metrics(ref, rnd, [region])
    assert result["regions"]["poly1"] == {"unsupported_geometry": "polygon"}


def test_mask_geometry_type_returns_unsupported(tmp_path):
    ref = make_solid_rgba(tmp_path, "ref.png", (0, 0, 0, 255))
    rnd = make_solid_rgba(tmp_path, "rnd.png", (255, 0, 0, 255))
    region = RegionConstraint(
        id="mask1",
        label="mask region",
        mode="modify",
        instruction="",
        geometry_type="mask",
        geometry={},
    )
    result = compute_region_metrics(ref, rnd, [region])
    assert result["regions"]["mask1"] == {"unsupported_geometry": "mask"}


# ---------------------------------------------------------------------------
# constraint_score with unsupported geometry protect region
# ---------------------------------------------------------------------------


def test_constraint_score_skips_unsupported_protect_region(tmp_path):
    """Unsupported geometry protect regions have no ssim → excluded from mean.
    If no valid protect ssim exists → constraint_score == 1.0."""
    ref = make_solid_rgba(tmp_path, "ref.png", (0, 0, 0, 255))
    rnd = make_solid_rgba(tmp_path, "rnd.png", (255, 0, 0, 255))
    region = RegionConstraint(
        id="poly_protect",
        label="",
        mode="protect",
        instruction="",
        geometry_type="polygon",
        geometry={},
    )
    result = compute_region_metrics(ref, rnd, [region])
    assert result["constraint_score"] == 1.0


# ---------------------------------------------------------------------------
# Clamped / edge cases
# ---------------------------------------------------------------------------


def test_rect_clamped_to_image_bounds(tmp_path):
    """A rect that slightly exceeds 1.0 is clamped and still computes metrics."""
    ref = make_solid_rgba(tmp_path, "ref.png", (0, 0, 0, 255))
    rnd = make_solid_rgba(tmp_path, "rnd.png", (100, 100, 100, 255))
    region = RegionConstraint(
        id="r1",
        label="",
        mode="modify",
        instruction="",
        geometry_type="rect",
        geometry={"x": 0.9, "y": 0.9, "w": 0.2, "h": 0.2},  # x+w=1.1, y+h=1.1
    )
    result = compute_region_metrics(ref, rnd, [region])
    assert "mse" in result["regions"]["r1"]


def test_zero_area_region_records_error(tmp_path):
    """A rect that clamps to zero area records error key."""
    ref = make_solid_rgba(tmp_path, "ref.png", (0, 0, 0, 255), size=(4, 4))
    rnd = make_solid_rgba(tmp_path, "rnd.png", (255, 0, 0, 255), size=(4, 4))
    # x=1.0 means x0==x1==W → zero width after clamping
    region = RegionConstraint(
        id="empty",
        label="",
        mode="modify",
        instruction="",
        geometry_type="rect",
        geometry={"x": 1.0, "y": 0.0, "w": 0.5, "h": 0.5},
    )
    result = compute_region_metrics(ref, rnd, [region])
    assert result["regions"]["empty"] == {"error": "empty_region"}


# ---------------------------------------------------------------------------
# Identical images → mse==0, ssim≈1
# ---------------------------------------------------------------------------


def test_identical_images_all_metrics_sane(tmp_path):
    ref = make_solid_rgba(tmp_path, "ref.png", (80, 120, 200, 255))
    region = make_region("r", 0.0, 0.0, 1.0, 1.0, mode="protect")
    result = compute_region_metrics(ref, ref, [region])
    r = result["regions"]["r"]
    assert r["mse"] == pytest.approx(0.0, abs=1e-6)
    assert r["ssim"] > 0.99
    assert result["constraint_score"] > 0.99
    md = r["mean_delta"]
    assert all(abs(v) < 1e-5 for v in md)


# ---------------------------------------------------------------------------
# Protect-region hard-veto core (Task 1)
# ---------------------------------------------------------------------------

from p2s_agent.orchestration.human_constraints import RegionConstraint
from p2s_agent.core.pipeline.region_metrics import (
    RegionVetoResult,
    protect_region_threshold,
    evaluate_protect_veto,
)
from PIL import Image


def _png(path, color, size=(64, 64)):
    Image.new("RGB", size, color).save(path)
    return path


def _protect_region(rid="r1", x=0.0, y=0.0, w=0.5, h=1.0, strength=0.5):
    return RegionConstraint(
        id=rid, label=rid, mode="protect", instruction="",
        geometry_type="rect", geometry={"x": x, "y": y, "w": w, "h": h},
        strength=strength,
    )


def test_protect_region_threshold_maps_strength():
    assert protect_region_threshold(0.0) == 0.85
    assert protect_region_threshold(0.5) == 0.90
    assert protect_region_threshold(1.0) == 0.95


def test_no_protect_regions_is_not_evaluated_and_not_vetoed(tmp_path):
    base = _png(tmp_path / "b.png", (10, 20, 30))
    cand = _png(tmp_path / "c.png", (200, 0, 0))
    modify = RegionConstraint(
        id="m", label="m", mode="modify", instruction="",
        geometry_type="rect", geometry={"x": 0, "y": 0, "w": 1, "h": 1}, strength=0.5,
    )
    res = evaluate_protect_veto(base, cand, [modify])
    assert isinstance(res, RegionVetoResult)
    assert res.vetoed is False and res.evaluated is False


def test_identical_protect_region_not_vetoed(tmp_path):
    base = _png(tmp_path / "b.png", (10, 20, 30))
    cand = _png(tmp_path / "c.png", (10, 20, 30))  # identical
    res = evaluate_protect_veto(base, cand, [_protect_region()])
    assert res.vetoed is False
    assert res.constraint_score > 0.95
    assert res.evaluated is True


def test_degraded_protect_region_is_vetoed(tmp_path):
    base = Image.new("RGB", (64, 64), (10, 20, 30))
    cand = Image.new("RGB", (64, 64), (10, 20, 30))
    for px in range(32):
        for py in range(64):
            cand.putpixel((px, py), (255, 255, 255))
    bpath = tmp_path / "b.png"; base.save(bpath)
    cpath = tmp_path / "c.png"; cand.save(cpath)
    region = _protect_region(x=0.0, w=0.5)  # protect the left half
    res = evaluate_protect_veto(bpath, cpath, [region])
    assert res.vetoed is True
    assert any(r["violated"] for r in res.regions)
    assert res.reason and "r1" in res.reason


def test_any_violated_region_triggers_veto(tmp_path):
    base = Image.new("RGB", (64, 64), (10, 20, 30))
    cand = Image.new("RGB", (64, 64), (10, 20, 30))
    for px in range(32):           # only left half changes
        for py in range(64):
            cand.putpixel((px, py), (255, 255, 255))
    bpath = tmp_path / "b.png"; base.save(bpath)
    cpath = tmp_path / "c.png"; cand.save(cpath)
    left = _protect_region(rid="left", x=0.0, w=0.5)    # degraded -> violated
    right = _protect_region(rid="right", x=0.5, w=0.5)  # untouched -> ok
    res = evaluate_protect_veto(bpath, cpath, [left, right])
    assert res.vetoed is True


def test_missing_baseline_is_not_evaluated(tmp_path):
    cand = _png(tmp_path / "c.png", (10, 20, 30))
    res = evaluate_protect_veto(tmp_path / "does_not_exist.png", cand, [_protect_region()])
    assert res.evaluated is False and res.vetoed is False
