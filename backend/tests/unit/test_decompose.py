"""Tests for structural image decomposition."""
import pytest

cv2 = pytest.importorskip("cv2")

from PIL import Image, ImageDraw

from app.dsl.compiler import compile_dsl
_decompose = pytest.importorskip(
    "app.pipeline.decompose", reason="accuracy plan Phase 2 Task 5 not implemented yet"
)
decompose_to_dsl = _decompose.decompose_to_dsl
fit_primitive_layer = _decompose.fit_primitive_layer
from app.dsl.validator import validate_dsl
import numpy as np


def _save(tmp_path, name, img):
    path = tmp_path / name
    img.save(path)
    return path


def test_decompose_red_circle_on_white(tmp_path):
    img = Image.new("RGB", (128, 128), (255, 255, 255))
    ImageDraw.Draw(img).ellipse((32, 32, 96, 96), fill=(255, 0, 0))
    path = _save(tmp_path, "circle.png", img)

    dsl = decompose_to_dsl(path, 512, 512, max_colors=4)

    assert dsl is not None
    r, g, b = (int(dsl["canvas"]["background"][i:i + 2], 16) for i in (1, 3, 5))
    assert min(r, g, b) > 230, "background should be near-white"
    assert len(dsl["layers"]) == 1
    layer = dsl["layers"][0]
    assert layer["type"] == "circle"
    cx, cy = layer["params"]["center"]
    assert abs(cx - 0.5) < 0.05 and abs(cy - 0.5) < 0.05
    assert abs(layer["params"]["radius"] - 0.25) < 0.05
    assert validate_dsl(dsl).valid
    assert compile_dsl(dsl).success


def test_decompose_two_shapes_ordered_by_area(tmp_path):
    img = Image.new("RGB", (128, 128), (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle((8, 8, 120, 80), fill=(0, 0, 255))      # big box
    d.ellipse((90, 90, 118, 118), fill=(255, 255, 0))   # small circle
    path = _save(tmp_path, "two.png", img)

    dsl = decompose_to_dsl(path, 512, 512, max_colors=4)

    assert dsl is not None
    assert len(dsl["layers"]) == 2
    assert dsl["layers"][0]["type"] == "box", "bigger shape must come first (bottom)"


def test_fit_primitive_prefers_box_for_rectangle():
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:80, 10:90] = True
    layer = fit_primitive_layer(mask, color_hex="#112233")
    assert layer is not None
    assert layer["type"] == "box"
    assert abs(layer["params"]["size"][0] - 0.80) < 0.05


def test_decompose_candidate_skips_photo_like(tmp_path):
    from app.candidates.decompose import generate_decompose_candidate
    img = Image.new("RGB", (64, 64), (255, 255, 255))
    ImageDraw.Draw(img).ellipse((16, 16, 48, 48), fill=(255, 0, 0))
    path = _save(tmp_path, "p.png", img)

    assert generate_decompose_candidate({"photo_like_score": 0.9}, path) is None

    dsl = generate_decompose_candidate({"photo_like_score": 0.1}, path)
    assert dsl is not None
    assert dsl["_meta"] == {"source": "decompose", "priority": 1}


def test_decompose_low_color_image_does_not_raise(tmp_path):
    """A flat / few-color image whose quantized palette has fewer than
    ``max_colors`` entries must not raise IndexError in _merge_similar_colors
    and must still return a usable DSL (or None), never crash."""
    img = Image.new("RGB", (128, 128), (10, 20, 30))
    path = _save(tmp_path, "flat.png", img)

    # Must not raise (regression: getpalette() len 3 vs max_colors=6 -> IndexError)
    dsl = decompose_to_dsl(path, 512, 512, max_colors=6)

    # A single flat color has no foreground regions, so None is acceptable;
    # the contract is "does not crash". When a DSL is produced it must be valid.
    if dsl is not None:
        assert validate_dsl(dsl).valid
        assert compile_dsl(dsl).success


def test_decompose_few_color_shape_returns_valid_dsl(tmp_path):
    """A simple two-color image (foreground shape on flat bg) where the
    quantized palette is smaller than max_colors yields a usable DSL."""
    img = Image.new("RGB", (128, 128), (10, 20, 30))
    ImageDraw.Draw(img).ellipse((40, 40, 88, 88), fill=(220, 30, 40))
    path = _save(tmp_path, "fewcolor.png", img)

    dsl = decompose_to_dsl(path, 512, 512, max_colors=6)

    assert dsl is not None
    assert len(dsl["layers"]) >= 1
    assert validate_dsl(dsl).valid
    assert compile_dsl(dsl).success
