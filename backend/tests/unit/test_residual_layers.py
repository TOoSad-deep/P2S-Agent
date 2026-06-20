"""Tests for residual-driven layer addition."""
import pytest

cv2 = pytest.importorskip("cv2")

import numpy as np
from PIL import Image, ImageDraw

from p2s_agent.core.dsl.renderer import render_dsl_to_image
from p2s_agent.core.pipeline.residual_layers import add_residual_layers


def _ref_two_circles(tmp_path):
    img = Image.new("RGBA", (128, 128), (0, 0, 0, 255))
    d = ImageDraw.Draw(img)
    d.ellipse((16, 16, 64, 64), fill=(255, 0, 0, 255))
    d.ellipse((80, 80, 120, 120), fill=(0, 255, 0, 255))
    path = tmp_path / "ref.png"
    img.save(path)
    return path


def _dsl_one_circle():
    return {
        "schema_version": 1,
        "canvas": {"width": 128, "height": 128, "background": "#000000"},
        "layers": [{
            "id": "c0", "type": "circle",
            "fill": {"type": "solid", "color": "#ff0000"},
            "params": {"center": [0.3125, 0.3125], "radius": 0.1875},
            "opacity": 1.0,
        }],
    }


def test_residual_adds_missing_circle(tmp_path):
    ref_path = _ref_two_circles(tmp_path)
    ref = np.asarray(Image.open(ref_path).convert("RGB"), dtype=np.float32) / 255.0
    counter = {"n": 0}

    def render_fn(dsl):
        counter["n"] += 1
        out = tmp_path / f"r{counter['n']}.png"
        return render_dsl_to_image(dsl, out, width=128, height=128)

    def score_fn(dsl):
        path = render_fn(dsl)
        rnd = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        return 1.0 - float(np.abs(ref - rnd).mean())

    result = add_residual_layers(
        _dsl_one_circle(), ref_path, score_fn=score_fn, render_fn=render_fn, max_added=3
    )

    assert result.layers_added >= 1
    assert result.final_score > result.initial_score
    new_ids = [l["id"] for l in result.final_dsl["layers"]]
    assert any(i.startswith("res_") for i in new_ids)


def test_residual_handles_dsl_without_layers_key(tmp_path):
    """A starting DSL with no 'layers' key must not raise KeyError when a
    residual layer is added (regression: current['layers'] was unguarded)."""
    ref_path = _ref_two_circles(tmp_path)
    counter = {"n": 0}

    def render_fn(dsl):
        counter["n"] += 1
        out = tmp_path / f"r{counter['n']}.png"
        # Always render black so the residual vs the two-circle ref is high,
        # guaranteeing a layer is fitted and line 106 is reached.
        Image.new("RGB", (128, 128), (0, 0, 0)).save(out)
        return out

    def score_fn(dsl):
        # Reward added layers so the fitted residual layer is accepted.
        return 0.3 + 0.1 * len(dsl.get("layers", []))

    dsl_no_layers = {
        "schema_version": 1,
        "canvas": {"width": 128, "height": 128, "background": "#000000"},
        # NOTE: intentionally no "layers" key.
    }
    result = add_residual_layers(
        dsl_no_layers, ref_path, score_fn=score_fn, render_fn=render_fn, max_added=1
    )
    assert "layers" in result.final_dsl
    assert result.layers_added >= 1
