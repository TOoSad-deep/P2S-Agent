"""Golden characterization + performance tests for the DSL renderer.

Golden PNGs are generated from the renderer BEFORE vectorization (plan
Task 2b Step 2), pinning the scalar implementation's behavior; the
vectorized renderer must match within tolerance.
"""
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.dsl.renderer import render_dsl_to_image
from app.dsl.schema import (
    FIXTURE_BOX_GRADIENT,
    FIXTURE_CIRCLE_SOLID,
    FIXTURE_GLOW_RING,
    FIXTURE_ROUNDEDBOX_VIGNETTE,
)

GOLDEN_DIR = Path(__file__).parent / "golden"
GOLDEN_SIZE = 96

GOLDEN_CASES = {
    "circle": FIXTURE_CIRCLE_SOLID,
    "box_gradient": FIXTURE_BOX_GRADIENT,
    "glow_ring": FIXTURE_GLOW_RING,
    "roundedbox_vignette": FIXTURE_ROUNDEDBOX_VIGNETTE,
    "ellipse": {
        "schema_version": 1,
        "canvas": {"width": 512, "height": 512, "background": "#102030"},
        "layers": [{
            "id": "e1", "type": "ellipse",
            "fill": {"type": "solid", "color": "#ff8800"},
            "params": {"center": [0.45, 0.55], "ab": [0.3, 0.18]},
            "opacity": 1.0,
        }],
    },
    "radial_3stop": {
        "schema_version": 1,
        "canvas": {"width": 512, "height": 512, "background": "#000000"},
        "layers": [{
            "id": "r1", "type": "box",
            "fill": {
                "type": "radialGradient",
                "center": [0.5, 0.5],
                "stops": [
                    {"color": "#ffffff", "position": 0.0},
                    {"color": "#220044", "position": 0.7},
                    {"color": "#000000", "position": 1.0},
                ],
            },
            "params": {"center": [0.5, 0.5], "size": [0.9, 0.9]},
            "opacity": 1.0,
        }],
    },
    "polygon": {
        "schema_version": 1,
        "canvas": {"width": 512, "height": 512, "background": "#101820"},
        "layers": [{
            "id": "p1", "type": "polygon",
            "fill": {"type": "solid", "color": "#ffaa22"},
            "params": {"center": [0.5, 0.5], "radius": 0.3, "sides": 6},
            "opacity": 1.0,
        }],
    },
    "grain": {
        "schema_version": 1,
        "canvas": {"width": 512, "height": 512, "background": "#202020"},
        "layers": [{
            "id": "g1", "type": "box",
            "fill": {"type": "solid", "color": "#8080c0"},
            "params": {"center": [0.5, 0.5], "size": [0.8, 0.8]},
            "opacity": 1.0,
            "effects": [{"type": "grain", "amount": 0.2}],
        }],
    },
}


@pytest.mark.parametrize("name", sorted(GOLDEN_CASES))
def test_render_matches_golden(name, tmp_path):
    golden_path = GOLDEN_DIR / f"{name}.png"
    assert golden_path.exists(), "run the golden generation snippet (Task 2b Step 2) first"
    out = render_dsl_to_image(
        GOLDEN_CASES[name], tmp_path / f"{name}.png",
        width=GOLDEN_SIZE, height=GOLDEN_SIZE,
    )
    got = np.asarray(Image.open(out).convert("RGB"), dtype=np.int16)
    want = np.asarray(Image.open(golden_path).convert("RGB"), dtype=np.int16)
    diff = np.abs(got - want)
    assert diff.mean() < 1.0, f"mean abs diff {diff.mean():.3f} exceeds 1/255"
    assert (diff > 8).mean() < 0.01, "over 1% of pixels deviate by more than 8/255"


def test_render_192_five_layers_is_fast(tmp_path):
    dsl = {
        "schema_version": 1,
        "canvas": {"width": 512, "height": 512, "background": "#101010"},
        "layers": [
            {"id": "l0", "type": "circle", "fill": {"type": "solid", "color": "#3366ff"},
             "params": {"center": [0.3, 0.3], "radius": 0.2}, "opacity": 0.9},
            {"id": "l1", "type": "ellipse", "fill": {"type": "solid", "color": "#ff6633"},
             "params": {"center": [0.6, 0.4], "ab": [0.25, 0.12]}, "opacity": 0.9},
            {"id": "l2", "type": "roundedBox", "fill": {"type": "solid", "color": "#33ff66"},
             "params": {"center": [0.5, 0.7], "size": [0.4, 0.2], "radius": 0.04}, "opacity": 0.9},
            {"id": "l3", "type": "ring", "fill": {"type": "solid", "color": "#ffcc00"},
             "params": {"center": [0.7, 0.7], "radius": 0.15, "thickness": 0.02}, "opacity": 0.9},
            {"id": "l4", "type": "polygon", "fill": {"type": "solid", "color": "#cc33ff"},
             "params": {"center": [0.25, 0.7], "radius": 0.15, "sides": 6}, "opacity": 0.9},
        ],
    }
    start = time.monotonic()
    render_dsl_to_image(dsl, tmp_path / "perf.png", width=192, height=192)
    elapsed = time.monotonic() - start
    assert elapsed < 0.5, f"192x192 render took {elapsed:.2f}s — renderer must be vectorized"
