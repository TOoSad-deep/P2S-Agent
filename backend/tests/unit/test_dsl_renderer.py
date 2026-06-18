"""Unit tests for the pure-Pillow PNG-to-Shader DSL renderer."""

from __future__ import annotations

from PIL import Image

from app.dsl.renderer import render_dsl_to_image
from app.dsl.schema import FIXTURE_CIRCLE_SOLID, FIXTURE_BOX_GRADIENT
from app.metrics.compute import check_nonblank_render


def test_render_dsl_to_image_writes_nonblank_png(tmp_path):
    output = tmp_path / "render.png"

    path = render_dsl_to_image(FIXTURE_CIRCLE_SOLID, output, width=64, height=64)

    assert path == output
    assert output.exists()
    assert Image.open(output).size == (64, 64)
    assert check_nonblank_render(output)


def test_render_dsl_to_image_handles_gradient_fill(tmp_path):
    output = tmp_path / "gradient.png"

    render_dsl_to_image(FIXTURE_BOX_GRADIENT, output, width=64, height=64)

    img = Image.open(output).convert("RGB")
    colors = img.getcolors(maxcolors=4096)
    assert colors is not None
    assert len(colors) > 2


def test_renderer_hex_to_rgb_parses_8_digit_rgba():
    """Bug 4 mirror: the renderer's _hex_to_rgb must parse the RGB part of an
    8-digit #RRGGBBAA color (matching the compiler), not collapse to white."""
    from app.dsl.renderer import _hex_to_rgb

    assert _hex_to_rgb("#112233ff") == (0x11, 0x22, 0x33)
