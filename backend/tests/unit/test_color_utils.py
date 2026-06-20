"""Tests for color_utils (migrated from the retired test_normalizer.py)."""

from __future__ import annotations

from p2s_agent.core.utils.color import normalize_color


def test_normalize_color_css_hex_6digit():
    assert normalize_color("#ff0000") == "#ff0000"
    assert normalize_color("#FF0000") == "#ff0000"
    assert normalize_color("#aabbcc") == "#aabbcc"


def test_normalize_color_css_hex_3digit():
    assert normalize_color("#abc") == "#aabbcc"
    assert normalize_color("#ABC") == "#aabbcc"
    assert normalize_color("#fff") == "#ffffff"
    assert normalize_color("#000") == "#000000"


def test_normalize_color_rgb_array_255():
    assert normalize_color([255, 0, 0]) == "#ff0000"
    assert normalize_color([0, 255, 0]) == "#00ff00"
    assert normalize_color([0, 0, 255]) == "#0000ff"
    assert normalize_color([128, 128, 128]) == "#808080"


def test_normalize_color_rgb_array_float():
    result = normalize_color([1.0, 0.0, 0.0])
    assert result == "#ff0000"
    result = normalize_color([0.0, 1.0, 0.0])
    assert result == "#00ff00"
    result = normalize_color([0.5, 0.5, 0.5])
    # 0.5 * 255 = 127.5 → rounds to 128 = 0x80
    assert result == "#808080"


def test_normalize_color_named_color():
    assert normalize_color("red") == "#ff0000"
    assert normalize_color("blue") == "#0000ff"
    assert normalize_color("white") == "#ffffff"
    assert normalize_color("black") == "#000000"
    assert normalize_color("RED") == "#ff0000"  # case-insensitive


def test_normalize_color_transparent():
    result = normalize_color("transparent")
    assert result is not None
    assert "00" in result  # has zeroed alpha or RGB


def test_normalize_color_returns_none_on_unknown():
    assert normalize_color("not-a-color") is None
    assert normalize_color(None) is None
    assert normalize_color({"r": 1}) is None
    assert normalize_color("#xyz") is None


def test_normalize_color_4_element_array():
    """RGBA array should produce 8-digit hex."""
    result = normalize_color([255, 0, 0, 128])
    assert result is not None
    assert result.startswith("#")
    assert len(result) == 9  # "#RRGGBBAA"
