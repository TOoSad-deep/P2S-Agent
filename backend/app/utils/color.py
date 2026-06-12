"""Color normalization utilities for PNG-to-Shader.

Extracted from the retired normalizer.py — normalize_color is the only
part of the old CanonicalSceneGraph normalization layer that the
pipeline actually uses (LLM scene candidate color cleanup).
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# CSS named colours (minimal common set)
# ---------------------------------------------------------------------------

_CSS_NAMED_COLORS: dict[str, str] = {
    "red": "#ff0000",
    "green": "#008000",
    "blue": "#0000ff",
    "white": "#ffffff",
    "black": "#000000",
    "yellow": "#ffff00",
    "cyan": "#00ffff",
    "magenta": "#ff00ff",
    "orange": "#ffa500",
    "purple": "#800080",
    "pink": "#ffc0cb",
    "brown": "#a52a2a",
    "gray": "#808080",
    "grey": "#808080",
    "silver": "#c0c0c0",
    "gold": "#ffd700",
    "transparent": "#00000000",
    "lime": "#00ff00",
    "maroon": "#800000",
    "navy": "#000080",
    "olive": "#808000",
    "teal": "#008080",
    "aqua": "#00ffff",
    "fuchsia": "#ff00ff",
    "indigo": "#4b0082",
    "violet": "#ee82ee",
    "coral": "#ff7f50",
    "salmon": "#fa8072",
    "khaki": "#f0e68c",
    "beige": "#f5f5dc",
    "ivory": "#fffff0",
    "lavender": "#e6e6fa",
    "turquoise": "#40e0d0",
    "tan": "#d2b48c",
    "wheat": "#f5deb3",
    "crimson": "#dc143c",
    "darkblue": "#00008b",
    "darkgreen": "#006400",
    "darkred": "#8b0000",
    "darkorange": "#ff8c00",
    "deeppink": "#ff1493",
    "dodgerblue": "#1e90ff",
    "firebrick": "#b22222",
    "forestgreen": "#228b22",
    "hotpink": "#ff69b4",
    "limegreen": "#32cd32",
    "midnightblue": "#191970",
    "orangered": "#ff4500",
    "royalblue": "#4169e1",
    "seagreen": "#2e8b57",
    "skyblue": "#87ceeb",
    "slategray": "#708090",
    "slategrey": "#708090",
    "springgreen": "#00ff7f",
    "steelblue": "#4682b4",
    "tomato": "#ff6347",
    "yellowgreen": "#9acd32",
}


def normalize_color(value: Any) -> "str | None":
    """Convert various colour representations to ``"#RRGGBB"`` or ``"#RRGGBBAA"``.

    Accepted forms:
    - CSS 3-digit hex: ``"#abc"`` → ``"#aabbcc"``
    - CSS 6-digit hex: ``"#aabbcc"`` → unchanged
    - CSS 8-digit hex: ``"#aabbccdd"`` → unchanged
    - RGB array of ints (0-255): ``[255, 0, 0]`` → ``"#ff0000"``
    - RGB array of floats (0.0-1.0): ``[1.0, 0.0, 0.0]`` → ``"#ff0000"``
    - CSS named colour: ``"red"`` → ``"#ff0000"``

    Returns ``None`` if the value cannot be parsed.
    """
    if value is None:
        return None

    # --- String inputs ---
    if isinstance(value, str):
        v = value.strip()

        # Named colours
        lower = v.lower()
        if lower in _CSS_NAMED_COLORS:
            return _CSS_NAMED_COLORS[lower]

        # Hex strings
        if v.startswith("#"):
            hex_body = v[1:]
            if len(hex_body) == 3 and re.fullmatch(r"[0-9a-fA-F]{3}", hex_body):
                r, g, b = hex_body
                return f"#{r}{r}{g}{g}{b}{b}".lower()
            if len(hex_body) == 6 and re.fullmatch(r"[0-9a-fA-F]{6}", hex_body):
                return v.lower()
            if len(hex_body) == 8 and re.fullmatch(r"[0-9a-fA-F]{8}", hex_body):
                return v.lower()
        return None

    # --- List/tuple inputs ---
    if isinstance(value, (list, tuple)) and len(value) in (3, 4):
        try:
            nums = [float(x) for x in value]
        except (TypeError, ValueError):
            return None

        # Detect whether values are 0-255 integers or 0.0-1.0 floats.
        # Priority: if any element is a Python float type, treat as 0.0-1.0.
        # Otherwise (all ints): treat as 0-255. Fallback: any value > 1 → int range.
        has_explicit_float = any(isinstance(x, float) for x in value)
        if has_explicit_float:
            # Explicit float literals → 0.0-1.0 range
            ints = [max(0, min(255, round(v * 255))) for v in nums]
        elif any(v > 1.0 for v in nums):
            # Any int channel > 1 → must be 0-255
            ints = [max(0, min(255, round(v))) for v in nums]
        else:
            # All int values are 0 or 1 → treat as 0-255 byte values
            ints = [max(0, min(255, round(v))) for v in nums]

        if len(ints) == 3:
            return f"#{ints[0]:02x}{ints[1]:02x}{ints[2]:02x}"
        else:
            return f"#{ints[0]:02x}{ints[1]:02x}{ints[2]:02x}{ints[3]:02x}"

    return None
