"""Image preprocessing for PNG-to-Shader.

Extracts structural features from a PNG image using only Pillow.
No LLM, no browser, no numpy required.
"""

from __future__ import annotations

import json
from collections import Counter, deque
from pathlib import Path
from typing import Any

from PIL import Image


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def preprocess_image(image_path: "str | Path") -> dict[str, Any]:
    """Extract structural features from a PNG (or any Pillow-readable) image.

    Returns a dict with these keys:

    - ``width``, ``height`` — original pixel dimensions.
    - ``has_alpha`` — True if the image has an alpha channel.
    - ``alpha_coverage`` — fraction of pixels with alpha > 0 (1.0 for opaque).
    - ``palette`` — top-5 most-common colours as ``"#RRGGBB"`` strings.
    - ``color_count_estimate`` — approximate distinct colours (32-level bins).
    - ``edge_sharpness`` — 0.0–1.0, fraction of "edge-like" pixels.
    - ``component_count_estimate`` — rough connected-component count.
    - ``texture_score`` — 0.0–1.0, high ↔ fine texture / noise.
    - ``photo_like_score`` — 0.0–1.0, high ↔ photographic content.
    - ``gradient_score`` — 0.0–1.0, high ↔ smooth colour gradients.
    """
    img = Image.open(image_path)
    width, height = img.size

    # ---- Alpha channel analysis ----------------------------------------
    has_alpha = img.mode in ("RGBA", "LA", "PA") or (
        img.mode == "P" and img.info.get("transparency") is not None
    )

    if has_alpha:
        rgba = img.convert("RGBA")
        pixels_rgba = list(rgba.getdata())
        total = len(pixels_rgba)
        non_transparent = sum(1 for p in pixels_rgba if p[3] > 0)
        alpha_coverage = non_transparent / total if total > 0 else 0.0
    else:
        alpha_coverage = 1.0

    # ---- Palette (top-5 colours, 16-level RGB quantisation) -------------
    rgb_img = img.convert("RGB")
    pixels_rgb = list(rgb_img.getdata())
    total = len(pixels_rgb)

    def _bin16(v: int) -> int:
        """Map 0-255 into one of 16 equal bins; returns bin centre."""
        b = min(v >> 4, 15)  # 0-15
        return b * 16 + 8     # centre of bin

    bin16_counts: Counter[tuple[int, int, int]] = Counter(
        (_bin16(r), _bin16(g), _bin16(b)) for r, g, b in pixels_rgb
    )
    top5 = bin16_counts.most_common(5)
    palette = [f"#{r:02X}{g:02X}{b:02X}" for (r, g, b), _ in top5]
    # Pad to exactly 5 entries if fewer distinct colours exist.
    while len(palette) < 5:
        palette.append("#000000")

    # ---- Color count estimate (32-level bins) ----------------------------
    def _bin32(v: int) -> int:
        return min(v >> 3, 31)  # 0-31

    bin32_set: set[tuple[int, int, int]] = set(
        (_bin32(r), _bin32(g), _bin32(b)) for r, g, b in pixels_rgb
    )
    color_count_estimate = len(bin32_set)

    # ---- Edge sharpness --------------------------------------------------
    # Compare each pixel with its right and bottom neighbours.
    edge_count = 0
    edge_total = 0
    for y in range(height):
        for x in range(width):
            idx = y * width + x
            pr, pg, pb = pixels_rgb[idx]

            # Right neighbour
            if x + 1 < width:
                nr, ng, nb = pixels_rgb[idx + 1]
                if (
                    abs(pr - nr) > 20
                    or abs(pg - ng) > 20
                    or abs(pb - nb) > 20
                ):
                    edge_count += 1
                edge_total += 1

            # Bottom neighbour
            if y + 1 < height:
                br, bg, bb = pixels_rgb[idx + width]
                if (
                    abs(pr - br) > 20
                    or abs(pg - bg) > 20
                    or abs(pb - bb) > 20
                ):
                    edge_count += 1
                edge_total += 1

    edge_sharpness = edge_count / edge_total if edge_total > 0 else 0.0

    # ---- Component count (BFS on downsampled mask) -----------------------
    component_count_estimate = _count_components(img, has_alpha, width, height)

    # ---- Texture score ---------------------------------------------------
    texture_score = _compute_texture_score(pixels_rgb, width, height)

    # ---- Derived scores --------------------------------------------------
    photo_raw = (
        0.4 * min(color_count_estimate / 100.0, 1.0)
        + 0.3 * (1.0 - edge_sharpness)
        + 0.3 * texture_score
    )
    photo_like_score = max(0.0, min(1.0, photo_raw))

    gradient_score = max(
        0.0, min(1.0, 0.7 * (1.0 - edge_sharpness) + 0.3 * (1.0 - texture_score))
    )

    return {
        "width": width,
        "height": height,
        "has_alpha": has_alpha,
        "alpha_coverage": alpha_coverage,
        "palette": palette,
        "color_count_estimate": color_count_estimate,
        "edge_sharpness": edge_sharpness,
        "component_count_estimate": component_count_estimate,
        "texture_score": texture_score,
        "photo_like_score": photo_like_score,
        "gradient_score": gradient_score,
    }


def save_preprocess_artifacts(
    preprocess_dict: dict[str, Any],
    run_dir: Path,
    image_path: Path,
) -> None:
    """Save preprocessing results and normalised images into *run_dir*.

    Writes:
    - ``preprocess.json`` — the full feature dict.
    - ``normalized_input.png`` — 128×128 center-crop / resize of the source.
    - ``alpha_mask.png`` — 128×128 grayscale alpha mask.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # preprocess.json
    json_path = run_dir / "preprocess.json"
    tmp = json_path.with_suffix(".json.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(preprocess_dict, fh, indent=2, sort_keys=True)
            fh.write("\n")
        import os
        os.replace(tmp, json_path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass

    # normalized_input.png — 128×128 thumbnail (center-crop then resize)
    img = Image.open(image_path)
    thumb = _center_crop_resize(img, 128, 128)
    _composite_for_model_view(thumb).save(run_dir / "normalized_input.png")
    _composite_for_model_view(img).save(run_dir / "llm_reference_input.png")

    # alpha_mask.png — 128×128 grayscale alpha
    if img.mode in ("RGBA", "LA"):
        alpha_img = _center_crop_resize(img, 128, 128)
        alpha_channel = alpha_img.split()[-1]  # last channel is alpha
    elif img.mode == "PA":
        alpha_img = _center_crop_resize(img.convert("RGBA"), 128, 128)
        alpha_channel = alpha_img.split()[-1]
    else:
        # Fully opaque — white mask
        alpha_channel = Image.new("L", (128, 128), 255)

    alpha_channel.save(run_dir / "alpha_mask.png")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _center_crop_resize(img: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Center-crop img to the target aspect ratio, then resize."""
    w, h = img.size
    src_aspect = w / h
    tgt_aspect = target_w / target_h

    if src_aspect > tgt_aspect:
        # Wider than target — crop left/right
        new_w = int(h * tgt_aspect)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    elif src_aspect < tgt_aspect:
        # Taller than target — crop top/bottom
        new_h = int(w / tgt_aspect)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))

    return img.resize((target_w, target_h), Image.LANCZOS)


def _composite_for_model_view(
    img: Image.Image,
    *,
    background: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    """Return an opaque RGB image matching the black-canvas preview.

    Some VLM providers mishandle transparent PNGs and effectively show the
    alpha mask instead of the intended color result. The UI and metrics use a
    black preview background, so the model-facing artifact should do the same.
    """
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and img.info.get("transparency") is not None):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (*background, 255))
        bg.alpha_composite(rgba)
        return bg.convert("RGB")
    return img.convert("RGB")


def _count_components(
    img: Image.Image,
    has_alpha: bool,
    width: int,
    height: int,
) -> int:
    """Estimate number of connected non-transparent components via BFS.

    Downsamples to at most 32×32 to keep runtime bounded.
    """
    MAX_DIM = 32
    scale_x = max(1, width // MAX_DIM)
    scale_y = max(1, height // MAX_DIM)
    ds_w = (width + scale_x - 1) // scale_x
    ds_h = (height + scale_y - 1) // scale_y

    # Build binary foreground mask on the downsampled grid
    if has_alpha:
        rgba = img.convert("RGBA")
        src_pixels = list(rgba.getdata())
        mask = [False] * (ds_w * ds_h)
        for dy in range(ds_h):
            for dx in range(ds_w):
                sx = min(dx * scale_x, width - 1)
                sy = min(dy * scale_y, height - 1)
                idx = sy * width + sx
                mask[dy * ds_w + dx] = src_pixels[idx][3] > 0
    else:
        # Use luminance threshold for opaque images.
        # Use midpoint between global min and max to handle uniform images
        # gracefully (uniform → small range → returns 1 component).
        gray = img.convert("L")
        src_pixels = list(gray.getdata())
        mask = [False] * (ds_w * ds_h)
        if src_pixels:
            lum_min = min(src_pixels)
            lum_max = max(src_pixels)
            if lum_max - lum_min < 20:
                # Nearly uniform image — treat as a single component
                mask = [True] * (ds_w * ds_h)
                src_pixels = None  # skip per-pixel loop below
            else:
                threshold = (lum_min + lum_max) / 2.0
        else:
            threshold = 128
        if src_pixels is not None:
            for dy in range(ds_h):
                for dx in range(ds_w):
                    sx = min(dx * scale_x, width - 1)
                    sy = min(dy * scale_y, height - 1)
                    idx = sy * width + sx
                    mask[dy * ds_w + dx] = src_pixels[idx] > threshold

    # BFS to count connected components
    visited = [False] * (ds_w * ds_h)
    components = 0

    for start in range(ds_w * ds_h):
        if not mask[start] or visited[start]:
            continue
        # New component found
        components += 1
        queue = deque([start])
        visited[start] = True
        while queue:
            cur = queue.popleft()
            cy, cx = divmod(cur, ds_w)
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= nx < ds_w and 0 <= ny < ds_h:
                    nidx = ny * ds_w + nx
                    if mask[nidx] and not visited[nidx]:
                        visited[nidx] = True
                        queue.append(nidx)

    return max(1, components)


def _compute_texture_score(
    pixels_rgb: list[tuple[int, int, int]],
    width: int,
    height: int,
) -> float:
    """Compute texture score as normalised mean std-dev of 8×8 blocks.

    High variance inside a block → high texture score.
    """
    block_size = 8
    if width < block_size or height < block_size:
        return 0.0

    block_stds: list[float] = []

    for by in range(0, height - block_size + 1, block_size):
        for bx in range(0, width - block_size + 1, block_size):
            vals: list[int] = []
            for dy in range(block_size):
                for dx in range(block_size):
                    r, g, b = pixels_rgb[(by + dy) * width + (bx + dx)]
                    lum = (r * 299 + g * 587 + b * 114) // 1000
                    vals.append(lum)
            mean = sum(vals) / len(vals)
            variance = sum((v - mean) ** 2 for v in vals) / len(vals)
            block_stds.append(variance ** 0.5)

    if not block_stds:
        return 0.0

    mean_std = sum(block_stds) / len(block_stds)
    # Normalise: std of 40 ≈ moderately textured, 80+ ≈ very textured
    score = min(1.0, mean_std / 80.0)
    return score
