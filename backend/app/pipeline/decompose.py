"""Structural image decomposition for PNG-to-Shader.

color quantization + connected components + primitive fitting:
produces a DSL whose geometry is *measured* from the input pixels
instead of guessed from global statistics (alpha coverage, palette).

Requires opencv. Callers must check DECOMPOSE_AVAILABLE or rely on the
candidate wrapper returning None.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import cv2
    DECOMPOSE_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    cv2 = None
    DECOMPOSE_AVAILABLE = False

ALPHA_THRESHOLD = 16
ANALYSIS_SIZE = 256       # decomposition runs at this square resolution
MIN_FIT_IOU = 0.55        # below this, fall back to moments ellipse
MERGE_COLOR_DIST = 50.0   # merge palette entries within this Euclidean RGB distance


def _palette_hex(palette: list[int], label: int) -> str:
    r, g, b = palette[label * 3: label * 3 + 3]
    return f"#{r:02x}{g:02x}{b:02x}"


def _merge_similar_colors(labels: np.ndarray, palette: list[int], max_colors: int, threshold: float) -> tuple[np.ndarray, list[int]]:
    """Merge palette entries that are within *threshold* Euclidean RGB distance.

    Returns (merged_labels, merged_palette) where palette has length max_colors
    but unused slots are filled with (0,0,0).
    """
    colors = np.array([[palette[i * 3], palette[i * 3 + 1], palette[i * 3 + 2]] for i in range(max_colors)], dtype=np.float32)
    merge_map = list(range(max_colors))

    # Single-linkage agglomerative merge
    for i in range(max_colors):
        if merge_map[i] != i:
            continue
        for j in range(i + 1, max_colors):
            if merge_map[j] != j:
                continue
            dist = float(np.linalg.norm(colors[i] - colors[j]))
            if dist < threshold:
                merge_map[j] = i

    # Build new label mapping
    new_labels = labels.copy()
    for old_label in range(max_colors):
        new_labels[labels == old_label] = merge_map[old_label]

    # Compact the palette
    used = sorted(set(merge_map))
    compact_map = {old: new for new, old in enumerate(used)}
    for old_label in range(max_colors):
        new_labels[new_labels == old_label] = compact_map[merge_map[old_label]]

    new_palette = []
    for label in used:
        new_palette.extend([palette[label * 3], palette[label * 3 + 1], palette[label * 3 + 2]])
    # Pad to max_colors * 3
    while len(new_palette) < max_colors * 3:
        new_palette.extend([0, 0, 0])

    return new_labels, new_palette


def _round_param(value):
    if isinstance(value, list):
        return [round(float(v), 4) for v in value]
    return round(float(value), 4)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum() / union)


def fit_primitive_layer(comp_mask: np.ndarray, *, color_hex: str) -> dict | None:
    """Fit circle / ellipse / box to a boolean component mask; pick best by IoU.

    Returns a DSL layer dict (without ``id``) or None for degenerate masks.
    Falls back to a moments-based ellipse when no primitive reaches MIN_FIT_IOU,
    so a detected region is never silently dropped.
    """
    if cv2 is None or not comp_mask.any():
        return None
    h, w = comp_mask.shape
    mask_u8 = comp_mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)

    candidates: list[tuple[float, str, dict]] = []

    # --- circle ---
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    cand = np.zeros_like(mask_u8)
    cv2.circle(cand, (int(round(cx)), int(round(cy))), int(round(radius)), 1, -1)
    candidates.append((
        _iou(comp_mask, cand.astype(bool)),
        "circle",
        {"center": [cx / w, cy / h], "radius": radius / w},
    ))

    # --- ellipse (axis-aligned only: DSL ellipse has no rotation) ---
    if len(contour) >= 5:
        (ex, ey), (d1, d2), angle = cv2.fitEllipse(contour)
        ang = angle % 180.0
        ab = None
        if ang < 15.0 or ang > 165.0:
            ab = [d1 / (2 * w), d2 / (2 * h)]
        elif abs(ang - 90.0) < 15.0:
            ab = [d2 / (2 * w), d1 / (2 * h)]
        if ab is not None and d1 > 0 and d2 > 0:
            cand = np.zeros_like(mask_u8)
            cv2.ellipse(cand, ((ex, ey), (d1, d2), angle), 1, -1)
            eiou = _iou(comp_mask, cand.astype(bool))
            # If the ellipse is nearly circular (aspect ratio < 1.05), treat as circle
            aspect = max(d1, d2) / max(1e-9, min(d1, d2))
            if aspect < 1.05:
                # Use minEnclosingCircle for circular shapes
                (mcx, mcy), mradius = cv2.minEnclosingCircle(contour)
                mcand = np.zeros_like(mask_u8)
                cv2.circle(mcand, (int(round(mcx)), int(round(mcy))), int(round(mradius)), 1, -1)
                miou = _iou(comp_mask, mcand.astype(bool))
                candidates[0] = (miou, "circle", {"center": [mcx / w, mcy / h], "radius": mradius / w})
            else:
                candidates.append((
                    eiou,
                    "ellipse",
                    {"center": [ex / w, ey / h], "ab": ab},
                ))

    # --- box ---
    bx, by, bw, bh = cv2.boundingRect(contour)
    cand = np.zeros_like(mask_u8)
    cand[by:by + bh, bx:bx + bw] = 1
    candidates.append((
        _iou(comp_mask, cand.astype(bool)),
        "box",
        {"center": [(bx + bw / 2.0) / w, (by + bh / 2.0) / h], "size": [bw / w, bh / h]},
    ))

    best_iou, best_type, best_params = max(candidates, key=lambda t: t[0])

    if best_iou < MIN_FIT_IOU:
        m = cv2.moments(mask_u8, binaryImage=True)
        if m["m00"] <= 0:
            return None
        mx, my = m["m10"] / m["m00"], m["m01"] / m["m00"]
        a = 2.0 * math.sqrt(max(m["mu20"] / m["m00"], 1e-9))
        b = 2.0 * math.sqrt(max(m["mu02"] / m["m00"], 1e-9))
        best_type = "ellipse"
        best_params = {"center": [mx / w, my / h], "ab": [a / w, b / h]}

    return {
        "type": best_type,
        "fill": {"type": "solid", "color": color_hex},
        "params": {k: _round_param(v) for k, v in best_params.items()},
        "opacity": 1.0,
    }


def decompose_to_dsl(
    image_path: "str | Path",
    canvas_width: int = 512,
    canvas_height: int = 512,
    *,
    max_colors: int = 6,
    min_area_frac: float = 0.004,
    max_layers: int = 10,
) -> dict | None:
    """Decompose an image into a DSL scene with measured geometry.

    Steps: median-cut color quantization -> background detection from the
    border -> per-color connected components -> per-component primitive fit
    -> layers ordered big-to-small (bottom-to-top).
    """
    if not DECOMPOSE_AVAILABLE:
        return None

    img = Image.open(Path(image_path)).convert("RGBA")
    img = img.resize((ANALYSIS_SIZE, ANALYSIS_SIZE), Image.LANCZOS)
    rgba = np.asarray(img, dtype=np.uint8)
    alpha_mask = rgba[..., 3] > ALPHA_THRESHOLD

    quantized = img.convert("RGB").quantize(colors=max_colors, method=Image.MEDIANCUT)
    labels = np.asarray(quantized, dtype=np.int32)
    palette = quantized.getpalette()

    # Merge similar colors to avoid over-splitting
    labels, palette = _merge_similar_colors(labels, palette, max_colors, MERGE_COLOR_DIST)

    h, w = labels.shape
    if alpha_mask.mean() < 0.95:
        # transparent background — every opaque region is a layer
        bg_label = -1
        bg_hex = "#000000"
    else:
        border = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
        bg_label = int(np.bincount(border).argmax())
        bg_hex = _palette_hex(palette, bg_label)

    fitted: list[tuple[float, dict]] = []
    n_colors = len(set(labels.ravel()))
    for label in range(n_colors):
        if label == bg_label:
            continue
        mask = (labels == label) & alpha_mask
        if not mask.any():
            continue
        n, comp_map, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        for comp in range(1, n):
            area_frac = stats[comp, cv2.CC_STAT_AREA] / float(h * w)
            if area_frac < min_area_frac:
                continue
            layer = fit_primitive_layer(comp_map == comp, color_hex=_palette_hex(palette, label))
            if layer is not None:
                fitted.append((area_frac, layer))

    if not fitted:
        return None

    fitted.sort(key=lambda t: -t[0])
    layers = []
    for i, (_, layer) in enumerate(fitted[:max_layers]):
        layer["id"] = f"dec_{i:02d}_{layer['type']}"
        layers.append(layer)

    logger.info("decompose: %d regions fitted, %d layers kept", len(fitted), len(layers))
    return {
        "schema_version": 1,
        "canvas": {"width": canvas_width, "height": canvas_height, "background": bg_hex},
        "layers": layers,
    }
