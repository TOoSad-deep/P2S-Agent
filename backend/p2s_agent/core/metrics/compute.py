"""Objective metrics for PNG-to-Shader evaluation (NumPy-vectorized, v2).

v2 changes vs v1:
- All metrics vectorized with NumPy (~100x faster than the pure-Python loops).
- RGB comparisons alpha-composite both images over a background color first,
  so RGB noise under transparent pixels no longer pollutes MSE/SSIM/histogram.
- New position-aware metrics: rmse, mask_iou, edge_iou, grid_color_sim.
- Legacy keys (mse, simple_ssim, alpha_coverage_diff, color_histogram_score,
  edge_density_diff) are kept so old artifacts and the v1 score formula
  still work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

PathLike = Union[str, Path]

ALPHA_THRESHOLD = 16            # alpha > 16/255 counts as foreground (v1 used > 0)
EDGE_THRESHOLD = 20.0 / 255.0   # same gradient threshold as v1, in [0,1] space (forward first-difference)
GRID_SIZE = 8                   # grid_color_sim cell grid
DELTA_E_SCALE = 60.0            # mean Lab deltaE mapped to similarity by 1 - dE/60


# ---------------------------------------------------------------------------
# Loading / preparation
# ---------------------------------------------------------------------------


def _load_rgba_array(path: PathLike) -> np.ndarray:
    """Load an image as float32 RGBA array in [0,1], shape (H, W, 4)."""
    img = Image.open(Path(path)).convert("RGBA")
    return np.asarray(img, dtype=np.float32) / 255.0


def _match_size_array(ref: np.ndarray, render: np.ndarray) -> np.ndarray:
    if render.shape[:2] != ref.shape[:2]:
        h, w = ref.shape[:2]
        img = Image.fromarray((render * 255.0 + 0.5).astype(np.uint8), "RGBA")
        render = np.asarray(img.resize((w, h), Image.LANCZOS), dtype=np.float32) / 255.0
    return render


def _composite_over(arr: np.ndarray, background_rgb: tuple[float, float, float]) -> np.ndarray:
    """Alpha-composite an RGBA array over a solid background. Returns (H, W, 3)."""
    alpha = arr[..., 3:4]
    bg = np.asarray(background_rgb, dtype=np.float32).reshape(1, 1, 3)
    return arr[..., :3] * alpha + bg * (1.0 - alpha)


def _foreground_mask(arr: np.ndarray) -> np.ndarray:
    return arr[..., 3] > (ALPHA_THRESHOLD / 255.0)


def _gray(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)


def hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    """Parse '#rrggbb' into a [0,1] RGB tuple. Invalid input falls back to black."""
    s = (hex_color or "").lstrip("#")
    if len(s) != 6:
        return (0.0, 0.0, 0.0)
    try:
        return (int(s[0:2], 16) / 255.0, int(s[2:4], 16) / 255.0, int(s[4:6], 16) / 255.0)
    except ValueError:
        return (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Core vectorized building blocks
# ---------------------------------------------------------------------------


def _box_filter(a: np.ndarray, radius: int) -> np.ndarray:
    """Mean filter with a (2r+1)^2 window via integral image, edge-padded."""
    pad = np.pad(a, radius, mode="edge")
    c = np.cumsum(np.cumsum(pad, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    k = 2 * radius + 1
    s = c[k:, k:] - c[k:, :-k] - c[:-k, k:] + c[:-k, :-k]
    return s / (k * k)


def _ssim_arrays(ref_gray: np.ndarray, rnd_gray: np.ndarray) -> float:
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    r = 2  # 5x5 window, matching v1
    mu1 = _box_filter(ref_gray, r)
    mu2 = _box_filter(rnd_gray, r)
    var1 = _box_filter(ref_gray * ref_gray, r) - mu1 * mu1
    var2 = _box_filter(rnd_gray * rnd_gray, r) - mu2 * mu2
    cov = _box_filter(ref_gray * rnd_gray, r) - mu1 * mu2
    num = (2 * mu1 * mu2 + C1) * (2 * cov + C2)
    den = (mu1 ** 2 + mu2 ** 2 + C1) * (var1 + var2 + C2)
    ssim_map = np.where(den != 0.0, num / np.where(den == 0.0, 1.0, den), 1.0)
    return float(np.clip(ssim_map.mean(), 0.0, 1.0))


def _mask_iou_arrays(m1: np.ndarray, m2: np.ndarray) -> float:
    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(m1, m2).sum() / union)


def _edge_map(gray: np.ndarray) -> np.ndarray:
    gx = np.abs(np.diff(gray, axis=1, append=gray[:, -1:]))
    gy = np.abs(np.diff(gray, axis=0, append=gray[-1:, :]))
    return np.maximum(gx, gy) > EDGE_THRESHOLD


def _dilate3(mask: np.ndarray) -> np.ndarray:
    p = np.pad(mask, 1)
    h, w = mask.shape
    out = np.zeros_like(mask)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            out |= p[dy:dy + h, dx:dx + w]
    return out


def _edge_iou_arrays(ref_gray: np.ndarray, rnd_gray: np.ndarray) -> float:
    """Symmetric F1 of edge maps with 1px dilation tolerance."""
    e_ref = _edge_map(ref_gray)
    e_rnd = _edge_map(rnd_gray)
    if not e_ref.any() and not e_rnd.any():
        return 1.0
    if not e_ref.any() or not e_rnd.any():
        return 0.0
    prec = np.logical_and(e_rnd, _dilate3(e_ref)).sum() / e_rnd.sum()
    rec = np.logical_and(e_ref, _dilate3(e_rnd)).sum() / e_ref.sum()
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert (..., 3) sRGB in [0,1] to CIE Lab (D65)."""
    linear = np.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )
    xyz = linear @ m.T
    white = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    t = xyz / white
    delta = 6.0 / 29.0
    f = np.where(t > delta ** 3, np.cbrt(t), t / (3 * delta ** 2) + 4.0 / 29.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def _block_mean(arr: np.ndarray, grid: int) -> np.ndarray:
    """Mean color per grid cell, shape (grid_h, grid_w, C). Works for any H, W.

    The grid is clamped per-axis so that images smaller than *grid* pixels in
    either dimension never produce zero-width cells (which would divide by zero).
    """
    h, w = arr.shape[:2]
    grid_h = min(grid, h)
    grid_w = min(grid, w)
    ys = (np.arange(grid_h) * h) // grid_h
    xs = (np.arange(grid_w) * w) // grid_w
    tmp = np.add.reduceat(arr, ys, axis=0)
    tmp = np.add.reduceat(tmp, xs, axis=1)
    counts_y = np.diff(np.append(ys, h)).astype(np.float32).reshape(-1, 1, 1)
    counts_x = np.diff(np.append(xs, w)).astype(np.float32).reshape(1, -1, 1)
    return tmp / (counts_y * counts_x)


def _grid_delta_e(ref_rgb: np.ndarray, rnd_rgb: np.ndarray, grid: int) -> np.ndarray:
    lab1 = _srgb_to_lab(_block_mean(ref_rgb, grid))
    lab2 = _srgb_to_lab(_block_mean(rnd_rgb, grid))
    return np.sqrt(((lab1 - lab2) ** 2).sum(axis=-1))


def _grid_color_similarity(ref_rgb: np.ndarray, rnd_rgb: np.ndarray, grid: int = GRID_SIZE) -> float:
    delta_e = _grid_delta_e(ref_rgb, rnd_rgb, grid)
    return float(np.clip(1.0 - delta_e.mean() / DELTA_E_SCALE, 0.0, 1.0))


def _histogram_score_arrays(ref_rgb: np.ndarray, rnd_rgb: np.ndarray) -> float:
    n_bins = 32
    scores = []
    for ch in range(3):
        hist_ref, _ = np.histogram(ref_rgb[..., ch], bins=n_bins, range=(0.0, 1.0))
        hist_rnd, _ = np.histogram(rnd_rgb[..., ch], bins=n_bins, range=(0.0, 1.0))
        total_ref = hist_ref.sum()
        score = np.minimum(hist_ref, hist_rnd).sum() / total_ref if total_ref > 0 else 1.0
        scores.append(score)
    return float(np.clip(np.mean(scores), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Path-based public API (kept for backward compatibility)
# ---------------------------------------------------------------------------


def compute_mse(ref_path: PathLike, render_path: PathLike) -> float:
    ref = _load_rgba_array(ref_path)
    rnd = _match_size_array(ref, _load_rgba_array(render_path))
    a, b = _composite_over(ref, (0, 0, 0)), _composite_over(rnd, (0, 0, 0))
    return float(np.clip(((a - b) ** 2).mean(), 0.0, 1.0))


def compute_simple_ssim(ref_path: PathLike, render_path: PathLike) -> float:
    ref = _load_rgba_array(ref_path)
    rnd = _match_size_array(ref, _load_rgba_array(render_path))
    return _ssim_arrays(_gray(_composite_over(ref, (0, 0, 0))), _gray(_composite_over(rnd, (0, 0, 0))))


def compute_alpha_coverage_diff(ref_path: PathLike, render_path: PathLike) -> float:
    ref = _load_rgba_array(ref_path)
    rnd = _match_size_array(ref, _load_rgba_array(render_path))
    return float(abs(_foreground_mask(ref).mean() - _foreground_mask(rnd).mean()))


def compute_color_histogram_score(ref_path: PathLike, render_path: PathLike) -> float:
    ref = _load_rgba_array(ref_path)
    rnd = _match_size_array(ref, _load_rgba_array(render_path))
    return _histogram_score_arrays(_composite_over(ref, (0, 0, 0)), _composite_over(rnd, (0, 0, 0)))


def compute_edge_density_diff(ref_path: PathLike, render_path: PathLike) -> float:
    ref = _load_rgba_array(ref_path)
    rnd = _match_size_array(ref, _load_rgba_array(render_path))
    e1 = _edge_map(_gray(_composite_over(ref, (0, 0, 0))))
    e2 = _edge_map(_gray(_composite_over(rnd, (0, 0, 0))))
    return float(abs(e1.mean() - e2.mean()))


def check_nonblank_render(render_path: PathLike) -> bool:
    arr = _load_rgba_array(render_path)
    visible = (arr[..., 3] > 0) & (arr[..., :3].max(axis=-1) > 0)
    return bool(visible.any())


# ---------------------------------------------------------------------------
# Region report for LLM refinement feedback
# ---------------------------------------------------------------------------

_ROW_NAMES = ["top", "upper-middle", "lower-middle", "bottom"]
_COL_NAMES = ["left", "center-left", "center-right", "right"]


def grid_color_report(
    ref_path: PathLike,
    render_path: PathLike,
    *,
    background_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0),
    top_n: int = 3,
    min_delta_e: float = 15.0,
) -> list[str]:
    """Return human-readable notes about the worst-matching 4x4 regions.

    Used as extra feedback for the LLM refinement loop so it knows *where*
    the render deviates, not just by how much.
    """
    grid = 4
    ref = _load_rgba_array(ref_path)
    rnd = _match_size_array(ref, _load_rgba_array(render_path))
    ref_rgb = _composite_over(ref, background_rgb)
    rnd_rgb = _composite_over(rnd, background_rgb)
    delta_e = _grid_delta_e(ref_rgb, rnd_rgb, grid)
    l_ref = _srgb_to_lab(_block_mean(ref_rgb, grid))[..., 0]
    l_rnd = _srgb_to_lab(_block_mean(rnd_rgb, grid))[..., 0]

    flat = [(float(delta_e[r, c]), r, c) for r in range(grid) for c in range(grid)]
    flat.sort(reverse=True)
    notes: list[str] = []
    for de, r, c in flat[:top_n]:
        if de < min_delta_e:
            break
        l_diff = float(l_rnd[r, c] - l_ref[r, c])
        if l_diff > 5.0:
            direction = "too bright"
        elif l_diff < -5.0:
            direction = "too dark"
        else:
            direction = "wrong hue"
        notes.append(
            f"region {_ROW_NAMES[r]}-{_COL_NAMES[c]}: render {direction} vs reference (deltaE={de:.0f})"
        )
    return notes


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------


def compute_objective_metrics(
    ref_path: PathLike,
    render_path: PathLike,
    *,
    shader_chars: int = 0,
    max_shader_chars: int = 12000,
    background_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict:
    """Compute all objective metrics (v1 keys + v2 keys) in one pass.

    v2 keys: rmse (lower better), mask_iou / edge_iou / grid_color_sim
    (higher better). RGB comparisons composite over *background_rgb*
    (pass the DSL canvas background for DSL renders).
    """
    ref = _load_rgba_array(ref_path)
    rnd = _match_size_array(ref, _load_rgba_array(render_path))
    ref_rgb = _composite_over(ref, background_rgb)
    rnd_rgb = _composite_over(rnd, background_rgb)
    ref_gray = _gray(ref_rgb)
    rnd_gray = _gray(rnd_rgb)
    e_ref = _edge_map(ref_gray)
    e_rnd = _edge_map(rnd_gray)
    mse = float(np.clip(((ref_rgb - rnd_rgb) ** 2).mean(), 0.0, 1.0))
    visible = (rnd[..., 3] > 0) & (rnd[..., :3].max(axis=-1) > 0)

    return {
        # --- v1 keys (semantics: now computed on composited RGB) ---
        "mse": mse,
        "simple_ssim": _ssim_arrays(ref_gray, rnd_gray),
        "alpha_coverage_diff": float(abs(_foreground_mask(ref).mean() - _foreground_mask(rnd).mean())),
        "color_histogram_score": _histogram_score_arrays(ref_rgb, rnd_rgb),
        "edge_density_diff": float(abs(e_ref.mean() - e_rnd.mean())),
        "nonblank_render": bool(visible.any()),
        "within_shader_budget": shader_chars <= max_shader_chars,
        # --- v2 keys ---
        "rmse": float(np.sqrt(mse)),
        "mask_iou": _mask_iou_arrays(_foreground_mask(ref), _foreground_mask(rnd)),
        "edge_iou": _edge_iou_arrays(ref_gray, rnd_gray),
        "grid_color_sim": _grid_color_similarity(ref_rgb, rnd_rgb),
    }
