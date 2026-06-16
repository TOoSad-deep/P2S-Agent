"""Per-region image-difference metrics for V4.2 Region/Mask Constraints.

Computes MSE, SSIM, mean_delta, and edge_delta restricted to each region's
normalized rectangle crop — metrics are computed over ONLY the region's pixels.

Design constraints (mirrors sibling pure modules):
- stdlib + numpy + PIL only (reuses helpers from app.metrics.compute).
- No FastAPI, no time()/random.
- Fully deterministic (safe for caching / testing).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.metrics.compute import (
    _load_rgba_array,
    _match_size_array,
    _composite_over,
    _gray,
    _ssim_arrays,
    _edge_map,
)
from app.pipeline.human_constraints import RegionConstraint


def compute_region_metrics(
    reference_path: "str | Path",
    render_path: "str | Path",
    regions: list[RegionConstraint],
) -> dict:
    """Compute image-difference metrics restricted to each region's rect crop.

    Parameters
    ----------
    reference_path:
        Path to the reference PNG (any size, RGBA).
    render_path:
        Path to the rendered PNG (resized to match reference if needed).
    regions:
        List of ``RegionConstraint`` objects.  Only ``geometry_type=="rect"``
        is processed; others record ``{"unsupported_geometry": <type>}``.

    Returns
    -------
    dict with keys:
      - ``"regions"``: mapping from region id → per-region metrics dict.
      - ``"constraint_score"``: float in [0,1]; mean SSIM of protect-mode
        regions that have a valid ssim.  Defaults to 1.0 when no protect
        regions exist or none produce a valid ssim.
    """
    if not regions:
        return {"regions": {}, "constraint_score": 1.0}

    # Load once and composite to RGB over black — mirrors compute_mse pattern.
    ref_rgba = _load_rgba_array(reference_path)
    rnd_rgba = _match_size_array(ref_rgba, _load_rgba_array(render_path))

    ref_rgb: np.ndarray = _composite_over(ref_rgba, (0, 0, 0))  # (H, W, 3)
    rnd_rgb: np.ndarray = _composite_over(rnd_rgba, (0, 0, 0))  # (H, W, 3)

    H, W = ref_rgb.shape[:2]

    region_results: dict[str, dict] = {}
    protect_ssims: list[float] = []

    for region in regions:
        rid = region.id

        if region.geometry_type != "rect":
            region_results[rid] = {"unsupported_geometry": region.geometry_type}
            continue

        geo = region.geometry
        x = float(geo.get("x", 0.0))
        y = float(geo.get("y", 0.0))
        w = float(geo.get("w", 0.0))
        h = float(geo.get("h", 0.0))

        # Convert normalized coords to pixel bounds.
        x0 = round(x * W)
        y0 = round(y * H)
        x1 = round((x + w) * W)
        y1 = round((y + h) * H)

        # Clamp to image bounds.
        x0 = max(0, min(x0, W))
        x1 = max(0, min(x1, W))
        y0 = max(0, min(y0, H))
        y1 = max(0, min(y1, H))

        # Guard zero-area crop.
        if x0 >= x1 or y0 >= y1:
            region_results[rid] = {"error": "empty_region"}
            continue

        # Crop both ref and render to the region pixels only.
        ref_crop = ref_rgb[y0:y1, x0:x1]   # (rH, rW, 3)
        rnd_crop = rnd_rgb[y0:y1, x0:x1]   # (rH, rW, 3)

        ref_gray = _gray(ref_crop)
        rnd_gray = _gray(rnd_crop)

        mse = float(np.clip(((ref_crop - rnd_crop) ** 2).mean(), 0.0, 1.0))
        ssim = _ssim_arrays(ref_gray, rnd_gray)
        mean_delta_arr = (rnd_crop - ref_crop).mean(axis=(0, 1))  # (3,)
        mean_delta = [float(mean_delta_arr[0]), float(mean_delta_arr[1]), float(mean_delta_arr[2])]
        edge_ref = _edge_map(ref_gray)
        edge_rnd = _edge_map(rnd_gray)
        edge_delta = float(abs(float(edge_ref.mean()) - float(edge_rnd.mean())))

        region_results[rid] = {
            "mse": mse,
            "ssim": ssim,
            "mean_delta": mean_delta,
            "edge_delta": edge_delta,
        }

        if region.mode == "protect":
            protect_ssims.append(ssim)

    constraint_score: float
    if protect_ssims:
        constraint_score = float(np.mean(protect_ssims))
    else:
        constraint_score = 1.0

    return {
        "regions": region_results,
        "constraint_score": constraint_score,
    }
