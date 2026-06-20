"""Per-region image-difference metrics for V4.2 Region/Mask Constraints.

Computes MSE, SSIM, mean_delta, and edge_delta restricted to each region's
normalized rectangle crop — metrics are computed over ONLY the region's pixels.

Design constraints (mirrors sibling pure modules):
- stdlib + numpy + PIL only (reuses helpers from app.metrics.compute).
- No FastAPI, no time()/random.
- Fully deterministic (safe for caching / testing).
"""

from __future__ import annotations

from dataclasses import dataclass
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
from app.pipeline.region_types import RegionConstraint


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


@dataclass
class RegionVetoResult:
    """Outcome of the protect-region hard-veto check for one candidate."""
    vetoed: bool
    constraint_score: float          # mean SSIM of evaluated protect regions vs baseline
    regions: list[dict]              # [{id, label, ssim, threshold, violated}]
    reason: str | None               # human-readable veto reason (fed to the LLM)
    evaluated: bool                  # False when baseline/candidate/geometry unusable


def protect_region_threshold(strength: float, *, floor: float = 0.85, ceil: float = 0.95) -> float:
    """Map a region's strength (0..1) to its minimum acceptable SSIM vs baseline.

    Higher strength = stricter (higher SSIM required). Default strength 0.5 -> 0.90.
    """
    s = min(1.0, max(0.0, float(strength)))
    # round to tame binary float artifacts (e.g. 0.85 + 0.05 -> 0.8999999999999999)
    return round(floor + (ceil - floor) * s, 10)


def evaluate_protect_veto(
    baseline_render: "str | Path",
    candidate_render: "str | Path",
    regions: list[RegionConstraint],
    *,
    floor: float = 0.85,
    ceil: float = 0.95,
) -> RegionVetoResult:
    """Hard-veto a candidate whose protect regions degraded vs the baseline render.

    Veto if ANY protect region's SSIM(candidate, baseline) < its strength threshold.
    Best-effort: missing files / unusable geometry -> evaluated=False, vetoed=False.
    """
    protect = [r for r in regions if getattr(r, "mode", None) == "protect"]
    if not protect:
        return RegionVetoResult(False, 1.0, [], None, evaluated=False)

    try:
        metrics = compute_region_metrics(baseline_render, candidate_render, protect)
    except Exception:  # missing/unreadable image, etc. — do not block on failure
        return RegionVetoResult(False, 1.0, [], None, evaluated=False)

    rows: list[dict] = []
    ssims: list[float] = []
    violated_labels: list[str] = []
    region_metrics_map = metrics.get("regions", {})
    for r in protect:
        rm = region_metrics_map.get(r.id, {})
        ssim = rm.get("ssim")
        if ssim is None:  # unsupported geometry / empty region / no valid ssim
            rows.append({"id": r.id, "label": r.label, "ssim": None, "threshold": None, "violated": False})
            continue
        thr = protect_region_threshold(r.strength, floor=floor, ceil=ceil)
        violated = ssim < thr
        ssims.append(float(ssim))
        rows.append({"id": r.id, "label": r.label, "ssim": float(ssim), "threshold": thr, "violated": violated})
        if violated:
            violated_labels.append(r.label or r.id)

    evaluated = len(ssims) > 0
    constraint_score = float(sum(ssims) / len(ssims)) if ssims else 1.0
    vetoed = len(violated_labels) > 0
    reason = ("protected regions degraded: " + ", ".join(violated_labels)) if vetoed else None
    return RegionVetoResult(vetoed, constraint_score, rows, reason, evaluated)
