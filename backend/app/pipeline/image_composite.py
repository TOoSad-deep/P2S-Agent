"""Composite target builder for V4.5 Local Fusion.

Blends source renders into a base render within feathered rectangular regions
to create a visual composite target for fusion refinement.

Constraints:
- Reuses ``_load_rgba_array`` / ``_match_size_array`` from app.metrics.compute.
- Uses numpy for mask arithmetic; PIL for save/load.
- No FastAPI, no time, no random — deterministic.
- Returns composite_target.png path; also writes region_masks/<region_id>.png.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from app.metrics.compute import _load_rgba_array, _match_size_array
from app.pipeline.fusion_plans import FusionRegion

PathLike = Union[str, Path]

# Regex for safe region-id filenames: allow alphanumeric, dot, underscore, dash, colon
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9._:\-]+$")


def _build_rect_mask(
    H: int,
    W: int,
    x: float,
    y: float,
    w: float,
    h: float,
    feather_px: int,
) -> np.ndarray:
    """Build a (H, W) float32 alpha mask for a normalized rect with feathering.

    Parameters
    ----------
    H, W:
        Image dimensions in pixels.
    x, y, w, h:
        Normalized rect bounds (0..1). ``(x, y)`` is top-left corner.
    feather_px:
        Feather width in pixels (>= 1). Linear ramp applied on the *inner*
        edge of the rect, transitioning from 0 at the outer border of the
        feather zone to 1 at the inner boundary.

    Returns
    -------
    mask : np.ndarray, shape (H, W), dtype float32, values in [0, 1].
    """
    # Convert normalized bounds to pixel coordinates (clamp to image)
    x0 = int(round(x * W))
    y0 = int(round(y * H))
    x1 = int(round((x + w) * W))
    y1 = int(round((y + h) * H))

    # Clamp
    x0 = max(0, min(W, x0))
    x1 = max(0, min(W, x1))
    y0 = max(0, min(H, y0))
    y1 = max(0, min(H, y1))

    mask = np.zeros((H, W), dtype=np.float32)

    if x1 <= x0 or y1 <= y0:
        return mask

    # Build coordinate grids
    col_idx = np.arange(W, dtype=np.float32)
    row_idx = np.arange(H, dtype=np.float32)

    if feather_px <= 0:
        # Hard binary mask
        mask[y0:y1, x0:x1] = 1.0
        return mask

    # Signed distance-to-interior along each axis, in pixels.
    # Positive = inside the rect; clipped to [0, feather_px].
    # col ramp: distance from left edge and right edge
    dist_left  = col_idx - x0           # positive inside, negative outside left
    dist_right = x1 - col_idx - 1.0     # positive inside, negative outside right
    col_dist = np.minimum(dist_left, dist_right)
    col_ramp = np.clip(col_dist, 0.0, feather_px) / feather_px  # [0,1]

    dist_top    = row_idx - y0
    dist_bottom = y1 - row_idx - 1.0
    row_dist = np.minimum(dist_top, dist_bottom)
    row_ramp = np.clip(row_dist, 0.0, feather_px) / feather_px  # [0,1]

    # 2D mask = min of the two 1D ramps (rectangular feather on all 4 sides)
    mask = np.minimum(col_ramp[np.newaxis, :], row_ramp[:, np.newaxis])
    return mask.astype(np.float32)


def build_composite_target(
    *,
    base_render_path: PathLike,
    source_render_paths: dict[str, Path],
    regions: list[FusionRegion],
    output_dir: PathLike,
) -> Path:
    """Composite source renders into the base over feathered rects.

    Returns
    -------
    Path
        ``output_dir/composite_target.png`` — always written (even with no regions).

    Side effects
    ------------
    - ``output_dir/composite_target.png`` — RGBA PNG.
    - ``output_dir/region_masks/<region.id>.png`` — grayscale uint8 mask per
      processed region (skipped regions produce no file).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / "region_masks"

    # Load base as float32 RGBA (H, W, 4) in [0, 1]
    base = _load_rgba_array(base_render_path)
    H, W = base.shape[:2]

    # Work in float; preserve base alpha channel throughout
    composite = base.copy()  # shape (H, W, 4), float32

    for region in regions:
        # Only "rect" geometry type is supported in V4.5
        if region.geometry_type != "rect":
            continue

        # Resolve source — skip if not in dict
        source_path = source_render_paths.get(region.source_run_id)
        if source_path is None:
            continue

        # Load and resize source to match base
        source_arr = _load_rgba_array(source_path)
        source_arr = _match_size_array(base, source_arr)  # (H, W, 4), float32

        # Extract rect geometry
        g = region.geometry
        try:
            rx = float(g["x"])
            ry = float(g["y"])
            rw = float(g["w"])
            rh = float(g["h"])
        except (KeyError, TypeError, ValueError):
            continue

        # Compute feather in pixels (at least 1 if feather > 0, else 0)
        feather_norm = float(region.feather)
        if feather_norm > 0.0:
            feather_px = max(1, int(feather_norm * min(H, W)))
        else:
            feather_px = 0

        # Build per-pixel alpha mask (H, W)
        mask = _build_rect_mask(H, W, rx, ry, rw, rh, feather_px)

        # Scale by strength
        strength = float(np.clip(region.strength, 0.0, 1.0))
        mask = mask * strength  # still in [0, 1]

        # Blend RGB channels: composite*(1-mask) + source*mask
        m = mask[:, :, np.newaxis]  # (H, W, 1) for broadcasting
        composite[:, :, :3] = (
            composite[:, :, :3] * (1.0 - m) + source_arr[:, :, :3] * m
        )
        # Alpha channel: preserve base alpha (do not blend alpha)

        # Save mask as grayscale uint8 PNG, guarding region.id for safety
        if _SAFE_ID_RE.match(region.id):
            masks_dir.mkdir(parents=True, exist_ok=True)
            mask_uint8 = (mask * 255.0 + 0.5).astype(np.uint8)
            mask_img = Image.fromarray(mask_uint8, mode="L")
            mask_img.save(masks_dir / f"{region.id}.png")

    # Save composite as RGBA PNG
    composite_uint8 = (np.clip(composite, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    out_path = output_dir / "composite_target.png"
    Image.fromarray(composite_uint8, mode="RGBA").save(out_path)

    return out_path
