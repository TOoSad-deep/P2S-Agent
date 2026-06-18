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


class CompositeTargetPath(type(Path())):  # type: ignore[misc]
    """A ``Path`` to ``composite_target.png`` that also carries per-region status.

    Behaves exactly like a :class:`pathlib.Path` (``isinstance(x, Path)`` is
    True, ``.name`` / ``.parent`` / ``.exists()`` all work) so existing callers
    that treat the return value as a plain path are unaffected. The extra
    ``region_status`` attribute lets callers surface which regions were applied
    vs skipped (and why) without changing the composite output.
    """

    # Default so the attribute always exists even if not populated.
    region_status: list[dict] = []


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

    # Cap the feather to the rect interior, per axis, BEFORE building the ramp.
    # The per-pixel distance-to-interior is ``min(idx - lo, hi - idx - 1)``; the
    # most-interior pixel along an axis of extent ``n`` reaches a distance of
    # ``(n - 1) // 2``. For that pixel to hit full strength the ramp may use at
    # most that many pixels, so cap ``feather`` to ``(n - 1) // 2``. Without
    # this, a rect narrower than ``2 * feather_px`` never reaches 1.0 and a
    # small enough one is zeroed entirely. When an axis is too thin to fit any
    # ramp (extent <= 2px → cap of 0) we fall back to a hard edge on that axis
    # rather than re-zeroing the region.
    rw_px = x1 - x0
    rh_px = y1 - y0
    col_feather = min(feather_px, (rw_px - 1) // 2)
    row_feather = min(feather_px, (rh_px - 1) // 2)

    # Signed distance-to-interior along each axis, in pixels.
    # Positive = inside the rect; clipped to [0, feather_eff].
    # col ramp: distance from left edge and right edge
    dist_left  = col_idx - x0           # positive inside, negative outside left
    dist_right = x1 - col_idx - 1.0     # positive inside, negative outside right
    col_dist = np.minimum(dist_left, dist_right)
    if col_feather <= 0:
        # Too thin to feather → hard edge inside [x0, x1).
        col_ramp = (col_dist >= 0.0).astype(np.float32)
    else:
        col_ramp = np.clip(col_dist, 0.0, col_feather) / col_feather  # [0,1]

    dist_top    = row_idx - y0
    dist_bottom = y1 - row_idx - 1.0
    row_dist = np.minimum(dist_top, dist_bottom)
    if row_feather <= 0:
        # Too thin to feather → hard edge inside [y0, y1).
        row_ramp = (row_dist >= 0.0).astype(np.float32)
    else:
        row_ramp = np.clip(row_dist, 0.0, row_feather) / row_feather  # [0,1]

    # 2D mask = min of the two 1D ramps (rectangular feather on all 4 sides)
    mask = np.minimum(col_ramp[np.newaxis, :], row_ramp[:, np.newaxis])
    return mask.astype(np.float32)


def build_composite_target(
    *,
    base_render_path: PathLike,
    source_render_paths: dict[str, Path],
    regions: list[FusionRegion],
    output_dir: PathLike,
) -> "CompositeTargetPath":
    """Composite source renders into the base over feathered rects.

    Returns
    -------
    CompositeTargetPath
        ``output_dir/composite_target.png`` — always written (even with no
        regions). This is a :class:`pathlib.Path` subclass; its
        ``region_status`` attribute holds a per-region list of
        ``{"index", "region_id", "applied", "reason"}`` dicts so callers can
        surface which regions were applied vs skipped (and why).

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

    # Per-region applied/skipped status, surfaced via the returned path.
    region_status: list[dict] = []

    def _record(region, index: int, applied: bool, reason: str = "") -> None:
        region_status.append({
            "index": index,
            "region_id": getattr(region, "id", None),
            "applied": applied,
            "reason": reason,
        })

    for index, region in enumerate(regions):
        # Only "rect" geometry type is supported in V4.5
        if region.geometry_type != "rect":
            _record(region, index, False, f"unsupported geometry_type '{region.geometry_type}'")
            continue

        # Resolve source — skip if not in dict
        source_path = source_render_paths.get(region.source_run_id)
        if source_path is None:
            _record(region, index, False, f"source run '{region.source_run_id}' not provided")
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
            _record(region, index, False, "invalid rect geometry")
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

        # A degenerate (zero-area) rect or zero strength contributes nothing —
        # report it as skipped so callers know the source was dropped.
        if not bool(mask.any()):
            _record(region, index, False, "degenerate region (empty mask, no contribution)")
            continue

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

        _record(region, index, True, "")

    # Save composite as RGBA PNG
    composite_uint8 = (np.clip(composite, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    out_path = CompositeTargetPath(output_dir / "composite_target.png")
    Image.fromarray(composite_uint8, mode="RGBA").save(out_path)

    out_path.region_status = region_status
    return out_path
