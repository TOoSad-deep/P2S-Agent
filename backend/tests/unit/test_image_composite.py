"""Unit tests for backend/app/pipeline/image_composite.py (V4.5 Local Fusion).

TDD: tests written before implementation.
Uses tmp_path for all I/O; never touches test_results/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.pipeline.fusion_plans import FusionRegion
from app.pipeline.image_composite import build_composite_target


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _solid_png(path: Path, color_rgba: tuple[int, int, int, int], size: tuple[int, int] = (100, 100)) -> Path:
    """Save a solid-colour RGBA PNG and return the path."""
    arr = np.full((*reversed(size), 4), color_rgba, dtype=np.uint8)
    img = Image.fromarray(arr, "RGBA")
    img.save(path)
    return path


def _load_rgb_array(path: Path) -> np.ndarray:
    """Return float32 (H,W,3) in [0,1]."""
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return arr


def _load_gray_array(path: Path) -> np.ndarray:
    """Return float32 (H,W) uint8 array [0,255] for mask inspection."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def _make_region(
    region_id: str,
    source_run_id: str,
    x: float, y: float, w: float, h: float,
    strength: float = 1.0,
    feather: float = 0.0,
    blend_mode: str = "soft",
) -> FusionRegion:
    return FusionRegion(
        id=region_id,
        label=region_id,
        source_run_id=source_run_id,
        instruction="test",
        geometry_type="rect",
        geometry={"x": x, "y": y, "w": w, "h": h},
        strength=strength,
        blend_mode=blend_mode,
        feather=feather,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOutputsExist:
    def test_composite_file_written(self, tmp_path: Path) -> None:
        base = _solid_png(tmp_path / "base.png", (255, 0, 0, 255))
        source = _solid_png(tmp_path / "src.png", (0, 0, 255, 255))
        region = _make_region("r1", "run_a", 0.25, 0.25, 0.5, 0.5)
        out = build_composite_target(
            base_render_path=base,
            source_render_paths={"run_a": source},
            regions=[region],
            output_dir=tmp_path / "out",
        )
        assert out.exists(), "composite_target.png must exist"
        assert out.name == "composite_target.png"

    def test_mask_file_written_per_region(self, tmp_path: Path) -> None:
        base = _solid_png(tmp_path / "base.png", (255, 0, 0, 255))
        source = _solid_png(tmp_path / "src.png", (0, 0, 255, 255))
        region = _make_region("r1", "run_a", 0.25, 0.25, 0.5, 0.5)
        out = build_composite_target(
            base_render_path=base,
            source_render_paths={"run_a": source},
            regions=[region],
            output_dir=tmp_path / "out",
        )
        mask_path = out.parent / "region_masks" / "r1.png"
        assert mask_path.exists(), "region_masks/r1.png must exist"

    def test_multiple_mask_files(self, tmp_path: Path) -> None:
        base = _solid_png(tmp_path / "base.png", (255, 0, 0, 255))
        src_a = _solid_png(tmp_path / "src_a.png", (0, 255, 0, 255))
        src_b = _solid_png(tmp_path / "src_b.png", (0, 0, 255, 255))
        regions = [
            _make_region("region-1", "run_a", 0.0, 0.0, 0.4, 0.4),
            _make_region("region-2", "run_b", 0.5, 0.5, 0.4, 0.4),
        ]
        out = build_composite_target(
            base_render_path=base,
            source_render_paths={"run_a": src_a, "run_b": src_b},
            regions=regions,
            output_dir=tmp_path / "out",
        )
        masks_dir = out.parent / "region_masks"
        assert (masks_dir / "region-1.png").exists()
        assert (masks_dir / "region-2.png").exists()


class TestAlphaBlend:
    def test_strength1_center_is_blue(self, tmp_path: Path) -> None:
        """strength=1.0, feather=0: center pixels must be nearly blue."""
        base = _solid_png(tmp_path / "base.png", (255, 0, 0, 255))
        source = _solid_png(tmp_path / "src.png", (0, 0, 255, 255))
        region = _make_region("r1", "run_a", 0.25, 0.25, 0.5, 0.5, strength=1.0, feather=0.0)
        out = build_composite_target(
            base_render_path=base,
            source_render_paths={"run_a": source},
            regions=[region],
            output_dir=tmp_path / "out",
        )
        arr = _load_rgb_array(out)
        H, W = arr.shape[:2]
        cy, cx = H // 2, W // 2
        center = arr[cy, cx]
        # Should be ~blue: R near 0, B near 1
        assert center[0] < 0.1, f"Red channel at center should be ~0, got {center[0]:.3f}"
        assert center[2] > 0.9, f"Blue channel at center should be ~1, got {center[2]:.3f}"

    def test_strength1_outside_is_red(self, tmp_path: Path) -> None:
        """strength=1.0, feather=0: far-outside pixels must be nearly red."""
        base = _solid_png(tmp_path / "base.png", (255, 0, 0, 255))
        source = _solid_png(tmp_path / "src.png", (0, 0, 255, 255))
        region = _make_region("r1", "run_a", 0.3, 0.3, 0.4, 0.4, strength=1.0, feather=0.0)
        out = build_composite_target(
            base_render_path=base,
            source_render_paths={"run_a": source},
            regions=[region],
            output_dir=tmp_path / "out",
        )
        arr = _load_rgb_array(out)
        # Top-left corner pixel — clearly outside the rect
        corner = arr[0, 0]
        assert corner[0] > 0.9, f"Red at corner should be ~1, got {corner[0]:.3f}"
        assert corner[2] < 0.1, f"Blue at corner should be ~0, got {corner[2]:.3f}"

    def test_strength_half_center_is_purple(self, tmp_path: Path) -> None:
        """strength=0.5: center should be halfway (equal R and B)."""
        base = _solid_png(tmp_path / "base.png", (255, 0, 0, 255))
        source = _solid_png(tmp_path / "src.png", (0, 0, 255, 255))
        region = _make_region("r1", "run_a", 0.25, 0.25, 0.5, 0.5, strength=0.5, feather=0.0)
        out = build_composite_target(
            base_render_path=base,
            source_render_paths={"run_a": source},
            regions=[region],
            output_dir=tmp_path / "out",
        )
        arr = _load_rgb_array(out)
        H, W = arr.shape[:2]
        cy, cx = H // 2, W // 2
        center = arr[cy, cx]
        # R and B should both be near 0.5
        assert abs(center[0] - 0.5) < 0.1, f"Red at center should be ~0.5, got {center[0]:.3f}"
        assert abs(center[2] - 0.5) < 0.1, f"Blue at center should be ~0.5, got {center[2]:.3f}"

    def test_strength_zero_leaves_base_unchanged(self, tmp_path: Path) -> None:
        """strength=0.0: composite should equal base."""
        base = _solid_png(tmp_path / "base.png", (255, 0, 0, 255))
        source = _solid_png(tmp_path / "src.png", (0, 0, 255, 255))
        region = _make_region("r1", "run_a", 0.0, 0.0, 1.0, 1.0, strength=0.0, feather=0.0)
        out = build_composite_target(
            base_render_path=base,
            source_render_paths={"run_a": source},
            regions=[region],
            output_dir=tmp_path / "out",
        )
        arr = _load_rgb_array(out)
        assert arr[:, :, 0].min() > 0.9, "All pixels should remain red (R~1)"
        assert arr[:, :, 2].max() < 0.1, "All pixels should remain red (B~0)"


class TestFeather:
    def test_feather_mask_has_intermediate_values(self, tmp_path: Path) -> None:
        """Nonzero feather: mask at the rect edge has values strictly between 0 and 255."""
        base = _solid_png(tmp_path / "base.png", (255, 0, 0, 255), size=(200, 200))
        source = _solid_png(tmp_path / "src.png", (0, 0, 255, 255), size=(200, 200))
        # feather=0.1 → 10% of min(H,W)=200 → 20 pixels of ramp
        region = _make_region("r1", "run_a", 0.2, 0.2, 0.6, 0.6, strength=1.0, feather=0.1)
        out = build_composite_target(
            base_render_path=base,
            source_render_paths={"run_a": source},
            regions=[region],
            output_dir=tmp_path / "out",
        )
        mask_path = out.parent / "region_masks" / "r1.png"
        mask = _load_gray_array(mask_path)
        # There should be pixels with values strictly between 0 and 255
        intermediate = (mask > 0) & (mask < 255)
        assert intermediate.any(), "Feathered mask must contain intermediate values (0 < v < 255)"

    def test_no_feather_mask_is_binary(self, tmp_path: Path) -> None:
        """feather=0.0: mask is exactly 0 or 255 — no intermediate values."""
        base = _solid_png(tmp_path / "base.png", (255, 0, 0, 255))
        source = _solid_png(tmp_path / "src.png", (0, 0, 255, 255))
        region = _make_region("r1", "run_a", 0.2, 0.2, 0.6, 0.6, strength=1.0, feather=0.0)
        out = build_composite_target(
            base_render_path=base,
            source_render_paths={"run_a": source},
            regions=[region],
            output_dir=tmp_path / "out",
        )
        mask_path = out.parent / "region_masks" / "r1.png"
        mask = _load_gray_array(mask_path)
        unique_vals = set(mask.ravel().astype(int))
        assert unique_vals <= {0, 255}, f"Zero-feather mask should be binary {{0,255}}, got {unique_vals}"

    def test_feather_transition_is_monotone_near_edge(self, tmp_path: Path) -> None:
        """Along a horizontal slice crossing the left rect edge, values go 0→255."""
        base = _solid_png(tmp_path / "base.png", (255, 0, 0, 255), size=(200, 200))
        source = _solid_png(tmp_path / "src.png", (0, 0, 255, 255), size=(200, 200))
        # rect left edge at x=0.3 → pixel 60; right edge at x=0.7 → pixel 140
        # feather=0.1 → feather_px=20
        # ramp goes from 0 at pixel 60 to 255 at pixel 80 (60+20)
        region = _make_region("r1", "run_a", 0.3, 0.0, 0.4, 1.0, strength=1.0, feather=0.1)
        out = build_composite_target(
            base_render_path=base,
            source_render_paths={"run_a": source},
            regions=[region],
            output_dir=tmp_path / "out",
        )
        mask_path = out.parent / "region_masks" / "r1.png"
        mask = _load_gray_array(mask_path)
        # Sample the middle row
        row = mask[100, :]  # H=200, mid row
        left_edge_px = int(round(0.3 * 200))   # 60
        feather_px = max(1, int(0.1 * 200))     # 20
        # Before ramp outer boundary: should be 0
        assert row[left_edge_px - 2] == 0.0, (
            f"Pixel just outside left edge should be 0, got {row[left_edge_px - 2]}"
        )
        # Well inside rect, past the full feather ramp: should be 255
        inside_px = left_edge_px + feather_px + 2  # 82
        assert row[inside_px] == 255.0, (
            f"Pixel well inside rect (past feather) should be 255, got {row[inside_px]}"
        )


class TestOverlapOrder:
    def test_later_region_wins_overlap(self, tmp_path: Path) -> None:
        """Two overlapping regions: blue first, green second → overlap is ~green."""
        base = _solid_png(tmp_path / "base.png", (255, 0, 0, 255))
        src_blue = _solid_png(tmp_path / "src_blue.png", (0, 0, 255, 255))
        src_green = _solid_png(tmp_path / "src_green.png", (0, 255, 0, 255))
        regions = [
            _make_region("r_blue", "run_blue", 0.1, 0.1, 0.8, 0.8, strength=1.0, feather=0.0),
            _make_region("r_green", "run_green", 0.1, 0.1, 0.8, 0.8, strength=1.0, feather=0.0),
        ]
        out = build_composite_target(
            base_render_path=base,
            source_render_paths={"run_blue": src_blue, "run_green": src_green},
            regions=regions,
            output_dir=tmp_path / "out",
        )
        arr = _load_rgb_array(out)
        H, W = arr.shape[:2]
        cy, cx = H // 2, W // 2
        center = arr[cy, cx]
        # Should be ~green (second region applied last)
        assert center[1] > 0.9, f"Green channel should be ~1 in overlap, got {center[1]:.3f}"
        assert center[2] < 0.1, f"Blue channel should be ~0 in overlap (green wins), got {center[2]:.3f}"


class TestMissingSource:
    def test_missing_source_skip_no_crash(self, tmp_path: Path) -> None:
        """Region whose source_run_id is absent from the dict: skipped, no crash."""
        base = _solid_png(tmp_path / "base.png", (255, 0, 0, 255))
        # Source dict is empty — region points to non-existent "run_missing"
        region = _make_region("r1", "run_missing", 0.0, 0.0, 1.0, 1.0, strength=1.0)
        out = build_composite_target(
            base_render_path=base,
            source_render_paths={},
            regions=[region],
            output_dir=tmp_path / "out",
        )
        assert out.exists()
        # Composite should look like base (red), since region was skipped
        arr = _load_rgb_array(out)
        assert arr[:, :, 0].min() > 0.9

    def test_missing_source_no_mask_written(self, tmp_path: Path) -> None:
        """Skipped region (missing source) must NOT write a mask file."""
        base = _solid_png(tmp_path / "base.png", (255, 0, 0, 255))
        region = _make_region("r1", "run_missing", 0.0, 0.0, 1.0, 1.0)
        out = build_composite_target(
            base_render_path=base,
            source_render_paths={},
            regions=[region],
            output_dir=tmp_path / "out",
        )
        mask_path = out.parent / "region_masks" / "r1.png"
        assert not mask_path.exists(), "No mask should be written for a skipped region"

    def test_partial_missing_source(self, tmp_path: Path) -> None:
        """One region has a source, another is missing: present one is applied."""
        base = _solid_png(tmp_path / "base.png", (255, 0, 0, 255))
        src_blue = _solid_png(tmp_path / "src_blue.png", (0, 0, 255, 255))
        regions = [
            _make_region("r_present", "run_blue", 0.0, 0.0, 1.0, 1.0, strength=1.0, feather=0.0),
            _make_region("r_missing", "run_ghost", 0.0, 0.0, 1.0, 1.0, strength=1.0, feather=0.0),
        ]
        out = build_composite_target(
            base_render_path=base,
            source_render_paths={"run_blue": src_blue},
            regions=regions,
            output_dir=tmp_path / "out",
        )
        arr = _load_rgb_array(out)
        # r_present applied (blue), r_missing skipped → stays blue
        assert arr[50, 50, 2] > 0.9, "Blue region should be applied"
        mask_present = out.parent / "region_masks" / "r_present.png"
        mask_missing = out.parent / "region_masks" / "r_missing.png"
        assert mask_present.exists()
        assert not mask_missing.exists()


class TestEmptyRegions:
    def test_empty_regions_composite_equals_base(self, tmp_path: Path) -> None:
        """No regions → composite is a copy of the base."""
        base = _solid_png(tmp_path / "base.png", (200, 100, 50, 255))
        out = build_composite_target(
            base_render_path=base,
            source_render_paths={},
            regions=[],
            output_dir=tmp_path / "out",
        )
        arr = _load_rgb_array(out)
        # Check all pixels match base color (200/255≈0.784, 100/255≈0.392, 50/255≈0.196)
        assert abs(arr[:, :, 0].mean() - 200 / 255) < 0.01
        assert abs(arr[:, :, 1].mean() - 100 / 255) < 0.01
        assert abs(arr[:, :, 2].mean() - 50 / 255) < 0.01

    def test_empty_regions_no_mask_dir_needed(self, tmp_path: Path) -> None:
        """No regions → region_masks dir may or may not exist, but no crash."""
        base = _solid_png(tmp_path / "base.png", (255, 255, 255, 255))
        out = build_composite_target(
            base_render_path=base,
            source_render_paths={},
            regions=[],
            output_dir=tmp_path / "out",
        )
        assert out.exists()

    def test_file_written_even_with_empty_regions(self, tmp_path: Path) -> None:
        base = _solid_png(tmp_path / "base.png", (0, 128, 0, 255))
        out = build_composite_target(
            base_render_path=base,
            source_render_paths={},
            regions=[],
            output_dir=tmp_path / "out",
        )
        assert out.name == "composite_target.png"
        assert out.stat().st_size > 0


class TestReturnPath:
    def test_returns_path_object(self, tmp_path: Path) -> None:
        base = _solid_png(tmp_path / "base.png", (255, 0, 0, 255))
        out = build_composite_target(
            base_render_path=base,
            source_render_paths={},
            regions=[],
            output_dir=tmp_path / "out",
        )
        assert isinstance(out, Path)

    def test_output_in_specified_dir(self, tmp_path: Path) -> None:
        base = _solid_png(tmp_path / "base.png", (255, 0, 0, 255))
        output_dir = tmp_path / "my_output"
        out = build_composite_target(
            base_render_path=base,
            source_render_paths={},
            regions=[],
            output_dir=output_dir,
        )
        assert out.parent == output_dir


class TestUnsafeRegionId:
    def test_unsafe_region_id_skips_mask_no_traversal(self, tmp_path: Path) -> None:
        """A region id containing '../' must be rejected by _SAFE_ID_RE.

        The composite_target.png IS written (the blend is still applied), but
        NO mask file is written outside the region_masks directory — in
        particular, no file at output_dir.parent / 'attack.png' is created.
        """
        base = _solid_png(tmp_path / "base.png", (255, 0, 0, 255))
        source = _solid_png(tmp_path / "src.png", (0, 0, 255, 255))
        region = FusionRegion(
            id="../attack",
            label="evil region",
            source_run_id="run_a",
            instruction="test",
            geometry_type="rect",
            geometry={"x": 0.0, "y": 0.0, "w": 0.5, "h": 0.5},
            strength=1.0,
            blend_mode="soft",
            feather=0.0,
        )
        output_dir = tmp_path / "out"
        # Must not raise
        out = build_composite_target(
            base_render_path=base,
            source_render_paths={"run_a": source},
            regions=[region],
            output_dir=output_dir,
        )
        # composite_target.png IS written
        assert out.exists(), "composite_target.png must be written even for unsafe id"
        assert out.name == "composite_target.png"
        # No traversal: attack.png must NOT appear at output_dir.parent / "attack.png"
        traversal_target = output_dir.parent / "attack.png"
        assert not traversal_target.exists(), (
            "Path traversal: attack.png must NOT be created outside output_dir"
        )


class TestNonRectRegion:
    def test_non_rect_geometry_type_skipped(self, tmp_path: Path) -> None:
        """geometry_type != 'rect' is not yet supported and should be skipped (no crash)."""
        base = _solid_png(tmp_path / "base.png", (255, 0, 0, 255))
        src = _solid_png(tmp_path / "src.png", (0, 0, 255, 255))
        region = FusionRegion(
            id="poly_r",
            label="polygon region",
            source_run_id="run_a",
            instruction="test",
            geometry_type="polygon",
            geometry={"x": 0.1, "y": 0.1, "w": 0.5, "h": 0.5},
            strength=1.0,
            blend_mode="soft",
            feather=0.0,
        )
        out = build_composite_target(
            base_render_path=base,
            source_render_paths={"run_a": src},
            regions=[region],
            output_dir=tmp_path / "out",
        )
        assert out.exists()
        # Base should be preserved since only geometry_type "rect" is supported
        arr = _load_rgb_array(out)
        assert arr[:, :, 0].min() > 0.9, "Non-rect region should be skipped, base preserved"
