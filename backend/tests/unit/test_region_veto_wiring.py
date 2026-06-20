from types import SimpleNamespace
from pathlib import Path
from PIL import Image
from p2s_agent.core.pipeline.graph import _build_region_veto_fn
from app.pipeline.human_constraints import RegionConstraint
from p2s_agent.core.pipeline.region_metrics import RegionVetoResult


def _protect(rid="r1", x=0.0, w=0.5, strength=0.5):
    return RegionConstraint(id=rid, label=rid, mode="protect", instruction="",
                            geometry_type="rect", geometry={"x": x, "y": 0.0, "w": w, "h": 1.0},
                            strength=strength)


def test_no_protect_regions_returns_none(tmp_path):
    sel = SimpleNamespace(render_path=str(tmp_path / "base.png"), dsl=None, compile_glsl="x")
    assert _build_region_veto_fn(sel, [], tmp_path, 64, 64, floor=0.85, ceil=0.95) is None


def test_none_selected_returns_none(tmp_path):
    assert _build_region_veto_fn(None, [_protect()], tmp_path, 64, 64, floor=0.85, ceil=0.95) is None


def test_builds_veto_fn_from_selected_render_path(tmp_path):
    # baseline (left half dark) vs degraded candidate (left half white) -> vetoed
    base = Image.new("RGB", (64, 64), (10, 20, 30)); bpath = tmp_path / "base.png"; base.save(bpath)
    cand = Image.new("RGB", (64, 64), (10, 20, 30))
    for px in range(32):
        for py in range(64):
            cand.putpixel((px, py), (255, 255, 255))
    cpath = tmp_path / "cand.png"; cand.save(cpath)
    sel = SimpleNamespace(render_path=str(bpath), dsl=None, compile_glsl="x")
    fn = _build_region_veto_fn(sel, [_protect(x=0.0, w=0.5)], tmp_path, 64, 64, floor=0.85, ceil=0.95)
    assert fn is not None
    res = fn(cpath)
    assert isinstance(res, RegionVetoResult) and res.vetoed is True


def test_identical_candidate_not_vetoed(tmp_path):
    base = Image.new("RGB", (64, 64), (10, 20, 30)); bpath = tmp_path / "base.png"; base.save(bpath)
    sel = SimpleNamespace(render_path=str(bpath), dsl=None, compile_glsl="x")
    fn = _build_region_veto_fn(sel, [_protect()], tmp_path, 64, 64, floor=0.85, ceil=0.95)
    assert fn is not None
    assert fn(bpath).vetoed is False   # candidate == baseline
