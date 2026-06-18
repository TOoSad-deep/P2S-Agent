# constraint_score protect-区域硬否决 gate — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 V4.2 protect 区域约束在 DSL/GLSL 两条 LLM 精修循环中真正硬否决——任何破坏 protect 区域的候选被拒绝，即使全局分提升或 directed VLM 判通过。

**Architecture:** 纯函数 `evaluate_protect_veto`（region_metrics.py）按 `strength` 派生的 per-region SSIM 阈值，比候选渲染与精修前 seed 渲染。两条循环新增可选注入 `region_veto_fn`（复用现有 judge 注入模式）；`graph._run_post_pipeline` 在存在 protect 区域时构造并注入。无 protect 区域 → `region_veto_fn=None` → 逐位等同今天，零回归。

**Tech Stack:** Python 3.9（系统解释器，无 venv）、numpy、PIL、pytest。基准：当前 main = `0f1379f`，1124 pytest 全绿。

> **设计依据:** `doc/2026-06-18-constraint-score-region-veto-design.md`（已评审）。
> **测试命令前缀:** 所有测试在 `backend/` 下运行：`cd backend && python3 -m pytest ...`。

---

## File Structure

| 文件 | 责任 | 改动 |
|---|---|---|
| `backend/app/pipeline/region_metrics.py` | 纯：区域指标 + 否决判定 | +`RegionVetoResult`、`protect_region_threshold`、`evaluate_protect_veto` |
| `backend/app/config.py` | 全局设置 | +`protect_veto_ssim_floor/_ceil` |
| `backend/app/pipeline/glsl_refinement.py` | GLSL 精修循环 | +`region_veto_fn` 参数 + 否决块 |
| `backend/app/pipeline/refinement.py` | DSL 精修循环 | +`region_veto_fn` 参数 + 否决块 |
| `backend/app/pipeline/graph.py` | post-pipeline 编排 | 构造 `region_veto_fn` 注入两循环；`run_png_shader_pipeline` +`protect_regions` → state；run 末写约束结果 |
| `backend/app/routers/png_shader.py` | 端点 → worker | branch-refine / explore-variants 筛 protect 区域 → `pipeline_extra` → pipeline |
| `backend/tests/unit/test_region_metrics.py` | 纯函数测试 | Task 1 用例 |
| `backend/tests/unit/test_glsl_refinement.py` | GLSL 集成测试 | Task 3 用例 |
| `backend/tests/unit/test_refinement.py`（若无则新建） | DSL 集成测试 | Task 4 用例 |

---

## Task 1: 纯否决核心（region_metrics.py）

**Files:**
- Modify: `backend/app/pipeline/region_metrics.py`
- Test: `backend/tests/unit/test_region_metrics.py`

- [ ] **Step 1: 写失败测试**

在 `backend/tests/unit/test_region_metrics.py` 末尾追加。先看文件顶部已有的 import 与构造 PNG 的 helper（复用之）；若没有构造 region 的 helper，本测试自带。

```python
from app.pipeline.human_constraints import RegionConstraint
from app.pipeline.region_metrics import (
    RegionVetoResult,
    protect_region_threshold,
    evaluate_protect_veto,
)
from PIL import Image


def _png(path, color, size=(64, 64)):
    Image.new("RGB", size, color).save(path)
    return path


def _protect_region(rid="r1", x=0.0, y=0.0, w=0.5, h=1.0, strength=0.5):
    return RegionConstraint(
        id=rid, label=rid, mode="protect", instruction="",
        geometry_type="rect", geometry={"x": x, "y": y, "w": w, "h": h},
        strength=strength,
    )


def test_protect_region_threshold_maps_strength():
    assert protect_region_threshold(0.0) == 0.85
    assert protect_region_threshold(0.5) == 0.90
    assert protect_region_threshold(1.0) == 0.95


def test_no_protect_regions_is_not_evaluated_and_not_vetoed(tmp_path):
    base = _png(tmp_path / "b.png", (10, 20, 30))
    cand = _png(tmp_path / "c.png", (200, 0, 0))
    modify = RegionConstraint(
        id="m", label="m", mode="modify", instruction="",
        geometry_type="rect", geometry={"x": 0, "y": 0, "w": 1, "h": 1}, strength=0.5,
    )
    res = evaluate_protect_veto(base, cand, [modify])
    assert isinstance(res, RegionVetoResult)
    assert res.vetoed is False and res.evaluated is False


def test_identical_protect_region_not_vetoed(tmp_path):
    base = _png(tmp_path / "b.png", (10, 20, 30))
    cand = _png(tmp_path / "c.png", (10, 20, 30))  # identical
    res = evaluate_protect_veto(base, cand, [_protect_region()])
    assert res.vetoed is False
    assert res.constraint_score > 0.95
    assert res.evaluated is True


def test_degraded_protect_region_is_vetoed(tmp_path):
    # Left half differs drastically between base and candidate.
    base = Image.new("RGB", (64, 64), (10, 20, 30))
    cand = Image.new("RGB", (64, 64), (10, 20, 30))
    for px in range(32):
        for py in range(64):
            cand.putpixel((px, py), (255, 255, 255))
    bpath = tmp_path / "b.png"; base.save(bpath)
    cpath = tmp_path / "c.png"; cand.save(cpath)
    region = _protect_region(x=0.0, w=0.5)  # protect the left half
    res = evaluate_protect_veto(bpath, cpath, [region])
    assert res.vetoed is True
    assert any(r["violated"] for r in res.regions)
    assert res.reason and "r1" in res.reason


def test_any_violated_region_triggers_veto(tmp_path):
    base = Image.new("RGB", (64, 64), (10, 20, 30))
    cand = Image.new("RGB", (64, 64), (10, 20, 30))
    for px in range(32):           # only left half changes
        for py in range(64):
            cand.putpixel((px, py), (255, 255, 255))
    bpath = tmp_path / "b.png"; base.save(bpath)
    cpath = tmp_path / "c.png"; cand.save(cpath)
    left = _protect_region(rid="left", x=0.0, w=0.5)    # degraded -> violated
    right = _protect_region(rid="right", x=0.5, w=0.5)  # untouched -> ok
    res = evaluate_protect_veto(bpath, cpath, [left, right])
    assert res.vetoed is True


def test_missing_baseline_is_not_evaluated(tmp_path):
    cand = _png(tmp_path / "c.png", (10, 20, 30))
    res = evaluate_protect_veto(tmp_path / "does_not_exist.png", cand, [_protect_region()])
    assert res.evaluated is False and res.vetoed is False
```

- [ ] **Step 2: 运行测试，确认失败**

Run: `cd backend && python3 -m pytest tests/unit/test_region_metrics.py -q`
Expected: FAIL（`ImportError: cannot import name 'RegionVetoResult'` 等）。

- [ ] **Step 3: 实现纯否决核心**

在 `backend/app/pipeline/region_metrics.py` 顶部 import 区加 `from dataclasses import dataclass`，并在文件末尾（`compute_region_metrics` 之后）追加：

```python
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
    return floor + (ceil - floor) * s


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
```

- [ ] **Step 4: 运行测试，确认通过**

Run: `cd backend && python3 -m pytest tests/unit/test_region_metrics.py -q`
Expected: PASS（全部，含原有用例）。

- [ ] **Step 5: 提交**

```bash
git add backend/app/pipeline/region_metrics.py backend/tests/unit/test_region_metrics.py
git commit -m "feat(region-veto): pure protect-region hard-veto core (evaluate_protect_veto)"
```

---

## Task 2: 配置旋钮（config.py）

**Files:**
- Modify: `backend/app/config.py`

- [ ] **Step 1: 加设置字段**

在 `Settings` 类里（其他 `*_base_url` / 数值字段附近）加两个字段，沿用该类现有的 pydantic 字段写法：

```python
    protect_veto_ssim_floor: float = 0.85
    protect_veto_ssim_ceil: float = 0.95
```

> 读取方式：env `PROTECT_VETO_SSIM_FLOOR` / `PROTECT_VETO_SSIM_CEIL`（pydantic settings 大小写不敏感，沿用本项目既有约定）。`region_metrics` 保持纯净（floor/ceil 为函数参数），由 graph.py 读 settings 后传入。

- [ ] **Step 2: 冒烟验证导入**

Run: `cd backend && python3 -c "from app.config import settings; print(settings.protect_veto_ssim_floor, settings.protect_veto_ssim_ceil)"`
Expected: 打印 `0.85 0.95`，无异常。

- [ ] **Step 3: 提交**

```bash
git add backend/app/config.py
git commit -m "feat(region-veto): add PROTECT_VETO_SSIM_FLOOR/CEIL settings"
```

---

## Task 3: GLSL 循环集成（glsl_refinement.py）

**Files:**
- Modify: `backend/app/pipeline/glsl_refinement.py`（签名 ~63-87；否决块插入在 render-failed 分支之后、`entry["score_after"]` 之前，~line 301）
- Test: `backend/tests/unit/test_glsl_refinement.py`

- [ ] **Step 1: 写失败测试**

复用该文件已有的 `VALID_GLSL_A` / `VALID_GLSL_B` 与 `_evaluate_by_r_with_render`（写真实 render 文件、`actual_render` 非 None）。在文件末尾追加：

```python
from app.pipeline.region_metrics import RegionVetoResult


def _veto_all(_render_path):
    return RegionVetoResult(
        vetoed=True, constraint_score=0.40,
        regions=[{"id": "r1", "label": "sky", "ssim": 0.40, "threshold": 0.90, "violated": True}],
        reason="protected regions degraded: sky", evaluated=True,
    )


def _veto_none(_render_path):
    return RegionVetoResult(True is False, 1.0, [], None, True)  # vetoed=False


def test_glsl_veto_rejects_globally_better_candidate(tmp_path, monkeypatch):
    # Candidate scores higher (0.6 > 0.3) but the protect region is degraded.
    monkeypatch.setattr(
        "app.candidates.llm_scene.generate_llm_glsl_refinement",
        lambda **k: {"glsl": VALID_GLSL_B, "_io": {}},
    )
    result = run_glsl_refinement_loop(
        VALID_GLSL_A, 0.30, {}, {"final_score": 0.30},
        tmp_path / "ref.png",
        evaluate_fn=_evaluate_by_r_with_render,  # writes a render, returns score by 'r'
        initial_render_path=tmp_path / "current.png",
        max_iterations=1, threshold=0.80, high_score_stop=0.92,
        no_improvement_patience=2, max_fresh_restarts=0,
        loop_dir=tmp_path / "loop",
        region_veto_fn=_veto_all,
    )
    assert result["best_glsl"] == VALID_GLSL_A          # not accepted
    assert result["history"][0].get("accepted") is not True
    assert result["history"][0].get("rejected_reason") == "protect_region_veto"


def test_glsl_veto_overrides_directed_acceptance(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.candidates.llm_scene.generate_llm_glsl_refinement",
        lambda **k: {"glsl": VALID_GLSL_B, "_io": {}},
    )
    result = run_glsl_refinement_loop(
        VALID_GLSL_A, 0.30, {}, {"final_score": 0.30},
        tmp_path / "ref.png",
        evaluate_fn=_evaluate_by_r_with_render,
        initial_render_path=tmp_path / "current.png",
        max_iterations=1, threshold=0.80, high_score_stop=0.92,
        no_improvement_patience=2, max_fresh_restarts=0,
        loop_dir=tmp_path / "loop",
        directed_acceptance={"score_drop_tolerance": 0.5},
        directed_pairwise_judge=lambda a, b: "B",   # would accept a drop
        region_veto_fn=_veto_all,                   # ...but veto overrides
    )
    assert result["best_glsl"] == VALID_GLSL_A
    assert result["history"][0].get("accepted") is not True
```

> 检查 `_evaluate_by_r_with_render` 的"按 GLSL 内容给分"约定（见文件 line ~410），确保 `VALID_GLSL_B` 得分 > `VALID_GLSL_A` 的初始 0.30；若它依赖 `_r`/标记，按既有用例同样的方式标注（参考 line ~437 已通过用例）。

- [ ] **Step 2: 运行测试，确认失败**

Run: `cd backend && python3 -m pytest tests/unit/test_glsl_refinement.py -q -k "veto"`
Expected: FAIL（`run_glsl_refinement_loop() got an unexpected keyword argument 'region_veto_fn'`）。

- [ ] **Step 3: 加参数**

在 `run_glsl_refinement_loop` 签名（line ~86，`on_iteration` 之后）加：

```python
    on_iteration: "Callable[[dict], None] | None" = None,
    region_veto_fn: "Callable[[Path], object] | None" = None,
) -> dict:
```

- [ ] **Step 4: 加否决块**

在循环里 render-failed 分支结束（`continue` 之后，原 `entry["score_after"] = round(new_score, 4)` 之前，~line 301）插入：

```python
        if region_veto_fn is not None and actual_render is not None:
            veto = region_veto_fn(actual_render)
            if getattr(veto, "vetoed", False):
                entry["score_after"] = round(new_score, 4)
                entry["delta"] = round(new_score - best_score, 4)
                entry["rejected_reason"] = "protect_region_veto"
                entry["region_veto"] = getattr(veto, "regions", [])
                entry["constraint_score"] = getattr(veto, "constraint_score", None)
                entry["accepted"] = False
                entry["best_score_after"] = round(best_score, 4)
                extra_feedback = [
                    "[PROTECT VIOLATION] Your last revision degraded a protected "
                    f"region ({getattr(veto, 'reason', None) or 'protected region'}). "
                    "You MUST keep those regions unchanged; revise elsewhere."
                ]
                no_improvement_count += 1
                _record(entry)
                if no_improvement_count >= no_improvement_patience and not _trigger_fresh_restart():
                    stop_reason = "no_improvement_patience"
                    break
                continue
```

- [ ] **Step 5: 运行测试，确认通过 + 无回归**

Run: `cd backend && python3 -m pytest tests/unit/test_glsl_refinement.py -q`
Expected: PASS（含新 veto 用例与全部既有用例 —— 既有用例不传 `region_veto_fn`，逐位不变）。

- [ ] **Step 6: 提交**

```bash
git add backend/app/pipeline/glsl_refinement.py backend/tests/unit/test_glsl_refinement.py
git commit -m "feat(region-veto): integrate protect-region hard-veto into GLSL refinement loop"
```

---

## Task 4: DSL 循环集成（refinement.py）

**Files:**
- Modify: `backend/app/pipeline/refinement.py`（签名 ~100-128；否决块插入在 `_evaluate_dsl(...)` 返回后、`entry["score_after"]` 之前，~line 396）
- Test: `backend/tests/unit/test_refinement.py`（若不存在则新建；否则追加）

- [ ] **Step 1: 写失败测试**

先读 `refinement.py` 顶部，确认 `run_dsl_refinement_loop` 的必填参数（`preprocess`、`initial_dsl`、`initial_score`、`initial_metrics`、`initial_quality`、`reference_path`、`canvas_width`、`canvas_height`、`loop_dir` 等）。本测试用 monkeypatch 替换 LLM 与渲染/评分依赖，注入一个总否决的 `region_veto_fn`，断言改进候选被拒。

```python
from pathlib import Path
import pytest
from PIL import Image
from app.pipeline.refinement import run_dsl_refinement_loop
from app.pipeline.region_metrics import RegionVetoResult

_DSL_A = {"schema_version": 1, "canvas": {"width": 64, "height": 64, "background": "#000000"},
          "layers": [{"id": "c0", "type": "circle", "fill": {"type": "solid", "color": "#ff0000"},
                      "params": {"center": [0.5, 0.5], "radius": 0.2}, "opacity": 1.0}]}
_DSL_B = {"schema_version": 1, "canvas": {"width": 64, "height": 64, "background": "#000000"},
          "layers": [{"id": "c0", "type": "circle", "fill": {"type": "solid", "color": "#00ff00"},
                      "params": {"center": [0.5, 0.5], "radius": 0.3}, "opacity": 1.0}]}


def _veto_all(_render_path):
    return RegionVetoResult(True, 0.4, [{"id": "r1", "label": "sky", "ssim": 0.4,
                            "threshold": 0.9, "violated": True}],
                            "protected regions degraded: sky", True)


def test_dsl_veto_rejects_globally_better_candidate(tmp_path, monkeypatch):
    # LLM always returns the "better" DSL_B.
    monkeypatch.setattr(
        "app.candidates.llm_scene.generate_llm_refinement",
        lambda **k: dict(_DSL_B),
    )
    # _evaluate_dsl renders to render_path and returns a higher score for B.
    def _fake_eval(dsl, glsl, ref, render_path, **kw):
        Image.new("RGB", (64, 64), (0, 0, 0)).save(render_path)   # ensure render exists
        score = 0.6 if dsl.get("layers", [{}])[0].get("fill", {}).get("color") == "#00ff00" else 0.3
        return {}, {"final_score": score}, score, render_path
    monkeypatch.setattr("app.pipeline.refinement._evaluate_dsl", _fake_eval)

    result = run_dsl_refinement_loop(
        preprocess={}, initial_dsl=dict(_DSL_A), initial_score=0.30,
        initial_metrics={}, initial_quality={"final_score": 0.30},
        reference_path=tmp_path / "ref.png",
        canvas_width=64, canvas_height=64, max_shader_chars=12000,
        max_iterations=1, threshold=0.80, high_score_stop=0.92,
        no_improvement_patience=2, loop_dir=tmp_path / "loop",
        protected_aspects=[],
        region_veto_fn=_veto_all,
    )
    assert result["best_dsl"] == _DSL_A                       # not accepted
    assert result["history"][0].get("accepted") is not True
    assert result["history"][0].get("rejected_reason") == "protect_region_veto"
```

> 若 `run_dsl_refinement_loop` 的关键字名与上面不完全一致（如 `protected_aspects` 是否必填），读签名后对齐；保持除 `region_veto_fn` 外的调用方式与 `graph.py:708` 一致。

- [ ] **Step 2: 运行测试，确认失败**

Run: `cd backend && python3 -m pytest tests/unit/test_refinement.py -q -k "veto"`
Expected: FAIL（`unexpected keyword argument 'region_veto_fn'`）。

- [ ] **Step 3: 加参数**

在 `run_dsl_refinement_loop` 签名末尾（最后一个关键字参数之后）加：

```python
    region_veto_fn: "Callable[[Path], object] | None" = None,
```

（确认文件已 `from typing import Callable`、`from pathlib import Path`；若缺则补 import。）

- [ ] **Step 4: 加否决块**

在 `_evaluate_dsl(...)` 返回之后（~line 395-396）、`entry["score_after"] = round(new_score, 4)` 之前插入：

```python
        if region_veto_fn is not None and render_path.exists():
            veto = region_veto_fn(render_path)
            if getattr(veto, "vetoed", False):
                entry["score_after"] = round(new_score, 4)
                entry["delta"] = round(new_score - best_score, 4)
                entry["rejected_reason"] = "protect_region_veto"
                entry["region_veto"] = getattr(veto, "regions", [])
                entry["constraint_score"] = getattr(veto, "constraint_score", None)
                entry["accepted"] = False
                entry["best_score_after"] = round(best_score, 4)
                extra_feedback = [
                    "[PROTECT VIOLATION] Your last revision degraded a protected "
                    f"region ({getattr(veto, 'reason', None) or 'protected region'}). "
                    "You MUST keep those regions unchanged; revise elsewhere."
                ]
                no_improvement_count += 1
                _record(entry)
                if no_improvement_count >= no_improvement_patience:
                    stop_reason = "no_improvement_patience"
                    break
                continue
```

- [ ] **Step 5: 运行测试，确认通过 + 无回归**

Run: `cd backend && python3 -m pytest tests/unit/test_refinement.py tests/unit/test_llm_glsl_refinement.py -q`
Expected: PASS（新 veto 用例 + 既有用例）。

- [ ] **Step 6: 提交**

```bash
git add backend/app/pipeline/refinement.py backend/tests/unit/test_refinement.py
git commit -m "feat(region-veto): integrate protect-region hard-veto into DSL refinement loop"
```

---

## Task 5: 编排接线（graph.py）

**Files:**
- Modify: `backend/app/pipeline/graph.py`
  - `_run_post_pipeline`（~351）：读 `state.get("protect_regions", [])`，构造 `region_veto_fn`，传入两处循环调用（DSL ~708、GLSL ~797）；run 末写约束结果到 `refinement_summary`。
  - `run_png_shader_pipeline`（~1026）：+`protect_regions` 参数 → 写入 state。

- [ ] **Step 1: 加 `protect_regions` 形参并入 state**

在 `run_png_shader_pipeline` 签名（~1041，`extra_artifacts` 之后）加：

```python
    extra_artifacts: dict | None = None,
    protect_regions: list | None = None,
) -> dict:
```

找到该函数内构建 `state` 字典的位置（与 `"protected_aspects": protected_aspects` 同一 dict，~line 1213 附近），加一行：

```python
        "protect_regions": list(protect_regions or []),
```

- [ ] **Step 2: 在 `_run_post_pipeline` 构造 `region_veto_fn`**

在 `_run_post_pipeline` 内、`should_refine` 分支之前（`protected_aspects = state.get(...)` 附近，~line 388）加：

```python
    from app.config import settings as _settings
    from app.pipeline.region_metrics import evaluate_protect_veto

    _protect_regions = [
        r for r in (state.get("protect_regions") or [])
        if getattr(r, "mode", None) == "protect"
    ]
    region_veto_fn = None
    if _protect_regions and selected is not None:
        _baseline = Path(selected.render_path) if getattr(selected, "render_path", None) else None
        if _baseline is None or not _baseline.exists():
            # DSL candidates may not carry a render_path: render the selected once.
            try:
                if selected.compile_glsl:
                    _baseline = None  # GLSL path without render — skip veto (best-effort)
                elif selected.dsl:
                    from app.dsl.renderer import render_dsl_to_image
                    _bpath = run_dir / "protect_baseline.png"
                    render_dsl_to_image(selected.dsl, _bpath,
                                        width=canvas_width, height=canvas_height)
                    _baseline = _bpath
            except Exception:
                logger.warning("protect-veto baseline render failed", exc_info=True)
                _baseline = None
        if _baseline is not None and _baseline.exists():
            _floor = float(_settings.protect_veto_ssim_floor)
            _ceil = float(_settings.protect_veto_ssim_ceil)
            region_veto_fn = (
                lambda cand, _r=_protect_regions, _b=_baseline, _f=_floor, _c=_ceil:
                evaluate_protect_veto(_b, cand, _r, floor=_f, ceil=_c)
            )
```

> `run_dir`、`canvas_width`、`canvas_height`、`selected`、`logger` 在 `_run_post_pipeline` 作用域已有（确认后使用）。

- [ ] **Step 3: 把 `region_veto_fn` 传入两处循环**

DSL 调用（~745，`on_iteration=_publish_iteration,` 之后）：

```python
            on_iteration=_publish_iteration,
            region_veto_fn=region_veto_fn,
        )
```

GLSL 调用（~820 区域，循环调用的最后一个关键字参数之后）同样加：

```python
            region_veto_fn=region_veto_fn,
        )
```

- [ ] **Step 4: run 末写约束结果**

在 `refinement_summary.update({...})`（DSL ~748、GLSL 对应处）之后，加（DSL 与 GLSL 两处各一份，或在两分支汇合后统一一次）：

```python
        _veto_iters = [h for h in refinement_history if h.get("rejected_reason") == "protect_region_veto"]
        if _protect_regions:
            refinement_summary["protect_regions"] = {
                "count": len(_protect_regions),
                "veto_count": len(_veto_iters),
                "last_constraint_score": (
                    _veto_iters[-1].get("constraint_score") if _veto_iters else None
                ),
            }
```

> `refinement_history` 变量在每个分支内已绑定；若 GLSL 分支用不同变量名，按其实际名引用。

- [ ] **Step 5: 跑全量后端回归**

Run: `cd backend && python3 -m pytest tests/unit/ -q`
Expected: PASS（1124 + 新增 ≈ 1135+；**无 FAIL**）。无 protect 区域时 `region_veto_fn=None`，既有路径逐位不变。

- [ ] **Step 6: 提交**

```bash
git add backend/app/pipeline/graph.py
git commit -m "feat(region-veto): wire protect-region veto into post-pipeline + run summary"
```

---

## Task 6: 端点透传 protect 区域（png_shader.py）

**Files:**
- Modify: `backend/app/routers/png_shader.py`
  - branch-refine（`_constraint_spec` 解析处 ~1048；`pipeline_extra` 构建处 ~1210）
  - explore-variants（`_ev_constraint_spec` 解析处 ~1482；其变体 worker 的 `pipeline_extra` 构建处）
  - `_run_png_shader_background`（~514/632）把 `protect_regions` 透传进 `run_png_shader_pipeline`

- [ ] **Step 1: 写失败/集成测试（端到端透传）**

在 `backend/tests/unit/test_router.py` 加一个测试：branch-refine 带一个 protect region 的 constraints，断言 worker 收到 `protect_regions`。最稳的做法是 monkeypatch `run_png_shader_pipeline` 捕获 kwargs：

```python
def test_branch_refine_forwards_protect_regions(monkeypatch, tmp_path):
    captured = {}
    import app.routers.png_shader as P

    def _fake_pipeline(*a, **k):
        captured.update(k)
        return {"run_id": k.get("run_id", "x"), "status": "completed", "selected": None,
                "scoreboard": {}, "refinement": {}}
    monkeypatch.setattr(P, "run_png_shader_pipeline", _fake_pipeline)
    # ... set up a parent run + checkpoint as the other branch-refine tests do ...
    # POST /branch-refine with constraints containing a protect region, run the
    # worker synchronously (the existing tests' pattern), then:
    assert "protect_regions" in captured
    assert captured["protect_regions"] and captured["protect_regions"][0].mode == "protect"
```

> 对齐既有 branch-refine 测试搭建父 run / checkpoint / 同步执行 worker 的方式（见 test_router.py 现有 branch-refine 用例）。若现有测试已 monkeypatch worker，可复用其夹具。

- [ ] **Step 2: 运行，确认失败**

Run: `cd backend && python3 -m pytest tests/unit/test_router.py -q -k "protect_regions"`
Expected: FAIL（`protect_regions` 不在 captured）。

- [ ] **Step 3: branch-refine 筛 protect 区域并入 pipeline_extra**

在 branch-refine 端点构建 `pipeline_extra` 的 dict（~1210，含 `"human_feedback_notes": notes`）里加：

```python
                "human_feedback_notes": notes,
                # ...
                "protect_regions": (
                    [r for r in _constraint_spec.regions if r.mode == "protect"]
                    if _constraint_spec is not None else []
                ),
```

- [ ] **Step 4: explore-variants 同样处理**

在 explore-variants 给每个变体 worker 构建 `pipeline_extra` 的位置（解析出 `_ev_constraint_spec` 之后），同样加：

```python
                "protect_regions": (
                    [r for r in _ev_constraint_spec.regions if r.mode == "protect"]
                    if _ev_constraint_spec is not None else []
                ),
```

- [ ] **Step 5: `_run_png_shader_background` 透传进 pipeline**

在 `_run_png_shader_background` 调用 `run_png_shader_pipeline(...)`（~632）处，从 `pipeline_extra` 取出并传入：

```python
                pipeline_result = run_png_shader_pipeline(
                    # ...existing args...
                    protect_regions=pipeline_extra.get("protect_regions"),
                )
```

> 确认 `pipeline_extra` 变量在该作用域可见（worker 既有透传 `human_feedback_notes` 等的同一 dict）；若 worker 用 `**pipeline_extra` 展开，则只需保证 `run_png_shader_pipeline` 接受 `protect_regions`（Task 5 已加），无需改这一行——读代码确认采用哪种方式。

- [ ] **Step 6: 运行测试，确认通过 + 全量回归**

Run: `cd backend && python3 -m pytest tests/unit/test_router.py -q && cd backend && python3 -m pytest tests/unit/ -q`
Expected: PASS（新透传用例 + 全量 1135+，无 FAIL）。

- [ ] **Step 7: 提交**

```bash
git add backend/app/routers/png_shader.py backend/tests/unit/test_router.py
git commit -m "feat(region-veto): forward protect regions from branch-refine/explore-variants into pipeline"
```

---

## Task 7: 全量门禁 + 收尾

**Files:** 无（仅验证）

- [ ] **Step 1: 后端全量**

Run: `cd backend && python3 -m pytest tests/unit/ -q`
Expected: PASS，0 失败。记录新总数（应为旧 1124 + 本次新增用例）。

- [ ] **Step 2: 前端门禁（确认未受影响）**

Run: `cd frontend && npm run build && npx vitest run`
Expected: build clean；vitest 125 passed（本特性纯后端，前端应不变）。

- [ ] **Step 3: 冒烟（可选，本地）**

带一个 protect 区域跑一次 branch-refine，确认 `refinement_summary.json` 出现 `protect_regions` 字段、且被否决的迭代 history 含 `rejected_reason="protect_region_veto"`。

- [ ] **Step 4: 最终提交（若有未提交收尾）**

```bash
git add -A backend
git commit -m "test(region-veto): full-suite green for protect-region hard-veto"
```

---

## Self-Review（计划自检结果）

- **Spec 覆盖：** D1 硬否决→Task3/4 否决块（命中即 continue，凌驾 directed，见 test_glsl_veto_overrides_directed_acceptance）；D2 两条循环→Task3+4；D3 基准=精修前渲染→Task5 `selected.render_path`/兜底渲染；D4/D5 per-region strength 阈值→Task1 `protect_region_threshold`+`evaluate_protect_veto`；D6 回滚+反馈+计数不终止→Task3/4 否决块；D7 可观测→Task3/4 history 字段 + Task5 summary；D8 no-op/兼容→所有循环用例不传 `region_veto_fn` 保持绿 + Task5 `region_veto_fn=None`；§9 配置→Task2；§10 测试→Task1/3/4/6 全覆盖；§11 文件→Task1-6 全覆盖。
- **占位符：** 无 TBD/TODO；每步含可运行代码/命令与期望输出。
- **类型一致：** `RegionVetoResult`(vetoed/constraint_score/regions/reason/evaluated)、`evaluate_protect_veto`、`protect_region_threshold`、`region_veto_fn` 在 Task1 定义、Task3/4/5 引用一致；`rejected_reason="protect_region_veto"` 三处一致。
- **已知实现待确认点（执行时读码对齐，非占位符）：** ① `selected.render_path` 对 DSL 是否已填（Task5 已含兜底渲染）；② `_evaluate_by_r_with_render` 给分约定（Task3 注明对齐既有用例）；③ worker 是否 `**pipeline_extra` 展开（Task6 Step5 注明两种写法）。
