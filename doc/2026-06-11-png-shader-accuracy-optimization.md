# P2S-Agent PNG-to-Shader 准确率优化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **适配说明（2026-06-12）：** 本文档适配自 VFX-Agent 的《png-shader 准确率优化计划》。P2S-Agent 重构建仓时已把原计划 **Phase 1 全部落地**（指标 v2、渲染器向量化、精修基线降采样、两停渐变 parity 修复、v2 评分公式、背景色透传、分区反馈注入），对应任务在本版中收敛为核查清单。**Phase 2（decompose + 残差增层）与 Phase 3（VLM 评判）未实现**——`tests/unit/test_decompose.py` 已预置为红灯测试，等待 `app.pipeline.decompose` 落地。本版同时适配了 P2S 的模块布局（见清理计划的映射表）与 **LangGraph 架构**（接线点从单体 `run_png_shader_pipeline` 改为 `node_selection` / `_run_post_pipeline` / state 透传），并在 Phase 0 新建 P2S 缺失的 E2E 脚手架。

**Goal:** 系统性提升 PNG→GLSL 的还原准确率：在已落地的指标 v2 之上，补齐打分阈值校准、"看得懂图"的构造前端（分解候选 + 残差增层）、并在关键决策点引入 VLM 评判。

**Architecture:** 三个独立可交付的阶段。Phase 0 搭建 E2E 验收脚手架并记录基线；Phase 1 核查已落地的指标 v2 并补校准脚本；Phase 2 新增基于颜色量化 + 连通域 + 图元拟合的 decompose 候选，并在优化阶段后增加残差驱动的贪心增层；Phase 3 新增 VLM 评判模块，仅在近平局裁决、精修仲裁、终稿门禁三个决策点调用，失败一律静默降级回客观分。

**Tech Stack:** Python 3.9+ / NumPy / Pillow / OpenCV（均已在 requirements.txt）/ 现有 `app/llm/client.py` 的 BaseAgent（Phase 3）。

**前置条件：** 先完成《[2026-06-11-png-shader-cleanup-refactor.md](2026-06-11-png-shader-cleanup-refactor.md)》（迁移收尾计划）——它修复测试安全网（当前 `pytest tests/unit/` 无法收集）与 3 处生产坏 import，并完成 git init。本计划所有任务以该计划完成、全量单测绿色为起点。

---

## 问题 → 方案总览（含 P2S 落地状态，2026-06-12 核查）

| # | 问题（来源：三轮分析） | 解决方案 | 状态 | 任务 |
|---|---|---|---|---|
| P1 | 指标为纯 Python 双重循环，慢 ~100× | NumPy 向量化全部指标 | ✅ 已落地 `app/metrics/compute.py` | 核查 1 |
| P2 | MSE/SSIM/直方图比较前丢弃 alpha | composite 到 canvas 背景色 | ✅ 已落地（同上 + `scoring.py:146` 透传） | 核查 1 |
| P3 | 5 指标中 3 个位置盲 | 新增 mask IoU、边缘 IoU、8×8 分块 Lab ΔE | ✅ 已落地 | 核查 1 |
| P4 | MSE 归一后饱和 | RMSE 且 v2 公式降权至 0.10 | ✅ 已落地 `quality_router.py:95-110` | 核查 1 |
| P5 | router 四档阈值为 v1 手拍，与 v2 公式失配 | 校准脚本按分数分布重定阈值 | ❌ 未做 | Task 4 |
| P6 | 确定性候选只用全局标量猜场景，几何先天不准 | decompose 候选：量化 + 连通域 + 图元 IoU 拟合 | ❌ 未做（测试已预置） | Task 5, 6 |
| P7 | 层结构一次性生成，无"哪里不像补哪里"机制 | 残差驱动贪心增层，逐层接受/拒绝 | ❌ 未做 | Task 7, 8 |
| P8 | 参数优化靠高斯随机扰动 | 增层参数解析初始化（质心/二阶矩/外接矩形） | ❌ 未做 | Task 5, 7 |
| P9 | failure_type 靠分数段瞎猜 | VLM rubric 评判输出 failure_type + revision_hints | ❌ 未做 | Task 9, 10 |
| P10 | 客观分近平局时选优是掷硬币 | VLM pairwise 裁决（换序去偏置） | ❌ 未做 | Task 9, 10 |
| P11 | "刷分解"无人拦截 | 终稿门禁 rubric 复核，`objective×0.7 + semantic×0.3` 混合 | ❌ 未做（公式已预留 `compute_final_score(metrics, semantic_scores)`） | Task 10 |
| P12 | 精修循环只喂全局指标数字 | `grid_color_report()` 分区反馈注入精修循环 | ✅ 已落地 `refinement.py:203-219` | 核查 1 |
| P13 | dsl_renderer 逐像素纯 Python 三重循环 | NumPy 向量化渲染器 + golden 表征测试 | ✅ 已落地 `app/dsl/renderer.py` + `tests/unit/golden/` | 核查 1 |
| P14 | 精修基线渲染在全画布分辨率 | 改用 `_metric_render_size` 有界尺寸 | ✅ 已落地 `refinement.py:129` | 核查 1 |
| P15 | 两停渐变 parity bug（渲染器 smoothstep vs 编译器线性 mix） | 编译器 2-stop 分支改 smoothstep | ✅ 已落地 `app/dsl/compiler.py:281-287` | 核查 1 |

**各阶段独立可交付**：Phase 1 收尾（Task 4）完成即可单独合入；Phase 2/3 各自同理。每个 Phase 结束跑一次 E2E batch 与 Phase 0 基线对比，按"单变量实验"原则决定合入或回滚。

## P2S 适配要点（执行者必读）

1. **模块路径**一律按清理计划的《VFX → P2S 模块映射表》解析；本文中的路径已全部换算为 P2S 路径。
2. **架构差异**：P2S 的 `app/pipeline/graph.py` 是 LangGraph StateGraph（preprocess→candidates→scoring→selection 四节点）+ `_run_post_pipeline`（优化/修订/精修，普通函数）。配置流向为：`run_png_shader_pipeline` 读取 quality_config + `strategy_clamp` → 写入 `write_manifest` 与 `initial_state` → 节点和 `_run_post_pipeline` 从 `state` 读取。因此新增配置键需要**四处**同步：`strategy_config.json`、`run_png_shader_pipeline` 读取段、`initial_state` dict、`app/state.py` 的 `P2SPipelineState`（TypedDict, total=False，漏写字段不报错但会失去类型提示）。
3. **接线锚点**：近平局裁决在 `node_selection`（graph.py:124）；残差增层与终稿门禁在 `_run_post_pipeline`；精修仲裁在 `refinement.py` 的 `run_dsl_refinement_loop`。`_run_post_pipeline` 内**没有** progress_callback，原计划的进度上报语句一律省去。
4. **测试**：单测在 `backend/tests/unit/` 扁平命名（无 png_shader 子目录）。`test_decompose.py` 已预置且带 importorskip 守卫（清理计划 Task 4），Task 5 落地模块后守卫自动放行。
5. **E2E**：P2S 没有继承 VFX 的 E2E 脚手架（VFX 版依赖 `/pipeline/*` 端点与本机外的样本目录），Phase 0 新建轻量版，走 P2S 的 `POST /png-shader/run`（端口 8001）。
6. **文档载体**：原计划中所有"更新 CLAUDE.md"改为更新 `README.md`；基线文档放 `doc/`。

---

## Phase 0：E2E 脚手架与基线测量

### Task 0a: 新建轻量 E2E batch 脚手架

**Files:**
- Create: `backend/tests/e2e/__init__.py`（空文件）
- Create: `backend/tests/e2e/run_batch.py`
- Create: `backend/tests/e2e/samples/*.png`

- [ ] **Step 1: 写 run_batch.py**

```python
"""Minimal E2E batch harness for P2S-Agent.

Usage:
  cd backend && python tests/e2e/run_batch.py                # all samples
  cd backend && python tests/e2e/run_batch.py circle box     # named samples

Prereq: backend running (./start.sh start), samples in tests/e2e/samples/*.png.
Writes a JSON report next to this file (report_latest.json).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8001"
SAMPLES_DIR = Path(__file__).parent / "samples"
REPORT_PATH = Path(__file__).parent / "report_latest.json"
TIMEOUT_S = 600
POLL_INTERVAL_S = 2


def run_sample(path: Path) -> tuple[str, dict]:
    with path.open("rb") as f:
        resp = httpx.post(
            f"{BASE_URL}/png-shader/run",
            files={"image": (path.name, f, "image/png")},
            timeout=30,
        )
    resp.raise_for_status()
    run_id = resp.json()["run_id"]
    deadline = time.monotonic() + TIMEOUT_S
    while time.monotonic() < deadline:
        status = httpx.get(f"{BASE_URL}/png-shader/status/{run_id}", timeout=10).json()
        if status.get("status") in {"completed", "failed"}:
            return run_id, status
        time.sleep(POLL_INTERVAL_S)
    return run_id, {"status": "timeout"}


def main() -> None:
    names = sys.argv[1:]
    samples = (
        [SAMPLES_DIR / f"{n}.png" for n in names]
        if names
        else sorted(SAMPLES_DIR.glob("*.png"))
    )
    missing = [p for p in samples if not p.exists()]
    if missing or not samples:
        print(f"no usable samples in {SAMPLES_DIR} (missing: {[p.name for p in missing]})")
        return

    rows = []
    for path in samples:
        run_id, status = run_sample(path)
        score = (status.get("quality_router") or {}).get("final_score")
        rows.append({
            "sample": path.stem,
            "run_id": run_id,
            "status": status.get("status"),
            "final_score": score,
            "selected_source": (status.get("scoreboard") or {}).get("selected_source"),
        })
        print(f"{path.stem:28s} {str(status.get('status')):10s} score={score}")

    scored = [r["final_score"] for r in rows if isinstance(r["final_score"], (int, float))]
    n_pass = sum(1 for s in scored if s >= 0.85)
    n_acceptable = sum(1 for s in scored if 0.55 <= s < 0.85)
    summary = {
        "n": len(rows),
        "avg_final_score": round(sum(scored) / len(scored), 4) if scored else None,
        "pass": n_pass,
        "acceptable": n_acceptable,
        "fail": len(rows) - n_pass - n_acceptable,
        "rows": rows,
    }
    REPORT_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"\nn={summary['n']} avg={summary['avg_final_score']} "
        f"PASS(>=0.85)={n_pass} ACCEPTABLE(>=0.55)={n_acceptable} FAIL={summary['fail']}"
        f"\nreport -> {REPORT_PATH}"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 生成合成样本 + 补充真实样本**

用 DSL fixtures 经渲染器生成 4 个合成样本：

```bash
cd backend && python - <<'EOF'
from pathlib import Path
from app.dsl.renderer import render_dsl_to_image
from app.dsl.schema import (
    FIXTURE_BOX_GRADIENT, FIXTURE_CIRCLE_SOLID,
    FIXTURE_GLOW_RING, FIXTURE_ROUNDEDBOX_VIGNETTE,
)
out = Path("tests/e2e/samples"); out.mkdir(parents=True, exist_ok=True)
cases = {
    "circle": FIXTURE_CIRCLE_SOLID, "box-gradient": FIXTURE_BOX_GRADIENT,
    "glow-ring": FIXTURE_GLOW_RING, "roundedbox-vignette": FIXTURE_ROUNDEDBOX_VIGNETTE,
}
for name, dsl in cases.items():
    render_dsl_to_image(dsl, out / f"{name}.png", width=512, height=512)
print(sorted(p.name for p in out.glob("*.png")))
EOF
```

**注意**：合成样本是渲染器自产，分数偏乐观，主要用于跨 Phase 的回归对比。请再人工拷入 6–10 张**真实 PNG**（扁平图标、简单插画、带渐变的 logo 等）到 `tests/e2e/samples/`，它们才是衡量真实还原率的样本。样本一旦入库**不要再改动**，保证三个 Phase 的对比口径一致。

- [ ] **Step 3: 冒烟验证**（前置：`./start.sh start` 后端已运行）

Run: `cd backend && python tests/e2e/run_batch.py circle`
Expected: 打印 circle 的 status=completed 与分数，生成 report_latest.json

- [ ] **Step 4: Commit**

```bash
git add backend/tests/e2e/
git commit -m "test(e2e): add minimal batch harness and sample set for accuracy tracking"
```

### Task 0b: 记录优化前基线

**Files:**
- Create: `doc/2026-06-12-accuracy-baseline.md`

- [ ] **Step 1: 确认单元测试全绿**

Run: `cd backend && python -m pytest tests/unit/ -q`
Expected: 全部 PASS（test_decompose skipped）。如有失败说明清理计划未完成，先回去完成。

- [ ] **Step 2: 跑 E2E batch 记录基线**（前置：后端运行中）

Run: `cd backend && python tests/e2e/run_batch.py`

- [ ] **Step 3: 把 report_latest.json 的汇总写入基线文档**

```markdown
# 准确率基线（优化前，2026-06-12）
- 样本数: N
- avg_final_score: X.XXXX
- PASS(≥0.85): n1 / ACCEPTABLE(≥0.55): n2 / FAIL: n3
- 每样本分数: <粘贴 rows>
```

- [ ] **Step 4: Commit**

```bash
git add doc/2026-06-12-accuracy-baseline.md
git commit -m "docs: record accuracy baseline before optimization"
```

---

## Phase 1：指标 v2（已落地）核查 + 阈值校准

### 核查 1: 确认指标 v2 全链路在位（原 Task 1/2/2b/2c/2d/3，重构时已落地）

逐项验证，不改代码；任何一项不符即停止并排查（说明重构版本与预期有偏差）：

- [ ] `backend/requirements.txt` 含 `numpy>=1.26` 与 `opencv-python-headless>=4.9`
- [ ] `app/metrics/compute.py` 为 NumPy v2 实现，导出 `rmse`/`mask_iou`/`edge_iou`/`grid_color_sim` 新键与全部 v1 旧键，且有 `grid_color_report`、`hex_to_rgb01`
  Run: `cd backend && python -m pytest tests/unit/test_metrics.py -q` → 全 PASS（含 5 个 `test_v2_*`）
- [ ] `app/dsl/renderer.py` 为向量化实现，golden 表征测试在位
  Run: `cd backend && python -m pytest tests/unit/test_dsl_renderer_golden.py -v` → 全 PASS（golden PNG 在 `tests/unit/golden/`）
- [ ] `app/pipeline/refinement.py:128-129` 基线渲染使用 `_metric_render_size`（非全画布）
- [ ] `app/dsl/compiler.py:281-287` 的 2-stop 渐变分支生成 `smoothstep(...)` 而非线性 `mix`
  Run: `cd backend && python -m pytest tests/unit/test_compiler.py -q` → 全 PASS
- [ ] `app/metrics/quality_router.py` 的 `compute_final_score` 检测到 `mask_iou` 走 v2 公式（ssim 0.30 / grid 0.25 / miou 0.20 / eiou 0.15 / (1-rmse) 0.10），无 v2 键回落 v1
  Run: `cd backend && python -m pytest tests/unit/test_quality_router.py -q` → 全 PASS
- [ ] `app/pipeline/scoring.py:146` 将 DSL canvas 背景色经 `hex_to_rgb01` 透传给 `compute_objective_metrics`
- [ ] `app/pipeline/refinement.py:203-219` 在 `generate_llm_refinement` 前注入 `grid_color_report` 分区反馈

### Task 4: 阈值校准脚本 + Phase 1 验收

**Files:**
- Create: `backend/scripts/calibrate_thresholds.py`

- [ ] **Step 1: 写校准脚本**

```python
"""Suggest quality-router band cutoffs from saved run artifacts.

Usage:  cd backend && python scripts/calibrate_thresholds.py [--results-dir test_results]

Collects selected-candidate final_score from every run's quality_router.json
and prints distribution percentiles to inform the 0.85/0.70/0.55/0.40 bands.
"""
import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="test_results")
    args = parser.parse_args()

    scores: list[float] = []
    for qr_path in sorted(Path(args.results_dir).glob("*/quality_router.json")):
        try:
            data = json.loads(qr_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, dict) and "final_score" in data:
            scores.append(float(data["final_score"]))

    if len(scores) < 10:
        print(f"only {len(scores)} runs found in {args.results_dir!r} — need >= 10")
        return

    scores.sort()

    def pct(p: float) -> float:
        return scores[min(len(scores) - 1, int(p * len(scores)))]

    print(f"n={len(scores)}  mean={statistics.mean(scores):.4f}  median={pct(0.5):.4f}")
    print(f"p90 (suggest 'excellent' cut): {pct(0.90):.4f}")
    print(f"p70 (suggest 'good' cut):      {pct(0.70):.4f}")
    print(f"p40 (suggest 'acceptable' cut):{pct(0.40):.4f}")
    print(f"p15 (suggest 'poor' floor):    {pct(0.15):.4f}")


if __name__ == "__main__":
    main()
```

（结果根目录是 `backend/test_results/`——`app/pipeline/artifacts.py:40` 的 `DEFAULT_RESULTS_ROOT`；每个 run 目录直接落 `quality_router.json`。）

- [ ] **Step 2: 跑 E2E 重建分数分布并校准**（前置：后端运行中）

Run: `cd backend && python tests/e2e/run_batch.py && python scripts/calibrate_thresholds.py`
Expected: 打印 n、mean 与四个分位数

- [ ] **Step 3: 按输出更新 router 阈值**

对照脚本输出，若分布明显偏移（例如 p90 落在 0.78 而不是 0.85），更新 `app/metrics/quality_router.py:215-251` 的四个分档常量（0.85/0.70/0.55/0.40），并同步更新 README 中相关描述与 `tests/e2e/run_batch.py` 的 PASS/ACCEPTABLE 阈值。**若分布与现有档位基本吻合则不改**——不要为改而改。

- [ ] **Step 4: 对比基线并记录**

把本次 E2E 平均分/通过数追加到 `doc/2026-06-12-accuracy-baseline.md` 的 "Phase 1 后" 小节。

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/calibrate_thresholds.py backend/app/metrics/quality_router.py README.md doc/2026-06-12-accuracy-baseline.md
git commit -m "feat(metrics): add threshold calibration script, recalibrate router bands"
```

---

## Phase 2：构造式分解候选 + 残差增层

### 注意事项（Phase 2）

1. **DSL 的 polygon 是正多边形**（`radius` + `sides`，见 `app/dsl/compiler.py`），**不能表达任意轮廓**。因此图元拟合只映射 circle/ellipse/box；拟合不佳时用矩量椭圆兜底。
2. **DSL 的 ellipse 不支持旋转**，倾斜椭圆只接受角度在 0°/90° ±15° 内的拟合，否则降级为矩量椭圆。
3. 分解在 256×256 正方形分析分辨率上进行，非正方形输入的 radius 归一化存在轻微纵横比误差——可接受（renderer 的 uv 同样是 0–1 归一），E2E 中宽高比悬殊的样本若退化优先查这里。
4. **残差增层会增加 layer_count**，与 `protected_aspects` 的 `layer_count` 语义冲突。这是有意的：protected_aspects 约束"修改已有结构的阶段"（optimizer/LLM 精修），残差增层是独立构造阶段，不受该约束。在 artifacts 的 `residual.json` 中记录每次增层。
5. 增层接受门槛 `ACCEPT_MIN_DELTA=0.003` 偏保守；若 E2E 显示增层全被拒绝，先检查渲染分辨率是否过低。
6. decompose 候选对 photo-like 输入（`photo_like_score > 0.7`）直接跳过。
7. **`tests/unit/test_decompose.py` 已预置**（含 importorskip 守卫），Task 5 不需要再写测试文件，落地模块后守卫自动放行——这正是本任务的红灯。

### Task 5: decompose.py — 量化 + 连通域 + 图元拟合

**Files:**
- Create: `backend/app/pipeline/decompose.py`
- Test: `backend/tests/unit/test_decompose.py`（已预置，无需创建）

- [ ] **Step 1: 确认预置测试当前为 skip（即红灯）**

Run: `cd backend && python -m pytest tests/unit/test_decompose.py -q`
Expected: 全部 skipped（`app.pipeline.decompose` 不存在）

- [ ] **Step 2: 实现 decompose.py**

```python
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


def _palette_hex(palette: list[int], label: int) -> str:
    r, g, b = palette[label * 3: label * 3 + 3]
    return f"#{r:02x}{g:02x}{b:02x}"


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
            candidates.append((
                _iou(comp_mask, cand.astype(bool)),
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
    for label in range(max_colors):
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
```

- [ ] **Step 3: 运行预置测试确认通过**

Run: `cd backend && python -m pytest tests/unit/test_decompose.py -q`
Expected: 3 PASS（importorskip 守卫自动放行）

- [ ] **Step 4: Commit**

```bash
git add backend/app/pipeline/decompose.py
git commit -m "feat(pipeline): add structural decomposition (quantize + components + primitive fit)"
```

### Task 6: decompose 候选注册进候选池

**Files:**
- Create: `backend/app/candidates/decompose.py`
- Modify: `backend/app/pipeline/pool.py`（import 区 + `run_candidate_pool`）
- Test: `backend/tests/unit/test_decompose.py`（追加）

- [ ] **Step 1: 写失败测试（追加到 test_decompose.py 末尾）**

```python
def test_decompose_candidate_skips_photo_like(tmp_path):
    from app.candidates.decompose import generate_decompose_candidate
    img = Image.new("RGB", (64, 64), (255, 255, 255))
    ImageDraw.Draw(img).ellipse((16, 16, 48, 48), fill=(255, 0, 0))
    path = _save(tmp_path, "p.png", img)

    assert generate_decompose_candidate({"photo_like_score": 0.9}, path) is None

    dsl = generate_decompose_candidate({"photo_like_score": 0.1}, path)
    assert dsl is not None
    assert dsl["_meta"] == {"source": "decompose", "priority": 1}
```

- [ ] **Step 2: 运行确认失败**

Run: `cd backend && python -m pytest tests/unit/test_decompose.py -q`
Expected: 新测试 FAIL（`ModuleNotFoundError: app.candidates.decompose`）

- [ ] **Step 3: 实现 app/candidates/decompose.py**

```python
"""Decomposition-based candidate: measured geometry via color quantization
+ connected components + primitive fitting (see app.pipeline.decompose)."""

from __future__ import annotations

import logging
from pathlib import Path

from app.pipeline.decompose import DECOMPOSE_AVAILABLE, decompose_to_dsl

logger = logging.getLogger(__name__)

PHOTO_LIKE_SKIP_THRESHOLD = 0.7


def generate_decompose_candidate(
    preprocess: dict,
    image_path: "str | Path | None",
    canvas_width: int = 512,
    canvas_height: int = 512,
) -> dict | None:
    if not DECOMPOSE_AVAILABLE or image_path is None:
        return None
    # photo-like inputs decompose into fragments — leave them to other candidates
    if float(preprocess.get("photo_like_score", 0.0)) > PHOTO_LIKE_SKIP_THRESHOLD:
        return None
    dsl = decompose_to_dsl(image_path, canvas_width, canvas_height)
    if dsl is None:
        return None
    dsl["_meta"] = {"source": "decompose", "priority": 1}
    return dsl
```

- [ ] **Step 4: pool.py 注册候选**

import 区（`from app.candidates.baseline import ...` 之后，pool.py:13-16 一带）添加：

```python
from app.candidates.decompose import generate_decompose_candidate
```

`run_candidate_pool` 中，"1b. Rule" 块结束（`candidates_raw.append(("rule_0", "rule", 1, rule_dsl, "dsl"))`，pool.py:126）之后、CV 候选块之前插入：

```python
    # 1b2. Decompose — measured-geometry candidate (needs opencv)
    if image_path is not None:
        try:
            dec_dsl = generate_decompose_candidate(
                preprocess, image_path, canvas_width, canvas_height
            )
            if dec_dsl is not None:
                candidates_raw.append(("decompose_0", "decompose", 1, dec_dsl, "dsl"))
                logger.info(
                    "candidate generated: source=decompose layers=%d",
                    len(dec_dsl.get("layers", []) or []),
                )
        except Exception:
            logger.warning("decompose candidate failed", exc_info=True)
```

- [ ] **Step 5: 运行测试**

Run: `cd backend && python -m pytest tests/unit/test_decompose.py tests/unit/test_graph.py tests/unit/test_candidates.py -q`
Expected: 全 PASS（如有"候选数量"硬断言，按 +1 修正）

- [ ] **Step 6: Commit**

```bash
git add backend/app/candidates/decompose.py backend/app/pipeline/pool.py backend/tests/unit/test_decompose.py
git commit -m "feat(candidates): register decompose candidate in the pool"
```

### Task 7: residual_layers.py — 残差驱动增层

**Files:**
- Create: `backend/app/pipeline/residual_layers.py`
- Create: `backend/tests/unit/test_residual_layers.py`

- [ ] **Step 1: 写失败测试**

```python
"""Tests for residual-driven layer addition."""
import pytest

cv2 = pytest.importorskip("cv2")

import numpy as np
from PIL import Image, ImageDraw

from app.dsl.renderer import render_dsl_to_image
from app.pipeline.residual_layers import add_residual_layers


def _ref_two_circles(tmp_path):
    img = Image.new("RGBA", (128, 128), (0, 0, 0, 255))
    d = ImageDraw.Draw(img)
    d.ellipse((16, 16, 64, 64), fill=(255, 0, 0, 255))
    d.ellipse((80, 80, 120, 120), fill=(0, 255, 0, 255))
    path = tmp_path / "ref.png"
    img.save(path)
    return path


def _dsl_one_circle():
    return {
        "schema_version": 1,
        "canvas": {"width": 128, "height": 128, "background": "#000000"},
        "layers": [{
            "id": "c0", "type": "circle",
            "fill": {"type": "solid", "color": "#ff0000"},
            "params": {"center": [0.3125, 0.3125], "radius": 0.1875},
            "opacity": 1.0,
        }],
    }


def test_residual_adds_missing_circle(tmp_path):
    ref_path = _ref_two_circles(tmp_path)
    ref = np.asarray(Image.open(ref_path).convert("RGB"), dtype=np.float32) / 255.0
    counter = {"n": 0}

    def render_fn(dsl):
        counter["n"] += 1
        out = tmp_path / f"r{counter['n']}.png"
        return render_dsl_to_image(dsl, out, width=128, height=128)

    def score_fn(dsl):
        path = render_fn(dsl)
        rnd = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        return 1.0 - float(np.abs(ref - rnd).mean())

    result = add_residual_layers(
        _dsl_one_circle(), ref_path, score_fn=score_fn, render_fn=render_fn, max_added=3
    )

    assert result.layers_added >= 1
    assert result.final_score > result.initial_score
    new_ids = [l["id"] for l in result.final_dsl["layers"]]
    assert any(i.startswith("res_") for i in new_ids)
```

- [ ] **Step 2: 运行确认失败**

Run: `cd backend && python -m pytest tests/unit/test_residual_layers.py -q`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 实现 residual_layers.py**

```python
"""Residual-driven layer addition (geometrize-style greedy construction).

After base optimization, repeatedly: render the current DSL, locate the
region deviating most from the reference, fit a primitive to that region
of the *reference* (analytic initialization — no random search), and keep
the new layer only when the objective score improves.

Unlike the optimizer/revision stages this stage intentionally changes
layer_count: it is a construction stage, not a mutation stage, so the
protected_aspects contract does not apply here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from app.pipeline.decompose import fit_primitive_layer

logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:  # pragma: no cover - environment-dependent
    cv2 = None

RESIDUAL_SIZE = 128       # residual analysis resolution
MIN_REGION_FRAC = 0.004   # ignore hot regions smaller than this
ACCEPT_MIN_DELTA = 0.003  # required score gain to keep a new layer


@dataclass
class ResidualAddResult:
    final_dsl: dict
    initial_score: float
    final_score: float
    layers_added: int
    log: list[dict] = field(default_factory=list)


def _load_rgb01(path, size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def add_residual_layers(
    dsl: dict,
    reference_path: "str | Path",
    *,
    score_fn,
    render_fn,
    max_added: int = 4,
    max_layers_total: int = 12,
) -> ResidualAddResult:
    """Greedily add primitives where the render deviates most from the reference.

    Args:
        dsl: Starting DSL (not mutated).
        reference_path: Reference image path.
        score_fn: callable(dsl) -> float, higher is better.
        render_fn: callable(dsl) -> Path | None rendering the DSL to an image.
        max_added: Max number of layers to add.
        max_layers_total: Hard cap on total layer count (shader budget guard).
    """
    initial_score = score_fn(dsl)
    if cv2 is None:
        return ResidualAddResult(dsl, initial_score, initial_score, 0)

    ref = _load_rgb01(reference_path, RESIDUAL_SIZE)
    current = dsl
    current_score = initial_score
    log: list[dict] = []

    for step in range(max_added):
        if len(current.get("layers", [])) >= max_layers_total:
            break
        render_path = render_fn(current)
        if render_path is None:
            break
        rnd = _load_rgb01(render_path, RESIDUAL_SIZE)
        residual = np.abs(ref - rnd).mean(axis=-1)
        residual = cv2.blur(residual, (5, 5))

        threshold = max(float(residual.mean() + 2.0 * residual.std()), 0.10)
        hot = (residual > threshold).astype(np.uint8)
        n, comp_map, stats, _ = cv2.connectedComponentsWithStats(hot, connectivity=8)
        if n <= 1:
            break
        comp = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        area_frac = stats[comp, cv2.CC_STAT_AREA] / float(hot.size)
        if area_frac < MIN_REGION_FRAC:
            break
        region_mask = comp_map == comp

        mean_color = (ref[region_mask].mean(axis=0) * 255.0 + 0.5).astype(int)
        color_hex = "#{:02x}{:02x}{:02x}".format(*np.clip(mean_color, 0, 255))
        layer = fit_primitive_layer(region_mask, color_hex=color_hex)
        if layer is None:
            break
        layer["id"] = f"res_{step:02d}_{layer['type']}"

        candidate = {**current, "layers": [*current["layers"], layer]}
        new_score = score_fn(candidate)
        accepted = new_score >= current_score + ACCEPT_MIN_DELTA
        log.append({
            "step": step + 1,
            "layer_id": layer["id"],
            "layer_type": layer["type"],
            "area_frac": round(float(area_frac), 4),
            "score_before": round(current_score, 4),
            "score_after": round(new_score, 4),
            "accepted": accepted,
        })
        logger.info(
            "residual layer step=%d layer=%s before=%.4f after=%.4f accepted=%s",
            step + 1, layer["id"], current_score, new_score, accepted,
        )
        if not accepted:
            break
        current = candidate
        current_score = new_score

    return ResidualAddResult(
        final_dsl=current,
        initial_score=initial_score,
        final_score=current_score,
        layers_added=sum(1 for e in log if e["accepted"]),
        log=log,
    )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend && python -m pytest tests/unit/test_residual_layers.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/residual_layers.py backend/tests/unit/test_residual_layers.py
git commit -m "feat(pipeline): residual-driven greedy layer addition"
```

### Task 8: 残差增层接入流水线 + 配置键

**Files:**
- Modify: `backend/app/strategy_config.json`
- Modify: `backend/app/state.py`
- Modify: `backend/app/pipeline/graph.py`

- [ ] **Step 1: strategy_config.json 的 params 中添加（`refinement_patience` 之后）**

```json
    "max_added_layers": {
      "default": 4,
      "min": 0,
      "max": 8,
      "step": 1,
      "integer": true,
      "label": "残差增层上限",
      "description": "Residual layer budget: 残差驱动增层阶段最多新增的图层数，0 表示关闭该阶段"
    }
```

并在 4 个 preset 中各加一行：`fast: 0`、`balanced: 4`、`quality: 6`、`aggressive: 6`。

- [ ] **Step 2: state.py 补字段**

在 `P2SPipelineState` 的 `refinement_patience: int` 之后添加：

```python
    max_added_layers: int
```

- [ ] **Step 3: graph.py 读取配置并透传 state**

import 区添加：

```python
from app.pipeline.residual_layers import add_residual_layers
```

`run_png_shader_pipeline` 配置读取段（`refinement_patience = ...` 块之后，graph.py:571 一带）添加：

```python
    max_added_layers = int(
        strategy_clamp(
            "max_added_layers",
            int(quality_config.get("max_added_layers", get_default("max_added_layers"))),
        )
    )
```

并把 `"max_added_layers": max_added_layers,` 同时加入 `write_manifest` 的 config dict 与 `initial_state` dict（两处都要，遗漏 initial_state 会导致 `_run_post_pipeline` 读到默认 0、增层永不触发）。

- [ ] **Step 4: _run_post_pipeline 接入**

在 `_run_post_pipeline` 中，revision 接受块（`if rev_result.success and rev_result.improved:` 整块，graph.py:287-299）结束之后、`# GLSL optimizer` 的 `elif` 之前，**仍在 `if selected.dsl and selected_quality:` 块内**追加：

```python
        # Residual-driven layer addition: construct what optimization can't fix.
        max_added_layers = int(state.get("max_added_layers", 0))
        if max_added_layers > 0 and selected.final_score < float(
            state.get("refinement_high_score_stop", 0.95)
        ):
            res_dir = run_dir / "residual_layers"
            res_render_fn = _make_render_dsl_fn(
                res_dir / "renders",
                canvas_width=canvas_width,
                canvas_height=canvas_height,
            )
            _res_score = _make_revision_scorer(
                res_dir / "scores",
                canvas_width=canvas_width,
                canvas_height=canvas_height,
                max_shader_chars=max_shader_chars,
                protected_aspects=protected_aspects,
            )
            try:
                res_result = add_residual_layers(
                    selected.dsl,
                    reference_path,
                    score_fn=lambda d: _res_score(d, reference_path),
                    render_fn=lambda d: res_render_fn(d, ""),
                    max_added=max_added_layers,
                )
            except Exception:
                logger.exception("residual layer addition failed")
                res_result = None

            if res_result is not None:
                save_json(res_dir / "residual.json", {
                    "initial_score": res_result.initial_score,
                    "final_score": res_result.final_score,
                    "layers_added": res_result.layers_added,
                    "log": res_result.log,
                })
                if res_result.layers_added > 0 and res_result.final_score > selected.final_score:
                    accepted = _accept_improvement(
                        selected,
                        res_result.final_dsl,
                        reference_path,
                        res_dir / "residual_render.png",
                        canvas_width=canvas_width,
                        canvas_height=canvas_height,
                        max_shader_chars=max_shader_chars,
                        protected_aspects=protected_aspects,
                        reason=(
                            f"residual layers (+{res_result.layers_added}) improved score "
                            f"{res_result.initial_score:.4f} -> {res_result.final_score:.4f}"
                        ),
                    )
                    if accepted is not None:
                        selected_dsl, selected_glsl, selected_metrics, selected_quality = accepted
```

（接受块直接复用 `_accept_improvement`——P2S 的 scoring.py 已有该帮助函数，与 optimizer/revision 的写法一致。`_run_post_pipeline` 没有 progress_callback，原计划的进度上报省去。）

- [ ] **Step 5: 运行全套单测**

Run: `cd backend && python -m pytest tests/unit/ -q`
Expected: 全 PASS

- [ ] **Step 6: Phase 2 E2E 验收并记录**

Run: `cd backend && python tests/e2e/run_batch.py`
把结果追加到 `doc/2026-06-12-accuracy-baseline.md` 的 "Phase 2 后" 小节，重点看：decompose 候选被选中的比例（report 的 `selected_source`）、`residual.json` 的增层接受率、是否有样本退化。**若 avg 提升但某类样本明显退化，回滚 Task 8 的 graph.py 接线（保留模块与测试），单独排查。**

- [ ] **Step 7: Commit**

```bash
git add backend/app/strategy_config.json backend/app/state.py backend/app/pipeline/graph.py doc/2026-06-12-accuracy-baseline.md
git commit -m "feat(pipeline): wire residual layer addition into post-pipeline with config budget"
```

---

## Phase 3：VLM 评判接入

### 注意事项（Phase 3）

1. **VLM 绝不进入内循环**（优化器/增层的 score_fn）。只在三个决策点调用：近平局选优裁决、精修小增益仲裁、终稿门禁。单次 run 的 VLM 调用应 ≤ 5 次。
2. **所有失败静默降级**：无 API key、超时、JSON 解析失败、维度缺失 → 返回 None，流水线行为与未开启时完全一致。宁可丢弃整个评判，不要只取部分字段。
3. pairwise 必须**换序问两次**，两次结论一致才采纳，否则判 tie——VLM 有显著的位置偏置。
4. 评判结果按文件内容 hash 缓存（进程内），同一 run 内重复比较零成本。
5. `vlm_judge_enabled` 默认 0（关）。strategy_config 的 clamp 机制只支持数值，布尔用 0/1 表示。
6. 复用 `settings.llm` 模型配置（与 LLM 候选同一通道，`app/llm/client.py` 的 BaseAgent 已处理图像与代理）；评判用 `temperature=0.0`。**需要 `LLM_SUPPORTS_IMAGE=true` 的多模态模型**，否则评判静默降级。
7. **上线前先做一致率验证**（Task 11）：人工标 30 对，judge-human 一致率 < 80% 时不要开启 pairwise 裁决，先迭代 prompt。

### Task 9: vlm_judge.py — rubric 评分 + pairwise 比较

**Files:**
- Create: `backend/app/llm/vlm_judge.py`
- Create: `backend/tests/unit/test_vlm_judge.py`

- [ ] **Step 1: 写失败测试**

```python
"""Tests for the VLM judge (all using injected fake clients — no network)."""
import json

from PIL import Image

import app.llm.vlm_judge as vj
from app.llm.vlm_judge import judge_pairwise, judge_rubric


def _img(tmp_path, name, color):
    path = tmp_path / name
    Image.new("RGB", (32, 32), color).save(path)
    return path


def setup_function(_fn):
    vj._CACHE.clear()


def test_rubric_parses_valid_response(tmp_path):
    ref = _img(tmp_path, "ref.png", (255, 0, 0))
    rnd = _img(tmp_path, "rnd.png", (0, 0, 255))

    def fake(system_prompt, user_prompt, image_paths):
        assert image_paths and len(image_paths) == 1
        return json.dumps({
            "differences": ["color wrong"],
            "shape_fidelity": 0.9, "position_layout": 0.8,
            "color_fidelity": 0.2, "effects_fidelity": 0.5,
            "failure_type": "color",
            "revision_hints": ["change fill color to red"],
        })

    out = judge_rubric(ref, rnd, work_dir=tmp_path, judge_client=fake)
    assert out is not None
    assert out["semantic_scores"]["color_fidelity"] == 0.2
    assert out["failure_type"] == "color"
    assert out["revision_hints"] == ["change fill color to red"]


def test_rubric_malformed_response_degrades_to_none(tmp_path):
    ref = _img(tmp_path, "ref.png", (255, 0, 0))
    rnd = _img(tmp_path, "rnd.png", (0, 0, 255))
    out = judge_rubric(ref, rnd, work_dir=tmp_path, judge_client=lambda s, u, i: "not json at all")
    assert out is None


def test_pairwise_position_bias_yields_tie(tmp_path):
    ref = _img(tmp_path, "ref.png", (255, 0, 0))
    a = _img(tmp_path, "a.png", (250, 0, 0))
    b = _img(tmp_path, "b.png", (0, 0, 255))
    # client always answers "A" regardless of order -> contradictory -> tie
    out = judge_pairwise(ref, a, b, work_dir=tmp_path, judge_client=lambda s, u, i: '{"winner": "A"}')
    assert out == "tie"


def test_pairwise_consistent_winner(tmp_path):
    ref = _img(tmp_path, "ref.png", (255, 0, 0))
    a = _img(tmp_path, "a.png", (250, 0, 0))
    b = _img(tmp_path, "b.png", (0, 0, 255))
    calls = []

    def fake(system_prompt, user_prompt, image_paths):
        calls.append(1)
        # fwd panel order (A=a, B=b): a is closer -> "A"
        # rev panel order (A=b, B=a): a is closer -> "B"
        return '{"winner": "A"}' if len(calls) == 1 else '{"winner": "B"}'

    out = judge_pairwise(ref, a, b, work_dir=tmp_path, judge_client=fake)
    assert out == "A"
    assert len(calls) == 2


def test_pairwise_result_is_cached(tmp_path):
    ref = _img(tmp_path, "ref.png", (255, 0, 0))
    a = _img(tmp_path, "a.png", (250, 0, 0))
    b = _img(tmp_path, "b.png", (0, 0, 255))
    calls = []

    def fake(system_prompt, user_prompt, image_paths):
        calls.append(1)
        return '{"winner": "A"}' if len(calls) % 2 == 1 else '{"winner": "B"}'

    first = judge_pairwise(ref, a, b, work_dir=tmp_path, judge_client=fake)
    second = judge_pairwise(ref, a, b, work_dir=tmp_path, judge_client=fake)
    assert first == second == "A"
    assert len(calls) == 2, "second call must hit the cache"
```

- [ ] **Step 2: 运行确认失败**

Run: `cd backend && python -m pytest tests/unit/test_vlm_judge.py -q`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 实现 vlm_judge.py**

```python
"""VLM-as-judge: rubric scoring and pairwise comparison of renders vs reference.

Design contract:
- NEVER called from inner optimization loops — decision points only
  (near-tie candidate selection, refinement arbitration, final gate).
- Every failure path returns None so callers degrade to objective metrics.
- Pairwise comparisons ask twice with panel order swapped; disagreement = tie.
- Results are cached in-process by content hash.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw

from app.config import settings

logger = logging.getLogger(__name__)

JudgeClient = Callable[[str, str, "list[str] | None"], "str | dict | None"]

_CACHE: dict[str, object] = {}

PANEL_HEIGHT = 384

RUBRIC_SYSTEM_PROMPT = """You are a strict visual QA judge for an image-to-shader system.
The image shows two labeled panels: REFERENCE (the target) and RENDER (the shader output).
First list concrete visual differences, then score the render against the reference.
Scoring anchors: 1.0 = visually identical, 0.8 = minor deviations, 0.5 = same concept but clearly off, 0.2 = barely related.
Respond ONLY with JSON:
{"differences": ["..."],
 "shape_fidelity": 0.0-1.0,
 "position_layout": 0.0-1.0,
 "color_fidelity": 0.0-1.0,
 "effects_fidelity": 0.0-1.0,
 "failure_type": "none" | "structure" | "parameter" | "color" | "layer_order",
 "revision_hints": ["actionable scene-level change", "..."]}"""

PAIRWISE_SYSTEM_PROMPT = """You compare two shader renders against a reference image.
The image shows three labeled panels: REFERENCE, A, B.
Decide which of A or B is visually closer to REFERENCE overall (shape, position, color, effects).
Respond ONLY with JSON: {"winner": "A" | "B" | "tie", "reason": "one sentence"}"""


def _file_digest(*paths) -> str:
    h = hashlib.sha1()
    for p in paths:
        h.update(Path(p).read_bytes())
    return h.hexdigest()


def _compose_panel(labeled_paths: "list[tuple[str, Path]]", out_path: Path) -> Path:
    """Concatenate labeled image panels horizontally into a single image."""
    panels = []
    for label, p in labeled_paths:
        img = Image.open(p).convert("RGB")
        scale = PANEL_HEIGHT / max(1, img.height)
        img = img.resize((max(1, int(img.width * scale)), PANEL_HEIGHT), Image.LANCZOS)
        labeled = Image.new("RGB", (img.width, PANEL_HEIGHT + 28), (24, 24, 24))
        labeled.paste(img, (0, 28))
        ImageDraw.Draw(labeled).text((8, 6), label, fill=(255, 255, 255))
        panels.append(labeled)
    total_w = sum(p.width for p in panels) + 12 * (len(panels) - 1)
    canvas = Image.new("RGB", (total_w, PANEL_HEIGHT + 28), (24, 24, 24))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width + 12
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def _default_client(system_prompt: str, user_prompt: str, image_paths: "list[str] | None"):
    if not settings.llm.api_key:
        return None
    from app.llm.client import BaseAgent

    agent = BaseAgent(settings.llm)
    paths = image_paths if image_paths and settings.llm_supports_image else None
    return agent.chat(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_paths=paths,
        temperature=0.0,
        max_tokens=1024,
        response_format={"type": "json_object"},
    )


def _parse_json(response) -> "dict | None":
    if response is None:
        return None
    if isinstance(response, dict):
        return response
    text = str(response).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def judge_rubric(
    reference_path,
    render_path,
    *,
    work_dir,
    judge_client: "JudgeClient | None" = None,
) -> "dict | None":
    """Score a render against the reference on semantic dimensions.

    Returns {"semantic_scores": {...4 dims...}, "failure_type": str,
    "revision_hints": [...], "differences": [...]} or None on any failure.
    """
    try:
        cache_key = "rubric:" + _file_digest(reference_path, render_path)
        if cache_key in _CACHE:
            return _CACHE[cache_key]  # type: ignore[return-value]
        panel = _compose_panel(
            [("REFERENCE", Path(reference_path)), ("RENDER", Path(render_path))],
            Path(work_dir) / "rubric_panel.png",
        )
        client = judge_client or _default_client
        raw = client(RUBRIC_SYSTEM_PROMPT, "Evaluate the RENDER against the REFERENCE.", [str(panel)])
        data = _parse_json(raw)
        if data is None:
            return None
        semantic: dict[str, float] = {}
        for dim in ("shape_fidelity", "position_layout", "color_fidelity", "effects_fidelity"):
            try:
                semantic[dim] = max(0.0, min(1.0, float(data[dim])))
            except (KeyError, TypeError, ValueError):
                return None  # malformed — degrade entirely rather than half-trust
        result = {
            "semantic_scores": semantic,
            "failure_type": str(data.get("failure_type", "none")),
            "revision_hints": [str(x) for x in data.get("revision_hints", [])][:5],
            "differences": [str(x) for x in data.get("differences", [])][:8],
        }
        _CACHE[cache_key] = result
        return result
    except Exception:
        logger.warning("VLM rubric judge failed", exc_info=True)
        return None


def judge_pairwise(
    reference_path,
    a_path,
    b_path,
    *,
    work_dir,
    judge_client: "JudgeClient | None" = None,
) -> "str | None":
    """Return "A", "B", or "tie" (order-debiased), or None on any failure."""
    try:
        cache_key = "pair:" + _file_digest(reference_path, a_path, b_path)
        if cache_key in _CACHE:
            return _CACHE[cache_key]  # type: ignore[return-value]
        client = judge_client or _default_client
        verdicts: list[str] = []
        for tag, first, second in (("fwd", a_path, b_path), ("rev", b_path, a_path)):
            panel = _compose_panel(
                [("REFERENCE", Path(reference_path)), ("A", Path(first)), ("B", Path(second))],
                Path(work_dir) / f"pair_panel_{tag}.png",
            )
            data = _parse_json(client(PAIRWISE_SYSTEM_PROMPT, "Which render is closer to REFERENCE?", [str(panel)]))
            if data is None:
                return None
            verdicts.append(str(data.get("winner", "tie")).strip().upper())
        fwd, rev = verdicts
        rev_mapped = {"A": "B", "B": "A"}.get(rev, "tie")  # rev call had panels swapped
        result = fwd if fwd == rev_mapped and fwd in ("A", "B") else "tie"
        _CACHE[cache_key] = result
        return result
    except Exception:
        logger.warning("VLM pairwise judge failed", exc_info=True)
        return None
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend && python -m pytest tests/unit/test_vlm_judge.py -q`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/llm/vlm_judge.py backend/tests/unit/test_vlm_judge.py
git commit -m "feat(llm): VLM judge with rubric scoring and order-debiased pairwise comparison"
```

### Task 10: 三个决策点接线 + 配置键

**Files:**
- Modify: `backend/app/strategy_config.json`
- Modify: `backend/app/state.py`
- Modify: `backend/app/pipeline/graph.py`
- Modify: `backend/app/pipeline/refinement.py`

- [ ] **Step 1: strategy_config.json 的 params 中添加（`max_added_layers` 之后）**

```json
    "vlm_judge_enabled": {
      "default": 0,
      "min": 0,
      "max": 1,
      "step": 1,
      "integer": true,
      "label": "VLM 评审",
      "description": "VLM judge: 1 开启模型评判（近平局裁决/精修仲裁/终稿复核），0 关闭。需多模态 LLM 配置可用"
    },
    "vlm_tie_epsilon": {
      "default": 0.05,
      "min": 0.0,
      "max": 0.2,
      "step": 0.01,
      "integer": false,
      "label": "近平局阈值",
      "description": "Pairwise trigger epsilon: 候选分差低于此值时才调用 VLM 裁决"
    }
```

4 个 preset 各加：`fast: vlm_judge_enabled=0`、`balanced: 0`、`quality: 1`、`aggressive: 1`（`vlm_tie_epsilon` 统一 0.05）。

- [ ] **Step 2: state.py 补字段 + graph.py 读取配置**

`P2SPipelineState` 的 `max_added_layers: int` 之后添加：

```python
    vlm_judge_enabled: bool
    vlm_tie_epsilon: float
```

graph.py import 区添加：

```python
from app.llm.vlm_judge import judge_pairwise, judge_rubric
from app.metrics.quality_router import compute_final_score
```

`run_png_shader_pipeline` 配置读取段（`max_added_layers = ...` 之后）添加：

```python
    vlm_judge_enabled = (
        bool(int(strategy_clamp(
            "vlm_judge_enabled",
            int(quality_config.get("vlm_judge_enabled", get_default("vlm_judge_enabled"))),
        )))
        and effective_llm_enabled
    )
    vlm_tie_epsilon = float(strategy_clamp(
        "vlm_tie_epsilon",
        float(quality_config.get("vlm_tie_epsilon", get_default("vlm_tie_epsilon"))),
    ))
```

把两个键加入 `write_manifest` 的 config dict 与 `initial_state` dict（同 Task 8，两处都要）。

- [ ] **Step 3: 决策点 1 — 近平局选优裁决（node_selection）**

在 `node_selection`（graph.py:124）中，`selected = select_best_candidate(...)` 行之前插入：

```python
    if state.get("vlm_judge_enabled"):
        run_dir = Path(state["run_dir"])
        reference_path = run_dir / "reference_input.png"
        ranked = sorted(
            [c for c in candidates if c.compile_success and c.render_path],
            key=lambda c: -c.final_score,
        )
        if (
            len(ranked) >= 2
            and (ranked[0].final_score - ranked[1].final_score)
            < float(state.get("vlm_tie_epsilon", 0.05))
        ):
            verdict = judge_pairwise(
                reference_path, ranked[0].render_path, ranked[1].render_path,
                work_dir=run_dir / "judge",
            )
            logger.info(
                "vlm near-tie arbitration: %s vs %s -> %s",
                ranked[0].id, ranked[1].id, verdict,
            )
            if verdict == "B":
                bump = ranked[0].final_score - ranked[1].final_score + 0.001
                ranked[1].final_score += bump
                ranked[1].reason.append(f"vlm pairwise judge won near-tie (+{bump:.4f})")
            elif verdict == "A":
                ranked[0].reason.append("vlm pairwise judge confirmed near-tie winner")
```

- [ ] **Step 4: 决策点 2 — 精修小增益仲裁（refinement.py）**

`run_dsl_refinement_loop` 签名添加参数（`strategy_reader` 之后）：

```python
    pairwise_judge: "Callable[[Path, Path], str | None] | None" = None,
```

循环体内，`entry["meaningful_improvement"] = delta >= min_improvement`（refinement.py:292）与其后的日志之后、`if delta > 0.0:`（refinement.py:304）之前插入：

```python
        # Arbitrate noise-level gains: objective metrics can't tell 0.005
        # improvement from rendering noise — let the judge veto.
        if (
            pairwise_judge is not None
            and 0.0 < delta < min_improvement
            and current_render_path is not None
            and render_path.exists()
        ):
            verdict = pairwise_judge(current_render_path, render_path)
            if verdict == "A":  # judge prefers the previous best
                entry["vlm_override"] = "veto_small_gain"
                delta = 0.0
                entry["improved"] = False
```

`_run_post_pipeline` 中调用 `run_dsl_refinement_loop(...)`（graph.py:377-395）处增加实参：

```python
            pairwise_judge=(
                (lambda cur, new: judge_pairwise(
                    reference_path, cur, new, work_dir=run_dir / "judge"
                ))
                if state.get("vlm_judge_enabled") else None
            ),
```

- [ ] **Step 5: 决策点 3 — 终稿门禁（_run_post_pipeline）**

在 `_run_post_pipeline` 中，`_sync_selected_record_for_response(...)` 调用（graph.py:422）**之前**插入：

```python
    judge_summary = None
    if state.get("vlm_judge_enabled") and selected is not None and selected.render_path:
        rubric = judge_rubric(
            reference_path, selected.render_path, work_dir=run_dir / "judge"
        )
        if rubric is not None:
            blended = compute_final_score(selected_metrics, rubric["semantic_scores"])
            judge_summary = {
                **rubric,
                "objective_score": float(selected.final_score),
                "blended_score": blended,
            }
            logger.info(
                "vlm final gate: objective=%.4f blended=%.4f failure_type=%s",
                float(selected.final_score), blended, rubric["failure_type"],
            )
            if selected_quality is not None:
                selected_quality = {
                    **selected_quality,
                    "final_score": blended,
                    "semantic_scores": rubric["semantic_scores"],
                    "vlm_failure_type": rubric["failure_type"],
                }
            selected.final_score = blended
            save_json(run_dir / "judge" / "final_rubric.json", judge_summary)
```

`_run_post_pipeline` 末尾的 return dict 中添加一行：

```python
        "vlm_judge": judge_summary,
```

`run_png_shader_pipeline` 末尾的 return dict 中添加一行（依赖清理计划 Task 2 的 `{**state, ...}` 修复使该键可见）：

```python
        "vlm_judge": state.get("vlm_judge"),
```

- [ ] **Step 6: 运行全套单测**

Run: `cd backend && python -m pytest tests/unit/ -q`
Expected: 全 PASS（`vlm_judge_enabled` 默认 0，现有 test_graph 路径行为不变）

- [ ] **Step 7: Commit**

```bash
git add backend/app/strategy_config.json backend/app/state.py backend/app/pipeline/graph.py backend/app/pipeline/refinement.py
git commit -m "feat(pipeline): wire VLM judge into selection, refinement, and final gate"
```

### Task 11: 一致率验证脚本 + 文档收尾

**Files:**
- Create: `backend/scripts/judge_agreement.py`
- Modify: `README.md`

- [ ] **Step 1: 写一致率脚本**

```python
"""Measure VLM judge vs human agreement on labeled render pairs.

CSV columns: reference,render_a,render_b,human   (human = A | B | tie)
Usage:  cd backend && python scripts/judge_agreement.py pairs.csv

Build pairs.csv by sampling candidate renders from backend/test_results/<run>/
candidates/ and labeling ~30 pairs by eye. Target agreement >= 85%; below
80% do NOT enable pairwise arbitration — iterate the judge prompt first.
"""
import csv
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.llm.vlm_judge import judge_pairwise


def main(csv_path: str) -> None:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    work_dir = Path(tempfile.mkdtemp(prefix="judge_cal_"))
    agree = total = 0
    for row in rows:
        verdict = judge_pairwise(
            row["reference"], row["render_a"], row["render_b"], work_dir=work_dir
        )
        if verdict is None:
            print(f"SKIP {row['render_a']}: judge call failed")
            continue
        human = row["human"].strip().upper()
        human = human if human in ("A", "B") else "tie"
        total += 1
        ok = verdict == human
        agree += ok
        print(f"{Path(row['render_a']).name} vs {Path(row['render_b']).name}: "
              f"judge={verdict} human={human} {'OK' if ok else 'MISS'}")
    if total:
        print(f"\nagreement: {agree}/{total} = {agree / total:.1%}  (target >= 85%)")
    else:
        print("no usable rows — check API config and CSV paths")


if __name__ == "__main__":
    main(sys.argv[1])
```

- [ ] **Step 2: 人工标注并验证**（需要多模态 LLM API 可用）

从历史 `backend/test_results/*/candidates/*_render.png` 中抽 30 对组成 `pairs.csv`，人工填 human 列后运行：

Run: `cd backend && python scripts/judge_agreement.py pairs.csv`
Expected: 打印每对判定与总一致率。**≥85% 才在 strategy preset 中保留 quality/aggressive 的 `vlm_judge_enabled=1`；<80% 改回 0 并记录原因。**

- [ ] **Step 3: 更新 README.md**

在 Pipeline Stages 的候选列表中加入 decompose（rule 与 cv 之间），流程描述中 Optimization 后加 "Residual layers（残差增层）"，并在配置说明加入 `max_added_layers`、`vlm_judge_enabled`、`vlm_tie_epsilon` 三个 strategy 键。

- [ ] **Step 4: 最终验收**

Run: `cd backend && python -m pytest tests/unit/ -q && python tests/e2e/run_batch.py`
把 Phase 3 结果追加到 `doc/2026-06-12-accuracy-baseline.md`，对比三个阶段的 avg_final_score / pass 数 / selected_source 分布。

- [ ] **Step 5: Commit**

```bash
git add backend/scripts/judge_agreement.py README.md doc/2026-06-12-accuracy-baseline.md
git commit -m "feat(llm): judge agreement script, docs for accuracy optimization"
```

---

## 风险与回滚预案

| 风险 | 信号 | 回滚动作 |
|---|---|---|
| decompose 候选在某类样本上产生碎片化场景 | E2E report 显示 decompose 被选中但分数低于 rule | 调高 `min_area_frac`/`MIN_FIT_IOU`；极端情况在 `generate_decompose_candidate` 返回 None |
| 残差增层撑爆 shader 字符预算 | quality_router 报 `budget` failure | `max_added_layers=0` 即关闭；`max_layers_total=12` 是硬上限 |
| VLM 评判拉低整体分数或翻车 | blended_score 系统性低于 objective_score 且人看渲染没问题 | `vlm_judge_enabled=0` 一键关闭；评判从不进入内循环，关闭后行为与 Phase 2 完全一致 |
| LangGraph state 透传遗漏导致新功能"静默不生效" | residual.json / judge 目录从不出现 | 检查四处同步（strategy_config / 读取段 / initial_state / state.py），尤其 initial_state |

## Self-Review 记录

- 原计划 P1–P15 全部有对应任务或核查项，见总览表；P1–P4、P12–P15 的"已落地"结论经 2026-06-12 代码核查（文件内容 + 行号），非推测。
- 已核对类型/签名一致性（按 P2S 实际代码）：`_make_render_dsl_fn` 返回 `fn(dsl, glsl) -> Path | None`（scoring.py:318），Task 8 的 `render_fn=lambda d: res_render_fn(d, "")` 与之匹配；`_make_revision_scorer` 返回 `fn(dsl, reference_path) -> float`（graph.py:280 同款用法）；`_accept_improvement` 签名含 `reason`（scoring.py:63），Task 8 复用；`compute_final_score(metrics, semantic_scores)` 的混合公式在 `app/metrics/quality_router.py` 已预留。
- P2S 与 VFX 的关键架构差异已逐处适配：配置走 state 四处同步、近平局裁决移入 `node_selection`、`_run_post_pipeline` 无 progress_callback、`BaseAgent`/`settings.llm`/`settings.llm_supports_image` 替代 `app.agents.base`/`settings.generate`/`settings.generate_supports_image`。
- 已知偏差声明：本计划不包含 "moderngl 替代 Playwright"、"DSL 扩展（描边/阴影/混合模式）"、"LLM best-of-n"；E2E 脚手架为 P2S 新建的轻量版（无 VFX 版的 50 样本集与报告对比工具），样本集需人工补充真实 PNG 后冻结。
- 前置依赖：清理计划必须先行——否则测试安全网不可用（收集失败）、Task 10 Step 5 依赖的 `{**state, ...}` 修复缺失会导致 `vlm_judge` 键无法进入 API 响应。
