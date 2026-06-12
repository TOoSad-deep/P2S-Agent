# GLSL 闭环精修 Implementation Plan（v2，对标 VFX-Agent 闭环）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 GLSL 候选补上与 DSL 对等且对标 VFX-Agent 最佳实践的 LLM 闭环精修能力：语义反馈驱动、梯度窗口、渲染失败回喂、停滞重启；同时修复现有 GLSL `#define` 优化器的两个缺陷、移除选择阶段对 GLSL 的精修偏置，并把语义反馈/梯度窗口回灌到 DSL 精修循环。

**Architecture:** 遵循代码库既有的"平行模块"模式（`optimizer.py` / `glsl_optimizer.py` 互为镜像），新增 `glsl_refinement.py` 镜像 `refinement.py` 的循环控制逻辑。GLSL 循环通过注入的 `evaluate_fn` 评分（生产环境是 WebGL 渲染，单测注入假函数）；LLM 修订函数 `generate_llm_glsl_refinement` 放在 `llm_scene.py`，复用现有 GLSL 解析/规范化链。两个共享反馈构建器（`build_semantic_notes` / `build_recent_history_notes`）放在 `refinement.py`，供 DSL/GLSL 两个循环共用。

**Tech Stack:** Python 3 / pytest / LangGraph 流水线（仅 post-pipeline 改动）/ 现有 WebGL 渲染评分（`_evaluate_glsl_with_webgl`）/ 现有 VLM 评审（`judge_rubric` / `judge_pairwise`）。

**测试命令约定:** 所有测试在 `backend/` 目录下运行：`cd backend && python -m pytest <path> -v`（如有虚拟环境先激活）。

---

## 设计依据：对标 VFX-Agent 闭环的差距分析

VFX-Agent 的闭环（`Generate GLSL → validate → WebGL render → VLM inspect → 语义反馈回 Generate`）有五个机制。逐一对照 P2S 现状：

| VFX 机制 | P2S 现状 | 本计划动作 |
|---|---|---|
| ① 语义反馈驱动（inspect 输出 visual_issues/visual_goals 回喂 Generate） | `judge_rubric` 已返回 `differences`+`revision_hints`（vlm_judge.py:143-148），**但精修循环完全没用**，只在终局做分数融合 | 两个精修循环都接入 `rubric_judge`，把 differences→`[VISUAL ISSUE]`、revision_hints→`[VISUAL GOAL]` 注入反馈（Task 4 / Task 8） |
| ② 梯度窗口（最近 N 轮摘要、不存完整 shader，防 prompt 膨胀和旧代码锚定） | 反馈只携带上一轮的单条 rollback note，无趋势信息 | `build_recent_history_notes`：最近 3 轮的 score/outcome/changes 摘要（不含代码体）注入反馈（Task 4 / Task 8） |
| ③ 物理回滚（checkpoint + score regression 时恢复最佳版） | **已等价实现且更优**：循环只接受提分（best 单调不降），拒绝即逻辑回滚 + `[ROLLBACK]` 注入 | 不需要改动；VFX 需要显式回滚是因为其 snapshot 即使退步也会前进，P2S 的 best-only 设计天然免疫 |
| ④ 低分/停滞触发重新解构（re-decompose + 失败日志） | 停滞 = 直接 break 结束，没有"换技术路线重来"的逃逸通道 | 停滞/耐心耗尽时触发一次 fresh restart：注入失败日志（近期分数、失败方向、历史最佳），`fresh_start=True` 让 LLM 弃锚重写（Task 3 / Task 4） |
| ⑤ 多层守门（validate → render → inspect 各自失败有不同回路） | 计划 v1 已有静态校验回喂；但渲染失败只表现为 0 分，会触发误导性的 rollback note | 区分渲染失败：`[RENDER FAILED]` 专用反馈，而不是当作"分数变差"（Task 4） |

附注：VFX 的通过阈值 0.85 vs P2S `refinement_threshold` 默认 0.5——属于配置调优，不在本计划内改默认值，可在落地后用真实样本校准。

**不做的事（YAGNI）:**
- 不把 `run_dsl_refinement_loop` 重构为通用回调骨架（沿用平行模块模式，仅共享两个纯函数反馈构建器）。
- 不在精修循环内部交替运行 `#define` 坐标下降（留作后续增强）。
- 不新增 strategy_config 配置项——GLSL 精修复用 `max_refinement_iterations` / `refinement_threshold` / `refinement_patience` 等现有参数；fresh restart 次数为函数默认参数（1 次）。
- DSL 循环不加 fresh restart（DSL 已有 revision patch / 残差补层作为结构性逃逸通道，且候选池本身是多路起点）。

---

## 文件结构总览

| 文件 | 操作 | 职责 |
|---|---|---|
| `backend/app/pipeline/glsl_optimizer.py` | 修改 | 早停修复 + 相对步长 |
| `backend/tests/unit/test_glsl_optimizer.py` | 修改 | 新增 3 个测试 |
| `backend/app/candidates/llm_scene.py` | 修改 | 新增 `generate_llm_glsl_refinement`（含 fresh_start 模式） |
| `backend/tests/unit/test_llm_glsl_refinement.py` | 新建 | LLM GLSL 修订函数测试 |
| `backend/app/pipeline/refinement.py` | 修改 | 新增共享反馈构建器；`_should_run_refinement` 放开 GLSL；DSL 循环接入语义反馈+梯度窗口 |
| `backend/app/pipeline/glsl_refinement.py` | 新建 | `run_glsl_refinement_loop` 闭环（语义反馈/梯度窗口/渲染失败回喂/fresh restart） |
| `backend/tests/unit/test_glsl_refinement.py` | 新建 | 循环控制逻辑测试 |
| `backend/app/pipeline/graph.py` | 修改 | 接线 GLSL 精修分支（含 rubric_judge）、移除选择偏置、更新流程图注释 |
| `backend/tests/unit/test_graph.py` | 修改 | 更新/新增 6 个测试 |

---

### Task 1: glsl_optimizer 早停条件修复

现状缺陷：[glsl_optimizer.py:363-369](../backend/app/pipeline/glsl_optimizer.py) 的早停条件含 `best_score == initial_score`，意味着只要发生过一次提升，循环就永远不会提前停止，后续无梯度轮次空耗渲染预算。

**Files:**
- Modify: `backend/app/pipeline/glsl_optimizer.py`
- Test: `backend/tests/unit/test_glsl_optimizer.py`

- [ ] **Step 1: 写失败测试**

在 `backend/tests/unit/test_glsl_optimizer.py` 的 `test_optimize_respects_max_iterations` 之后添加两个测试。第一个是守护测试（现状恰好通过），第二个暴露缺陷：

```python
def test_optimize_early_stops_after_full_round_without_improvement(monkeypatch, tmp_path):
    """A full pass over every param with no accepted trial must stop the loop —
    even when max_iterations would allow many more renders."""
    glsl = """
#define A 0.2
#define B 0.3
#define C 0.4
""".strip()

    _make_score_driver(monkeypatch, lambda _g: 0.5)

    result = optimize_glsl_candidate(
        glsl,
        tmp_path / "ref.png",
        render_glsl_fn=lambda _g: tmp_path / "render.png",
        max_iterations=100,
    )
    # 3 params x 2 directions = 6 renders for one full no-improvement round.
    assert result.iterations_run <= 6


def test_optimize_early_stops_after_improvement_then_plateau(monkeypatch, tmp_path):
    """After an improvement is found and the gradient dries up, the optimizer
    must still stop after one fruitless full round (old code compared against
    initial_score and never stopped once any improvement happened)."""
    glsl = """
#define A 0.2
#define B 0.3
""".strip()

    def score_for_glsl(g: str) -> float:
        # Reward A == 0.25 exactly; B has no effect. After the first accepted
        # step (A: 0.2 -> 0.25) no further trial can improve.
        for line in g.splitlines():
            if line.startswith("#define A"):
                val = float(line.split()[-1])
                return 1.0 - abs(0.25 - val)
        return 0.0

    _make_score_driver(monkeypatch, score_for_glsl)

    result = optimize_glsl_candidate(
        glsl,
        tmp_path / "ref.png",
        render_glsl_fn=lambda _g: tmp_path / "render.png",
        max_iterations=100,
        scale=0.05,
    )
    assert result.improved is True
    # Improvement round (<= 4 renders) + one fruitless full round (4 renders).
    assert result.iterations_run <= 8
```

- [ ] **Step 2: 运行测试确认第二个失败**

Run: `cd backend && python -m pytest tests/unit/test_glsl_optimizer.py::test_optimize_early_stops_after_improvement_then_plateau -v`
Expected: FAIL，`iterations_run` 接近 100。

- [ ] **Step 3: 实现修复**

在 `backend/app/pipeline/glsl_optimizer.py` 中，把 `optimize_glsl_candidate` 内的

```python
    step_count = 0
    param_idx = 0
    seen_param_keys: set[str] = set()
```

改为

```python
    step_count = 0
    param_idx = 0
    params_since_improvement = 0
```

把循环尾部的

```python
        if best_trial_value is not None:
            best_glsl = update_glsl_define(
                best_glsl, param.name, best_trial_value, param.glsl_type
            )
            best_score = best_trial_score
            loss_curve.append(best_score)
            collected = parse_glsl_defines(best_glsl)

        # Stop when one full round produced no improvement and we've visited
        # every param at least once. Prevents redundant later rounds on the
        # same defines when there's no gradient left.
        seen_param_keys.add(param.name)
        if (
            len(seen_param_keys) >= len(collected)
            and best_score == initial_score
            and step_count > 0
        ):
            break
```

改为

```python
        if best_trial_value is not None:
            best_glsl = update_glsl_define(
                best_glsl, param.name, best_trial_value, param.glsl_type
            )
            best_score = best_trial_score
            loss_curve.append(best_score)
            collected = parse_glsl_defines(best_glsl)
            params_since_improvement = 0
        else:
            params_since_improvement += 1

        # Stop once a full pass over every param produced no accepted trial
        # since the last improvement — no gradient left to follow.
        if collected and params_since_improvement >= len(collected) and step_count > 0:
            break
```

- [ ] **Step 4: 运行整个优化器测试文件确认全部通过**

Run: `cd backend && python -m pytest tests/unit/test_glsl_optimizer.py -v`
Expected: 全部 PASS（含既有测试）。

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/glsl_optimizer.py backend/tests/unit/test_glsl_optimizer.py
git commit -m "fix(glsl-optimizer): early-stop after fruitless full round, not only at initial score"
```

---

### Task 2: glsl_optimizer 相对步长

现状缺陷：`scale=0.05` 是绝对步长，对量级大于 1 的参数（如 `#define RADIUS 10.0`）相当于 0.5% 的微扰，基本无效。

**Files:**
- Modify: `backend/app/pipeline/glsl_optimizer.py`
- Test: `backend/tests/unit/test_glsl_optimizer.py`

- [ ] **Step 1: 写失败测试**

```python
def test_optimize_uses_relative_step_for_large_values(monkeypatch, tmp_path):
    """Params outside [0,1] must be perturbed proportionally to their
    magnitude (0.05 * 10.0 = 0.5), not by the absolute 0.05 scale."""
    glsl = "#define BIG_RADIUS 10.0\nvoid main() {}"

    def score_for_glsl(g: str) -> float:
        for line in g.splitlines():
            if line.startswith("#define BIG_RADIUS"):
                return float(line.split()[-1]) / 100.0
        return 0.0

    _make_score_driver(monkeypatch, score_for_glsl)

    result = optimize_glsl_candidate(
        glsl,
        tmp_path / "ref.png",
        render_glsl_fn=lambda _g: tmp_path / "render.png",
        max_iterations=2,
        scale=0.05,
    )
    accepted = [s for s in result.optimizer_log if s.accepted]
    assert accepted, "larger BIG_RADIUS scores higher, the + trial must be accepted"
    assert abs(float(accepted[0].new_value) - 10.0) >= 0.49
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend && python -m pytest tests/unit/test_glsl_optimizer.py::test_optimize_uses_relative_step_for_large_values -v`
Expected: FAIL，`new_value` 为 10.05，步长断言不满足。

- [ ] **Step 3: 实现**

在 `glsl_optimizer.py` 的 `_clamp_value` 之后添加：

```python
def _step_for(value: float, scale: float) -> float:
    """Perturbation step: absolute for in-[0,1] values, relative elsewhere.

    A flat 0.05 step on e.g. RADIUS=10.0 is a 0.5% nudge with no visible
    effect; scaling by the magnitude keeps the step meaningful.
    """
    if 0.0 <= value <= 1.0:
        return scale
    return scale * max(abs(value), 1.0)
```

把 `optimize_glsl_candidate` 中的 trial 构造段

```python
        if param.glsl_type == "float":
            original = float(param.value)  # type: ignore[arg-type]
            plus = _clamp_value(original + scale, original, param.name)
            minus = _clamp_value(original - scale, original, param.name)
            trials: list[float | list[float]] = [plus, minus]
        else:
            original_components = list(param.value)  # type: ignore[arg-type]
            plus = [
                _clamp_value(v + scale, v, param.name) for v in original_components
            ]
            minus = [
                _clamp_value(v - scale, v, param.name) for v in original_components
            ]
            trials = [plus, minus]
```

改为

```python
        if param.glsl_type == "float":
            original = float(param.value)  # type: ignore[arg-type]
            step = _step_for(original, scale)
            plus = _clamp_value(original + step, original, param.name)
            minus = _clamp_value(original - step, original, param.name)
            trials: list[float | list[float]] = [plus, minus]
        else:
            original_components = list(param.value)  # type: ignore[arg-type]
            plus = [
                _clamp_value(v + _step_for(v, scale), v, param.name)
                for v in original_components
            ]
            minus = [
                _clamp_value(v - _step_for(v, scale), v, param.name)
                for v in original_components
            ]
            trials = [plus, minus]
```

- [ ] **Step 4: 运行全文件确认通过**

Run: `cd backend && python -m pytest tests/unit/test_glsl_optimizer.py -v`
Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/glsl_optimizer.py backend/tests/unit/test_glsl_optimizer.py
git commit -m "feat(glsl-optimizer): use magnitude-relative step for params outside [0,1]"
```

---

### Task 3: `generate_llm_glsl_refinement`（LLM GLSL 修订函数，含 fresh_start 模式）

`fresh_start=True` 是 VFX re-decompose 的 GLSL 等价物：不把当前 shader 放进 prompt，指示模型换技术路线从头写，避免锚定在失败的实现上。

**Files:**
- Modify: `backend/app/candidates/llm_scene.py`（在 `generate_llm_refinement` 函数之后追加）
- Create: `backend/tests/unit/test_llm_glsl_refinement.py`

- [ ] **Step 1: 写失败测试**

创建 `backend/tests/unit/test_llm_glsl_refinement.py`：

```python
"""Tests for generate_llm_glsl_refinement (LLM-driven GLSL revision)."""

from __future__ import annotations

import json

from app.candidates.llm_scene import generate_llm_glsl_refinement

VALID_GLSL = (
    "#define R 0.5\n"
    "void mainImage(out vec4 fragColor, in vec2 fragCoord) { fragColor = vec4(R); }"
)


def test_returns_normalized_glsl_with_io():
    def fake_client(system_prompt, user_prompt, image_paths):
        # LLM wraps the shader in a fenced block inside the JSON value —
        # the parser chain must strip the fences.
        return json.dumps({"glsl": "```glsl\n" + VALID_GLSL + "\n```"})

    result = generate_llm_glsl_refinement(
        current_glsl=VALID_GLSL,
        metrics={"mse": 0.4, "ssim": 0.5},
        quality_router={"final_score": 0.4, "quality_band": "low"},
        llm_client=fake_client,
    )

    assert result is not None
    assert "void mainImage" in result["glsl"]
    assert "```" not in result["glsl"]
    assert result["_io"]["mode"] == "glsl_refinement"
    assert "current_glsl" in result["_io"]["user_prompt"]


def test_fresh_start_omits_current_glsl_and_requests_rewrite():
    captured = {}

    def fake_client(system_prompt, user_prompt, image_paths):
        captured["system"] = system_prompt
        captured["user"] = user_prompt
        return json.dumps({"glsl": VALID_GLSL})

    result = generate_llm_glsl_refinement(
        current_glsl=VALID_GLSL,
        metrics={},
        quality_router={"final_score": 0.3},
        fresh_start=True,
        llm_client=fake_client,
    )

    assert result is not None
    assert "current_glsl" not in captured["user"]
    assert "from scratch" in captured["system"]
    assert result["_io"]["fresh_start"] is True


def test_returns_none_on_empty_response():
    result = generate_llm_glsl_refinement(
        current_glsl=VALID_GLSL,
        metrics={},
        quality_router={},
        llm_client=lambda *args: "",
    )
    assert result is None


def test_returns_none_when_no_mainimage():
    def fake_client(system_prompt, user_prompt, image_paths):
        return json.dumps({"glsl": "float x = 1.0;"})

    result = generate_llm_glsl_refinement(
        current_glsl=VALID_GLSL,
        metrics={},
        quality_router={},
        llm_client=fake_client,
    )
    assert result is None
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend && python -m pytest tests/unit/test_llm_glsl_refinement.py -v`
Expected: FAIL with `ImportError: cannot import name 'generate_llm_glsl_refinement'`

- [ ] **Step 3: 实现**

在 `backend/app/candidates/llm_scene.py` 文件末尾（`generate_llm_refinement` 之后）添加。说明：复用 `_build_feedback_issues` 的指标反馈；解析链与 `generate_llm_scene_candidate` 的 GLSL 分支一致；`max_tokens=6144` 因为 GLSL 全文上限 12000 字符，DSL 用的 3072 不够。

```python
def generate_llm_glsl_refinement(
    current_glsl: str,
    metrics: dict,
    quality_router: dict,
    *,
    reference_image_path: "str | Path | None" = None,
    current_render_path: "str | Path | None" = None,
    extra_feedback: "list[str] | None" = None,
    fresh_start: bool = False,
    llm_client: LlmClient | None = None,
) -> dict | None:
    """Ask the LLM to revise a Shadertoy GLSL candidate based on feedback.

    Mirrors ``generate_llm_refinement`` but for raw GLSL: when reference and
    current-render images are available (and the model supports image input),
    both are attached for a visual diff; metric issues are always included.

    ``fresh_start=True`` is the GLSL analog of the VFX pipeline's re-decompose:
    the current shader is withheld from the prompt and the model is told to
    rewrite from scratch with a different technique instead of iterating.

    Returns ``{"glsl": <normalized>, "postprocess_warnings": [...], "_io": {...}}``
    or None on failure. The caller must pop ``_io`` before further use.
    """
    issues = _build_feedback_issues(metrics, quality_router)
    if extra_feedback:
        issues = list(extra_feedback) + issues
    image_paths = _normalize_image_paths([reference_image_path, current_render_path])
    has_images = bool(image_paths) and settings.llm_supports_image

    restart_clause = (
        "The previous incremental approach has stalled. Write a completely new "
        "shader from scratch with a different technique; do not anchor on the "
        "old implementation. "
        if fresh_start
        else ""
    )

    system_prompt = (
        "You are a Shadertoy GLSL expert. "
        + (
            "Two images are attached: image 1 is the TARGET (reference PNG), "
            "image 2 is the CURRENT rendered output. "
            "Diff them visually and produce GLSL whose render matches "
            "image 1 more closely. "
            if has_images
            else "Given quality feedback about the current render, produce "
                 "GLSL that better matches the reference image. "
        )
        + restart_clause
        + "Rules: keep a valid `void mainImage(out vec4 fragColor, in vec2 fragCoord)` "
        "entry point. Do NOT declare iTime/iResolution/iMouse/iFrame uniforms "
        "(they are auto-injected). Do NOT use discard. Every float literal must "
        "contain a decimal point (write 2.0, never 2). Keep all visually tunable "
        "constants as `#define NAME value` lines at the top of the shader. "
        "The target is a static image: do not animate with iTime. "
        "Return ONLY a JSON object {\"glsl\": \"<the full shader>\"} "
        "with no markdown or prose outside the JSON."
    )

    user_prompt = json.dumps(
        {
            **({} if fresh_start else {"current_glsl": current_glsl}),
            "feedback": {
                "current_score": round(float(quality_router.get("final_score", 0.0)), 4),
                "quality_band": quality_router.get("quality_band", "unknown"),
                "failure_type": quality_router.get("failure_type", "unknown"),
                "issues": issues,
                "instruction": f"Fix: {'; '.join(issues[:3])}.",
            },
        },
        ensure_ascii=False,
        indent=2,
    )

    response = _call_llm(
        system_prompt,
        user_prompt,
        image_paths=image_paths if has_images else None,
        llm_client=llm_client,
        max_tokens=6144,
        response_format={"type": "json_object"},
    )
    content = _response_content(response)
    if not content:
        return None

    payload = parse_glsl_response_payload(content)
    glsl = _extract_glsl(str(payload.get("glsl") or ""))
    if not glsl:
        glsl = _extract_glsl(content)
    if not glsl:
        return None

    normalized = normalize_shadertoy_glsl(glsl)
    if "void mainImage" not in normalized.glsl:
        return None

    return {
        "glsl": normalized.glsl,
        "postprocess_warnings": list(normalized.warnings),
        "_io": {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "raw_response": content,
            "mode": "glsl_refinement",
            "fresh_start": fresh_start,
            "image_paths": image_paths if has_images else [],
        },
    }
```

注：`parse_glsl_response_payload`、`normalize_shadertoy_glsl`、`_extract_glsl`、`_build_feedback_issues`、`_normalize_image_paths`、`_call_llm`、`_response_content`、`LlmClient`、`settings`、`json` 均已在该文件中导入/定义，无需新增 import。

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend && python -m pytest tests/unit/test_llm_glsl_refinement.py -v`
Expected: 5 个测试全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add backend/app/candidates/llm_scene.py backend/tests/unit/test_llm_glsl_refinement.py
git commit -m "feat(llm): add generate_llm_glsl_refinement with visual diff and fresh-start mode"
```

---

### Task 4: 共享反馈构建器 + `run_glsl_refinement_loop`（GLSL 闭环主循环）

循环对标 VFX 的四项机制：语义反馈（`rubric_judge`）、梯度窗口（`build_recent_history_notes`）、渲染失败专用回喂（`[RENDER FAILED]`）、停滞/耐心耗尽时一次 fresh restart（`[FRESH RESTART]` 失败日志）。逻辑回滚（best 单调不降 + `[ROLLBACK]` 注入）沿用 DSL 循环的既有模式。

**Files:**
- Modify: `backend/app/pipeline/refinement.py`（添加两个共享反馈构建器）
- Create: `backend/app/pipeline/glsl_refinement.py`
- Create: `backend/tests/unit/test_glsl_refinement.py`

- [ ] **Step 1: 写失败测试**

创建 `backend/tests/unit/test_glsl_refinement.py`：

```python
"""Tests for the GLSL LLM refinement loop."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.pipeline.glsl_refinement import _diff_glsl_summary, run_glsl_refinement_loop
from app.pipeline.refinement import build_recent_history_notes, build_semantic_notes

VALID_GLSL_A = (
    "#define R 0.30\n"
    "void mainImage(out vec4 fragColor, in vec2 fragCoord) { fragColor = vec4(R); }"
)
VALID_GLSL_B = (
    "#define R 0.50\n"
    "void mainImage(out vec4 fragColor, in vec2 fragCoord) { fragColor = vec4(R); }"
)


def _evaluate_by_r(glsl: str, render_path: Path):
    """Deterministic fake scorer: score equals the R define's value."""
    for line in glsl.splitlines():
        if line.startswith("#define R"):
            score = float(line.split()[-1])
            return {"mse": 1.0 - score}, {"final_score": score}, score, None
    return {}, {}, 0.0, None


def test_loop_accepts_improvement_and_stops_at_threshold(tmp_path, monkeypatch):
    calls: list[dict] = []

    def fake_refine(**kwargs):
        calls.append(kwargs)
        return {"glsl": VALID_GLSL_B, "_io": {"mode": "glsl_refinement"}}

    monkeypatch.setattr(
        "app.candidates.llm_scene.generate_llm_glsl_refinement", fake_refine
    )

    result = run_glsl_refinement_loop(
        VALID_GLSL_A,
        0.30,
        {"mse": 0.7},
        {"final_score": 0.30},
        tmp_path / "ref.png",
        evaluate_fn=_evaluate_by_r,
        max_iterations=3,
        threshold=0.45,
        high_score_stop=0.92,
        loop_dir=tmp_path / "loop",
    )

    assert result["best_glsl"] == VALID_GLSL_B
    assert result["best_score"] == pytest.approx(0.50)
    assert result["stop_reason"] == "threshold_reached"
    assert len(result["history"]) == 1
    entry = result["history"][0]
    assert entry["improved"] is True
    assert entry["llm_io"] == {"mode": "glsl_refinement"}
    assert entry["compile_glsl"] == VALID_GLSL_B
    assert calls[0]["fresh_start"] is False
    assert (tmp_path / "loop" / "iter_1.json").exists()


def test_loop_rolls_back_and_feeds_back(tmp_path, monkeypatch):
    worse = VALID_GLSL_A.replace("0.30", "0.10")
    calls: list[dict] = []

    def fake_refine(**kwargs):
        calls.append(kwargs)
        return {"glsl": worse, "_io": {}}

    monkeypatch.setattr(
        "app.candidates.llm_scene.generate_llm_glsl_refinement", fake_refine
    )

    result = run_glsl_refinement_loop(
        VALID_GLSL_A,
        0.30,
        {},
        {"final_score": 0.30},
        tmp_path / "ref.png",
        evaluate_fn=_evaluate_by_r,
        max_iterations=5,
        threshold=0.80,
        high_score_stop=0.92,
        no_improvement_patience=2,
        max_fresh_restarts=0,
        loop_dir=tmp_path / "loop",
    )

    assert result["best_glsl"] == VALID_GLSL_A
    assert result["stop_reason"] == "no_improvement_patience"
    assert len(calls) == 2
    second_call_feedback = calls[1]["extra_feedback"]
    assert second_call_feedback and any("[ROLLBACK]" in n for n in second_call_feedback)


def test_loop_fresh_restart_after_patience(tmp_path, monkeypatch):
    """Patience exhaustion with restarts left must trigger one fresh-start
    generation (VFX re-decompose analog), then stop on the second exhaustion."""
    worse = VALID_GLSL_A.replace("0.30", "0.10")
    calls: list[dict] = []

    def fake_refine(**kwargs):
        calls.append(kwargs)
        return {"glsl": worse, "_io": {}}

    monkeypatch.setattr(
        "app.candidates.llm_scene.generate_llm_glsl_refinement", fake_refine
    )

    result = run_glsl_refinement_loop(
        VALID_GLSL_A,
        0.30,
        {},
        {"final_score": 0.30},
        tmp_path / "ref.png",
        evaluate_fn=_evaluate_by_r,
        max_iterations=6,
        threshold=0.80,
        high_score_stop=0.92,
        no_improvement_patience=2,
        max_fresh_restarts=1,
        loop_dir=tmp_path / "loop",
    )

    assert result["stop_reason"] == "no_improvement_patience"
    assert len(calls) == 4  # 2 rejected + restart + 2 rejected (4th call total)
    assert calls[2]["fresh_start"] is True
    assert any("[FRESH RESTART]" in n for n in calls[2]["extra_feedback"])
    assert calls[3]["fresh_start"] is False
    assert result["best_glsl"] == VALID_GLSL_A


def test_loop_feeds_render_failure_back(tmp_path, monkeypatch):
    """A render failure is not a 'worse score' — the model must be told the
    shader failed to render so it fixes the code instead of the visuals."""
    calls: list[dict] = []

    def fake_refine(**kwargs):
        calls.append(kwargs)
        return {"glsl": VALID_GLSL_B, "_io": {}}

    monkeypatch.setattr(
        "app.candidates.llm_scene.generate_llm_glsl_refinement", fake_refine
    )

    def evaluate_fail(glsl: str, render_path: Path):
        return {}, {}, 0.0, None

    result = run_glsl_refinement_loop(
        VALID_GLSL_A,
        0.30,
        {},
        {"final_score": 0.30},
        tmp_path / "ref.png",
        evaluate_fn=evaluate_fail,
        max_iterations=4,
        threshold=0.80,
        high_score_stop=0.92,
        no_improvement_patience=2,
        max_fresh_restarts=0,
        loop_dir=tmp_path / "loop",
    )

    assert result["best_glsl"] == VALID_GLSL_A
    assert result["history"][0]["error_type"] == "render_failed"
    assert len(calls) == 2
    assert any("[RENDER FAILED]" in n for n in calls[1]["extra_feedback"])


def test_loop_injects_semantic_feedback_from_rubric(tmp_path, monkeypatch):
    calls: list[dict] = []

    def fake_refine(**kwargs):
        calls.append(kwargs)
        return {"glsl": VALID_GLSL_B, "_io": {}}

    monkeypatch.setattr(
        "app.candidates.llm_scene.generate_llm_glsl_refinement", fake_refine
    )

    run_glsl_refinement_loop(
        VALID_GLSL_A,
        0.30,
        {},
        {"final_score": 0.30},
        tmp_path / "ref.png",
        evaluate_fn=_evaluate_by_r,
        initial_render_path=tmp_path / "current.png",
        rubric_judge=lambda render: {
            "differences": ["edges too sharp"],
            "revision_hints": ["soften the edges"],
        },
        max_iterations=1,
        threshold=0.80,
        high_score_stop=0.92,
        loop_dir=tmp_path / "loop",
    )

    feedback = calls[0]["extra_feedback"]
    assert any("[VISUAL ISSUE] edges too sharp" in n for n in feedback)
    assert any("[VISUAL GOAL] soften the edges" in n for n in feedback)


def test_loop_includes_recent_history_notes(tmp_path, monkeypatch):
    worse = VALID_GLSL_A.replace("0.30", "0.10")
    calls: list[dict] = []

    def fake_refine(**kwargs):
        calls.append(kwargs)
        return {"glsl": worse, "_io": {}}

    monkeypatch.setattr(
        "app.candidates.llm_scene.generate_llm_glsl_refinement", fake_refine
    )

    run_glsl_refinement_loop(
        VALID_GLSL_A,
        0.30,
        {},
        {"final_score": 0.30},
        tmp_path / "ref.png",
        evaluate_fn=_evaluate_by_r,
        max_iterations=2,
        threshold=0.80,
        high_score_stop=0.92,
        no_improvement_patience=3,
        max_fresh_restarts=0,
        loop_dir=tmp_path / "loop",
    )

    assert len(calls) == 2
    assert any("[HISTORY iter 1]" in n for n in calls[1]["extra_feedback"])


def test_loop_skips_invalid_glsl_with_compile_feedback(tmp_path, monkeypatch):
    calls: list[dict] = []

    def fake_refine(**kwargs):
        calls.append(kwargs)
        return {"glsl": "void broken() {", "_io": {}}  # no mainImage, brace mismatch

    monkeypatch.setattr(
        "app.candidates.llm_scene.generate_llm_glsl_refinement", fake_refine
    )

    result = run_glsl_refinement_loop(
        VALID_GLSL_A,
        0.30,
        {},
        {"final_score": 0.30},
        tmp_path / "ref.png",
        evaluate_fn=_evaluate_by_r,
        max_iterations=5,
        threshold=0.80,
        high_score_stop=0.92,
        no_improvement_patience=2,
        max_fresh_restarts=0,
        loop_dir=tmp_path / "loop",
    )

    assert result["best_glsl"] == VALID_GLSL_A
    assert result["stop_reason"] == "no_improvement_patience"
    assert result["history"][0]["error"].startswith("GLSL invalid")
    assert len(calls) == 2
    assert any("[COMPILE FEEDBACK]" in n for n in calls[1]["extra_feedback"])


def test_loop_stops_when_llm_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.candidates.llm_scene.generate_llm_glsl_refinement",
        lambda **kwargs: None,
    )

    result = run_glsl_refinement_loop(
        VALID_GLSL_A,
        0.30,
        {},
        {"final_score": 0.30},
        tmp_path / "ref.png",
        evaluate_fn=_evaluate_by_r,
        max_iterations=3,
        threshold=0.80,
        high_score_stop=0.92,
        loop_dir=tmp_path / "loop",
    )

    assert result["stop_reason"] == "llm_returned_none"
    assert result["history"][0]["error_type"] == "llm_returned_none"


def test_loop_high_score_stops_without_llm_call(tmp_path, monkeypatch):
    def fail_refine(**kwargs):
        raise AssertionError("LLM must not be called above high_score_stop")

    monkeypatch.setattr(
        "app.candidates.llm_scene.generate_llm_glsl_refinement", fail_refine
    )

    result = run_glsl_refinement_loop(
        VALID_GLSL_A,
        0.95,
        {},
        {"final_score": 0.95},
        tmp_path / "ref.png",
        evaluate_fn=_evaluate_by_r,
        max_iterations=3,
        threshold=0.80,
        high_score_stop=0.92,
        loop_dir=tmp_path / "loop",
    )

    assert result["stop_reason"] == "high_score_stop"
    assert result["history"] == []


def test_diff_glsl_summary_reports_define_changes():
    summary = _diff_glsl_summary(VALID_GLSL_A, VALID_GLSL_B)
    assert "changed lines" in summary
    assert "#define R" in summary
    assert _diff_glsl_summary(VALID_GLSL_A, VALID_GLSL_A) == "no changes"


def test_build_semantic_notes_maps_rubric_fields():
    notes = build_semantic_notes(
        {"differences": ["bg mismatch"], "revision_hints": ["make bg white"]}
    )
    assert notes == ["[VISUAL ISSUE] bg mismatch", "[VISUAL GOAL] make bg white"]


def test_build_recent_history_notes_excludes_code_and_caps_entries():
    history = [
        {"iteration": i, "score_before": 0.3, "score_after": 0.3,
         "improved": False, "changes_summary": f"change {i}", "error": None}
        for i in range(1, 6)
    ]
    notes = build_recent_history_notes(history, max_entries=3)
    assert len(notes) == 3
    assert "[HISTORY iter 3]" in notes[0]
    assert "[HISTORY iter 5]" in notes[2]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend && python -m pytest tests/unit/test_glsl_refinement.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.pipeline.glsl_refinement'`

- [ ] **Step 3a: 在 `refinement.py` 添加共享反馈构建器**

在 `backend/app/pipeline/refinement.py` 的 `_diff_dsl_summary` 之后添加（模块顶部已有所需 import，无需新增）：

```python
def build_semantic_notes(rubric: dict) -> list[str]:
    """Convert a judge_rubric result into LLM feedback lines.

    Mirrors the VFX pipeline's visual_issues / visual_goals feedback:
    concrete differences become issues, revision hints become goals.
    """
    notes: list[str] = []
    for diff in list(rubric.get("differences", []))[:4]:
        notes.append(f"[VISUAL ISSUE] {diff}")
    for hint in list(rubric.get("revision_hints", []))[:3]:
        notes.append(f"[VISUAL GOAL] {hint}")
    return notes


def build_recent_history_notes(history: list[dict], max_entries: int = 3) -> list[str]:
    """Summarize recent iterations without shader/DSL bodies.

    The gradient-window idea from the VFX pipeline: the model sees the
    trajectory (scores, outcomes, change summaries) without prompt bloat or
    anchoring on stale code.
    """
    notes: list[str] = []
    for h in history[-max_entries:]:
        if h.get("improved"):
            outcome = "accepted"
        elif h.get("error"):
            outcome = f"failed ({h.get('error_type') or 'error'})"
        else:
            outcome = "rejected"
        notes.append(
            f"[HISTORY iter {h.get('iteration')}] score "
            f"{h.get('score_before')} -> {h.get('score_after')} ({outcome}); "
            f"changes: {(h.get('changes_summary') or 'n/a')[:160]}"
        )
    return notes
```

- [ ] **Step 3b: 创建 `backend/app/pipeline/glsl_refinement.py`（完整文件）**

```python
"""LLM-driven GLSL refinement loop for PNG-to-Shader.

Mirrors the DSL refinement loop (``refinement.py``) but operates on raw
Shadertoy GLSL, with the feedback mechanisms of the VFX pipeline's closed
loop: semantic VLM feedback (rubric_judge), a gradient window of recent
iteration summaries, render-failure feedback distinct from low scores, and
a one-shot fresh restart when incremental revision stalls. The best shader
is monotonic — rejected revisions roll back and the rollback reason is fed
to the next LLM call.
"""

from __future__ import annotations

import difflib
import logging
import time
from pathlib import Path
from typing import Callable

from app.pipeline.artifacts import save_json
from app.pipeline.refinement import build_recent_history_notes, build_semantic_notes
from app.services.shader_validator import validate_shader_static

logger = logging.getLogger(__name__)


def _short_exception(exc: Exception) -> str:
    message = str(exc).strip()
    if len(message) > 500:
        message = message[:497] + "..."
    return f"{exc.__class__.__name__}: {message}" if message else exc.__class__.__name__


def _diff_glsl_summary(old_glsl: str, new_glsl: str) -> str:
    """Concise human-readable summary of what changed between two shaders."""
    diff = difflib.unified_diff(
        old_glsl.splitlines(), new_glsl.splitlines(), lineterm="", n=0
    )
    changed = [
        line for line in diff
        if line[:1] in "+-" and not line.startswith(("+++", "---"))
    ]
    if not changed:
        return "no changes"
    define_changes = [line.strip() for line in changed if "#define" in line]
    summary = f"{len(changed)} changed lines"
    if define_changes:
        summary += "; " + "; ".join(define_changes[:4])
    return summary


def _build_failure_log_note(history: list[dict], best_score: float) -> str:
    """Failure log injected before a fresh restart — the GLSL analog of the
    VFX pipeline's re-decompose failure log."""
    recent_scores = [
        h.get("score_after") for h in history[-3:]
        if h.get("score_after") is not None
    ]
    failed_directions = [
        f"iter {h.get('iteration')}: {h.get('changes_summary') or h.get('error') or 'n/a'}"
        for h in history[-3:]
    ]
    return (
        "[FRESH RESTART] Incremental revision has stalled "
        f"(recent scores: {recent_scores}, best so far: {best_score:.3f}). "
        "Discard the current implementation approach and write a NEW shader "
        "from scratch for the same reference image using a different "
        "technique. Avoid repeating these failed directions: "
        + "; ".join(failed_directions[:3])
    )


def run_glsl_refinement_loop(
    initial_glsl: str,
    initial_score: float,
    initial_metrics: dict,
    initial_quality: dict,
    reference_path: Path,
    *,
    evaluate_fn: "Callable[[str, Path], tuple[dict, dict, float, Path | None]]",
    initial_render_path: "Path | None" = None,
    max_iterations: int = 3,
    threshold: float = 0.80,
    high_score_stop: float = 0.92,
    min_improvement: float = 0.01,
    no_improvement_patience: int = 2,
    max_fresh_restarts: int = 1,
    force_first_iteration: bool = False,
    loop_dir: Path,
    strategy_reader: "Callable[[], dict] | None" = None,
    pairwise_judge: "Callable[[Path, Path], str | None] | None" = None,
    rubric_judge: "Callable[[Path], dict | None] | None" = None,
) -> dict:
    """Drive the LLM to iteratively revise a GLSL candidate.

    ``evaluate_fn(glsl, render_path)`` must return
    ``(metrics, quality_dict, score, actual_render_path)`` and must not raise
    (return a 0.0 score with ``None`` render path on failure instead).

    ``rubric_judge(render_path)`` optionally returns a judge_rubric dict whose
    differences/revision_hints are injected as semantic feedback.

    Returns dict with keys: best_glsl, best_score, best_metrics, best_quality,
    history, stop_reason — the same shape as ``run_dsl_refinement_loop``.
    """
    from app.candidates.llm_scene import generate_llm_glsl_refinement

    loop_dir.mkdir(parents=True, exist_ok=True)
    render_dir = loop_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "glsl refinement start: initial_score=%.4f threshold=%.2f high_stop=%.2f max_iter=%d",
        float(initial_score), float(threshold), float(high_score_stop), int(max_iterations),
    )

    best_glsl = initial_glsl
    best_score = initial_score
    best_metrics = dict(initial_metrics)
    best_quality = dict(initial_quality)
    current_render_path = initial_render_path

    history: list[dict] = []
    stop_reason = "max_iterations"
    no_improvement_count = 0
    extra_feedback: list[str] = []
    fresh_restarts_left = max_fresh_restarts
    pending_fresh_start = False
    stagnation_anchor = 0

    def _trigger_fresh_restart() -> bool:
        """Consume one fresh restart if available; reset counters/windows."""
        nonlocal fresh_restarts_left, pending_fresh_start
        nonlocal stagnation_anchor, no_improvement_count, extra_feedback
        if fresh_restarts_left <= 0:
            return False
        fresh_restarts_left -= 1
        pending_fresh_start = True
        stagnation_anchor = len(history)
        no_improvement_count = 0
        extra_feedback = [_build_failure_log_note(history, best_score)]
        logger.info("glsl refinement: fresh restart triggered")
        return True

    for i in range(max_iterations):
        # Read latest strategy + stop flag (one-shot per iteration).
        if strategy_reader is not None:
            try:
                live = strategy_reader()
            except Exception:
                live = {}
            if live.get("stop_requested"):
                stop_reason = "user_stop"
                break
            live_strategy = live.get("strategy") or {}
            if "refinement_threshold" in live_strategy:
                threshold = float(live_strategy["refinement_threshold"])
            if "refinement_high_score_stop" in live_strategy:
                high_score_stop = float(live_strategy["refinement_high_score_stop"])
            if "refinement_min_improvement" in live_strategy:
                min_improvement = float(live_strategy["refinement_min_improvement"])
            if "refinement_patience" in live_strategy:
                no_improvement_patience = int(live_strategy["refinement_patience"])
            if "max_refinement_iterations" in live_strategy:
                if i >= int(live_strategy["max_refinement_iterations"]):
                    stop_reason = "user_lowered_cap"
                    break

        if best_score >= high_score_stop:
            stop_reason = "high_score_stop"
            break
        if best_score >= threshold and not (force_first_iteration and not history):
            stop_reason = "threshold_reached"
            break

        # Stagnation over entries since the last fresh restart.
        scored_entries = [
            h["score_after"] for h in history[stagnation_anchor:]
            if h.get("score_after") is not None
        ]
        if len(scored_entries) >= 3:
            recent = scored_entries[-3:]
            if max(recent) - min(recent) < 0.02:
                if not _trigger_fresh_restart():
                    stop_reason = "stagnation"
                    break

        was_fresh = pending_fresh_start
        pending_fresh_start = False

        entry: dict = {
            "iteration": i + 1,
            "score_before": round(best_score, 4),
            "score_after": None,
            "delta": None,
            "improved": False,
            "meaningful_improvement": False,
            "fresh_start": was_fresh,
            "changes_summary": None,
            "llm_io": None,
            "llm_duration_ms": None,
            "error": None,
            "error_type": None,
            "compile_glsl": None,
        }

        region_notes: list[str] = []
        if current_render_path is not None:
            try:
                from app.metrics.compute import grid_color_report
                region_notes = grid_color_report(reference_path, current_render_path)
            except Exception:
                logger.warning("grid_color_report failed", exc_info=True)

        semantic_notes: list[str] = []
        if rubric_judge is not None and current_render_path is not None:
            try:
                rubric = rubric_judge(current_render_path)
            except Exception:
                rubric = None
                logger.warning("rubric judge failed", exc_info=True)
            if rubric:
                semantic_notes = build_semantic_notes(rubric)

        history_notes = build_recent_history_notes(history)

        llm_start = time.monotonic()
        try:
            revised = generate_llm_glsl_refinement(
                current_glsl=best_glsl,
                metrics=best_metrics,
                quality_router=best_quality,
                reference_image_path=reference_path,
                current_render_path=current_render_path,
                extra_feedback=(
                    extra_feedback + history_notes + semantic_notes + region_notes
                ) or None,
                fresh_start=was_fresh,
            )
        except Exception as exc:
            entry["llm_duration_ms"] = int((time.monotonic() - llm_start) * 1000)
            entry["error_type"] = exc.__class__.__name__
            entry["error"] = f"LLM call failed: {_short_exception(exc)}"
            history.append(entry)
            save_json(loop_dir / f"iter_{i + 1}.json", entry)
            stop_reason = "llm_call_failed"
            break

        entry["llm_duration_ms"] = int((time.monotonic() - llm_start) * 1000)
        if revised and isinstance(revised, dict):
            entry["llm_io"] = revised.pop("_io", None)

        revised_glsl = revised.get("glsl") if isinstance(revised, dict) else None
        if not revised_glsl:
            entry["error_type"] = "llm_returned_none"
            entry["error"] = (
                "LLM returned no usable GLSL: response content was empty, was "
                "not valid JSON, or did not contain a mainImage entry point."
            )
            logger.info("glsl refinement iter=%d skipped: llm_returned_none", i + 1)
            history.append(entry)
            save_json(loop_dir / f"iter_{i + 1}.json", entry)
            stop_reason = "llm_returned_none"
            break

        static = validate_shader_static(revised_glsl)
        if not static["valid"]:
            entry["error"] = f"GLSL invalid: {static['errors'][:2]}"
            logger.info(
                "glsl refinement iter=%d skipped: static_invalid errors=%s",
                i + 1, static["errors"][:2],
            )
            history.append(entry)
            save_json(loop_dir / f"iter_{i + 1}.json", entry)
            no_improvement_count += 1
            extra_feedback = [
                "[COMPILE FEEDBACK] Your last revision failed static validation: "
                + "; ".join(static["errors"][:3])
                + ". Fix these issues and return the full corrected shader."
            ]
            if no_improvement_count >= no_improvement_patience and not _trigger_fresh_restart():
                stop_reason = "no_improvement_patience"
                break
            continue

        # Per-iteration GLSL so the frontend can preview exactly this
        # iteration's shader (same contract as the DSL loop).
        entry["compile_glsl"] = revised_glsl

        render_path = render_dir / f"iter_{i + 1}.png"
        new_metrics, new_quality, new_score, actual_render = evaluate_fn(
            revised_glsl, render_path
        )

        # Render failure is a code problem, not a visual one — feed it back
        # as such instead of producing a misleading rollback note.
        if actual_render is None and new_score <= 0.0:
            entry["error_type"] = "render_failed"
            entry["error"] = (
                "render failed: WebGL produced no screenshot "
                "(compile or runtime error)"
            )
            entry["score_after"] = 0.0
            history.append(entry)
            save_json(loop_dir / f"iter_{i + 1}.json", entry)
            no_improvement_count += 1
            extra_feedback = [
                "[RENDER FAILED] The revised shader passed static checks but "
                "failed to render in WebGL (likely a GLSL compile/runtime "
                "error: undefined symbol, int/float mismatch, or an "
                "out-of-range loop). Fix the shader so it renders; keep the "
                "same visual intent."
            ]
            if no_improvement_count >= no_improvement_patience and not _trigger_fresh_restart():
                stop_reason = "no_improvement_patience"
                break
            continue

        entry["score_after"] = round(new_score, 4)
        delta = new_score - best_score
        entry["delta"] = round(delta, 4)
        entry["improved"] = delta > 0.0
        entry["meaningful_improvement"] = delta >= min_improvement
        entry["changes_summary"] = _diff_glsl_summary(best_glsl, revised_glsl)
        logger.info(
            "glsl refinement iter=%d before=%.4f after=%.4f delta=%+.4f improved=%s fresh=%s changes=%s",
            i + 1, float(entry["score_before"]), float(new_score), float(delta),
            bool(entry["improved"]), was_fresh, (entry["changes_summary"] or "")[:120],
        )

        # Arbitrate noise-level gains, same as the DSL loop.
        if (
            pairwise_judge is not None
            and 0.0 < delta < min_improvement
            and current_render_path is not None
            and actual_render is not None
        ):
            verdict = pairwise_judge(current_render_path, actual_render)
            if verdict == "A":  # judge prefers the previous best
                entry["vlm_override"] = "veto_small_gain"
                delta = 0.0
                entry["improved"] = False

        if delta > 0.0:
            best_glsl = revised_glsl
            best_score = new_score
            best_metrics = new_metrics
            best_quality = new_quality
            if actual_render is not None:
                current_render_path = actual_render
            extra_feedback = []
        else:
            extra_feedback = [
                f"[ROLLBACK] Your last revision dropped the score from "
                f"{best_score:.3f} to {new_score:.3f}. The system reverted to "
                f"the previous best version. Changes were: {entry['changes_summary']}. "
                f"Do NOT repeat the same approach. Try a different strategy."
            ]

        if delta >= min_improvement:
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        history.append(entry)
        save_json(loop_dir / f"iter_{i + 1}.json", entry)

        if no_improvement_count >= no_improvement_patience and not _trigger_fresh_restart():
            stop_reason = "no_improvement_patience"
            break

    logger.info(
        "glsl refinement done: stop_reason=%s best_score=%.4f iters=%d",
        stop_reason, float(best_score), len(history),
    )
    return {
        "best_glsl": best_glsl,
        "best_score": best_score,
        "best_metrics": best_metrics,
        "best_quality": best_quality,
        "history": history,
        "stop_reason": stop_reason,
    }
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend && python -m pytest tests/unit/test_glsl_refinement.py -v`
Expected: 12 个测试全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/glsl_refinement.py backend/app/pipeline/refinement.py backend/tests/unit/test_glsl_refinement.py
git commit -m "feat(pipeline): add GLSL refinement loop with semantic feedback, gradient window, fresh restart"
```

---

### Task 5: `_should_run_refinement` 放开 GLSL 候选

**Files:**
- Modify: `backend/app/pipeline/refinement.py`（`_should_run_refinement`）
- Test: `backend/tests/unit/test_graph.py`

- [ ] **Step 1: 修改既有测试 + 新增测试**

在 `backend/tests/unit/test_graph.py` 中，把 `test_should_run_refinement_skips_non_dsl_candidate`（约 161-173 行）整体替换为：

```python
def test_should_run_refinement_allows_glsl_candidate():
    """GLSL candidates with compiled shader text are now refinable."""
    candidate = _make_refinement_candidate(score=0.3, has_dsl=False)

    should_run, reason = _should_run_refinement(
        "on",
        candidate,
        {"final_score": 0.3},
        threshold=0.8,
        high_score_stop=0.92,
    )

    assert should_run is True
    assert reason == "force_enabled"


def test_should_run_refinement_skips_unrefinable_candidate():
    """No DSL and no compiled GLSL -> nothing to refine."""
    candidate = _make_refinement_candidate(score=0.3, has_dsl=False)
    candidate.compile_glsl = ""
    candidate.compile_success = False

    should_run, reason = _should_run_refinement(
        "on",
        candidate,
        {"final_score": 0.3},
        threshold=0.8,
        high_score_stop=0.92,
    )

    assert should_run is False
    assert reason == "selected_candidate_not_refinable"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend && python -m pytest tests/unit/test_graph.py::test_should_run_refinement_allows_glsl_candidate tests/unit/test_graph.py::test_should_run_refinement_skips_unrefinable_candidate -v`
Expected: 两个都 FAIL（现状返回 `selected_candidate_is_not_dsl`）。

- [ ] **Step 3: 实现**

在 `backend/app/pipeline/refinement.py` 中，把 `_should_run_refinement` 的开头

```python
    if refinement_mode == "off":
        return False, "refinement_mode_off"
    if selected is None:
        return False, "no_selected_candidate"
    if selected.dsl is None:
        return False, "selected_candidate_is_not_dsl"
    if selected_quality is None:
        return False, "missing_quality_router"
```

改为

```python
    if refinement_mode == "off":
        return False, "refinement_mode_off"
    if selected is None:
        return False, "no_selected_candidate"
    is_dsl = selected.dsl is not None
    is_glsl = (
        not is_dsl
        and selected.output_kind == "glsl"
        and selected.compile_success
        and bool(selected.compile_glsl)
    )
    if not is_dsl and not is_glsl:
        return False, "selected_candidate_not_refinable"
    if selected_quality is None:
        return False, "missing_quality_router"
```

其余分数门控逻辑不变。同时把函数 docstring 改为 `"""Decide whether to enter the LLM refinement loop (DSL or GLSL)."""`。

- [ ] **Step 4: 运行 test_graph 全文件确认通过**

Run: `cd backend && python -m pytest tests/unit/test_graph.py -v`
Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/refinement.py backend/tests/unit/test_graph.py
git commit -m "feat(refinement): allow GLSL candidates through the refinement gate"
```

---

### Task 6: `node_selection` 移除精修对 GLSL 的偏置

现状：开启精修时 `prefer_output_kind` 不会设为 `"glsl"`，等于变相禁用 GLSL 路线。GLSL 现在可精修，该偏置应删除。

**Files:**
- Modify: `backend/app/pipeline/graph.py`（`node_selection`）
- Test: `backend/tests/unit/test_graph.py`

- [ ] **Step 1: 写失败测试**

在 `backend/tests/unit/test_graph.py` 顶部 import 区域追加 `node_selection`（来源 `app.pipeline.graph`）。然后添加测试：

```python
def test_node_selection_prefers_glsl_even_when_refinement_requested():
    """GLSL preference must not be disabled by an active refinement config —
    GLSL candidates are refinable now. Equal scores: preference decides."""
    glsl_cand = CandidateRecord(
        id="llm_0", source="llm", enabled=True, priority=5,
        dsl=None, output_kind="glsl",
        validation_valid=True, validation_errors=[],
        compile_success=True,
        compile_glsl="void mainImage(out vec4 fragColor, in vec2 fragCoord){fragColor=vec4(1.0);}",
        compile_errors=[], final_score=0.7, selected=False,
    )
    dsl_cand = CandidateRecord(
        id="rule_0", source="rule", enabled=True, priority=1,
        dsl={"layers": []}, output_kind="dsl",
        validation_valid=True, validation_errors=[],
        compile_success=True,
        compile_glsl="void mainImage(out vec4 fragColor, in vec2 fragCoord){fragColor=vec4(0.0);}",
        compile_errors=[], final_score=0.7, selected=False,
    )
    state = {
        "candidates": [dsl_cand, glsl_cand],
        "glsl_render_enabled": True,
        "refinement_mode": "on",
        "max_refinement_iterations": 3,
    }

    result = node_selection(state)

    assert result["selected_candidate_id"] == "llm_0"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend && python -m pytest tests/unit/test_graph.py::test_node_selection_prefers_glsl_even_when_refinement_requested -v`
Expected: FAIL——现状 `refinement_requested=True` 导致 `prefer_output_kind=None`，同分时按 priority 选中 `rule_0`。

- [ ] **Step 3: 实现**

在 `backend/app/pipeline/graph.py` 的 `node_selection` 中，把

```python
    # Determine preference for GLSL output
    prefer_output_kind = None
    refinement_requested = (
        state.get("refinement_mode", "auto") != "off"
        and state.get("max_refinement_iterations", 0) > 0
    )

    if state.get("glsl_render_enabled", False) and not refinement_requested:
```

改为

```python
    # Determine preference for GLSL output. GLSL candidates are refinable via
    # run_glsl_refinement_loop, so an active refinement config no longer
    # disables the preference.
    prefer_output_kind = None
    if state.get("glsl_render_enabled", False):
```

（`best_glsl_score` / `best_dsl_score` 的计算和判断保持不变。）

- [ ] **Step 4: 运行 test_graph 全文件确认通过**

Run: `cd backend && python -m pytest tests/unit/test_graph.py -v`
Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/graph.py backend/tests/unit/test_graph.py
git commit -m "fix(selection): stop suppressing GLSL preference when refinement is enabled"
```

---

### Task 7: `_run_post_pipeline` 接线 GLSL 精修分支（含 rubric_judge）

**Files:**
- Modify: `backend/app/pipeline/graph.py`（import 区、`_run_post_pipeline`、`CORE_PIPELINE_FLOWCHART`）
- Test: `backend/tests/unit/test_graph.py`

- [ ] **Step 1: 写失败测试**

在 `backend/tests/unit/test_graph.py` 添加（`_run_post_pipeline` 从 `app.pipeline.graph` 导入，加入顶部 import）：

```python
def test_run_post_pipeline_runs_glsl_refinement(tmp_path, monkeypatch):
    glsl = "void mainImage(out vec4 fragColor, in vec2 fragCoord){fragColor=vec4(1.0);}"
    cand = CandidateRecord(
        id="llm_0", source="llm", enabled=True, priority=5,
        dsl=None, output_kind="glsl",
        validation_valid=True, validation_errors=[],
        compile_success=True, compile_glsl=glsl,
        compile_errors=[], final_score=0.4, selected=True,
        quality_router={"final_score": 0.4, "next_action": "accept"},
    )
    improved_glsl = "#define R 0.9\n" + glsl

    def fake_loop(*args, **kwargs):
        return {
            "best_glsl": improved_glsl,
            "best_score": 0.8,
            "best_metrics": {"mse": 0.1},
            "best_quality": {"final_score": 0.8, "next_action": "accept"},
            "history": [{"iteration": 1, "score_after": 0.8, "improved": True}],
            "stop_reason": "threshold_reached",
        }

    monkeypatch.setattr("app.pipeline.graph.run_glsl_refinement_loop", fake_loop)

    state = {
        "selected_candidate_id": "llm_0",
        "candidates": [cand],
        "run_dir": str(tmp_path),
        "selected_dsl": None,
        "selected_glsl": glsl,
        "selected_metrics": {"mse": 0.5},
        "selected_quality": {"final_score": 0.4, "next_action": "accept"},
        "refinement_mode": "on",
        "max_refinement_iterations": 2,
        "llm_enabled": False,
        "glsl_render_enabled": True,
        "vlm_judge_enabled": False,
    }

    result = _run_post_pipeline(state)

    assert result["selected_glsl"] == improved_glsl
    assert result["refinement_summary"]["enabled"] is True
    assert result["refinement_summary"]["decision"] == "force_enabled"
    assert result["refinement_summary"]["improved"] is True
    assert result["refinement_summary"]["stop_reason"] == "threshold_reached"
    assert cand.final_score == 0.8
    assert cand.compile_glsl == improved_glsl
    assert (tmp_path / "selected_shader.glsl").read_text(encoding="utf-8") == improved_glsl


def test_run_post_pipeline_skips_glsl_refinement_when_render_disabled(tmp_path, monkeypatch):
    """Without the WebGL renderer there is no way to score revisions."""
    glsl = "void mainImage(out vec4 fragColor, in vec2 fragCoord){fragColor=vec4(1.0);}"
    cand = CandidateRecord(
        id="llm_0", source="llm", enabled=True, priority=5,
        dsl=None, output_kind="glsl",
        validation_valid=True, validation_errors=[],
        compile_success=True, compile_glsl=glsl,
        compile_errors=[], final_score=0.4, selected=True,
        quality_router={"final_score": 0.4, "next_action": "accept"},
    )

    def fail_loop(*args, **kwargs):
        raise AssertionError("refinement loop must not run without the renderer")

    monkeypatch.setattr("app.pipeline.graph.run_glsl_refinement_loop", fail_loop)

    state = {
        "selected_candidate_id": "llm_0",
        "candidates": [cand],
        "run_dir": str(tmp_path),
        "selected_dsl": None,
        "selected_glsl": glsl,
        "selected_metrics": {},
        "selected_quality": {"final_score": 0.4, "next_action": "accept"},
        "refinement_mode": "on",
        "max_refinement_iterations": 2,
        "llm_enabled": False,
        "glsl_render_enabled": False,
        "vlm_judge_enabled": False,
    }

    result = _run_post_pipeline(state)

    assert result["selected_glsl"] == glsl
    assert result["refinement_summary"]["decision"] == "glsl_render_disabled"
    assert result["refinement_summary"]["enabled"] is False
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend && python -m pytest tests/unit/test_graph.py::test_run_post_pipeline_runs_glsl_refinement -v`
Expected: FAIL with `AttributeError: <module 'app.pipeline.graph'> does not have the attribute 'run_glsl_refinement_loop'`

- [ ] **Step 3: 实现接线**

3a. `backend/app/pipeline/graph.py` import 区，在 `from app.pipeline.glsl_optimizer import (...)` 之后添加：

```python
from app.pipeline.glsl_refinement import run_glsl_refinement_loop
```

3b. 在 `_run_post_pipeline` 中，找到 DSL 精修块（`if should_refine and selected and selected.dsl:` 整块），在它结束之后、`# VLM final gate` 注释之前，插入 GLSL 分支：

```python
    elif should_refine and selected and selected.output_kind == "glsl" and selected.compile_glsl:
        if not state.get("glsl_render_enabled", False):
            refinement_summary["enabled"] = False
            refinement_summary["decision"] = "glsl_render_disabled"
        else:
            initial_refinement_score = selected.final_score

            def _glsl_refinement_evaluate(
                glsl: str, render_path: Path
            ) -> tuple[dict, dict, float, "Path | None"]:
                try:
                    return _evaluate_glsl_with_webgl(
                        glsl,
                        reference_path,
                        render_path,
                        canvas_width=canvas_width,
                        canvas_height=canvas_height,
                        max_shader_chars=max_shader_chars,
                    )
                except Exception:
                    logger.exception("glsl refinement evaluation failed")
                    return {}, {}, 0.0, None

            ref_result = run_glsl_refinement_loop(
                selected.compile_glsl,
                selected.final_score,
                dict(selected.objective_metrics),
                dict(selected.quality_router) if selected.quality_router else {},
                reference_path,
                evaluate_fn=_glsl_refinement_evaluate,
                initial_render_path=Path(selected.render_path) if selected.render_path else None,
                max_iterations=max_refinement_iterations,
                threshold=refinement_threshold,
                high_score_stop=refinement_high_score_stop,
                min_improvement=refinement_min_improvement,
                no_improvement_patience=refinement_patience,
                force_first_iteration=effective_refinement_mode == "on",
                loop_dir=run_dir / "glsl_refinement",
                strategy_reader=strategy_reader,
                pairwise_judge=(
                    (lambda cur, new: judge_pairwise(
                        reference_path, cur, new, work_dir=run_dir / "judge"
                    ))
                    if state.get("vlm_judge_enabled") else None
                ),
                rubric_judge=(
                    (lambda render: judge_rubric(
                        reference_path, render, work_dir=run_dir / "judge"
                    ))
                    if state.get("vlm_judge_enabled") else None
                ),
            )
            refinement_history = ref_result.get("history", [])
            refinement_summary.update({
                "kind": "glsl",
                "iterations": len(refinement_history),
                "initial_score": initial_refinement_score,
                "final_score": ref_result.get("best_score", initial_refinement_score),
                "improved": ref_result.get("best_score", 0) > initial_refinement_score,
                "stop_reason": ref_result.get("stop_reason"),
            })

            if ref_result.get("best_score", 0) > selected.final_score:
                selected.compile_glsl = ref_result["best_glsl"]
                selected.objective_metrics = ref_result["best_metrics"]
                selected.quality_router = ref_result["best_quality"]
                selected.final_score = ref_result["best_score"]
                selected_glsl = selected.compile_glsl
                selected_metrics = ref_result["best_metrics"]
                selected_quality = ref_result["best_quality"]
```

注意：`refinement_threshold` / `refinement_high_score_stop` / `refinement_min_improvement` / `refinement_patience` / `effective_refinement_mode` / `max_refinement_iterations` 这些局部变量在 `_run_post_pipeline` 中已于精修配置读取段定义，直接复用；`judge_rubric` 已在 graph.py 顶部 import。

3c. 同步维护开发者流程图（graph.py 文件头注释块和 `CORE_PIPELINE_FLOWCHART`，文件头明确要求同步）。注释块中 `LLM refinement` 段落的"闭环精修 DSL"改为"闭环精修 DSL/GLSL"；`CORE_PIPELINE_FLOWCHART` mermaid 中把

```
        Y["run_dsl_refinement_loop<br/>支持运行中策略更新和停止"]
```

改为

```
        Y["refinement loop（DSL/GLSL）<br/>支持运行中策略更新和停止"]
```

- [ ] **Step 4: 运行 test_graph 全文件确认通过**

Run: `cd backend && python -m pytest tests/unit/test_graph.py -v`
Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/graph.py backend/tests/unit/test_graph.py
git commit -m "feat(pipeline): wire GLSL refinement loop into post-pipeline with VLM rubric feedback"
```

---

### Task 8: DSL 精修循环回灌语义反馈 + 梯度窗口

把 VFX 机制 ①② 同样应用到既有 DSL 循环：`rubric_judge` 语义反馈 + 最近 3 轮历史摘要（此前 DSL 循环的反馈只有指标数字、分区颜色报告和单条 rollback note）。

**Files:**
- Modify: `backend/app/pipeline/refinement.py`（`run_dsl_refinement_loop`）
- Modify: `backend/app/pipeline/graph.py`（DSL 精修调用处传入 `rubric_judge`）
- Test: `backend/tests/unit/test_graph.py`

- [ ] **Step 1: 写失败测试**

在 `backend/tests/unit/test_graph.py` 添加：

```python
def test_dsl_refinement_feeds_history_and_semantic_notes(tmp_path, monkeypatch):
    from types import SimpleNamespace

    captured: list[dict] = []

    def fake_refinement(**kwargs):
        captured.append(kwargs)
        return {"layers": [], "_io": {}}

    monkeypatch.setattr(
        "app.candidates.llm_scene.generate_llm_refinement", fake_refinement
    )
    monkeypatch.setattr(
        "app.pipeline.refinement.render_dsl_to_image", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "app.dsl.validator.validate_dsl",
        lambda d: SimpleNamespace(valid=True, errors=[]),
    )
    monkeypatch.setattr(
        "app.dsl.compiler.compile_dsl",
        lambda d: SimpleNamespace(success=True, glsl="void mainImage(){}", errors=[]),
    )
    monkeypatch.setattr(
        "app.pipeline.refinement._evaluate_dsl",
        lambda *a, **k: ({}, {"final_score": 0.2}, 0.2, None),
    )

    run_dsl_refinement_loop(
        preprocess={},
        initial_dsl={"layers": [{"id": "a", "type": "circle", "params": {}}]},
        initial_score=0.5,
        initial_metrics={},
        initial_quality={"final_score": 0.5},
        reference_path=tmp_path / "ref.png",
        canvas_width=512,
        canvas_height=512,
        max_shader_chars=12000,
        max_iterations=2,
        threshold=0.9,
        high_score_stop=0.95,
        no_improvement_patience=3,
        loop_dir=tmp_path / "loop",
        rubric_judge=lambda render: {
            "differences": ["edges too sharp"],
            "revision_hints": ["soften the edges"],
        },
    )

    assert len(captured) == 2
    second_fb = captured[1]["extra_feedback"]
    assert any("[HISTORY iter 1]" in n for n in second_fb)
    assert any("[VISUAL ISSUE] edges too sharp" in n for n in second_fb)
    assert any("[ROLLBACK]" in n for n in second_fb)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend && python -m pytest tests/unit/test_graph.py::test_dsl_refinement_feeds_history_and_semantic_notes -v`
Expected: FAIL with `TypeError: run_dsl_refinement_loop() got an unexpected keyword argument 'rubric_judge'`

- [ ] **Step 3: 实现**

3a. 在 `backend/app/pipeline/refinement.py` 的 `run_dsl_refinement_loop` 签名中，`pairwise_judge` 参数之后添加：

```python
    rubric_judge: "Callable[[Path], dict | None] | None" = None,
```

3b. 在循环体内，找到

```python
        region_notes: list[str] = []
        if current_render_path is not None:
            try:
                from app.metrics.compute import grid_color_report
                region_notes = grid_color_report(reference_path, current_render_path)
            except Exception:
                logger.warning("grid_color_report failed", exc_info=True)
```

在其后添加：

```python
        semantic_notes: list[str] = []
        if rubric_judge is not None and current_render_path is not None:
            try:
                rubric = rubric_judge(current_render_path)
            except Exception:
                rubric = None
                logger.warning("rubric judge failed", exc_info=True)
            if rubric:
                semantic_notes = build_semantic_notes(rubric)

        history_notes = build_recent_history_notes(history)
```

并把紧随其后的 `generate_llm_refinement(...)` 调用中的

```python
                extra_feedback=(extra_feedback + region_notes) or None,
```

改为

```python
                extra_feedback=(
                    extra_feedback + history_notes + semantic_notes + region_notes
                ) or None,
```

（`build_semantic_notes` / `build_recent_history_notes` 在 Task 4 已加入本文件，直接使用。）

3c. 在 `backend/app/pipeline/graph.py` 的 `_run_post_pipeline` 中，DSL 精修调用 `run_dsl_refinement_loop(...)` 的 `pairwise_judge=...` 实参之后添加：

```python
            rubric_judge=(
                (lambda render: judge_rubric(
                    reference_path, render, work_dir=run_dir / "judge"
                ))
                if state.get("vlm_judge_enabled") else None
            ),
```

- [ ] **Step 4: 运行 test_graph 全文件确认通过**

Run: `cd backend && python -m pytest tests/unit/test_graph.py -v`
Expected: 全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/refinement.py backend/app/pipeline/graph.py backend/tests/unit/test_graph.py
git commit -m "feat(refinement): feed VLM semantic notes and gradient-window history into the DSL loop"
```

---

### Task 9: 全量回归 + 收尾

**Files:**
- 无新增代码；必要时修复回归。

- [ ] **Step 1: 跑全部单测**

Run: `cd backend && python -m pytest tests/unit -v`
Expected: 全部 PASS。若有失败，按失败信息修复（最可能受影响的是依赖 `node_selection` 旧偏置行为或 `_should_run_refinement` 旧 reason 字符串的测试）。

- [ ] **Step 2: 跑 e2e（如环境允许）**

Run: `cd backend && python -m pytest tests/e2e -v`
Expected: PASS 或与改动无关的既有跳过/失败。涉及 WebGL/LLM 的用例若因环境缺失跳过属正常。

- [ ] **Step 3: Commit（如有修复）**

```bash
git add -A backend
git commit -m "test: fix regressions from GLSL refinement wiring"
```

---

## 验收清单（对照需求 + VFX 对标）

- [x] 修复 glsl_optimizer 问题 → Task 1（早停）、Task 2（相对步长）
- [x] 新增 GLSL LLM 精修 → Task 3（修订函数 + fresh_start）、Task 4（闭环循环）
- [x] VFX ① 语义反馈驱动 → Task 4（GLSL）、Task 8（DSL），复用既有 `judge_rubric` 的 differences/revision_hints
- [x] VFX ② 梯度窗口 → Task 4 / Task 8 的 `build_recent_history_notes`（不含代码体）
- [x] VFX ③ 物理回滚 → 既有 best-only 设计已等价，无需改动
- [x] VFX ④ 停滞重启（re-decompose 等价物）→ Task 3 fresh_start + Task 4 失败日志触发
- [x] VFX ⑤ 多层守门 → 静态校验回喂（Task 4）+ 渲染失败专用反馈（Task 4）
- [x] 精修门控放开 GLSL → Task 5
- [x] 解除选择偏置 → Task 6
- [x] 接线 + 流程图同步 → Task 7
- [x] 回归 → Task 9

## 后续增强（本计划不做）

- 精修循环内交替运行小预算 `#define` 坐标下降（LLM 改结构 + 数值收敛）。
- vec 参数逐分量扰动（当前所有分量同向平移）。
- `refinement_threshold` 默认值校准（VFX 用 0.85 通过阈值，P2S 默认 0.5——待真实样本基线验证后调整）。
- DSL 循环的 fresh restart（DSL 已有 revision patch / 残差补层作为结构性逃逸通道，优先级低）。
