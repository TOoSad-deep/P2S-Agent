# Seed-GLSL 闭环优化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增「PNG + 已有 GLSL → 闭环优化」入口：用外部 GLSL 伪造一个 selected GLSL 候选，跳过候选池，直接复用现有 `_run_post_pipeline`（`#define` 优化器 + LLM 精修循环 + VLM 门）。

**Architecture:** 在 `run_png_shader_pipeline` 增加 `seed_glsl` 分支：preprocess → 格式适配(`adapt_seed_glsl`) → 合成 `CandidateRecord` → `_evaluate_glsl_with_webgl` 评分 → 写 `selected_*` → `_run_post_pipeline`（原样复用）。前端在现有上传界面加 seed 输入，HTTP 复用 `/png-shader/run`（新增 `seed_glsl` Form 字段）。

**Tech Stack:** Python 3 / pytest / FastAPI / LangGraph（仅 post-pipeline 复用）/ 现有 WebGL 渲染评分 / React + Vite + TypeScript（前端无测试 runner，门禁=tsc+eslint）。

**测试命令约定:** 后端 `cd backend && python -m pytest <path> -v`；前端 `cd frontend && npm run build && npm run lint`。

**设计依据:** [doc/2026-06-14-seed-glsl-closed-loop-design.md](2026-06-14-seed-glsl-closed-loop-design.md)

---

## 文件结构总览

| 文件 | 操作 | 职责 |
|---|---|---|
| `backend/app/pipeline/seed_glsl.py` | 新建 | `SeedAdaptResult`、`adapt_seed_glsl`（normalize→wrap→LLM 兜底）、`build_seed_candidate` |
| `backend/tests/unit/test_seed_glsl.py` | 新建 | 适配与候选构造单测 |
| `backend/app/pipeline/graph.py` | 修改 | `run_png_shader_pipeline` 增 `seed_glsl` 参数 + `_run_seed_glsl_path` 助手；更新编排注释 |
| `backend/tests/unit/test_graph.py` | 修改 | seed 路径集成测试（跳过候选池、进入精修循环） |
| `backend/app/routers/png_shader.py` | 修改 | `/png-shader/run` 增 `seed_glsl` Form 字段；seed 默认 `refinement_mode="on"`；透传到后台 worker |
| `backend/tests/unit/test_router.py` | 修改 | seed 请求契约测试 |
| `frontend/src/hooks/usePngShader.ts` | 修改 | `runPngShader(file, seedGlsl?)` 追加 `seed_glsl` 到 FormData |
| `frontend/src/components/PngShaderView.tsx` | 修改 | seed 开关 + 文本框 + `.glsl` 文件读入；`onRun` 透传 seedGlsl |
| `frontend/src/App.tsx` | 修改 | `handleRun(file, seedGlsl?)` 透传 |

---

## Task 1: `adapt_seed_glsl` 格式适配（normalize → 确定性包装 → LLM 兜底）

**Files:**
- Create: `backend/app/pipeline/seed_glsl.py`
- Test: `backend/tests/unit/test_seed_glsl.py`

- [ ] **Step 1: 写失败测试**

创建 `backend/tests/unit/test_seed_glsl.py`：

```python
"""Tests for seed-GLSL adaptation and candidate construction."""

from __future__ import annotations

import json

from app.pipeline.seed_glsl import (
    SeedAdaptResult,
    adapt_seed_glsl,
    build_seed_candidate,
)

VALID_SHADERTOY = (
    "#define R 0.30\n"
    "void mainImage(out vec4 fragColor, in vec2 fragCoord) { fragColor = vec4(R); }"
)
LEGACY_MAIN = "void main() { gl_FragColor = vec4(0.3, 0.4, 0.5, 1.0); }"
UNWRAPPABLE = "float helper(float x) { return x * 2.0; }"


def test_valid_shadertoy_passes_through_as_normalized():
    result = adapt_seed_glsl(VALID_SHADERTOY)
    assert isinstance(result, SeedAdaptResult)
    assert result.valid is True
    assert result.adapted_by == "normalized"
    assert "void mainImage" in result.glsl


def test_legacy_main_is_wrapped_into_mainimage():
    result = adapt_seed_glsl(LEGACY_MAIN)
    assert result.valid is True
    assert result.adapted_by == "wrapped"
    assert "void mainImage(out vec4 fragColor, in vec2 fragCoord)" in result.glsl
    assert "fragColor = vec4(0.3, 0.4, 0.5, 1.0)" in result.glsl
    assert "gl_FragColor" not in result.glsl


def test_unwrappable_falls_back_to_llm_port():
    def fake_client(system_prompt, user_prompt, image_paths=None):
        return json.dumps({"glsl": VALID_SHADERTOY})

    result = adapt_seed_glsl(UNWRAPPABLE, llm_client=fake_client)
    assert result.valid is True
    assert result.adapted_by == "llm_ported"
    assert "void mainImage" in result.glsl


def test_invalid_when_all_stages_fail():
    def empty_client(system_prompt, user_prompt, image_paths=None):
        return ""

    result = adapt_seed_glsl(UNWRAPPABLE, llm_client=empty_client)
    assert result.valid is False
    assert result.adapted_by == "failed"
    assert result.errors


def test_empty_source_is_invalid():
    result = adapt_seed_glsl("   ")
    assert result.valid is False
    assert result.adapted_by == "failed"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend && python -m pytest tests/unit/test_seed_glsl.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.pipeline.seed_glsl'`

- [ ] **Step 3: 实现 `seed_glsl.py`（适配部分）**

创建 `backend/app/pipeline/seed_glsl.py`：

```python
"""Seed-GLSL adaptation and candidate construction.

Turns an externally-supplied GLSL shader (possibly not in Shadertoy form)
into a renderable Shadertoy ``mainImage`` shader and wraps it as a single
``CandidateRecord``, so the existing post-pipeline closed loop can refine it
without running candidate generation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable

from app.pipeline.pool import CandidateRecord
from app.services.shader_validator import validate_shader_static
from app.utils.glsl_postprocess import normalize_shadertoy_glsl

logger = logging.getLogger(__name__)


@dataclass
class SeedAdaptResult:
    """Outcome of adapting a seed shader to renderable Shadertoy GLSL."""

    glsl: str
    valid: bool
    adapted_by: str  # "normalized" | "wrapped" | "llm_ported" | "failed"
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


# `void main()` / `void main(void)` signature of a legacy fragment shader.
_MAIN_SIG_RE = re.compile(r"\bvoid\s+main\s*\(\s*(?:void)?\s*\)")


def _wrap_legacy_main(glsl: str) -> str | None:
    """Best-effort rewrite of a ``void main(){...gl_FragColor...}`` fragment
    shader into a Shadertoy ``mainImage``.

    Returns the rewritten shader, or ``None`` when *glsl* is not a recognizable
    legacy fragment shader (already has ``mainImage``, no ``main()``, or never
    writes ``gl_FragColor``). The transform is textual and best-effort: when it
    produces something that still fails validation, the caller falls back to the
    LLM port stage.
    """
    if "void mainImage" in glsl:
        return None
    if not _MAIN_SIG_RE.search(glsl):
        return None
    if "gl_FragColor" not in glsl and "gl_FragData" not in glsl:
        return None
    wrapped = _MAIN_SIG_RE.sub(
        "void mainImage(out vec4 fragColor, in vec2 fragCoord)", glsl, count=1
    )
    wrapped = re.sub(r"\bgl_FragColor\b", "fragColor", wrapped)
    wrapped = re.sub(r"\bgl_FragData\s*\[\s*0\s*\]", "fragColor", wrapped)
    # gl_FragCoord (vec4) -> vec4(fragCoord, 0.0, 1.0); a trailing `.xy` swizzle
    # on the vec4 literal stays valid GLSL.
    wrapped = re.sub(r"\bgl_FragCoord\b", "vec4(fragCoord, 0.0, 1.0)", wrapped)
    return wrapped


def _llm_port_to_shadertoy(
    source: str, *, llm_client: "Callable | None" = None
) -> str | None:
    """Ask the LLM to rewrite arbitrary GLSL as a Shadertoy ``mainImage`` shader.

    Reuses ``generate_llm_glsl_refinement`` with a port instruction so we do not
    duplicate the GLSL parse/normalize chain. Returns the normalized GLSL string
    or ``None`` on any failure.
    """
    from app.candidates.llm_scene import generate_llm_glsl_refinement

    try:
        result = generate_llm_glsl_refinement(
            current_glsl=source,
            metrics={},
            quality_router={"final_score": 0.0},
            extra_feedback=[
                "[PORT] The shader above is NOT in Shadertoy format. Rewrite it "
                "as a Shadertoy shader with a `void mainImage(out vec4 fragColor, "
                "in vec2 fragCoord)` entry point, preserving the original visual "
                "intent. Map gl_FragColor->fragColor and gl_FragCoord.xy->fragCoord."
            ],
            fresh_start=False,
            llm_client=llm_client,
        )
    except Exception:
        logger.warning("seed LLM port failed", exc_info=True)
        return None
    if not result:
        return None
    result.pop("_io", None)
    return result.get("glsl") or None


def adapt_seed_glsl(
    source: str,
    *,
    llm_client: "Callable | None" = None,
) -> SeedAdaptResult:
    """Adapt an arbitrary GLSL string into renderable Shadertoy GLSL.

    Strategy (design decision 5): deterministic normalize -> deterministic wrap
    -> LLM port fallback. Each stage's output is re-checked with
    ``validate_shader_static``; the first that passes wins.
    """
    if not source or not source.strip():
        return SeedAdaptResult(
            glsl="", valid=False, adapted_by="failed", errors=["seed GLSL is empty"]
        )

    # Stage 1: normalize (strip markdown / conflicting uniforms / #version, etc.)
    normalized = normalize_shadertoy_glsl(source)
    static = validate_shader_static(normalized.glsl)
    if static["valid"]:
        return SeedAdaptResult(
            glsl=normalized.glsl,
            valid=True,
            adapted_by="normalized",
            warnings=list(normalized.warnings),
        )

    # Stage 2: deterministic wrap of a legacy main() shader, then re-normalize.
    wrapped = _wrap_legacy_main(normalized.glsl)
    if wrapped is not None:
        renorm = normalize_shadertoy_glsl(wrapped)
        if validate_shader_static(renorm.glsl)["valid"]:
            return SeedAdaptResult(
                glsl=renorm.glsl,
                valid=True,
                adapted_by="wrapped",
                warnings=list(normalized.warnings)
                + list(renorm.warnings)
                + ["wrapped_legacy_main"],
            )

    # Stage 3: LLM port fallback.
    ported = _llm_port_to_shadertoy(source, llm_client=llm_client)
    if ported is not None and validate_shader_static(ported)["valid"]:
        return SeedAdaptResult(
            glsl=ported, valid=True, adapted_by="llm_ported", warnings=["llm_ported"]
        )

    return SeedAdaptResult(
        glsl=normalized.glsl,
        valid=False,
        adapted_by="failed",
        warnings=list(normalized.warnings),
        errors=list(static["errors"]),
    )
```

注：`generate_llm_glsl_refinement` 的 `llm_client` 形参签名为 `(system_prompt, user_prompt, image_paths)`，测试注入的 fake 与之兼容。

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend && python -m pytest tests/unit/test_seed_glsl.py -v`
Expected: 5 个测试全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/seed_glsl.py backend/tests/unit/test_seed_glsl.py
git commit -m "feat(seed): add adapt_seed_glsl (normalize -> wrap -> LLM port)"
```

---

## Task 2: `build_seed_candidate` 合成候选

**Files:**
- Modify: `backend/app/pipeline/seed_glsl.py`
- Test: `backend/tests/unit/test_seed_glsl.py`

- [ ] **Step 1: 追加失败测试**

在 `backend/tests/unit/test_seed_glsl.py` 末尾追加：

```python
def test_build_seed_candidate_fields():
    candidate = build_seed_candidate(
        VALID_SHADERTOY, adapted_by="normalized", warnings=["w1"]
    )
    assert candidate.id == "seed_0"
    assert candidate.source == "seed"
    assert candidate.output_kind == "glsl"
    assert candidate.dsl is None
    assert candidate.compile_success is True
    assert candidate.compile_glsl == VALID_SHADERTOY
    assert candidate.selected is True
    assert candidate.glsl_metadata["adapted_by"] == "normalized"
    assert candidate.glsl_metadata["warnings"] == ["w1"]
```

- [ ] **Step 2: 运行确认失败**

Run: `cd backend && python -m pytest tests/unit/test_seed_glsl.py::test_build_seed_candidate_fields -v`
Expected: FAIL with `ImportError: cannot import name 'build_seed_candidate'`（若 Task 1 已在导入行写了 `build_seed_candidate`，则失败原因为 `AttributeError`/未定义）。

- [ ] **Step 3: 实现 `build_seed_candidate`**

在 `backend/app/pipeline/seed_glsl.py` 末尾追加：

```python
def build_seed_candidate(
    glsl: str,
    *,
    adapted_by: str = "normalized",
    warnings: "list[str] | None" = None,
) -> CandidateRecord:
    """Wrap adapted seed GLSL as the single selected GLSL candidate.

    The seed path never runs ``select_best_candidate``; ``priority`` is purely
    cosmetic (scoreboard ordering / display).
    """
    return CandidateRecord(
        id="seed_0",
        source="seed",
        enabled=True,
        priority=100,
        dsl=None,
        output_kind="glsl",
        validation_valid=True,
        validation_errors=[],
        compile_success=True,
        compile_glsl=glsl,
        compile_errors=[],
        final_score=0.0,
        selected=True,
        reason=[f"seed shader (adapted_by={adapted_by})"],
        glsl_metadata={"adapted_by": adapted_by, "warnings": list(warnings or [])},
    )
```

- [ ] **Step 4: 运行确认通过**

Run: `cd backend && python -m pytest tests/unit/test_seed_glsl.py -v`
Expected: 6 个测试全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/seed_glsl.py backend/tests/unit/test_seed_glsl.py
git commit -m "feat(seed): build_seed_candidate wraps adapted GLSL as a CandidateRecord"
```

---

## Task 3: `run_png_shader_pipeline` 增加 seed 路径

**Files:**
- Modify: `backend/app/pipeline/graph.py`（`run_png_shader_pipeline` 签名 + 分支；新增 `_run_seed_glsl_path`）
- Test: `backend/tests/unit/test_graph.py`

- [ ] **Step 1: 写失败测试**

在 `backend/tests/unit/test_graph.py` 末尾追加。该测试通过 monkeypatch 把 WebGL 渲染与 LLM 替换为确定性假函数，断言 seed 路径跳过候选池并进入 GLSL 精修循环：

```python
def test_seed_glsl_pipeline_skips_pool_and_refines(tmp_path, monkeypatch):
    """A seed_glsl run must build one 'seed' candidate (no pool generation)
    and drive it through run_glsl_refinement_loop."""
    import app.pipeline.graph as graph_mod

    png = make_solid_png(tmp_path, color=(120, 60, 30, 255))

    seed = (
        "#define R 0.30\n"
        "void mainImage(out vec4 fragColor, in vec2 fragCoord) { fragColor = vec4(R); }"
    )
    improved = seed.replace("0.30", "0.50")

    def fake_eval(glsl, ref_path, output_path, *, canvas_width, canvas_height, max_shader_chars):
        score = 0.30
        for line in glsl.splitlines():
            if line.startswith("#define R"):
                score = float(line.split()[-1])
        return ({"mse": 1.0 - score}, {"final_score": score, "next_action": "refine"}, score, output_path)

    def fake_refine(**kwargs):
        return {"glsl": improved, "_io": {"mode": "glsl_refinement"}}

    monkeypatch.setattr(graph_mod, "_evaluate_glsl_with_webgl", fake_eval)
    monkeypatch.setattr(
        "app.candidates.llm_scene.generate_llm_glsl_refinement", fake_refine
    )

    spec = build_input_spec(
        png,
        quality={"refinement_mode": "on", "max_refinement_iterations": 2, "refinement_patience": 1},
        candidates={"glsl_render_enabled": True},
    )

    result = run_png_shader_pipeline(png, spec, run_id="seedtest", seed_glsl=seed)

    details = result["candidate_details"]
    assert len(details) == 1
    assert details[0]["source"] == "seed"
    assert result["refinement_summary"]["enabled"] is True
    assert "0.50" in (result["selected_glsl"] or "")


def test_seed_glsl_invalid_raises(tmp_path, monkeypatch):
    """A seed that cannot be adapted (and no LLM client) must raise so the
    background worker marks the run failed."""
    png = make_solid_png(tmp_path)
    monkeypatch.setattr(
        "app.candidates.llm_scene.generate_llm_glsl_refinement",
        lambda **kwargs: None,
    )
    spec = build_input_spec(png, candidates={"glsl_render_enabled": True})

    with pytest.raises(ValueError, match="seed GLSL"):
        run_png_shader_pipeline(
            png, spec, run_id="seedbad", seed_glsl="float helper(){ return 1.0; }"
        )
```

注：需在该测试文件顶部的导入中加入 `build_input_spec`：把第 12-22 行的 `from app.pipeline.graph import (...)` 之后追加一行 `from app.pipeline.input_spec import build_input_spec`（若已存在则跳过）。

- [ ] **Step 2: 运行确认失败**

Run: `cd backend && python -m pytest tests/unit/test_graph.py::test_seed_glsl_pipeline_skips_pool_and_refines -v`
Expected: FAIL with `TypeError: run_png_shader_pipeline() got an unexpected keyword argument 'seed_glsl'`

- [ ] **Step 3a: 新增 `_run_seed_glsl_path` 助手**

在 `backend/app/pipeline/graph.py` 的 `_run_post_pipeline` 定义之后、`_build_graph` 之前，新增：

```python
def _run_seed_glsl_path(
    seed_glsl: str,
    state: P2SPipelineState,
    *,
    strategy_reader: Callable[[], dict] | None = None,
) -> P2SPipelineState:
    """Seed entry: refine an externally-supplied GLSL instead of generating
    candidates. Runs preprocess + adapt + score to synthesize a single selected
    GLSL candidate, then hands off to the unchanged ``_run_post_pipeline``.

    Raises ValueError when the seed cannot be adapted to renderable Shadertoy
    GLSL, so the caller/worker marks the run failed.
    """
    from app.pipeline.seed_glsl import adapt_seed_glsl, build_seed_candidate

    run_dir = Path(state["run_dir"])
    image_path = Path(state["image_path"])

    # Preprocess: target features + canvas context (artifacts mirror node_preprocess).
    preprocess = preprocess_image(image_path)
    save_preprocess_artifacts(preprocess, run_dir, image_path)
    preprocess["llm_reference_background"] = "#000000"
    state["preprocess"] = preprocess
    state["llm_image_path"] = str(run_dir / "llm_reference_input.png")

    # Adapt the seed to renderable Shadertoy GLSL.
    adapted = adapt_seed_glsl(seed_glsl)
    (run_dir / "seed_input.glsl").write_text(seed_glsl, encoding="utf-8")
    if not adapted.valid:
        raise ValueError(
            "seed GLSL could not be adapted to renderable Shadertoy GLSL: "
            f"{adapted.errors[:3]}"
        )
    (run_dir / "seed_adapted.glsl").write_text(adapted.glsl, encoding="utf-8")

    candidate = build_seed_candidate(
        adapted.glsl, adapted_by=adapted.adapted_by, warnings=adapted.warnings
    )

    # Score once to seed the loop (real initial score/metrics/quality).
    reference_path = run_dir / "reference_input.png"
    candidate_dir = run_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    metrics, quality, score, render_path = _evaluate_glsl_with_webgl(
        adapted.glsl,
        reference_path,
        candidate_dir / f"{candidate.id}_webgl.png",
        canvas_width=state.get("canvas_width", 512),
        canvas_height=state.get("canvas_height", 512),
        max_shader_chars=state.get("max_shader_chars", 12000),
    )
    candidate.objective_metrics = metrics
    candidate.quality_router = quality
    candidate.final_score = score
    candidate.render_path = str(render_path) if render_path else None

    state["candidates"] = [candidate]
    state["selected_candidate_id"] = candidate.id
    state["selected_dsl"] = None
    state["selected_glsl"] = candidate.compile_glsl
    state["selected_metrics"] = dict(metrics)
    state["selected_quality"] = dict(quality)

    return _run_post_pipeline(state, strategy_reader=strategy_reader)
```

- [ ] **Step 3b: 在 `run_png_shader_pipeline` 接线 seed 分支**

在 `backend/app/pipeline/graph.py` 中，把 `run_png_shader_pipeline` 的签名：

```python
def run_png_shader_pipeline(
    image_path: str | Path,
    input_spec: dict | None = None,
    run_id: str | None = None,
    *,
    llm_enabled: bool | None = None,
    llm_implementation: Implementation | None = None,
    progress_callback: Callable[[str], None] | None = None,
    strategy_reader: Callable[[], dict] | None = None,
) -> dict:
```

改为（新增 `seed_glsl` 形参）：

```python
def run_png_shader_pipeline(
    image_path: str | Path,
    input_spec: dict | None = None,
    run_id: str | None = None,
    *,
    seed_glsl: str | None = None,
    llm_enabled: bool | None = None,
    llm_implementation: Implementation | None = None,
    progress_callback: Callable[[str], None] | None = None,
    strategy_reader: Callable[[], dict] | None = None,
) -> dict:
```

在该函数内，找到这一段（计算 `effective_glsl_render_enabled` 之后、`optimizer_iterations = ...` 之前）：

```python
    effective_glsl_render_enabled = requested_glsl_render_enabled or auto_glsl_render_enabled
```

在其后插入 seed 模式强制项：

```python
    # Seed-GLSL mode: refine an externally-supplied shader. Force the GLSL
    # closed loop on (WebGL render scoring + LLM available); refinement_mode
    # defaults to "on" upstream in the router.
    if seed_glsl is not None:
        effective_glsl_render_enabled = True
        effective_llm_enabled = True
        input_spec = {**input_spec, "seed_glsl": seed_glsl}
```

然后找到运行 LangGraph 的这一段：

```python
    state = _pipeline_graph.invoke(initial_state)

    # Run post-pipeline (optimization, revision, refinement)
    if progress_callback:
        progress_callback("optimizing")

    state = _run_post_pipeline(state, strategy_reader=strategy_reader)
```

改为：

```python
    if seed_glsl is not None:
        if progress_callback:
            progress_callback("optimizing")
        state = _run_seed_glsl_path(
            seed_glsl, initial_state, strategy_reader=strategy_reader
        )
    else:
        state = _pipeline_graph.invoke(initial_state)

        # Run post-pipeline (optimization, revision, refinement)
        if progress_callback:
            progress_callback("optimizing")

        state = _run_post_pipeline(state, strategy_reader=strategy_reader)
```

- [ ] **Step 3c: 更新编排注释**

在 `backend/app/pipeline/graph.py` 顶部模块 docstring（第 1-8 行）的流程描述后，追加一行说明 seed 入口：

```python
"""P2S-Agent Pipeline Orchestrator

核心 LangGraph 流程:
  preprocess -> candidates -> scoring -> selection

优化、修订、残差补层、LLM 精修和 VLM 评审作为 post-pipeline
同步函数运行。修改下方编排逻辑时，请同步维护 CORE_PIPELINE_FLOWCHART。

Seed-GLSL 入口（run_png_shader_pipeline(seed_glsl=...)）跳过 LangGraph
核心链路，经 _run_seed_glsl_path 合成单个 GLSL 候选后直接进入 post-pipeline。
"""
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend && python -m pytest tests/unit/test_graph.py -v`
Expected: 全部 PASS（含既有测试 + 2 个新测试）。

- [ ] **Step 5: Commit**

```bash
git add backend/app/pipeline/graph.py backend/tests/unit/test_graph.py
git commit -m "feat(pipeline): seed-GLSL entry path in run_png_shader_pipeline"
```

---

## Task 4: 路由 `seed_glsl` Form 字段

**Files:**
- Modify: `backend/app/routers/png_shader.py`
- Test: `backend/tests/unit/test_router.py`

- [ ] **Step 1: 写失败测试**

在 `backend/tests/unit/test_router.py` 末尾追加：

```python
def test_run_accepts_seed_glsl_and_defaults_refinement_on(tmp_path, monkeypatch):
    _run_store.clear()
    captured: dict = {}

    def fake_pipeline(image_path, input_spec=None, run_id=None, *, seed_glsl=None, **kwargs):
        captured["seed_glsl"] = seed_glsl
        captured["refinement_mode"] = (
            (input_spec or {}).get("quality", {}).get("refinement_mode")
        )
        return {
            "run_id": run_id,
            "selected_glsl": seed_glsl or "",
            "scoreboard": {},
            "quality_router": {},
            "refinement_summary": {},
        }

    monkeypatch.setattr(
        "app.routers.png_shader.run_png_shader_pipeline", fake_pipeline
    )
    client = _client()
    seed = "void mainImage(out vec4 c, in vec2 p){ c = vec4(0.3); }"

    response = client.post(
        "/png-shader/run",
        data={"seed_glsl": seed},
        files={"image": ("input.png", _png_bytes(tmp_path), "image/png")},
    )
    assert response.status_code == 200
    data = _wait_for_completion(client, response.json()["run_id"])
    assert data["status"] == "completed"
    assert "mainImage" in captured["seed_glsl"]
    assert captured["refinement_mode"] == "on"


def test_run_rejects_blank_seed_glsl(tmp_path):
    _run_store.clear()
    client = _client()
    response = client.post(
        "/png-shader/run",
        data={"seed_glsl": "   "},
        files={"image": ("input.png", _png_bytes(tmp_path), "image/png")},
    )
    assert response.status_code == 422
```

- [ ] **Step 2: 运行确认失败**

Run: `cd backend && python -m pytest tests/unit/test_router.py::test_run_accepts_seed_glsl_and_defaults_refinement_on -v`
Expected: FAIL —— `seed_glsl` 当前未被接收，`captured` 为空（KeyError）。

- [ ] **Step 3a: 路由接收 `seed_glsl` 并默认 refinement_mode**

在 `backend/app/routers/png_shader.py` 的 `run_png_shader` 签名中加入 `seed_glsl` 字段：

```python
@router.post("/run")
async def run_png_shader(
    image: UploadFile = File(...),
    input_spec_json: Optional[str] = Form(default=None),
    seed_glsl: Optional[str] = Form(default=None),
) -> dict:
```

**紧接** `input_spec_json` 的 `try/except` 解析块之后、`run_id = "run_" + ...` 之前插入 seed 处理（务必在 `upload_dir = Path(tempfile.mkdtemp(...))` 之前，使空 seed 的 422 不会泄漏临时目录）：

```python
    if seed_glsl is not None:
        if not seed_glsl.strip():
            raise HTTPException(
                status_code=422,
                detail="seed_glsl must be a non-empty string when provided",
            )
        # Default the seed run to always-refine unless the caller set the mode.
        overrides = dict(input_spec) if isinstance(input_spec, dict) else {}
        quality = dict(overrides.get("quality") or {})
        quality.setdefault("refinement_mode", "on")
        overrides["quality"] = quality
        input_spec = overrides
```

- [ ] **Step 3b: 把 `seed_glsl` 透传到后台 worker 与 pipeline**

在 `_run_png_shader_background` 的签名加入 `seed_glsl`：

```python
def _run_png_shader_background(
    *,
    run_id: str,
    image_path: Path,
    upload_dir: Path,
    pipeline_input_spec: Optional[dict],
    seed_glsl: Optional[str],
    trace_input: dict,
    trace_metadata: dict,
) -> None:
```

把其中的 `run_png_shader_pipeline(...)` 调用：

```python
                pipeline_result = run_png_shader_pipeline(
                    image_path,
                    pipeline_input_spec,
                    run_id=run_id,
                    progress_callback=_progress,
                    strategy_reader=_strategy_reader,
                )
```

改为：

```python
                pipeline_result = run_png_shader_pipeline(
                    image_path,
                    pipeline_input_spec,
                    run_id=run_id,
                    seed_glsl=seed_glsl,
                    progress_callback=_progress,
                    strategy_reader=_strategy_reader,
                )
```

再在 `run_png_shader` 里启动线程处把 kwargs 补上 `seed_glsl`：

```python
        threading.Thread(
            target=_run_png_shader_background,
            kwargs={
                "run_id": run_id,
                "image_path": image_path,
                "upload_dir": upload_dir,
                "pipeline_input_spec": pipeline_input_spec,
                "seed_glsl": seed_glsl,
                "trace_input": trace_input,
                "trace_metadata": trace_metadata,
            },
            daemon=True,
        ).start()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd backend && python -m pytest tests/unit/test_router.py -v`
Expected: 全部 PASS（含既有测试 + 2 个新测试）。

- [ ] **Step 5: Commit**

```bash
git add backend/app/routers/png_shader.py backend/tests/unit/test_router.py
git commit -m "feat(router): accept seed_glsl on /png-shader/run, default refinement_mode=on"
```

---

## Task 5: 前端 seed 输入

前端无测试 runner，门禁为 `npm run build`（tsc 类型检查）与 `npm run lint`（eslint，`--max-warnings 0`）。

**Files:**
- Modify: `frontend/src/hooks/usePngShader.ts`
- Modify: `frontend/src/components/PngShaderView.tsx`
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: `usePngShader.runPngShader` 接收 seedGlsl**

在 `frontend/src/hooks/usePngShader.ts` 中把 `runPngShader` 的定义：

```typescript
  const runPngShader = useCallback(async (file: File): Promise<void> => {
```

改为：

```typescript
  const runPngShader = useCallback(async (file: File, seedGlsl?: string): Promise<void> => {
```

并在同一函数内 `formData.append("input_spec_json", ...)` 之后追加：

```typescript
      if (seedGlsl && seedGlsl.trim()) {
        formData.append("seed_glsl", seedGlsl);
      }
```

- [ ] **Step 2: `App.handleRun` 透传 seedGlsl**

在 `frontend/src/App.tsx` 中把 `handleRun`（约第 32-40 行）：

```typescript
  const handleRun = useCallback((file: File) => {
```

改为接收并透传 seedGlsl（保留原有 body，仅改签名与最后一行调用）：

```typescript
  const handleRun = useCallback((file: File, seedGlsl?: string) => {
```

并把该回调内的 `runPngShader(file)` 改为：

```typescript
    runPngShader(file, seedGlsl)
```

依赖数组保持 `[runPngShader]` 不变（若原本含其它项，原样保留）。

- [ ] **Step 3: `PngShaderView` 增加 seed 开关 + 文本框 + 文件读入**

在 `frontend/src/components/PngShaderView.tsx`：

(a) 更新 `Props` 类型中 `onRun` 的签名（文件上方的 props 接口里）：把
`onRun: (file: File) => void;`
改为
`onRun: (file: File, seedGlsl?: string) => void;`

(b) 在组件内 `const [selectedFile, setSelectedFile] = useState<File | null>(null);` 之后，新增 seed 状态：

```typescript
  const [seedEnabled, setSeedEnabled] = useState(false);
  const [seedGlsl, setSeedGlsl] = useState("");

  const handleSeedFile = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    file.text().then((text) => setSeedGlsl(text));
  }, []);
```

(c) 把 `handleRun`（约第 98-100 行）：

```typescript
  const handleRun = useCallback(() => {
    if (selectedFile) onRun(selectedFile);
  }, [selectedFile, onRun]);
```

改为：

```typescript
  const handleRun = useCallback(() => {
    if (selectedFile) onRun(selectedFile, seedEnabled ? seedGlsl : undefined);
  }, [selectedFile, onRun, seedEnabled, seedGlsl]);
```

(d) 在上传区下方的 `<div className="mt-2 flex flex-col gap-2">` 容器内、`{/* LLM mode selector */}` 之前，插入 seed 输入块：

```tsx
          {/* Seed GLSL: start the closed loop from an existing shader */}
          <div className="flex flex-col gap-2 px-3 py-2 bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-lg">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={seedEnabled}
                onChange={(e) => setSeedEnabled(e.target.checked)}
                disabled={loading}
                className="accent-emerald-500"
              />
              <span className="text-xs font-medium text-[var(--text-primary)]">
                从已有着色器开始
                <span className="ml-2 text-[var(--text-muted)] font-normal">Seed GLSL</span>
              </span>
            </label>
            {seedEnabled && (
              <>
                <textarea
                  value={seedGlsl}
                  onChange={(e) => setSeedGlsl(e.target.value)}
                  disabled={loading}
                  placeholder="粘贴已有 GLSL（Shadertoy mainImage 或普通 main 片元着色器）"
                  rows={6}
                  className="w-full text-xs font-mono p-2 rounded bg-[var(--bg-tertiary)] border border-[var(--border-color)] text-[var(--text-primary)] resize-y"
                />
                <input
                  type="file"
                  accept=".glsl,.frag,.txt,text/plain"
                  onChange={handleSeedFile}
                  disabled={loading}
                  className="text-xs text-[var(--text-muted)]"
                />
              </>
            )}
          </div>
```

- [ ] **Step 4: 类型检查 + lint 通过**

Run: `cd frontend && npm run build && npm run lint`
Expected: tsc 无类型错误；eslint 0 warning。

- [ ] **Step 5: Commit**

```bash
git add frontend/src/hooks/usePngShader.ts frontend/src/components/PngShaderView.tsx frontend/src/App.tsx
git commit -m "feat(frontend): seed-GLSL input on the run panel"
```

---

## Task 6: 全量回归 + 手动验证

- [ ] **Step 1: 后端全量测试**

Run: `cd backend && python -m pytest -q`
Expected: 全绿（新增模块/路由/管线测试与既有测试均通过）。

- [ ] **Step 2: 前端构建**

Run: `cd frontend && npm run build && npm run lint`
Expected: 构建成功、lint 0 warning。

- [ ] **Step 3: 端到端手动冒烟（需 WebGL 渲染服务可用）**

按 `start.sh` 启动前后端，打开界面：勾选「从已有着色器开始」，粘贴一段 Shadertoy `mainImage` GLSL，上传目标 PNG，点击运行。验证：scoreboard 仅 1 个 `seed` 候选；精修历史出现迭代；`selected_shader.glsl` 与 `seed_input.glsl` / `seed_adapted.glsl` 落入 run_dir。

- [ ] **Step 4: 最终提交（如有遗留改动）**

```bash
git add -A && git commit -m "test: seed-GLSL closed-loop regression + manual smoke notes"
```

---

## 自检覆盖对照（spec → task）

| 设计要求 | 落地任务 |
|---|---|
| 纯 seed：跳过候选池 | Task 3（`_run_seed_glsl_path` 不调 `run_candidate_pool`/选择） |
| `#define` 优化器 + LLM 精修 | Task 3（复用 `_run_post_pipeline` GLSL 分支，含优化器前置 + 精修循环） |
| 混合格式适配（确定性包装 + LLM 兜底） | Task 1（`adapt_seed_glsl` 三阶段） |
| 默认 refinement_mode=on | Task 4（路由 `setdefault("refinement_mode","on")`） |
| Web UI seed 输入 | Task 5 |
| seed 校验失败 → 硬失败 | Task 3（`adapt` 无效则 `raise ValueError` → worker 标记 failed） |
| 审计：原始 + 适配后 GLSL 落盘 | Task 3（写 `seed_input.glsl` / `seed_adapted.glsl`） |
| 结果字典形状不变、前端零下游改动 | Task 3 复用 `_run_post_pipeline` 序列化；Task 5 仅加输入 |
