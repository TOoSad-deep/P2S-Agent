# 闭环优化进度实时增量推送 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让闭环优化每完成一步（基线 + 每次精修迭代）就把部分结果写入内存 store，使前端已有的 1 秒轮询能实时渲染进度，配合现有 stop / strategy-patch 实现人工及时介入。

**Architecture:** 新增一个尽力而为的回调 `publish_partial(partial: dict)`，按 `strategy_reader` 同样的方式从路由穿透到 `run_png_shader_pipeline → _run_post_pipeline → 两个精修循环`。两个精修循环新增 `on_iteration` 回调，在每次 `history.append` 后吐出当前最优快照；`_run_post_pipeline` 发布基线 + 把快照翻译成 store 结构并发布。路由层加锁合并到 `_run_store`（仅 running 态）。最终整包写入保持权威。

**Tech Stack:** Python 3 / pytest / FastAPI / 现有内存 store + 1 秒轮询 / React + Vite + TypeScript（前端核心零改动）。

**测试命令约定:** 后端 `cd backend && python -m pytest <path> -v`；前端（仅可选任务用）`cd frontend && npm run build && npm run lint`。

**设计依据:** [doc/2026-06-14-realtime-optimization-progress-design.md](2026-06-14-realtime-optimization-progress-design.md)

> **与设计的一处实现细化（重要）：** 设计稿写「`on_iteration` 闭包同步 `selected` 记录」。实际实现**不**修改 `selected.final_score`，因为精修循环结束后有 `if ref_result["best_score"] > selected.final_score:` 的「提升才接受」判定（[graph.py:655](../backend/app/pipeline/graph.py) / [graph.py:733](../backend/app/pipeline/graph.py)）；若迭代中把 `selected.final_score` 提前改成 best，会让该判定恒为 False，导致最终结果回退。因此发布闭包**只从循环快照构造 partial**，不触碰 `selected`，外部行为（实时部分更新）不变。

---

## 文件结构总览

| 文件 | 操作 | 职责 |
|---|---|---|
| `backend/app/pipeline/glsl_refinement.py` | 修改 | `run_glsl_refinement_loop` 增 `on_iteration` 参数 + `_record` 助手（append+save+emit），替换 5 处 append/save |
| `backend/tests/unit/test_glsl_refinement.py` | 修改 | GLSL 循环 `on_iteration` 每迭代触发测试 |
| `backend/app/pipeline/refinement.py` | 修改 | `run_dsl_refinement_loop` 增 `on_iteration` 参数 + `_record` 助手，替换 5 处 append/save |
| `backend/app/pipeline/graph.py` | 修改 | `_run_post_pipeline` 增 `publish_partial`：基线发布 ×2 + `_publish_iteration` 闭包，传给两个循环；`run_png_shader_pipeline` 与 `_run_seed_glsl_path` 透传 `publish_partial` |
| `backend/tests/unit/test_graph.py` | 修改 | DSL 循环 `on_iteration` 测试 + `_run_post_pipeline` 基线/迭代发布测试 + 管线/seed 透传测试 |
| `backend/app/routers/png_shader.py` | 修改 | 模块级 `_publish_partial_to_store(run_id, partial)`；worker 内定义 `_publish_partial` 并传入管线 |
| `backend/tests/unit/test_router.py` | 修改 | `_publish_partial_to_store` 合并/保留/no-op 单测 |
| `frontend/src/components/LlmIOPanel.tsx` | 可选 | 「迭代 N/M · 当前最优分数」实时指示器 |

---

## Task 1: GLSL 精修循环新增 `on_iteration` 回调

**Files:**
- Modify: `backend/app/pipeline/glsl_refinement.py`
- Test: `backend/tests/unit/test_glsl_refinement.py`

- [ ] **Step 1: 写失败测试**

在 `backend/tests/unit/test_glsl_refinement.py` 末尾追加（复用文件顶部已有的 `VALID_GLSL_A` / `_evaluate_by_r`）：

```python
def test_loop_invokes_on_iteration_each_iteration(tmp_path, monkeypatch):
    glsls = [VALID_GLSL_A.replace("0.30", v) for v in ("0.40", "0.55")]
    seq = iter(glsls)

    def fake_refine(**kwargs):
        return {"glsl": next(seq), "_io": {}}

    monkeypatch.setattr(
        "app.candidates.llm_scene.generate_llm_glsl_refinement", fake_refine
    )

    snaps: list[dict] = []
    run_glsl_refinement_loop(
        VALID_GLSL_A,
        0.30,
        {"mse": 0.7},
        {"final_score": 0.30},
        tmp_path / "ref.png",
        evaluate_fn=_evaluate_by_r,
        max_iterations=2,
        threshold=0.99,
        high_score_stop=0.999,
        min_improvement=0.001,
        no_improvement_patience=5,
        loop_dir=tmp_path / "loop",
        on_iteration=lambda s: snaps.append(
            {"len": len(s["history"]), "best": s["best_score"], "glsl": s["best_glsl"]}
        ),
    )

    assert [s["len"] for s in snaps] == [1, 2]
    assert snaps[-1]["best"] == pytest.approx(0.55)
    assert "0.55" in snaps[-1]["glsl"]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend && python -m pytest tests/unit/test_glsl_refinement.py::test_loop_invokes_on_iteration_each_iteration -v`
Expected: FAIL — `TypeError: run_glsl_refinement_loop() got an unexpected keyword argument 'on_iteration'`

- [ ] **Step 3: 加参数**

在 `backend/app/pipeline/glsl_refinement.py` 的 `run_glsl_refinement_loop` 签名里，在 `rubric_judge` 参数之后新增一行（保持现有缩进与逗号风格）：

```python
    rubric_judge: "Callable[[Path], dict | None] | None" = None,
    on_iteration: "Callable[[dict], None] | None" = None,
) -> dict:
```

- [ ] **Step 4: 加 `_record` 助手并替换 5 处 append/save**

在 `history: list[dict] = []`（约第 98 行）与紧随其后的状态初始化之后、`for i in range(max_iterations):`（约第 119 行）**之前**，插入：

```python
    def _record(entry: dict) -> None:
        history.append(entry)
        save_json(loop_dir / f"iter_{i + 1}.json", entry)
        if on_iteration is None:
            return
        try:
            on_iteration({
                "best_glsl": best_glsl,
                "best_score": best_score,
                "best_metrics": best_metrics,
                "best_quality": best_quality,
                "history": history,
            })
        except Exception:
            logger.warning("on_iteration publish failed", exc_info=True)
```

然后把循环体内全部 **5 处** 如下两行（`history.append(entry)` 紧跟一行 `save_json(loop_dir / f"iter_{i + 1}.json", entry)`）替换为单行 `_record(entry)`，保持各处原缩进。这 5 处分别在：LLM 调用失败、LLM 返回空、静态校验失败、渲染失败、正常迭代结束。`history.append(entry)` 在本函数内恰好出现 5 次，均为待替换点。例如正常迭代处：

```python
        history.append(entry)
        save_json(loop_dir / f"iter_{i + 1}.json", entry)

        if no_improvement_count >= no_improvement_patience and not _trigger_fresh_restart():
```

改为：

```python
        _record(entry)

        if no_improvement_count >= no_improvement_patience and not _trigger_fresh_restart():
```

- [ ] **Step 5: 运行测试确认通过**

Run: `cd backend && python -m pytest tests/unit/test_glsl_refinement.py -v`
Expected: PASS（新测试通过，且原有 GLSL 循环测试全部仍通过——`on_iteration` 默认 None 时 `_record` 与原 append+save 行为一致）

- [ ] **Step 6: 提交**

```bash
cd backend
git add app/pipeline/glsl_refinement.py tests/unit/test_glsl_refinement.py
git commit -m "feat(glsl-refinement): emit per-iteration snapshot via on_iteration"
```

---

## Task 2: DSL 精修循环新增 `on_iteration` 回调

**Files:**
- Modify: `backend/app/pipeline/refinement.py`
- Test: `backend/tests/unit/test_graph.py`（DSL 循环的单测就在此文件，复用其 monkeypatch 模式）

- [ ] **Step 1: 写失败测试**

在 `backend/tests/unit/test_graph.py` 末尾追加：

```python
def test_dsl_loop_invokes_on_iteration_each_iteration(tmp_path, monkeypatch):
    from types import SimpleNamespace

    def fake_refinement(**kwargs):
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
    scores = iter([0.6, 0.7])
    monkeypatch.setattr(
        "app.pipeline.refinement._evaluate_dsl",
        lambda *a, **k: ({}, {"final_score": 0.6}, next(scores), None),
    )

    snaps: list[int] = []
    run_dsl_refinement_loop(
        preprocess={},
        initial_dsl={"layers": [{"id": "a", "type": "circle", "params": {}}]},
        initial_score=0.5,
        initial_metrics={},
        initial_quality={"final_score": 0.5},
        reference_path=tmp_path / "ref.png",
        canvas_width=64,
        canvas_height=64,
        max_shader_chars=12000,
        max_iterations=2,
        threshold=0.99,
        high_score_stop=0.999,
        min_improvement=0.001,
        no_improvement_patience=5,
        loop_dir=tmp_path / "loop",
        on_iteration=lambda s: snaps.append(len(s["history"])),
    )

    assert snaps == [1, 2]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend && python -m pytest tests/unit/test_graph.py::test_dsl_loop_invokes_on_iteration_each_iteration -v`
Expected: FAIL — `TypeError: run_dsl_refinement_loop() got an unexpected keyword argument 'on_iteration'`

- [ ] **Step 3: 加参数**

在 `backend/app/pipeline/refinement.py` 的 `run_dsl_refinement_loop` 签名里，在 `rubric_judge` 参数之后新增一行：

```python
    rubric_judge: "Callable[[Path], dict | None] | None" = None,
    on_iteration: "Callable[[dict], None] | None" = None,
) -> dict:
```

- [ ] **Step 4: 加 `_record` 助手并替换 5 处 append/save**

在 `history: list[dict] = []`（约第 171 行）之后、`for i in range(max_iterations):`（约第 176 行）之前插入：

```python
    def _record(entry: dict) -> None:
        history.append(entry)
        save_json(loop_dir / f"iter_{i + 1}.json", entry)
        if on_iteration is None:
            return
        try:
            on_iteration({
                "best_dsl": best_dsl,
                "best_glsl": best_glsl,
                "best_score": best_score,
                "best_metrics": best_metrics,
                "best_quality": best_quality,
                "history": history,
            })
        except Exception:
            logger.warning("on_iteration publish failed", exc_info=True)
```

然后把循环体内全部 **5 处** `history.append(entry)` + 紧随的 `save_json(loop_dir / f"iter_{i + 1}.json", entry)` 两行，替换为单行 `_record(entry)`，保持原缩进。这 5 处分别在：LLM 调用失败、LLM 返回 None、DSL 校验失败、编译失败、正常迭代结束。`history.append(entry)` 在本函数内恰好出现 5 次。例如正常迭代处：

```python
        history.append(entry)
        save_json(loop_dir / f"iter_{i + 1}.json", entry)

        if no_improvement_count >= no_improvement_patience:
```

改为：

```python
        _record(entry)

        if no_improvement_count >= no_improvement_patience:
```

- [ ] **Step 5: 运行测试确认通过**

Run: `cd backend && python -m pytest tests/unit/test_graph.py -k "dsl_loop_invokes or dsl_refinement or refinement_loop" -v`
Expected: PASS（新测试通过；`test_dsl_refinement_feeds_history_and_semantic_notes`、`test_refinement_loop_records_llm_call_exception` 等原有测试仍通过）

- [ ] **Step 6: 提交**

```bash
cd backend
git add app/pipeline/refinement.py tests/unit/test_graph.py
git commit -m "feat(dsl-refinement): emit per-iteration snapshot via on_iteration"
```

---

## Task 3: `_run_post_pipeline` 发布基线 + 迭代 partial，并透传 `publish_partial`

**Files:**
- Modify: `backend/app/pipeline/graph.py`
- Test: `backend/tests/unit/test_graph.py`

- [ ] **Step 1: 写失败测试**

在 `backend/tests/unit/test_graph.py` 末尾追加（复刻已有 `test_run_post_pipeline_runs_glsl_refinement` 的 state 形状，额外传 `publish_partial` 并让 fake 循环回调 `on_iteration`）：

```python
def test_run_post_pipeline_publishes_baseline_and_iterations(tmp_path, monkeypatch):
    glsl = "void mainImage(out vec4 fragColor, in vec2 fragCoord){fragColor=vec4(1.0);}"
    cand = CandidateRecord(
        id="llm_0",
        source="llm",
        enabled=True,
        priority=5,
        dsl=None,
        output_kind="glsl",
        validation_valid=True,
        validation_errors=[],
        compile_success=True,
        compile_glsl=glsl,
        compile_errors=[],
        final_score=0.4,
        selected=True,
        objective_metrics={"mse": 0.5},
        quality_router={"final_score": 0.4, "next_action": "accept"},
    )
    improved_glsl = "#define R 0.9\n" + glsl

    def fake_loop(*args, **kwargs):
        on_iteration = kwargs.get("on_iteration")
        if on_iteration is not None:
            on_iteration({
                "best_glsl": improved_glsl,
                "best_score": 0.8,
                "best_metrics": {"mse": 0.1},
                "best_quality": {"final_score": 0.8, "next_action": "accept"},
                "history": [{"iteration": 1, "score_after": 0.8, "improved": True}],
            })
        return {
            "best_glsl": improved_glsl,
            "best_score": 0.8,
            "best_metrics": {"mse": 0.1},
            "best_quality": {"final_score": 0.8, "next_action": "accept"},
            "best_render_path": None,
            "history": [{"iteration": 1, "score_after": 0.8, "improved": True}],
            "stop_reason": "threshold_reached",
        }

    monkeypatch.setattr("app.pipeline.graph.run_glsl_refinement_loop", fake_loop)

    published: list[dict] = []
    result = _run_post_pipeline(
        {
            "selected_candidate_id": "llm_0",
            "candidates": [cand],
            "run_dir": str(tmp_path),
            "preprocess": {"palette": ["#ffffff"]},
            "selected_dsl": None,
            "selected_glsl": glsl,
            "selected_metrics": {"mse": 0.5},
            "selected_quality": {"final_score": 0.4, "next_action": "accept"},
            "refinement_mode": "on",
            "max_refinement_iterations": 2,
            "llm_enabled": False,
            "glsl_render_enabled": True,
            "vlm_judge_enabled": False,
        },
        publish_partial=lambda p: published.append(p),
    )

    # baseline: scoreboard published before any iteration
    assert any("scoreboard" in p for p in published)
    # iteration: refinement_history published with the current best shader
    iter_partials = [p for p in published if "refinement_history" in p]
    assert iter_partials, "expected at least one per-iteration partial"
    last = iter_partials[-1]
    assert last["refinement_history"][0]["iteration"] == 1
    assert last["selected_glsl"] == improved_glsl
    assert last["refinement_summary"]["iterations"] == 1
    assert last["refinement_summary"]["enabled"] is True
    # final result unchanged: improvement still accepted
    assert result["selected_glsl"] == improved_glsl
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend && python -m pytest tests/unit/test_graph.py::test_run_post_pipeline_publishes_baseline_and_iterations -v`
Expected: FAIL — `TypeError: _run_post_pipeline() got an unexpected keyword argument 'publish_partial'`

- [ ] **Step 3: 给 `_run_post_pipeline` 加参数**

把 `_run_post_pipeline` 签名（约第 349 行）改为：

```python
def _run_post_pipeline(
    state: P2SPipelineState,
    *,
    strategy_reader: Callable[[], dict] | None = None,
    publish_partial: Callable[[dict], None] | None = None,
) -> P2SPipelineState:
```

- [ ] **Step 4: 加基线发布 #1 与 `_publish_iteration` 闭包**

在 `refinement_history: list = []`（约第 399 行）之后、`# Optimization and revision`（约第 401 行）之前插入：

```python
    def _publish_iteration(snapshot: dict) -> None:
        if publish_partial is None:
            return
        hist = list(snapshot.get("history") or [])
        best = snapshot.get("best_score")
        try:
            publish_partial({
                "refinement_history": hist,
                "refinement_summary": {
                    **refinement_summary,
                    "enabled": True,
                    "iterations": len(hist),
                    "final_score": best if best is not None
                    else refinement_summary.get("final_score"),
                },
                "selected_glsl": snapshot.get("best_glsl") or None,
                "objective_metrics": dict(snapshot.get("best_metrics") or {}),
                "quality_router": dict(snapshot.get("best_quality") or {}),
            })
        except Exception:
            logger.warning("publish_partial (iteration) failed", exc_info=True)

    # Baseline #1: surface candidates + initial selection ASAP.
    if publish_partial is not None:
        try:
            publish_partial({
                "preprocess": state.get("preprocess", {}),
                "scoreboard": build_scoreboard(candidates),
                "selected_candidate_id": selected.id,
                "selected_glsl": selected.compile_glsl,
                "objective_metrics": dict(selected.objective_metrics),
                "quality_router": dict(selected.quality_router) if selected.quality_router else {},
            })
        except Exception:
            logger.warning("publish_partial (baseline) failed", exc_info=True)
```

- [ ] **Step 5: 加基线发布 #2（精修前，体现优化/修订/补层提升）**

在 `should_refine, refinement_decision = _should_run_refinement(`（约第 596 行）**之前**插入：

```python
    # Baseline #2: reflect optimizer / revision / residual gains before refinement.
    if publish_partial is not None:
        try:
            publish_partial({
                "scoreboard": build_scoreboard(candidates),
                "selected_glsl": selected_glsl,
                "objective_metrics": dict(selected_metrics),
                "quality_router": dict(selected_quality) if selected_quality else {},
            })
        except Exception:
            logger.warning("publish_partial (pre-refine) failed", exc_info=True)
```

- [ ] **Step 6: 把 `_publish_iteration` 传给两个精修循环**

在 `run_dsl_refinement_loop(` 调用（约第 614 行）的关键字参数中，于 `rubric_judge=...,` 之后加一行：

```python
            on_iteration=_publish_iteration,
```

在 `run_glsl_refinement_loop(` 调用（约第 694 行）的关键字参数中，于 `rubric_judge=...,` 之后加一行：

```python
                on_iteration=_publish_iteration,
```

（注意 GLSL 调用点在 `else` 分支内，缩进比 DSL 多一级；与该调用其它 kwargs 对齐即可。）

- [ ] **Step 7: 运行测试确认通过**

Run: `cd backend && python -m pytest tests/unit/test_graph.py -k "post_pipeline" -v`
Expected: PASS（新测试通过；`test_run_post_pipeline_runs_glsl_refinement`、`test_run_post_pipeline_skips_glsl_refinement_when_render_disabled` 等仍通过——它们不传 `publish_partial`，默认 None，发布全部跳过）

- [ ] **Step 8: 给 `run_png_shader_pipeline` 与 `_run_seed_glsl_path` 透传 `publish_partial`**

写一个透传集成测试，追加到 `backend/tests/unit/test_graph.py` 末尾：

```python
def test_pipeline_threads_publish_partial_baseline(tmp_path):
    png = make_solid_png(tmp_path)
    seen: list[dict] = []
    run_png_shader_pipeline(
        png, run_id="pub_smoke", publish_partial=lambda p: seen.append(p)
    )
    # deterministic path (no LLM) still publishes at least the baseline scoreboard
    assert any("scoreboard" in p for p in seen)


def test_seed_pipeline_publishes_iterations(tmp_path, monkeypatch):
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
    seen: list[dict] = []
    run_png_shader_pipeline(
        png, spec, run_id="seedpub", seed_glsl=seed, publish_partial=lambda p: seen.append(p)
    )

    assert any("refinement_history" in p for p in seen)
```

Run（应先失败：`run_png_shader_pipeline` 尚无 `publish_partial` 参数）:
`cd backend && python -m pytest tests/unit/test_graph.py::test_pipeline_threads_publish_partial_baseline tests/unit/test_graph.py::test_seed_pipeline_publishes_iterations -v`
Expected: FAIL — `TypeError: run_png_shader_pipeline() got an unexpected keyword argument 'publish_partial'`

- [ ] **Step 9: 实现透传**

在 `run_png_shader_pipeline` 签名（约第 913 行）的 `strategy_reader` 参数之后加一行：

```python
    strategy_reader: Callable[[], dict] | None = None,
    publish_partial: Callable[[dict], None] | None = None,
) -> dict:
```

把 seed 分支调用（约第 1096 行）改为传 `publish_partial`：

```python
        state = _run_seed_glsl_path(
            seed_glsl, initial_state, strategy_reader=strategy_reader,
            publish_partial=publish_partial,
        )
```

把普通分支的后处理调用（约第 1106 行）改为：

```python
        state = _run_post_pipeline(
            state, strategy_reader=strategy_reader, publish_partial=publish_partial
        )
```

给 `_run_seed_glsl_path` 签名（约第 809 行）加参数并透传到它内部的 `_run_post_pipeline` 调用（约第 878 行）：

```python
def _run_seed_glsl_path(
    seed_glsl: str,
    state: P2SPipelineState,
    *,
    strategy_reader: Callable[[], dict] | None = None,
    publish_partial: Callable[[dict], None] | None = None,
) -> P2SPipelineState:
```

```python
    return _run_post_pipeline(
        state, strategy_reader=strategy_reader, publish_partial=publish_partial
    )
```

- [ ] **Step 10: 运行测试确认通过**

Run: `cd backend && python -m pytest tests/unit/test_graph.py -v`
Expected: PASS（全部，含 seed 既有测试 `test_seed_glsl_pipeline_skips_pool_and_refines` / `test_seed_glsl_invalid_raises`）

- [ ] **Step 11: 更新后处理流程注释（轻量）**

在 graph.py 的「Post-pipeline processing」节注释（`_run_post_pipeline` 的 docstring，约第 354 行）补一句：

```python
    """Run optimization, revision, and refinement after selection.

    This is called after the LangGraph pipeline completes. When
    ``publish_partial`` is provided, a baseline snapshot is published before
    refinement and one partial per refinement iteration via ``on_iteration``,
    so a polling client can render progress live.
    """
```

- [ ] **Step 12: 提交**

```bash
cd backend
git add app/pipeline/graph.py tests/unit/test_graph.py
git commit -m "feat(post-pipeline): publish baseline + per-iteration partials"
```

---

## Task 4: 路由层加锁合并 partial 到内存 store

**Files:**
- Modify: `backend/app/routers/png_shader.py`
- Test: `backend/tests/unit/test_router.py`

- [ ] **Step 1: 写失败测试**

在 `backend/tests/unit/test_router.py` 末尾追加（导入新函数）：

```python
def test_publish_partial_merges_into_running_store():
    from app.routers.png_shader import _publish_partial_to_store

    _run_store.clear()
    _run_store["run_x"] = {
        "run_id": "run_x",
        "status": "running",
        "strategy": {"refinement_threshold": 0.8},
        "stop_requested": False,
        "strategy_revision": 1,
    }

    _publish_partial_to_store(
        "run_x",
        {"scoreboard": {"selected_id": "seed_0"}, "refinement_history": [{"iteration": 1}]},
    )

    stored = _run_store["run_x"]
    assert stored["status"] == "running"
    assert stored["scoreboard"]["selected_id"] == "seed_0"
    assert stored["refinement_history"] == [{"iteration": 1}]
    # control fields preserved
    assert stored["strategy"] == {"refinement_threshold": 0.8}
    assert stored["stop_requested"] is False
    assert stored["strategy_revision"] == 1


def test_publish_partial_noop_when_terminal():
    from app.routers.png_shader import _publish_partial_to_store

    _run_store.clear()
    _run_store["run_done"] = {"run_id": "run_done", "status": "completed"}
    _publish_partial_to_store("run_done", {"scoreboard": {"x": 1}})
    assert "scoreboard" not in _run_store["run_done"]


def test_publish_partial_noop_when_missing():
    from app.routers.png_shader import _publish_partial_to_store

    _run_store.clear()
    _publish_partial_to_store("ghost", {"scoreboard": {}})
    assert "ghost" not in _run_store
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd backend && python -m pytest tests/unit/test_router.py -k publish_partial -v`
Expected: FAIL — `ImportError: cannot import name '_publish_partial_to_store'`

- [ ] **Step 3: 加模块级合并函数**

在 `backend/app/routers/png_shader.py` 的 `_store_run`（约第 37 行）之后新增：

```python
def _publish_partial_to_store(run_id: str, partial: dict) -> None:
    """Merge a partial pipeline result into a still-running run's store entry.

    Best-effort: silently no-ops when the run was evicted or already reached a
    terminal state, so a late partial can't resurrect a finished run. Only data
    fields are merged; control fields (strategy / stop_requested /
    strategy_revision / status / ...) are preserved.
    """
    with _run_store_lock:
        stored = _run_store.get(run_id)
        if stored is None or stored.get("status") != "running":
            return
        stored.update(partial)
```

- [ ] **Step 4: worker 内定义 `_publish_partial` 并传入管线**

在 `_run_png_shader_background` 内，`_strategy_reader` 定义（约第 63–69 行）之后新增：

```python
    def _publish_partial(partial: dict) -> None:
        _publish_partial_to_store(run_id, partial)
```

并把 `run_png_shader_pipeline(...)` 调用（约第 81–88 行）补上该回调：

```python
                pipeline_result = run_png_shader_pipeline(
                    image_path,
                    pipeline_input_spec,
                    run_id=run_id,
                    seed_glsl=seed_glsl,
                    progress_callback=_progress,
                    strategy_reader=_strategy_reader,
                    publish_partial=_publish_partial,
                )
```

- [ ] **Step 5: 运行测试确认通过**

Run: `cd backend && python -m pytest tests/unit/test_router.py -v`
Expected: PASS（新 3 个测试通过；原有路由契约测试全部仍通过）

- [ ] **Step 6: 提交**

```bash
cd backend
git add app/routers/png_shader.py tests/unit/test_router.py
git commit -m "feat(router): merge per-step partials into the running run store"
```

---

## Task 5: 后端全量回归

**Files:** 无（仅运行）

- [ ] **Step 1: 跑全部后端单测**

Run: `cd backend && python -m pytest tests/unit -q`
Expected: 全绿。若有失败，定位到具体测试修复后重跑。

- [ ] **Step 2: 提交（仅当有修复改动）**

```bash
cd backend
git add -A
git commit -m "test: fix regressions from progress-streaming wiring"
```

（若 Step 1 全绿无改动，跳过本步。）

---

## Task 6（可选，可跳过）: 前端「迭代 N/M · 当前最优分数」实时指示器

> 核心功能在 Task 1–4 完成后，前端**无需任何改动**即生效（已有 1 秒轮询 + 按 `result` 字段渲染）。本任务仅为锦上添花，用户已确认为可选；若不做，直接进入收尾。

**Files:**
- Modify: `frontend/src/components/LlmIOPanel.tsx`

- [ ] **Step 1: 阅读现有渲染**

先 Read `frontend/src/components/LlmIOPanel.tsx`，确认它如何接收 `refinementSummary` / `refinementHistory` props（见 [PngShaderView.tsx:362](../frontend/src/components/PngShaderView.tsx) 的传参）。

- [ ] **Step 2: 在面板顶部加运行中指示器**

在 `LlmIOPanel` 渲染顶部，当存在 `refinementSummary` 且 `refinementSummary.enabled` 为真时，显示一行：

```tsx
{refinementSummary?.enabled ? (
  <div className="text-[11px] text-[var(--text-muted)] px-2 py-1">
    迭代 {Number(refinementSummary.iterations ?? 0)}
    {" · 当前最优 "}
    {typeof refinementSummary.final_score === "number"
      ? refinementSummary.final_score.toFixed(3)
      : "—"}
  </div>
) : null}
```

（`refinementSummary` 字段为 `Record<string, unknown>`，按上面方式做 `Number(...)` / `typeof` 守卫即可，无需新增类型字段。）

- [ ] **Step 3: 构建 + lint**

Run: `cd frontend && npm run build && npm run lint`
Expected: 构建与 lint 均通过。

- [ ] **Step 4: 提交**

```bash
cd frontend
git add src/components/LlmIOPanel.tsx
git commit -m "feat(frontend): live iteration/score indicator during refinement"
```

---

## 收尾

完成 Task 1–5（核心）后：

- [ ] 实机验证（可选）：跑一次带 seed-GLSL 或 LLM-on 的优化，打开前端观察候选/迭代卡片/预览在跑完前就逐步出现；中途点「停止运行」「实时调参」确认仍生效。
- [ ] 用 superpowers:finishing-a-development-branch 决定合并/PR/清理。

---

## Self-Review（对照 spec）

**1. Spec 覆盖：**
- 决策1（仅实时可见 + 现有控制）→ 不新增介入动作；stop / strategy-patch 零改动 ✓（Task 4 明确不动这些端点）。
- 决策2（基线 + 每次迭代）→ Task 3 基线发布 ×2 + `on_iteration` 每迭代发布（Task 1/2 提供回调）✓。
- 决策3（轮询 + store 合并，不引 SSE）→ Task 4 `_publish_partial_to_store` ✓。
- 决策4（前端基本零改动；指示器可选）→ Task 6 标注可选 ✓。
- 错误处理（尽力而为、终态保护）→ `_record`/`_publish_iteration`/baseline 均 try/except；`_publish_partial_to_store` 仅 running 态合并 ✓。
- 最终整包写入保持权威 → 不改 [png_shader.py:89](../backend/app/routers/png_shader.py) ✓。
- 测试（路由合并 / 循环回调 / post-pipeline 基线+迭代 / 透传）→ Task 1/2/3/4 各有对应测试 ✓。

**2. 占位符扫描：** 无 TBD/TODO；每个代码步均给出完整代码与精确命令/期望。

**3. 类型/命名一致性：** 全链路统一 `publish_partial`（管线/后处理/路由 worker）、`on_iteration`（两个循环）、`_publish_iteration`（闭包）、`_publish_partial_to_store`（路由模块级）、`_record`（循环助手）。快照字典键 `best_glsl` / `best_dsl` / `best_score` / `best_metrics` / `best_quality` / `history` 在 Task 1/2 产出、Task 3 消费，一致。

**4. 与 spec 的偏差：** 已在顶部「实现细化」记录——发布闭包不修改 `selected`，从快照构造 partial，避免破坏精修「提升才接受」判定；外部行为不变。设计的测试节提到 `test_refinement.py`，实际 DSL 循环单测位于 `test_graph.py`（仓库现状），本计划据此放置。
