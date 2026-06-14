# 闭环优化进度实时增量推送（Real-time Optimization Progress）设计

> **状态:** 设计已定稿，待评审 → 进入实现计划（writing-plans）。
> **For agentic workers:** 实现阶段请用 superpowers:writing-plans 把本设计拆成 task-by-task 计划。

## Goal

当前只有在闭环优化**完全跑完**后，前端才能看到整个优化过程和每一步情况。本功能让**每完成一步优化就及时把结果推到前端**，使用户能实时查看优化进度，并配合已有的「停止运行」「实时调参」按钮**及时人工介入**。

目标：最大化复用现有「后台线程 + 内存 store + 1 秒轮询」架构，对前端尽量零破坏。

## 关键决策（已与用户确认）

| # | 决策 | 选择 |
|---|---|---|
| 1 | 人工介入范围 | **仅实时可见 + 现有控制**（复用已有 stop / strategy-patch，不新增介入动作） |
| 2 | 实时更新粒度 | **基线 + 每次迭代**（核心流程选定候选后先推基线，之后每次精修迭代完成再推一次） |
| 3 | 传输方式 | **轮询 + 部分结果写入内存 store**（不引入 SSE/WebSocket） |
| 4 | 前端改动 | **基本零改动**（现有组件按 `result` 字段渲染，不依赖 `status === "completed"`）；「迭代 N/M」指示器为可选锦上添花 |

## 现状（问题根因）

现有 `POST /png-shader/run`（[backend/app/routers/png_shader.py:133](../backend/app/routers/png_shader.py)）起后台线程并立即返回 `status="running"`。前端 `usePngShader` 已经每 1 秒轮询 `GET /png-shader/status/{run_id}` 并 `setResult(data)`（[frontend/src/hooks/usePngShader.ts:200](../frontend/src/hooks/usePngShader.ts)）。

问题在于：运行期间后台只更新了 `current_phase`（[png_shader.py:57](../backend/app/routers/png_shader.py)），真正的数据 —— `scoreboard / refinement_history / refinement_summary / selected_glsl / objective_metrics / quality_router` —— **只在最末尾一次性写入 store**（[png_shader.py:89](../backend/app/routers/png_shader.py)）。因此轮询期间前端面板始终为空，直到全部跑完才突然填满。

两个精修循环（[refinement.py:385](../backend/app/pipeline/refinement.py)、[glsl_refinement.py:322](../backend/app/pipeline/glsl_refinement.py)）本就在每次迭代后 `history.append(entry)` 并落盘 `iter_N.json` —— 这是天然的发布钩子点。

## 方案选择（及为什么）

- **A. 轮询 + 部分结果写入 store（采用）** —— 每步优化结果合并进现有 `_run_store`，前端用已有 1 秒轮询渲染。无新传输层，复用 stop / strategy-patch，延迟约 1 秒。
- **B. SSE/WebSocket 推送（不选）** —— 真正实时，但要新增传输层并改写前端轮询逻辑，对「1 秒级可见」需求过度。
- **C. 状态接口扫描磁盘 `iter_N.json`（不选）** —— 免穿透回调，但把 `/status` 接口与产物目录结构耦合，且每次轮询重读磁盘、重复组装逻辑。

A 最优：改动最小，契合已有回调模式（`progress_callback` / `strategy_reader` 已用同样方式穿透），且前端本就按 `result` 字段渲染。

## Architecture

按 `strategy_reader` 同样的方式，新增一个**尽力而为**的回调 `publish_partial(partial: dict)`，从路由穿透到精修循环：

```
router worker ──publish_partial──▶ run_png_shader_pipeline
                                    └─▶ _run_post_pipeline
                                         ├─（基线发布 ×1~2）
                                         └─▶ 精修循环 ──on_iteration──▶ publish_partial
```

### 1. 路由层（`png_shader.py`）

在 `_run_png_shader_background` 内定义 `_publish_partial(partial: dict)`，与现有 `_progress` / `_strategy_reader` 并列：

```python
def _publish_partial(partial: dict) -> None:
    with _run_store_lock:
        stored = _run_store.get(run_id)
        if stored is None or stored.get("status") != "running":
            return  # run 已被替换或已终态：不复活
        stored.update(partial)          # partial 只含数据字段
        # status 保持 "running"；控制字段（strategy/stop_requested/
        # strategy_revision/submitted_at/filename/current_phase）天然保留
```

传入 `run_png_shader_pipeline(..., publish_partial=_publish_partial)`。`/status`、`/runs/{id}/stop`、`/runs/{id}/strategy` **零改动**——实时停止/策略热更自动可用。最终完成时的整包写入（[png_shader.py:89](../backend/app/routers/png_shader.py)）保持不变，仍是权威结果。

### 2. `run_png_shader_pipeline`（`graph.py`）

新增参数 `publish_partial: Callable[[dict], None] | None = None`，直接转发给 `_run_post_pipeline`。seed-GLSL 路径（`_run_seed_glsl_path`）也走 `_run_post_pipeline`，因此**自动覆盖**。

### 3. `_run_post_pipeline`（`graph.py`，store 形状的唯一归口）

它在末尾已经构建了完整的结果字典（scoreboard / refinement_* / selected_* 等）。本功能复用同一套组装逻辑产出 **partial**：

1. **基线发布**：拿到 `selected` 后发布一次 ——
   `scoreboard`(由 `build_scoreboard(candidates)`)、`selected_candidate_id`、`selected_glsl`、`objective_metrics`、`quality_router`、`preprocess`。
   进入精修循环**之前**再发布一次基线，体现 `#define` 优化器 / revision / 残差补层带来的提升。
2. **`on_iteration(snapshot)` 闭包**：交给精修循环按迭代回调。snapshot 含循环内当前最优（`best_score / best_glsl(或best_dsl→compile) / best_metrics / best_quality / history`）。闭包负责：
   - 用 snapshot 同步 `selected` 记录（compile_glsl / objective_metrics / quality_router / final_score）；
   - 重建 `scoreboard`；
   - 组装 partial：`refinement_history`(当前 history 拷贝)、`refinement_summary`(实时 `iterations` 计数 + `initial_score` + 当前 `final_score`)、`scoreboard`、`selected_glsl`、`objective_metrics`、`quality_router`；
   - 调 `publish_partial(partial)`。

> store 形状逻辑集中在 `_run_post_pipeline`（与最终结果同源），循环只吐原始 snapshot，避免两处漂移。

### 4. 两个精修循环（`refinement.py` / `glsl_refinement.py`）

新增可选参数 `on_iteration: Callable[[dict], None] | None = None`。在**每次** `history.append(entry)` 之后（含提前 `break` 前的那几处 append：llm 失败 / 返回空 / 校验失败 / 渲染失败 / 正常迭代）调用：

```python
if on_iteration is not None:
    try:
        on_iteration({
            "best_score": best_score,
            "best_glsl": best_glsl,          # glsl 循环
            # "best_dsl": best_dsl,          # dsl 循环（闭包内 compile 取 glsl）
            "best_metrics": best_metrics,
            "best_quality": best_quality,
            "history": history,
            "iteration": entry["iteration"],
        })
    except Exception:
        logger.warning("on_iteration publish failed", exc_info=True)  # 绝不影响优化
```

`on_iteration is None`（测试 / 直接调用 / 不需要推送）时跳过，行为与现状完全一致。

## 前端表现（基本零改动）

下一次轮询时现有组件自动渲染部分数据：

- `CandidateScoreboard` ← `result.scoreboard`（基线候选立即可见）
- `ImageDiffPanel` / 预览链 ← `result.selected_glsl`（当前最优着色器，前端自带 WebGL 渲染，无需服务端 PNG）
- `LlmIOPanel` ← `result.refinement_summary` + `result.refinement_history`（不断增长的迭代卡片）
- `QualityRouterPanel` ← `result.quality_router`
- `SceneGraphPanel` ← `result.preprocess`

**可选锦上添花（非核心，可延后）**：在 `LlmIOPanel` 或运行状态条显示「迭代 N/M · 当前最优分数」，数据取自 `refinement_summary.iterations` / `max_refinement_iterations` / `final_score`。若做，需要 partial 里带上 `max_refinement_iterations`（或前端从 strategy 读取）。

## 数据流（结果字典形状不变）

partial 字段是最终结果字典字段的**子集**，前端 `PngShaderResult` 结构无需新增字段（除非做可选指示器时加 1~2 个可选字段）。前端下游（轮询、面板、预览）全部复用。

## 错误处理与并发

- `publish_partial` 与 `on_iteration` 严格**尽力而为**：循环侧 try/except 包裹，发布失败仅 warning，绝不打断优化。
- 合并前检查 `status == "running"`：run 已被 `_store_run` 淘汰、或已 `failed`/`completed` 时跳过，**迟到的 partial 不能复活已结束的 run**。
- 所有 store 读写经 `_run_store_lock`（与现有 `_progress` / `patch_strategy` / `stop_run` 一致），无新并发面。
- 最终整包写入（[png_shader.py:89](../backend/app/routers/png_shader.py)）保持权威：partial 只是中间态。

## 后端改动清单

| 文件 | 操作 | 职责 |
|---|---|---|
| `backend/app/routers/png_shader.py` | 修改 | `_run_png_shader_background` 内新增 `_publish_partial`，加锁合并 partial（仅 running 态）；传入 `run_png_shader_pipeline(..., publish_partial=...)`。`/status`、`/stop`、`/strategy` 零改动 |
| `backend/app/pipeline/graph.py` | 修改 | `run_png_shader_pipeline` 增参 `publish_partial`；转发给 `_run_post_pipeline`。`_run_post_pipeline` 增参 `publish_partial`：基线发布 ×（1~2）+ 构造 `on_iteration` 闭包（同步 selected → 重建 scoreboard → 组装并发布 partial），分别传给 DSL/GLSL 两个精修调用点 |
| `backend/app/pipeline/refinement.py` | 修改 | `run_dsl_refinement_loop` 增参 `on_iteration`；每次 `history.append` 后 try/except 回调 |
| `backend/app/pipeline/glsl_refinement.py` | 修改 | `run_glsl_refinement_loop` 增参 `on_iteration`；每次 `history.append` 后 try/except 回调 |

## 前端改动清单（可选）

| 文件 | 操作 | 职责 |
|---|---|---|
| `frontend/src/components/LlmIOPanel.tsx` | 可选 | 顶部显示「迭代 N/M · 当前最优分数」实时指示器（取自 `refinement_summary`） |
| `frontend/src/hooks/usePngShader.ts` | 可选 | 若指示器需要 `max_refinement_iterations`，在 `PngShaderResult` / partial 增可选字段 |

> 核心功能下，前端可**完全不改**即生效。

## 测试

- **单元** `backend/tests/unit/test_router.py`（扩展）：`_publish_partial`（或其等价封装）—— 合并数据字段、保留控制字段、run 缺失/终态时 no-op、status 保持 `running`。
- **单元** `backend/tests/unit/test_refinement.py` / `test_glsl_refinement.py`（扩展或新建）：用打桩 LLM/渲染，断言 `on_iteration` 每完成一次迭代触发一次，且传入的 `history` 单调增长；`on_iteration=None` 时行为不变。
- **集成** `backend/tests/unit/test_graph.py`（扩展）：用捕获式 fake `publish_partial`，断言 `_run_post_pipeline` 至少发布 1 次基线 + 每次迭代 1 次，partial 字段形状正确（覆盖 DSL 与 GLSL/seed 两条路径）。
- **回归**：现有 `test_router.py` / `test_seed_glsl.py` 保持通过。
- 约定：`cd backend && python -m pytest <path> -v`。

## 不做（YAGNI，本期范围外）

- 不引入 SSE/WebSocket（决策 3）。
- 不新增人工介入动作（如「采纳当前并停止」「迭代间暂停 + 手动编辑」）——仅复用现有 stop / strategy-patch（决策 1）。
- 不在 `#define` 优化器 / revision / 残差补层的**每个内部步**发布；这些是快速确定性步骤，统一在「进入精修前的基线」里体现一次即可。
- 不改最终整包写入与结果字典对外形状。
- 不做断点续跑 / checkpoint。

## 关键文件索引

- [backend/app/routers/png_shader.py](../backend/app/routers/png_shader.py) — `_run_png_shader_background`(46) / 最终写入(89) / `_progress`(57) / `_store_run`(37)
- [backend/app/pipeline/graph.py](../backend/app/pipeline/graph.py) — `run_png_shader_pipeline`(913) / `_run_post_pipeline`(349, DSL 精修调用点 612, GLSL 精修调用点 671) / `_run_seed_glsl_path`(809)
- [backend/app/pipeline/refinement.py](../backend/app/pipeline/refinement.py) — `run_dsl_refinement_loop`(99, `history.append` 在 269/287/296/308/385)
- [backend/app/pipeline/glsl_refinement.py](../backend/app/pipeline/glsl_refinement.py) — `run_glsl_refinement_loop`(63, `history.append` 在 217/233/242/267/322)
- [backend/app/pipeline/pool.py](../backend/app/pipeline/pool.py) — `build_scoreboard` / `_candidate_detail`（partial 组装复用）
- [frontend/src/hooks/usePngShader.ts](../frontend/src/hooks/usePngShader.ts) — `pollStatus`(200, 已有 1 秒轮询) / `PngShaderResult`(114)
- [frontend/src/components/PngShaderView.tsx](../frontend/src/components/PngShaderView.tsx) — 面板按 `result` 字段渲染，不依赖完成态
- [frontend/src/components/LlmIOPanel.tsx](../frontend/src/components/LlmIOPanel.tsx) — `refinement_history` / `refinement_summary` 渲染（可选指示器落点）
