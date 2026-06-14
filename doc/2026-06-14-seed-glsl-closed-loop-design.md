# 以已有 GLSL 为起点的闭环优化（Seed-GLSL Closed Loop）设计

> **状态:** 设计已定稿，待评审 → 进入实现计划（writing-plans）。
> **For agentic workers:** 实现阶段请用 superpowers:writing-plans 把本设计拆成 task-by-task 计划。

## Goal

给 PNG-to-Shader 流水线新增一条入口：用户提供 **目标 PNG + 一段已有 GLSL**，系统以该 GLSL 为**起点**直接进入闭环优化（`#define` 参数微调 + LLM 语义精修 + VLM 评审），而不是每次都从零生成候选。目标是把现有「后处理闭环」复用到外部着色器上，最大化复用、对前端零破坏。

## 关键决策（已与用户确认）

| # | 决策 | 选择 |
|---|---|---|
| 1 | seed 与候选池的关系 | **纯 seed：完全跳过候选池**（不生成 baseline/cv/llm/fallback） |
| 2 | 对 seed 跑哪些优化 | **`#define` 坐标下降 + LLM 闭环精修**（完整 GLSL 后处理链） |
| 3 | 入口形式 | **Web UI 新增 seed 输入**（复用现有 PNG 上传界面） |
| 4 | 输入 GLSL 格式 | **混合/不确定**：需要格式适配层 |
| 5 | 非 Shadertoy 格式适配 | **确定性包装 + LLM 兜底**（规则包装失败才调一次 LLM 转换） |
| 6 | seed 运行默认精修模式 | **默认 `on`**（即使初始分过阈值也至少精修 1 轮） |

## Architecture

### 复用方案：合成 seed 候选 + 复刻不动的 `_run_post_pipeline`

现有 `run_png_shader_pipeline`（[backend/app/pipeline/graph.py:838](../backend/app/pipeline/graph.py)）分两段：
1. **LangGraph 核心链路** `preprocess → candidates → scoring → selection`——其唯一职责是「产出一个 selected 候选」。
2. **`_run_post_pipeline`**（[graph.py:346](../backend/app/pipeline/graph.py)）——闭环优化的全部逻辑（GLSL 优化器 + `run_glsl_refinement_loop` + VLM 终局门 + 序列化），**对候选来源无任何假设**，只读 `selected.compile_glsl` / `selected.output_kind` / `selected.quality_router` 等字段。

这正是本功能利用的接缝：seed 路径只需「用外部 GLSL 伪造一个 selected GLSL 候选」，就能让 `_run_post_pipeline` 原封不动地完成闭环。

```
seed_glsl + PNG
   │
   ├─ node_preprocess(state)            # 复用：提供目标特征(供 LLM 反馈) + 画布尺寸
   ├─ adapt_seed_glsl(seed_glsl)        # 新增：normalize → 适配混合格式 → 静态校验
   ├─ build_seed_candidate(adapted)     # 新增：合成 1 个 CandidateRecord(source="seed")
   ├─ _evaluate_glsl_with_webgl(...)    # 复用：渲染 + 评分 → 真实初始 score/metrics/quality
   ├─ 写入 state.selected_*             # 让 _run_post_pipeline 找到 selected
   └─ _run_post_pipeline(state, ...)    # 复用 100%：#define 优化器 + 精修循环 + VLM 门 + 落盘
```

跳过了 `candidates_step` / `scoring_step` / `selection_step` 与全部 from-scratch 生成器 → 满足「纯 seed」。

### 为什么不选另外两条路

- **Option B（全新轻量 orchestrator，绕过 `_run_post_pipeline`）**：最解耦，但会重复 run_dir 搭建、evaluate 闭包、VLM 门接线、结果字典组装——两条路径易漂移；且默认跳过 `#define` 优化器。
- **Option C（注入 `run_candidate_pool` 作为 external 源）**：最贴流水线，但仍会跑完整候选生成（更慢、违背「不从零」目标），且 seed 可能在 selection 阶段被生成候选击败——对「优化我的着色器」语义反直觉。

## 闭环本体（复用，不改）

`run_glsl_refinement_loop`（[backend/app/pipeline/glsl_refinement.py](../backend/app/pipeline/glsl_refinement.py)）签名仅需 5 个起点输入——`initial_glsl, initial_score, initial_metrics, initial_quality, reference_path`——其余皆为配置/回调。它**不依赖 `CandidateRecord` / `P2SPipelineState` / LangGraph**，因此 seed 路径喂给它的与现有调用点（[graph.py:691](../backend/app/pipeline/graph.py)）完全同构。机制：语义反馈（`rubric_judge`）、梯度窗口（`build_recent_history_notes`）、渲染失败专用回喂、停滞 fresh restart、best 单调不降的逻辑回滚。停止条件、`evaluate_fn` 契约、`history` 结构均保持不变。

## 进入闭环的门控（已核对）

`_should_run_refinement`（[backend/app/pipeline/refinement.py:411](../backend/app/pipeline/refinement.py)）放行 GLSL 候选的条件：

```python
is_glsl = (not is_dsl) and output_kind == "glsl" and compile_success and bool(compile_glsl)
# 且 selected_quality is not None
# 且 refinement_mode == "on" → force_enabled（除非 score >= high_score_stop）
```

合成 seed 候选只要满足 `output_kind="glsl"` / `compile_success=True` / `compile_glsl` 非空，并经 `_evaluate_glsl_with_webgl` 评分得到非 None 的 `quality_router`，即可直接进入循环。默认 `refinement_mode="on"`（决策 6）保证至少精修一轮。

## 格式适配（决策 4 + 5）

`normalize_shadertoy_glsl`（[backend/app/utils/glsl_postprocess.py:80](../backend/app/utils/glsl_postprocess.py)）能清洗 Shadertoy GLSL（去 markdown / 去冲突 uniform / 去 `#version` / int→float / 注入缺失 define），但**不会**把 `void main()`/`gl_FragColor` 着色器转成 `mainImage`——只会 warn `missing_mainImage`。

新增 `adapt_seed_glsl(source) -> (glsl, warnings, adapted_by)`：
1. **normalize**：先跑 `normalize_shadertoy_glsl`。若结果含 `void mainImage` → 通过。
2. **确定性包装**：若无 `mainImage` 但含 `void main()` 且写入 `gl_FragColor`/out 变量 → 规则包装为
   `void mainImage(out vec4 fragColor, in vec2 fragCoord){ ... }`，把 `gl_FragCoord` 映射到 `fragCoord`、`gl_FragColor` 映射到 `fragColor`，剥离 `#version`/uniform 声明（与现有渲染契约一致）。
3. **LLM 兜底**：包装仍无法得到可校验的 `mainImage` → 调一次 `generate_llm_glsl_refinement`（或专用 port 提示）做「port to Shadertoy mainImage」转换。
4. **静态校验**：`validate_shader_static`（[backend/app/services/shader_validator.py:13](../backend/app/services/shader_validator.py)）。失败 → seed 视为不可用（见失败处理）。

`adapted_by` 取值 `normalized` / `wrapped` / `llm_ported`，写入 artifacts 供审计。

## 后端改动清单

| 文件 | 操作 | 职责 |
|---|---|---|
| `backend/app/pipeline/seed_glsl.py` | **新建** | `adapt_seed_glsl(source)`、`build_seed_candidate(glsl) -> CandidateRecord`（`source="seed"`, `output_kind="glsl"`, `compile_success=True`；seed 路径不跑 `select_best_candidate`，`priority` 仅占位/展示用） |
| `backend/app/pipeline/graph.py` | 修改 | `run_png_shader_pipeline` 增参 `seed_glsl: str \| None = None`；非空时走 seed 路径（preprocess → adapt → 合成候选 → 评分 → set selected_* → `_run_post_pipeline`），跳过 `_pipeline_graph.invoke()`。默认 `refinement_mode="on"`、`glsl_render_enabled=True`、`output_kind="glsl"`。`_run_post_pipeline` 不动 |
| `backend/app/routers/png_shader.py` | 修改 | `POST /png-shader/run` 增 `seed_glsl: str = Form(default=None)`；经现有 `_run_png_shader_background` 透传到 `run_png_shader_pipeline(..., seed_glsl=...)`。`/status`、`/runs/{id}/stop`、`/runs/{id}/strategy` **零改动**——实时停止/策略热更自动可用 |
| `backend/app/pipeline/input_spec.py` | 修改 | seed 时无 DSL canvas，可从 PNG 自动探测画布尺寸（preprocess 已含 width/height），`target.resolution` 显式给出时优先；把 `seed_glsl` 落入 `input_spec.json` 审计 |
| `backend/app/pipeline/artifacts.py` | 复用 | 额外保存 `seed_input.glsl`（原始）与 `seed_adapted.glsl`（适配后）到 run_dir |

## 前端改动（最小）

- `frontend/src/components/PngShaderParamPanel.tsx` / `PngShaderView.tsx`：新增「从已有着色器开始」开关 + GLSL 文本框（粘贴），可选 `.glsl` 文件加载填入文本框。
- `frontend/src/hooks/usePngShader.ts`：`runPngShader(file, seedGlsl?)` 在现有 `FormData` 追加 `seed_glsl` 字段。
- **下游零改动**：轮询（`pollStatus`）、`PngShaderResult` 结构、scoreboard、`refinement_history` 面板、`ShaderPreview` 全部复用——scoreboard 只显示 1 个 `seed` 候选行。

## 数据流（结果字典形状不变）

seed 路径复用 `_run_post_pipeline` 的序列化，返回字典与现有 `run` 完全同构：`run_id / run_dir / preprocess / scoreboard / selected_candidate_id / selected_glsl / objective_metrics / quality_router / optimization / refinement_summary / refinement_history / candidate_details`。前端无需新字段。

## 失败处理与默认值

- **seed 校验/渲染失败**：硬失败该 run（`status="failed"` + 明确 error，如「seed GLSL 无法适配为 Shadertoy mainImage / 渲染失败」），**不**静默回退到从零生成（违背纯 seed 语义，且对用户意外）。
- **默认精修模式**：`on`（决策 6），可被 `input_spec.quality.refinement_mode` 覆盖。
- **`#define` 优化器前置**：保留（决策 2）。触发条件是 `next_action ∈ {optimize, revise}` 且 `optimizer_iterations(=max_iterations) > 0`——seed run 需保证 `max_iterations > 0`，否则只跑 LLM 循环、跳过坐标下降。
- **画布尺寸**：默认从 PNG 自动探测；`input_spec.target.resolution` 可覆盖。
- **`glsl_render_enabled=False`（无 WebGL）**：seed 路径需要 WebGL 渲染评分，缺失时直接报错说明，而非空跑。
- **审计**：原始 + 适配后 GLSL 落盘到 run_dir。

## 测试

- **单元** `backend/tests/unit/test_seed_glsl.py`（新建）：
  - `adapt_seed_glsl`：mainImage 直通 / `main()`+`gl_FragColor` 规则包装 / 规则失败走 LLM 兜底（注入 fake client）/ 无法校验→失败。
  - `build_seed_candidate`：字段正确（`output_kind="glsl"`、`compile_success`、`source="seed"`）。
- **集成** `backend/tests/unit/test_graph.py`（扩展）：`run_png_shader_pipeline(seed_glsl=...)` 注入 fake 渲染/LLM，断言：跳过候选池（无 baseline/cv 候选）、seed 进入 `run_glsl_refinement_loop`、返回字典形状正确、`refinement_mode` 默认 `on`。
- **复用**：闭环本体已由 `test_glsl_refinement.py` 覆盖，不重复。
- 约定：`cd backend && python -m pytest <path> -v`。

## 不做（YAGNI，v1 范围外）

- 不支持「已有 DSL + PNG」入口（本期仅 GLSL；后续可复用 `run_dsl_refinement_loop`）。
- 不支持「seed 与生成候选同台竞争」模式（即 Option C 的 `external_seed_glsl` 比赛模式）。
- 不支持多 seed 批量同 run。
- 不支持闭环中途 resume/checkpoint。
- 不新增 strategy_config 配置项——复用 `max_iterations` / `max_refinement_iterations` / `refinement_threshold` 等现有参数。

## 关键文件索引

- [backend/app/pipeline/graph.py](../backend/app/pipeline/graph.py) — `run_png_shader_pipeline`(838) / `_run_post_pipeline`(346, GLSL 精修调用点 668–740)
- [backend/app/pipeline/glsl_refinement.py](../backend/app/pipeline/glsl_refinement.py) — `run_glsl_refinement_loop`（闭环本体，原样复用）
- [backend/app/pipeline/scoring.py](../backend/app/pipeline/scoring.py) — `_evaluate_glsl_with_webgl`(174，evaluate_fn 原语)
- [backend/app/pipeline/pool.py](../backend/app/pipeline/pool.py) — `CandidateRecord`（seed 要伪造的数据结构）
- [backend/app/pipeline/refinement.py](../backend/app/pipeline/refinement.py) — `_should_run_refinement`(411，门控)
- [backend/app/utils/glsl_postprocess.py](../backend/app/utils/glsl_postprocess.py) — `normalize_shadertoy_glsl`(80)
- [backend/app/services/shader_validator.py](../backend/app/services/shader_validator.py) — `validate_shader_static`(13) / `validate_shader`(194)
- [backend/app/candidates/llm_scene.py](../backend/app/candidates/llm_scene.py) — `generate_llm_glsl_refinement`（LLM 兜底转换可复用）
- [backend/app/routers/png_shader.py](../backend/app/routers/png_shader.py) — `run_png_shader` 路由 + `_run_png_shader_background`
- [frontend/src/hooks/usePngShader.ts](../frontend/src/hooks/usePngShader.ts) — `runPngShader`（FormData 注入点）
