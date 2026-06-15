# Human-in-the-loop V1-V4 实现顺序与规划

> **状态:** 执行规划，待评审。
>
> **关联方案:**
> - [V1 branch refinement](2026-06-15-human-in-loop-branch-refinement-design.md)
> - [V2 branch workspace](2026-06-15-human-in-loop-v2-branch-workspace-design.md)
> - [V3 variant exploration](2026-06-15-human-in-loop-v3-variant-exploration-design.md)
> - [V4 local control/preferences](2026-06-15-human-in-loop-v4-local-control-preferences-design.md)

## 总体原则

1. 先做可运行闭环，再做复杂 UI。
2. 先保证数据可追溯，再做分支树和多分支。
3. 先做结构化输入，再做 mask 和偏好学习。
4. 每个版本都必须能独立验收，不能留下“只能和下个版本一起用”的半成品。
5. 不跳过测试门禁：后端单测优先，前端至少 `npm run build`。

## 推荐总体顺序

```text
M0: Preflight / 现有能力稳定
M1: V1.1 单 checkpoint -> 单 child run
M2: V1.2 directed acceptance
M3: V2.1 run index + timeline API
M4: V2.2 branch workspace UI
M5: V3.1 VariantGroup backend
M6: V3.2 VariantExplorer frontend
M7: V4.1 structured constraints
M8: V4.2 region/mask constraints
M9: V4.3 preference events/profile
M10: V4.4 preference-assisted generation/ranking
```

## M0: Preflight

目标：确认现有 seed GLSL、实时 partial、前端轮询基础可用，再开始 human-in-loop。

必须完成：

- 跑当前后端关键测试：
  - `cd backend && python -m pytest tests/unit/test_seed_glsl.py -v`
  - `cd backend && python -m pytest tests/unit/test_glsl_refinement.py -v`
  - `cd backend && python -m pytest tests/unit/test_router.py -v`
- 跑前端构建：
  - `cd frontend && npm run build`
- 手动确认 `/png-shader/run` 普通模式和 seed GLSL 模式都能完成。
- 确认 running status partial 中至少有 `scoreboard/refinement_history/selected_glsl`。

验收：

- 不修 human-in-loop 之前，主流程必须是绿的。
- 若已有测试失败，先修主流程，不要把失败混进 V1。

## M1: V1.1 单 checkpoint 分支

目标：从某个 checkpoint 创建一个 child run，child run 复用 seed GLSL 路径。

后端顺序：

1. 新增 `backend/app/pipeline/checkpoints.py`。
2. 实现 `list_checkpoints(result)` / `resolve_checkpoint(result, checkpoint_id)`。
3. 新增 `backend/app/pipeline/human_feedback.py`。
4. `run_png_shader_pipeline` 增加 `human_feedback_notes`、`lineage`、`force_first_refinement_iteration` 参数。
5. DSL/GLSL refinement loop 增加 `initial_extra_feedback`。
6. `_run_post_pipeline` partial 带 `run_dir`、`lineage`。
7. 抽出 `_start_pipeline_worker(...)`，兼容 `upload_dir: Path | None`。
8. 新增 `POST /png-shader/runs/{run_id}/branch-refine`。
9. 新增 `GET /png-shader/runs/{run_id}/checkpoints`。
10. 保存 artifacts：`lineage.json`、`branch_request.json`、`source_checkpoint.glsl/json`。

前端顺序：

1. `usePngShader` 增加 `fetchCheckpoints`、`branchRefine`。
2. 新增 `HumanLoopPanel`。
3. `PngShaderView` 增加 `branchCheckpointId`。
4. 候选表/iteration card/final result 能设置起点。
5. branch 创建成功后切到 child run 并复用现有 polling。

验收：

- completed parent 可以从 `final:selected` 分支。
- running parent 可以在已有 `refinement:iter:n` 后分支。
- child run 不覆盖 parent。
- child run 不删除 parent `run_dir/reference_input.png`。
- child result 有 lineage。

测试门禁：

- `cd backend && python -m pytest tests/unit/test_checkpoints.py tests/unit/test_human_feedback.py tests/unit/test_router.py -v`
- `cd frontend && npm run build`

## M2: V1.2 Directed Acceptance

目标：用户目标可以影响接受策略，避免“语义更符合但小幅降分被回滚”。

后端顺序：

1. `directed_acceptance` 只保留 JSON 配置。
2. `_run_post_pipeline` 按 run context 构造 `judge_directed_pairwise` callable。
3. DSL/GLSL loop 接收 `directed_acceptance`。
4. 修改 accept 逻辑：小幅降分 + VLM judge 选 B 时可接受。
5. 增加 `force_first_refinement_iteration` 对 `_should_run_refinement` 和 loop 高分早停的影响。
6. history entry 增加 `human_goal_override`、`accepted`、`best_score_after`。

验收：

- 高分 checkpoint 仍至少执行一轮。
- VLM 不可用时回退 metric-only。
- 降分超过 tolerance 不接受。
- `directed_acceptance` 能写入 JSON artifacts。

测试门禁：

- `cd backend && python -m pytest tests/unit/test_glsl_refinement.py tests/unit/test_graph.py -v`

## M3: V2.1 Run Index + Timeline API

目标：分支历史可恢复，run 内 checkpoint 有统一 timeline。

后端顺序：

1. 新增 `backend/app/pipeline/run_index.py`。
2. 支持 pending record：`run_dir=null`。
3. worker 创建 run_dir 后更新 run index。
4. 成功/失败后更新 final status/score。
5. `checkpoints.py` 增加 `build_timeline(result)`。
6. 新增 `GET /png-shader/runs/{run_id}/timeline`。
7. 新增 `GET /png-shader/runs/{run_id}/branches`。
8. 新增 `PATCH /png-shader/runs/{run_id}/metadata`。
9. 新增安全 artifact resolver。

验收：

- 服务重启后能从 run index 恢复 branch tree。
- `run_dir is None` 时 artifact/timeline 不做路径解析。
- timeline endpoint 从 store 或 artifacts 现算，不依赖 status partial 中的 timeline 字段。

测试门禁：

- `cd backend && python -m pytest tests/unit/test_run_index.py tests/unit/test_checkpoints.py tests/unit/test_router.py -v`

## M4: V2.2 Branch Workspace UI

目标：前端有可用的分支工作台。

前端顺序：

1. `usePngShader` 增加 `fetchTimeline`、`fetchBranches`、`switchRun`、`updateRunMetadata`。
2. 新增 `CheckpointTimeline`。
3. 新增 `BranchTree`。
4. 新增 `BranchWorkspacePanel`。
5. 增加 parent/child 切换。
6. 增加 favorite/title/tags。
7. 增加基础 compare strip。

验收：

- 能看到当前 run 的 timeline。
- 能看到 root 下的 branch tree。
- 点击 branch tree 节点可切换 active run。
- 分支元数据可编辑。

测试门禁：

- `cd frontend && npm run build`

## M5: V3.1 VariantGroup Backend

目标：一次从同一 checkpoint 创建多个 variant child runs。

后端顺序：

1. 新增 `backend/app/pipeline/variant_groups.py`。
2. 实现 `build_variant_strategies`。
3. 增加 VariantGroup 持久化。
4. `RunLineageRecord` 增加 `variant_group_id/index/label`。
5. 新增 `POST /png-shader/runs/{run_id}/explore-variants`。
6. 新增 `GET /png-shader/variant-groups/{group_id}`。
7. 新增 group stop/winner/rating endpoints。
8. 增加 variant worker semaphore。
9. queued child 在 acquire 前可被 stop。

验收：

- 创建 N 个 child runs。
- 每个 child lineage 包含 variant metadata。
- group status 可聚合 queued/running/completed/failed。
- winner/rating 写入 group events，不依赖 V4 preferences。

测试门禁：

- `cd backend && python -m pytest tests/unit/test_variant_groups.py tests/unit/test_router.py tests/unit/test_run_index.py -v`

## M6: V3.2 Variant Explorer UI

目标：用户能比较 variants、选 winner、继续优化。

前端顺序：

1. `usePngShader` 增加 variant group API。
2. 新增 `VariantExplorerPanel`。
3. 新增 `VariantGrid` / `VariantCard`。
4. group status 每 2 秒轮询。
5. 支持 preview variant。
6. 支持 select winner。
7. 支持 continue from winner。

验收：

- variants 逐个完成时 UI 实时更新。
- failed variant 不阻塞其他 variant。
- winner 选中后切换 active run。

测试门禁：

- `cd frontend && npm run build`

## M7: V4.1 Structured Constraints

目标：全局结构化约束进入 prompt 和 artifacts。

后端顺序：

1. 新增 `backend/app/pipeline/human_constraints.py`。
2. 定义 `HumanConstraintSpec`。
3. 校验 locks/targets/edit_strength。
4. `branch-refine` / `explore-variants` 接收 `constraints`。
5. constraints 转成 prompt notes。
6. 保存 `constraints.json`。

前端顺序：

1. 新增 `FineControlPanel`。
2. 支持 locks、targets、edit strength。
3. constraints 传给 branchRefine/exploreVariants。

验收：

- 不传 constraints 行为不变。
- constraints 能写入 artifacts。
- constraints notes 进入 LLM prompt。

## M8: V4.2 Region / Mask Constraints

目标：支持 rectangle 区域约束和局部指标。

后端顺序：

1. 支持 `RegionConstraint`。
2. 新增 `POST /png-shader/runs/{run_id}/region-mask`。
3. V4.2 首版只支持 normalized rect。
4. 新增 `region_metrics.py`。
5. region metrics 写入 `objective_metrics.region_metrics`。
6. protected region 明显变差时作为 hard veto。

前端顺序：

1. 新增 `RegionMaskEditor`。
2. 在 `ImageDiffPanel` 上覆盖 rectangle editor。
3. 每个 region 可设置 mode/instruction/strength。

验收：

- rect 坐标归一化正确。
- 越界 region 返回 422。
- region metrics 只计算区域内像素。

## M9: V4.3 Preference Events / Profile

目标：把用户选择沉淀为可审计偏好。

后端顺序：

1. 新增 `backend/app/pipeline/preferences.py`。
2. 持久化 `events.jsonl` / `profile.json`。
3. 从 V3 winner/rating group events backfill PreferenceEvent。
4. 新增 profile CRUD/rebuild/clear endpoints。
5. deterministic rebuild profile。

前端顺序：

1. 新增 `PreferencePanel`。
2. 展示 profile。
3. 支持启用/禁用、编辑、重建、清空。

验收：

- winner/rating 能生成 preference event。
- profile 可编辑。
- `enabled=false` 时不注入 preference notes。

## M10: V4.4 Preference-assisted Generation / Ranking

目标：偏好辅助 prompt 和 variant ranking。

后端顺序：

1. `build_preference_notes(profile)`。
2. branch/variant 请求默认带 profile snapshot。
3. V3 variants status 增加 preference-assisted rank。
4. 可选 LLM summarizer，但必须保留 raw events。

前端顺序：

1. HumanLoopPanel 增加 `Use preferences`。
2. VariantExplorer 显示 preference recommendation。
3. 用户可覆盖推荐。

验收：

- profile snapshot 落盘。
- 用户可禁用偏好。
- preference ranking 不自动替用户选 winner。

## 不建议并行的任务

- 不要在 V1 branch-refine 未稳定前做 V3 variants。
- 不要在 V2 run index 未完成前做复杂 branch tree。
- 不要在 V4.1 constraints 未完成前做 mask。
- 不要在 V3 group events 未稳定前做 preference profile。

## 可以并行的任务

- V1 后端 checkpoint resolver 与前端 HumanLoopPanel 原型可以并行。
- V2 run index 与 BranchWorkspace UI mock 可以并行。
- V3 backend VariantGroup 与前端 VariantCard 静态组件可以并行。
- V4 PreferencePanel UI 可以在后端 preferences endpoint 前用 mock 数据并行。

## 最小首发范围

建议首发做到 M1 + M2：

- 从任意 checkpoint 创建 child run。
- 用户反馈注入。
- 强制首轮定向 refine。
- directed acceptance。
- 简单 HumanLoopPanel。

这是 human-in-loop 的最小闭环。V2 以后是体验和规模化能力。

