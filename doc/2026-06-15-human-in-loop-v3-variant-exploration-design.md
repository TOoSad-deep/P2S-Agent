# V3: Human-in-the-loop 多分支探索技术方案

> **状态:** 技术方案，待评审。
>
> **依赖:** V1 branch refine + V2 branch workspace。若已完成 [V2.1 Branch Canvas Workspace](2026-06-16-human-in-loop-v2-1-branch-canvas-workspace-design.md)，V3 前端应优先以 Branch Canvas 承载 VariantGroup；V3 复用 child run、lineage、timeline、branch tree/canvas，并新增 VariantGroup 编排层。
>
> **目标读者:** 后端/前端实现者，以及后续把 V3 拆成 implementation plan 的 agentic worker。
>
> **V3.5 补充:** V3 正在执行时，不建议打断 VariantGroup 主线。[V3.5 批量抽卡式多结果生成](2026-06-16-human-in-loop-v3-5-batch-draw-design.md) 在 V3 完成后作为产品化增强，复用 VariantGroup 和 child runs，新增 DrawSession、追加抽卡、单卡重抽和卡片事件。

## Goal

V3 的目标是让用户从一个 checkpoint 和一段反馈出发，一次生成多个不同优化方向，并在统一界面中比较、选择、继续。

核心能力：

1. 从同一个 checkpoint 创建 N 个 variant child runs。
2. 每个 variant 拥有不同的探索策略或提示词侧重点。
3. 系统聚合 variant 状态、分数、缩略图、关键变化摘要。
4. 用户可以选择 winner，并把 winner 作为 active branch 或继续定向优化。
5. variant 选择结果作为后续 V4 偏好学习的数据来源。

V3 完成的是“多分支探索底座”；V3.5 完成的是“抽卡式批量生成体验”。两者使用同一套 child run / lineage / group status 能力。

## Non-goals

- 不做局部 mask 或区域约束；这是 V4。
- 不训练模型，只记录用户选择数据。
- 不引入复杂任务队列；v3 仍使用当前后台线程模型，但增加并发上限。
- 不让 variant 互相覆盖；每个 variant 都是独立 run。
- 不要求所有 variant 同时完成后才能预览；完成一个展示一个。

## 与 V1/V2 的关系

V1：

- 单个 checkpoint -> 单个 child run。
- 用户反馈注入闭环。
- directed acceptance。

V2：

- child run 进入 branch tree。
- timeline/checkpoint 管理。
- run metadata 和 artifact 读取。

V2.1：

- branch tree/timeline 被映射为 Branch Canvas。
- checkpoint、run、branch action 都是画布节点。
- V3 的 VariantGroup 在画布中表现为 group node，可展开为多个 variant child run 节点。

V3：

- 在 V1 branch-refine 外包一层 `VariantGroup`。
- 一个 group 创建多个 child runs。
- group 管理聚合状态和对比。

V3.5：

- 在 `VariantGroup` 之上增加 `DrawSession` 产品层。
- 一个 draw session 可包含一个或多个 variant groups。
- 支持 Draw more、Redraw card、Favorite、Eliminate、Use as fusion source。

```text
checkpoint + feedback
   │
   └─ VariantGroup group_123
        ├─ child run A: conservative polish
        ├─ child run B: stronger semantic change
        ├─ child run C: color/lighting focus
        └─ child run D: structure/detail focus
```

## 用户体验

### 入口

在 V2 `HumanLoopPanel` 中新增模式：

- `Refine one`
- `Explore variants`

选择 `Explore variants` 后展示：

- variant 数量：2/3/4/6，默认 4。
- diversity：low / medium / high，默认 medium。
- 每个 variant 最大迭代数，默认 2 到 3。
- 可选：自动停止 parent。

### 对比界面

若尚未启用 V2.1，`VariantExplorerPanel` 可作为独立 panel 展示：

```text
Variant Group: "make reflection stronger"
status: 2/4 completed, 2 running

┌────────────┬────────────┬────────────┬────────────┐
│ Variant A  │ Variant B  │ Variant C  │ Variant D  │
│ thumbnail  │ thumbnail  │ thumbnail  │ thumbnail  │
│ score 0.71 │ score 0.69 │ running... │ failed     │
│ conservative │ semantic │ color      │ detail     │
│ [Preview] [Select winner] [Continue]              │
└────────────┴────────────┴────────────┴────────────┘
```

若已启用 V2.1，则推荐 canvas-first 展示：

```text
checkpoint: refinement:iter:2
   │
   └─ VariantGroup: "make reflection stronger"  status 2/4 completed
        ├─ Variant A conservative    score 0.71  ★ winner
        ├─ Variant B semantic        score 0.69
        ├─ Variant C lighting_color  running
        └─ Variant D detail_texture  failed
```

`VariantExplorerPanel` 在该模式下降级为右侧 inspector/detail view：展示 group 设置、variant 卡片、winner/rating 操作；主关系仍由 Branch Canvas 表达。

用户动作：

| 动作 | 行为 |
|---|---|
| Preview variant | 切换主 preview 到该 child run 的 selected GLSL |
| Select winner | 标记 group winner，并把该 run favorite=true |
| Continue from winner | 以 winner final checkpoint 进入 V1 branch refine |
| Compare with parent | 打开 parent vs variant 对比 |
| Stop group | 对 running child runs 批量 stop |

V3.5 会把这里的 variant 对比进一步产品化为抽卡卡片墙，但 V3 的 Preview / Select winner / Continue / Stop group 动作保持不变。

## 数据模型

### VariantGroupRecord

建议放在 `backend/app/pipeline/variant_groups.py`。

```python
from dataclasses import dataclass, field

@dataclass
class VariantGroupRecord:
    group_id: str
    root_run_id: str
    parent_run_id: str
    source_checkpoint_id: str
    feedback: str
    mode: str
    variant_count: int
    diversity: str
    status: str  # "queued" | "running" | "completed" | "partial_failed" | "failed" | "cancelled"
    child_run_ids: list[str] = field(default_factory=list)
    winner_run_id: str | None = None
    created_at: float = 0.0
    completed_at: float | None = None
```

### VariantRunMetadata

每个 child run 的 lineage 增加：

```json
{
  "variant_group_id": "group_123",
  "variant_index": 0,
  "variant_label": "conservative",
  "variant_strategy": {
    "diversity": "medium",
    "prompt_focus": "preserve composition; polish reflections",
    "score_drop_tolerance": 0.01
  }
}
```

### VariantGroupStatus

API 返回：

```ts
export interface VariantGroupStatus {
  group_id: string;
  parent_run_id: string;
  source_checkpoint_id: string;
  feedback: string;
  status: "queued" | "running" | "completed" | "partial_failed" | "failed" | "cancelled";
  winner_run_id?: string | null;
  variants: Array<{
    run_id: string;
    variant_index: number;
    label: string;
    status: string;
    final_score?: number | null;
    current_score?: number | null;
    selected_glsl?: string | null;
    thumbnail_url?: string | null;
    changes_summary?: string | null;
    error?: string | null;
    favorite?: boolean;
  }>;
}
```

## Variant 策略生成

V3 需要保证多个 child run 不只是重复调用同一 prompt。

新增 helper：

```python
def build_variant_strategies(
    *,
    feedback: str,
    count: int,
    diversity: str,
    mode: str,
) -> list[dict]:
    ...
```

默认 4 个策略：

| label | prompt_focus | acceptance |
|---|---|---|
| `conservative` | 保持构图和调色，只做小幅改动 | `score_drop_tolerance=0.005` |
| `semantic` | 更强烈满足用户反馈，可适度改变局部表现 | `score_drop_tolerance=0.03` |
| `lighting_color` | 优先调整亮度、对比、色彩、反射/阴影 | `score_drop_tolerance=0.02` |
| `detail_texture` | 优先增强纹理、边缘、局部细节 | `score_drop_tolerance=0.02` |

如果 `diversity=high`：

- 允许 `fresh_start` 较早触发。
- feedback notes 加入“try a different rendering technique”。
- `score_drop_tolerance` 上限可到 `0.05`，但仍需要 directed VLM judge。

如果 `diversity=low`：

- 所有策略都加 `small_edits_only`。
- 不允许大幅结构改动。

## API 设计

### 1. 创建 variants

```http
POST /png-shader/runs/{run_id}/explore-variants
Content-Type: application/json
```

请求：

```json
{
  "checkpoint_id": "refinement:iter:2",
  "feedback": "保持云雾层次，但让水面反射更明显",
  "variant_count": 4,
  "diversity": "medium",
  "mode": "explore",
  "quality": {
    "max_refinement_iterations": 3
  },
  "stop_parent": false
}
```

响应：

```json
{
  "group_id": "group_7f23a1",
  "status": "running",
  "parent_run_id": "run_a",
  "source_checkpoint_id": "refinement:iter:2",
  "child_run_ids": ["run_b", "run_c", "run_d", "run_e"]
}
```

### 2. 获取 group 状态

```http
GET /png-shader/variant-groups/{group_id}
```

返回 `VariantGroupStatus`。

状态计算：

- 全部 queued -> `queued`
- 任一 running 或 queued，且未全部 terminal -> `running`
- 至少一个 completed 且还有 running -> `running`
- 全部 completed -> `completed`
- 有 completed 也有 failed -> `partial_failed`
- 全部 failed -> `failed`
- 用户批量停止 -> `cancelled` 或 `partial_failed`

### 3. 停止 group

```http
POST /png-shader/variant-groups/{group_id}/stop
```

行为：对所有 queued/running child run 设置 `stop_requested=true`；queued child 在 acquire semaphore 前读到 stop flag 时直接标记为 cancelled/failed，不再启动 pipeline。

### 4. 选择 winner

```http
POST /png-shader/variant-groups/{group_id}/winner
Content-Type: application/json
```

请求：

```json
{
  "winner_run_id": "run_c",
  "reason": "reflection is stronger without darkening the image"
}
```

行为：

- group `winner_run_id=run_c`
- run_c metadata `favorite=true`
- 写入 VariantGroup event；V4 启用后再镜像/汇总为 PreferenceEvent

### 5. 评价 variant

```http
POST /png-shader/variant-groups/{group_id}/ratings
Content-Type: application/json
```

请求：

```json
{
  "run_id": "run_c",
  "rating": 1,
  "reason": "best balance",
  "tags": ["reflection", "not-too-dark"]
}
```

`rating` 建议取值：

- `1`: like
- `0`: neutral
- `-1`: dislike

## 后端改动

### `backend/app/pipeline/variant_groups.py` 新增

职责：

- 创建 group id。
- 生成 variant strategies。
- 持久化 group record。
- 聚合 child run 状态。
- 写 winner/rating group event。V3 不直接依赖 V4 `preferences.py`；V4 可从这些 group events backfill preference events。

持久化：

```text
backend/test_results/variant_groups/<group_id>.json
backend/test_results/variant_groups/<group_id>_events.jsonl
```

### `backend/app/pipeline/run_index.py` 扩展

`RunLineageRecord` 增加：

```python
variant_group_id: str | None = None
variant_index: int | None = None
variant_label: str | None = None
```

`build_branch_tree` 可以选择：

- 默认显示每个 variant child。
- UI 可将同 group 的 sibling 折叠成一个 group node。

V2.1 canvas 模式下不建议修改 run index 的树语义。前端 `buildBranchCanvasModel` 根据 `variant_group_id/index/label` 把 sibling child runs 聚合为 `VariantGroupNode`，展开时再显示每个 `VariantRunNode`。

### `backend/app/routers/png_shader.py` 扩展

新增 endpoints：

- `POST /png-shader/runs/{run_id}/explore-variants`
- `GET /png-shader/variant-groups/{group_id}`
- `POST /png-shader/variant-groups/{group_id}/stop`
- `POST /png-shader/variant-groups/{group_id}/winner`
- `POST /png-shader/variant-groups/{group_id}/ratings`

### 启动 child runs

不要复制 pipeline worker 逻辑；复用 V2 抽出的 `_start_pipeline_worker`。

伪代码：

```python
strategies = build_variant_strategies(...)
child_run_ids = []
for idx, strategy in enumerate(strategies):
    child_run_id = "run_" + uuid4().hex[:8]
    lineage = {
        **base_lineage,
        "variant_group_id": group_id,
        "variant_index": idx,
        "variant_label": strategy["label"],
        "variant_strategy": strategy,
    }
    notes = build_human_feedback_notes(...) + strategy["notes"]
    directed_acceptance = build_directed_acceptance(..., strategy=strategy)
    _start_pipeline_worker(..., run_id=child_run_id, seed_glsl=checkpoint.glsl, ...)
    child_run_ids.append(child_run_id)
```

### 并发控制

当前用后台线程，V3 必须限制并发，避免 LLM/API/WebGL 同时打爆。

新增简单 semaphore：

```python
_variant_worker_semaphore = threading.Semaphore(2)
```

每个 child worker 在真正跑 pipeline 前 acquire，finally release。

为避免 UI 把等待 semaphore 的 child 误判为正在推理，`_start_pipeline_worker` 应先把 child store status 设为 `queued` 或 `running + current_phase="queued"`；acquire 成功后再切到 `running + current_phase="preprocessing"`。

配置项建议：

```json
{
  "max_variant_count": 6,
  "max_variant_concurrency": 2
}
```

先写在后端常量即可，后续进入 strategy config。

### 自动排序

group status 返回 variants 时按：

1. winner first
2. completed > running > failed
3. final_score desc
4. variant_index asc

可选 V3.1：完成后跑 VLM tournament：

```python
judge_variant_group(reference, parent_render, variant_renders, feedback)
```

输出 `auto_rank`，但不要自动选 winner，只作为 UI 推荐。

## 前端改动

### hooks

`usePngShader.ts` 新增：

- `exploreVariants(parentRunId, request)`
- `fetchVariantGroup(groupId)`
- `stopVariantGroup(groupId)`
- `selectVariantWinner(groupId, runId, reason?)`
- `rateVariant(groupId, runId, rating, reason?)`

### 组件

新增：

- `VariantExplorerPanel.tsx`
- `VariantGrid.tsx`
- `VariantCard.tsx`
- `VariantCompareModal.tsx`
- `VariantGroupCanvasNode.tsx`，V2.1 canvas 模式新增
- `VariantRunCanvasNode.tsx`，V2.1 canvas 模式新增
- `VariantInspector.tsx`，可复用 `VariantExplorerPanel` 的详情与操作

`VariantCard` 信息：

- label
- status
- score/current score
- thumbnail 或 shader preview
- changes summary
- action buttons: Preview / Select / Continue / Like / Dislike

### 状态管理

新增：

```ts
const [activeVariantGroupId, setActiveVariantGroupId] = useState<string | null>(null);
const [variantGroup, setVariantGroup] = useState<VariantGroupStatus | null>(null);
```

轮询：

- active run status 仍然每秒。
- active group status 每 2 秒。
- group 全部 terminal 后停止 group polling。

### Winner 后动作

选择 winner 后：

1. 更新 group status。
2. 调 `switchRun(winnerRunId)`，让主工作台展示 winner。
3. 刷新 branch tree/canvas model。
4. winner 节点高亮并写入 favorite metadata。
5. HumanLoopPanel/Inspector 的默认 checkpoint 设为 `final:selected`。

## 测试计划

### 后端

`backend/tests/unit/test_variant_groups.py`

- `build_variant_strategies` 数量、label、notes 正确。
- diversity low/medium/high 参数不同。
- group record 持久化和读取。
- group status 聚合 running/completed/failed。
- winner 写入 group event；V4 可从 group event 回填 PreferenceEvent。

`backend/tests/unit/test_router.py`

- `/explore-variants` 创建 N 个 child runs。
- 每个 child lineage 包含 group id/index/label。
- variant_count 超上限返回 422。
- `/variant-groups/{id}/stop` 设置所有 queued/running child stop flag。
- `/winner` 拒绝不属于该 group 的 run_id。

`backend/tests/unit/test_run_index.py`

- 同 group child runs 在 branch tree 可折叠/可查询。

### 前端

- `npm run build`
- Explore Variants request body 正确。
- group polling 展示逐个完成的 variant。
- Select winner 后调用 API 并 switch run。
- failed variant 卡片不会阻塞其他 variants。
- V2.1 canvas 模式下，group 能显示为可折叠 `VariantGroupNode`，展开后每个 child run 有独立节点。
- winner 选择后 canvas 高亮 winner 节点，且 active run 与主 preview 同步。

## 失败处理

| 场景 | 行为 |
|---|---|
| 部分 child submit 失败 | group 标为 `partial_failed`，已创建的 child 继续 |
| 全部 child failed | group `failed` |
| 用户 stop group | running child 设置 stop，completed child 保留 |
| 一个 variant LLM 失败 | 该 child run failed，不影响 group 其他 child |
| parent/checkpoint 不存在 | `/explore-variants` 返回 404/422，不创建 group |

## 指标与日志

新增 log events：

- `variant_group_created`
- `variant_child_submitted`
- `variant_group_status`
- `variant_group_stopped`
- `variant_winner_selected`
- `variant_rated`

统计字段：

- group completion time
- best score among variants
- winner score rank
- user selected winner vs auto top score 是否一致

这些数据直接服务 V4 偏好学习。

V3.5 会额外记录 draw session 事件，例如 `draw_card_favorited`、`draw_card_redrawn`、`draw_card_used_for_fusion`。这些事件同样可在 V4.3/V4.4 中进入 preference profile。

## 实现顺序

1. `variant_groups.py` + 单测。
2. `/explore-variants` 创建 grouped child runs。
3. `/variant-groups/{id}` 聚合状态。
4. stop/winner/rating endpoints。
5. 前端 hooks 接入 group API。
6. 若未启用 V2.1，先实现 `VariantExplorerPanel` / `VariantGrid`。
7. 若已启用 V2.1，优先实现 `VariantGroupCanvasNode` / `VariantRunCanvasNode` / `VariantInspector`，`VariantExplorerPanel` 作为 fallback/detail。
8. winner promotion 和 branch tree/canvas 联动。
9. 可选 VLM tournament 自动推荐。
10. V3 主线稳定后，再实施 V3.5 DrawSession；不要在 VariantGroup backend 未稳定前引入 draw-more/redraw 语义。

## 关键文件索引

- [backend/app/routers/png_shader.py](../backend/app/routers/png_shader.py) — variant group endpoints 与 worker 启动
- [backend/app/pipeline/checkpoints.py](../backend/app/pipeline/checkpoints.py) — source checkpoint resolver
- [backend/app/pipeline/run_index.py](../backend/app/pipeline/run_index.py) — lineage 与 branch tree
- [backend/app/pipeline/glsl_refinement.py](../backend/app/pipeline/glsl_refinement.py) — variant child run 复用闭环
- [frontend/src/hooks/usePngShader.ts](../frontend/src/hooks/usePngShader.ts) — variant group API
- [frontend/src/components/PngShaderView.tsx](../frontend/src/components/PngShaderView.tsx) — variant explorer/canvas 挂载点
- `frontend/src/lib/branchCanvasModel.ts` — V2.1 canvas 模式下把 variant groups 映射为节点和边
- [V3.5 batch draw supplement](2026-06-16-human-in-loop-v3-5-batch-draw-design.md) — 抽卡式批量生成、DrawSession、重抽与追加抽卡
