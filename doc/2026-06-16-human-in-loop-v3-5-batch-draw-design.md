# V3.5: 批量抽卡式多结果生成补充方案

> **状态:** V3 补充方案，待评审。
>
> **背景:** V3 正在执行，已围绕 `VariantGroup` 实现“一次从同一 checkpoint 创建多个 child runs”。V3.5 不重写 V3，而是把 VariantGroup 产品化为“抽卡式批量生成”，补齐抽卡数量、追加抽卡、重抽单卡、收藏、筛选、批次管理和 Branch Canvas 展示。
>
> **依赖:** V1 branch-refine、V2 run index/timeline、已完成的 V2.1 Branch Canvas、V3 VariantGroup backend/frontend。

## Goal

V3.5 的目标是让用户像生图抽卡一样，从一个 checkpoint 和一段目标反馈出发，一次生成多张候选结果，并快速筛选、收藏、重抽、继续优化。

核心能力：

1. 用户可以选择抽卡数量：4/8/12，默认 8。
2. 每张卡仍是独立 child run，复用 V3 `VariantGroup`。
3. 支持追加抽卡：在同一目标下再追加一组结果。
4. 支持单卡重抽：替换某个失败或不满意的 variant。
5. 支持收藏、淘汰、评分、标签、winner。
6. 支持按 score、用户收藏、策略、状态筛选和排序。
7. 在已完成的 Branch Canvas 基础上扩展 `DrawSessionNode` / `DrawCardNode`，把抽卡批次和卡片结果放入画布。
8. 为 V4.5 局部融合提供稳定来源：每张卡都可作为局部素材来源。

## Non-goals

- 不改变 V3 的基本 worker 模型。
- 不要求所有卡同时完成；完成一张展示一张。
- 不自动把多张卡融合成最终结果；局部融合属于 V4.5。
- 不训练偏好模型；收藏/淘汰/评分事件可供 V4.3/V4.4 使用。
- 不把抽卡做成纯图片生成；每张卡必须对应可追溯 child run 和 shader。

## 与 V3 的关系

V3 的 `VariantGroup` 是编排层，V3.5 的 `DrawSession` 是产品层。

```text
V3:
checkpoint + feedback -> VariantGroup -> child runs

V3.5:
checkpoint + feedback -> DrawSession -> one or more VariantGroups -> cards
```

首版可以让一个 `DrawSession` 对应一个 `VariantGroup`。当用户追加抽卡或重抽时，再把新的 group 追加到同一个 session 下：

```text
DrawSession draw_123 "增强水面反射"
  ├─ VariantGroup group_a: initial 8 cards
  ├─ VariantGroup group_b: add 4 more cards
  └─ VariantGroup group_c: redraw card 3
```

这样可以复用 V3 已有接口，又能让用户看到“这是同一次抽卡目标下的多轮尝试”。

## 用户体验

### 抽卡入口

在 `BranchCanvasInspector` 的 checkpoint/run 节点上新增入口：

```text
Batch Draw
Feedback: [让水面反射更明显，整体不要变暗]
Cards: 8
Diversity: medium
Max iterations per card: 2
Locks: preserve layout, preserve cloud layering

[Start draw]
```

若未启用 V2.1，则放在 `VariantExplorerPanel` / `HumanLoopPanel` 中。

### 抽卡结果

```text
Draw Session: "让水面反射更明显"
status: 5/8 completed, 2 running, 1 failed

┌──────────┬──────────┬──────────┬──────────┐
│ Card 1   │ Card 2 ★ │ Card 3   │ Card 4   │
│ score .71│ score .76│ running  │ failed   │
│ 云层好    │ 整体最佳  │          │ [重抽]   │
│ [预览] [收藏] [局部选取] [继续]             │
└──────────┴──────────┴──────────┴──────────┘
```

用户动作：

| 动作 | 行为 |
|---|---|
| Preview | 主 preview 显示该卡对应 child run |
| Favorite | 写 run metadata favorite/tag |
| Eliminate | 在 DrawSession event 中标记 eliminated，不删除 run |
| Redraw card | 基于同一 checkpoint/feedback 创建 replacement group 或 child run |
| Draw more | 在同一 session 下追加新的 VariantGroup |
| Select winner | 复用 V3 winner endpoint，并写 draw session event |
| Use as base | 将该卡作为 V4.5 fusion base |
| Use region | 将该卡作为 V4.5 某个 region 的 source |

## 数据模型

### DrawSessionRecord

新增建议文件：`backend/app/pipeline/draw_sessions.py`。

```python
from dataclasses import dataclass, field

@dataclass
class DrawSessionRecord:
    draw_id: str
    root_run_id: str
    parent_run_id: str
    source_checkpoint_id: str
    feedback: str
    status: str  # "queued" | "running" | "completed" | "partial_failed" | "failed" | "cancelled"
    requested_count: int
    diversity: str
    mode: str = "batch_draw"
    group_ids: list[str] = field(default_factory=list)
    card_run_ids: list[str] = field(default_factory=list)
    winner_run_id: str | None = None
    created_at: float = 0.0
    updated_at: float | None = None
    completed_at: float | None = None
    metadata: dict = field(default_factory=dict)
```

### DrawCardStatus

前端展示结构：

```ts
export interface DrawCardStatus {
  card_id: string;
  run_id: string;
  group_id: string;
  index: number;
  status: string;
  label: string;
  strategy_label?: string | null;
  final_score?: number | null;
  current_score?: number | null;
  thumbnail_url?: string | null;
  feedback?: string | null;
  favorite?: boolean;
  eliminated?: boolean;
  tags?: string[];
  replacement_of_run_id?: string | null;
  can_use_for_fusion: boolean;
}
```

### DrawSessionStatus

```ts
export interface DrawSessionStatus {
  draw_id: string;
  parent_run_id: string;
  source_checkpoint_id: string;
  feedback: string;
  status: "queued" | "running" | "completed" | "partial_failed" | "failed" | "cancelled";
  requested_count: number;
  completed_count: number;
  running_count: number;
  failed_count: number;
  winner_run_id?: string | null;
  group_ids: string[];
  cards: DrawCardStatus[];
}
```

## API 设计

V3.5 可以在 V3 endpoints 外新增 draw session endpoints，内部复用 `explore-variants`。

### 1. 创建抽卡 session

```http
POST /png-shader/runs/{run_id}/draw-session
Content-Type: application/json
```

请求：

```json
{
  "checkpoint_id": "refinement:iter:2",
  "feedback": "保持云雾层次，但让水面反射更明显，整体不要变暗",
  "card_count": 8,
  "diversity": "medium",
  "quality": {"max_refinement_iterations": 2},
  "constraints": {
    "locks": {"preserve_layout": true, "preserve_background": true}
  },
  "stop_parent": false
}
```

响应：

```json
{
  "draw_id": "draw_7f23a1",
  "status": "running",
  "group_ids": ["group_a"],
  "card_run_ids": ["run_1", "run_2", "run_3"]
}
```

实现规则：

- `card_count` 默认 8，允许 2 到 12。
- 超过后端并发上限时仍只排队，不一次性放开 worker。
- 首版可把 `card_count` 传给 V3 `variant_count`；如果 V3 上限仍是 6，则 V3.5 endpoint 将 8/12 拆成多个 group。

### 2. 获取抽卡 session

```http
GET /png-shader/draw-sessions/{draw_id}
```

返回 `DrawSessionStatus`。

### 3. 追加抽卡

```http
POST /png-shader/draw-sessions/{draw_id}/draw-more
Content-Type: application/json
```

请求：

```json
{
  "card_count": 4,
  "diversity": "high",
  "quality": {"max_refinement_iterations": 2}
}
```

行为：

- 继承原 session 的 parent/checkpoint/feedback/constraints。
- 创建新的 VariantGroup。
- 把新 group id 和 child runs append 到 DrawSessionRecord。

### 4. 单卡重抽

```http
POST /png-shader/draw-sessions/{draw_id}/redraw
Content-Type: application/json
```

请求：

```json
{
  "run_id": "run_bad_card",
  "reason": "too dark",
  "diversity": "medium"
}
```

行为：

- 不删除原卡。
- 创建一个 replacement child run 或单卡 group。
- 新 card 的 `replacement_of_run_id` 指向原卡。
- 原卡可自动标记 eliminated。

### 5. 卡片事件

```http
POST /png-shader/draw-sessions/{draw_id}/cards/{run_id}/event
Content-Type: application/json
```

请求：

```json
{
  "event_type": "favorite",
  "value": true,
  "reason": "best cloud layering",
  "tags": ["cloud", "candidate-for-fusion"]
}
```

支持事件：

- `favorite`
- `eliminate`
- `tag`
- `note`
- `use_as_fusion_base`
- `use_as_region_source`

## 后端设计

### `draw_sessions.py`

职责：

- 创建 draw id。
- 持久化 DrawSessionRecord。
- 复用 V3 `build_variant_strategies` 创建 group。
- 聚合多个 group 的 child runs。
- 写 draw session event。
- 输出 DrawSessionStatus。

持久化：

```text
backend/test_results/draw_sessions/<draw_id>.json
backend/test_results/draw_sessions/<draw_id>_events.jsonl
```

### 与 VariantGroup 的复用

不要复制 V3 worker 逻辑。V3.5 应调用同一套 group creation helper：

```python
group = create_variant_group(
    parent_run_id=...,
    source_checkpoint_id=...,
    feedback=...,
    variant_count=batch_size,
    diversity=...,
    mode="batch_draw",
    draw_session_id=draw_id,
)
```

`VariantGroupRecord` 增加可选字段：

```python
draw_session_id: str | None = None
```

`RunLineageRecord` 增加可选字段：

```python
draw_session_id: str | None = None
draw_card_index: int | None = None
replacement_of_run_id: str | None = None
```

### 抽卡策略

V3 的 4 个策略适合探索，但抽卡需要更多去重。建议 V3.5 新增 strategy fan-out：

```python
def build_draw_strategies(
    *,
    feedback: str,
    count: int,
    diversity: str,
    constraints: dict | None,
) -> list[dict]:
    ...
```

策略生成原则：

- 基础策略来自 V3：conservative、semantic、lighting_color、detail_texture。
- count > 4 时，在每个策略下增加 seed hint、rendering technique hint、局部 focus hint。
- diversity=low 时只允许小幅变化。
- diversity=high 时允许更明显的 shader technique 差异，但仍遵守 locks。
- 每个 card 的 prompt notes 必须包含 card index 和 strategy label，方便审计。

### 状态聚合

DrawSession status 根据所有 card 聚合：

- 全部 queued -> `queued`
- 有 running/queued -> `running`
- 全部 completed -> `completed`
- completed + failed 混合 -> `partial_failed`
- 全部 failed -> `failed`
- 用户 stop -> `cancelled` 或 `partial_failed`

## 前端设计

### Canvas 表达

V3.5 在已完成的 Branch Canvas 基础上新增节点，不回改 V2.1 基座文档：

- `DrawSessionNode`
- `DrawCardNode`

```text
checkpoint: iter2
   │
   └─ DrawSession: "增强水面反射"  5/8 done
        ├─ Card 1 conservative
        ├─ Card 2 lighting_color ★
        ├─ Card 3 running
        └─ Card 4 failed [redraw]
```

节点关系：

- checkpoint -> DrawSession: `draw_from`
- DrawSession -> DrawCard: `draw_card`
- replacement card -> original card: `replacement_of`
- selected winner/favorite 可高亮。

### Canvas model 扩展

V3.5 前端扩展 `branchCanvasModel.ts` 的 union，而不是要求 V2.1 预置这些类型：

```ts
type DrawCanvasNodeType = "draw_session" | "draw_card";
type DrawCanvasEdgeRelation = "draw_from" | "draw_card" | "replacement_of";
```

节点数据建议复用现有 run/card 信息：

```ts
interface DrawCanvasNodeData {
  draw_id?: string;
  run_id?: string;
  group_id?: string;
  card_id?: string;
  status: string;
  label: string;
  favorite?: boolean;
  eliminated?: boolean;
  replacement_of_run_id?: string | null;
  can_use_for_fusion?: boolean;
}
```

V3.5 的 `buildBranchCanvasModel` 增量读取 `DrawSessionStatus`，生成 draw nodes/edges；V2.1 的 run/checkpoint/branch 行为保持不变。

### DrawSessionInspector

右侧 inspector 显示：

- prompt/feedback
- cards progress
- Draw more
- Stop session
- Sort/filter
- winner
- fusion actions

### DrawCard

卡片操作：

- Preview
- Favorite
- Eliminate
- Redraw
- Continue
- Use as base，V4.5 启用后显示
- Use region，V4.5 启用后显示

### 筛选与排序

支持：

- all / favorite / completed / running / failed / eliminated
- score desc
- created order
- strategy label
- tagged for fusion

## 与 V4.5 的衔接

V3.5 的每张卡都可以成为 V4.5 的来源：

- `base_run_id`: 选择整体最好的 card。
- `source_run_id`: 选择局部最好的 card。
- DrawSession event 中记录 `use_as_fusion_base` / `use_as_region_source`，便于回溯用户为什么选择这些卡。

V3.5 不生成融合结果，只提供候选池和用户选择信号。

## 测试计划

### 后端

`backend/tests/unit/test_draw_sessions.py`

- 创建 draw session 会创建一个或多个 VariantGroup。
- card_count 超过 V3 单 group 上限时能拆分 group。
- draw-more 会 append group 和 cards。
- redraw 不删除原 run，replacement_of_run_id 正确。
- DrawSessionStatus 聚合 queued/running/completed/failed 正确。
- card event 写 JSONL，favorite 同步 run metadata。

`backend/tests/unit/test_variant_groups.py`

- `draw_session_id` 能写入 group 和 child lineage。
- batch_draw mode 不影响普通 explore-variants。

### 前端

- `npm run build`
- 创建 draw session 请求体正确。
- card grid/canvas 展示逐个完成。
- draw-more 追加卡片不覆盖旧卡。
- redraw 生成 replacement 节点。
- favorite/eliminate/tag 事件能更新 UI。
- V4.5 未启用时 fusion actions 隐藏或 disabled。

## 失败处理

| 场景 | 行为 |
|---|---|
| 部分卡失败 | session `partial_failed`，用户可重抽失败卡 |
| 全部卡失败 | session `failed`，保留错误摘要 |
| draw-more submit 部分失败 | 已创建 group 继续，失败事件写入 session |
| redraw 失败 | 原卡不受影响 |
| V3 worker 并发满 | 新卡进入 queued |
| 用户离开页面 | draw session 可从持久化记录恢复 |

## 指标与日志

新增事件：

- `draw_session_created`
- `draw_more_requested`
- `draw_card_submitted`
- `draw_card_completed`
- `draw_card_redrawn`
- `draw_card_favorited`
- `draw_card_eliminated`
- `draw_card_used_for_fusion`

统计字段：

- cards requested/completed/failed
- favorite ratio
- redraw count
- winner rank by score
- fusion source usage，供 V4.5/V4.3 使用

## 实现顺序

1. `draw_sessions.py` + 持久化 + 单测。
2. 抽出 V3 group creation helper，供 `/explore-variants` 和 `/draw-session` 复用。
3. 新增 create/get/draw-more/redraw/card-event endpoints。
4. `VariantGroupRecord` / `RunLineageRecord` 增加 draw session 可选字段。
5. 前端 hook 增加 draw session API。
6. V3.5 扩展 Branch Canvas：增加 `DrawSessionNode` / `DrawCardNode` 和 draw edges。
7. Inspector 增加 DrawSessionInspector / DrawCard 操作。
8. 加入 fusion action 占位，为 V4.5 对接。

## 关键文件索引

- `backend/app/pipeline/draw_sessions.py` — V3.5 draw session 编排与持久化
- [backend/app/pipeline/variant_groups.py](../backend/app/pipeline/variant_groups.py) — 复用 VariantGroup 创建与状态聚合
- [backend/app/pipeline/run_index.py](../backend/app/pipeline/run_index.py) — draw session lineage 字段
- [backend/app/routers/png_shader.py](../backend/app/routers/png_shader.py) — draw session endpoints
- [frontend/src/hooks/usePngShader.ts](../frontend/src/hooks/usePngShader.ts) — draw session API
- `frontend/src/components/DrawSessionInspector.tsx` — 抽卡批次 inspector
- `frontend/src/components/DrawCard.tsx` — 抽卡结果卡片
- `frontend/src/components/DrawSessionCanvasNode.tsx` — canvas session 节点
- `frontend/src/components/DrawCardCanvasNode.tsx` — canvas card 节点
- `frontend/src/lib/branchCanvasModel.ts` — 将 draw session 映射为 canvas nodes/edges
