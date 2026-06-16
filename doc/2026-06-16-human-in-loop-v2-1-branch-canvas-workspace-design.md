# V2.1: Branch Canvas Workspace 优化补充方案

> **状态:** V2 体验升级方案，待评审。
>
> **背景:** V1/V2 已完成后，现有能力已经支持 checkpoint 分支、run lineage、timeline、branch tree 和元数据。V2.1 不改变优化 pipeline，而是把 V2 的列表/树式工作台升级为自由画布式分支工作台，为 V3 variants 和 V4 局部控制提供统一承载界面。
>
> **依赖:** V1 branch-refine、V2 run index/timeline/branches/metadata/artifacts 已可用。

## Goal

V2.1 的目标是让 human-in-loop 的前端从“看列表操作”升级成“在优化地图上操作”：

1. 用自由画布展示 root run、checkpoint、child run、分支关系和当前 active run。
2. 支持拖拽、缩放、自动布局、节点选择、节点预览、节点对比、从节点继续分支。
3. 保留 V2 timeline/branch tree 的数据语义，但将 tree 变成画布视图模型。
4. 为 V3 `VariantGroup` 和 V4 `RegionConstraint` / `Preference` 预留节点类型和交互区域。
5. 首版尽量前端化：优先由现有 `/timeline`、`/branches`、`/status`、`/artifacts` 组合生成 canvas，不强制新增后端 pipeline 逻辑。

## Non-goals

- 不重写 V1/V2 后端接口。
- 不改变 child run 创建、评分、接受策略和 lineage 语义。
- 不做多人协同画布。
- 不把 Shader 编辑器、参数面板、主 preview 全部塞进画布节点。
- 不在 V2.1 实现 V3 多分支编排或 V4 区域 mask；只预留节点和插槽。

## 总体界面

推荐采用三栏工作台：

```text
┌────────────────────────────────────────────────────────────────────────────┐
│ Top Toolbar: active run / status / zoom / fit / auto layout / search       │
├──────────────────────────┬─────────────────────────────────┬───────────────┤
│ Main Preview             │ Branch Canvas                   │ Inspector     │
│                          │                                 │               │
│ Reference / Render       │ Input PNG                       │ Selected node │
│ Compare / Diff           │   ↓                             │ Preview info  │
│ Shader output            │ Candidate:selected              │ Feedback      │
│ Region overlay(V4)       │   ↓                             │ Locks         │
│                          │ Iter 1 → Iter 2 ── child run B  │ Refine        │
│                          │   ↓          │                  │ Explore(V3)   │
│                          │ Final A      └─ child run C     │ Controls(V4)  │
└──────────────────────────┴─────────────────────────────────┴───────────────┘
```

布局原则：

- 主 preview 负责高保真查看当前 active node，不跟随画布缩放。
- 画布负责展示关系、选择节点、发起动作。
- 右侧 inspector 负责配置动作，不在节点里塞复杂表单。
- timeline 不再作为独立长列表主入口，而是成为 run node 展开后的 checkpoint lane。

## 用户交互

| 动作 | 行为 |
|---|---|
| 单击节点 | 设置 `selectedCanvasNodeId`，主 preview 显示该节点对应 render/shader，右侧 inspector 显示详情 |
| 双击 run 节点 | `switchRun(run_id)`，该 run 成为 active run 并恢复 status polling |
| 点击 checkpoint 节点 | 设置 branch 起点，可直接在 inspector 输入 feedback 后分支 |
| 从节点拖出连线到空白处 | 创建 branch draft，松手后打开 inspector 的 branch form |
| 框选两个节点 | 打开 compare mode，主 preview 显示 A/B 或 diff |
| 标记 favorite | 写 V2 metadata，节点出现 favorite 标记 |
| 自动布局 | 按 lineage + timeline 重新排列节点，不修改后端 lineage |
| 手动拖拽节点 | 只保存前端 layout preference，不影响 run index |

## 推荐技术选型

前端建议使用 `React Flow`：

- 内置节点/边、拖拽、缩放、fit view、minimap、selection。
- 自定义节点易实现。
- 能从 V2 branch tree 直接映射为 graph。
- 后续 V3 group node、V4 constraint node 可增量加入。

避免手写 canvas 的原因：

- hit testing、pan/zoom、selection、edge routing、keyboard interaction 都会很快变复杂。
- V2.1 的价值在产品工作流，不在底层画布渲染。

## Canvas 视图模型

V2.1 不要求后端新增强绑定 schema。前端可建立 adapter：

```ts
export type BranchCanvasNodeType =
  | "input"
  | "run"
  | "checkpoint"
  | "branch_action"
  | "variant_group"
  | "variant_run"
  | "region_constraint"
  | "preference";

export interface BranchCanvasNodeData {
  type: BranchCanvasNodeType;
  run_id?: string;
  checkpoint_id?: string;
  parent_run_id?: string | null;
  source_checkpoint_id?: string | null;
  title?: string | null;
  label: string;
  status?: string;
  score?: number | null;
  delta?: number | null;
  accepted?: boolean | null;
  favorite?: boolean;
  feedback?: string | null;
  thumbnail_artifact_id?: string | null;
  shader_artifact_id?: string | null;
  group_id?: string | null;
  collapsed?: boolean;
}

export interface BranchCanvasEdgeData {
  relation:
    | "timeline_next"
    | "branch_from"
    | "active_run"
    | "variant_child"
    | "constraint_applies"
    | "preference_influences";
  label?: string;
}
```

React Flow 内部使用：

```ts
type BranchCanvasNode = Node<BranchCanvasNodeData>;
type BranchCanvasEdge = Edge<BranchCanvasEdgeData>;
```

## 数据来源

V2.1 首版使用现有 V2 API：

| 数据 | 来源 | 用途 |
|---|---|---|
| root/parent/child | `GET /png-shader/runs/{run_id}/branches` | run 节点、branch edge |
| run 内 checkpoint | `GET /png-shader/runs/{run_id}/timeline` | checkpoint lane |
| 当前渲染/score/status | `GET /png-shader/status/{run_id}` | active 节点实时状态 |
| thumbnail/render/shader | `GET /png-shader/runs/{run_id}/artifacts/{artifact_id}` | 节点缩略图和 preview |
| title/favorite/tags | `PATCH /png-shader/runs/{run_id}/metadata` | 节点展示元数据 |

### 可选后端聚合接口

如果前端聚合逻辑膨胀，后续可以新增只读 endpoint：

```http
GET /png-shader/runs/{run_id}/canvas
```

返回：

```json
{
  "root_run_id": "run_a",
  "active_run_id": "run_c",
  "nodes": [],
  "edges": [],
  "layout_version": 1
}
```

但 V2.1 首版不依赖该接口。推荐先在前端实现 `buildBranchCanvasModel(...)`，等 V3/V4 节点越来越多时再考虑后端聚合。

## 节点设计

### RunNode

表达一个 run 的整体状态：

- title / run_id 短码
- status：queued/running/completed/failed/cancelled
- final_score 或 current_score
- source checkpoint
- feedback 摘要
- favorite 标记

RunNode 操作：

- `Open` / 双击切换 active run
- `Rename`
- `Favorite`
- `Continue from final`
- `Collapse checkpoints`

### CheckpointNode

表达 run 内一个可分支起点：

- kind：candidate / refinement_iter / final
- iteration
- accepted/rejected
- score/delta
- changes_summary
- thumbnail

CheckpointNode 操作：

- `Preview`
- `Compare with previous`
- `Refine from here`
- `Explore variants`，V3 启用后显示

### BranchActionNode

拖线或点击 `Refine from here` 后产生的临时节点。它不写后端，只用于表达用户正在配置的分支草稿：

- source run
- source checkpoint
- mode
- feedback draft
- locks draft

提交成功后删除 draft node，并刷新 branches/timeline，显示真实 child RunNode。

### VariantGroupNode，V3 预留

V3 完成后，一个 `VariantGroup` 在画布中是一个 group node，可展开为多个 `VariantRunNode`。

### RegionConstraintNode，V4 预留

V4 完成后，区域约束不是独立 run，而是挂在某个 checkpoint/run 节点旁边的附属节点，用 edge `constraint_applies` 连接。

## 画布生成规则

### 输入

```ts
interface BuildBranchCanvasInput {
  activeRunId: string;
  branchTree: BranchTreeNode | null;
  timelinesByRunId: Record<string, CheckpointTimelineEntry[]>;
  statusesByRunId: Record<string, PngShaderStatus | null>;
  collapsedRunIds: Set<string>;
  layoutOverrides: Record<string, { x: number; y: number }>;
}
```

### 输出

```ts
interface BuildBranchCanvasOutput {
  nodes: BranchCanvasNode[];
  edges: BranchCanvasEdge[];
}
```

### 规则

1. branch tree 中每个 run 生成一个 `run` node。
2. active run 默认展开 timeline；非 active run 默认可折叠。
3. 展开的 run 为每个 timeline entry 生成 checkpoint node。
4. run 内 checkpoint 使用 `timeline_next` edge 串联。
5. child run 使用 `branch_from` edge 连接到 parent 的 `source_checkpoint_id`；如果 parent timeline 未加载，则连接到 parent run node。
6. manual layout override 优先于 auto layout。
7. running run 的 active/current checkpoint 可用高亮边框，但不能改变后端状态。

## 自动布局

首版建议采用 deterministic layered layout，不必引入重型图布局库：

- root/input 放左侧。
- 每个 run 是一条纵向 lane。
- checkpoint 沿 Y 轴按时间排列。
- child run 放在 parent checkpoint 右侧一列。
- sibling branch 按创建时间错开。
- variant group 在 V3 中横向展开。

需要满足：

- 每次刷新相同数据时位置稳定。
- 用户拖动过的节点不被自动布局覆盖，除非点击 `Reset layout`。
- 节点数量超过阈值时自动折叠历史 checkpoint，只展开 active run 和 favorite run。

建议阈值：

```ts
const MAX_EXPANDED_RUNS = 3;
const MAX_VISIBLE_CHECKPOINTS_PER_RUN = 8;
```

超过时保留：

- `candidate:selected`
- accepted iterations
- favorite/referenced checkpoint
- `final:selected`

## 状态管理

`PngShaderView` 或新建 `BranchCanvasWorkspace` 管理：

```ts
const [selectedCanvasNodeId, setSelectedCanvasNodeId] = useState<string | null>(null);
const [compareNodeIds, setCompareNodeIds] = useState<[string, string] | null>(null);
const [collapsedRunIds, setCollapsedRunIds] = useState<Set<string>>(new Set());
const [layoutOverrides, setLayoutOverrides] = useState<Record<string, XYPosition>>({});
const [branchDraft, setBranchDraft] = useState<BranchDraft | null>(null);
```

持久化建议：

- layout overrides 先存 `localStorage`，key 带 `root_run_id`。
- title/favorite/tags 仍走 V2 metadata endpoint。
- 不把节点坐标写入 run index，避免 UI 状态污染 lineage。

## Inspector 设计

右侧 inspector 根据节点类型切换：

| 节点类型 | Inspector 内容 |
|---|---|
| input | 原图信息、重新运行入口 |
| run | run 状态、score、feedback、metadata、switch run、continue from final |
| checkpoint | score、accepted、changes、preview、refine from here、compare |
| branch_action | feedback、mode、locks、submit/cancel |
| variant_group | V3 group status、stop、winner |
| variant_run | V3 variant preview、select winner、continue |
| region_constraint | V4 region instruction、mode、strength |
| preference | V4 preference profile 摘要与启用状态 |

V2.1 首版至少实现 `run`、`checkpoint`、`branch_action`。

## 与现有 V2 UI 的关系

V2 已有的 `BranchWorkspacePanel` 可以保留为 fallback 或 compact mode。V2.1 新增：

- `BranchCanvasWorkspace.tsx`
- `BranchCanvas.tsx`
- `BranchCanvasNode.tsx`
- `BranchCanvasInspector.tsx`
- `branchCanvasModel.ts`
- `branchCanvasLayout.ts`

推荐迁移方式：

1. 保留 `CheckpointTimeline` 和 `BranchTree` 的数据 hook。
2. 新建 `BranchCanvasWorkspace` 使用同一批 hooks。
3. 在页面中提供 `Canvas / List` 分段切换。
4. Canvas 稳定后把 List 降级为 accessibility/fallback 视图。

## 与 V3 的衔接

V3 不再把 `VariantExplorerPanel` 当成独立主视图，而是：

- 在 inspector 中提供 `Explore variants` 表单。
- 创建成功后在画布上生成 `VariantGroupNode`。
- group 展开后显示多个 `VariantRunNode`。
- winner 选择后给 winner 节点加 favorite/highlight，并可一键 `Continue from winner`。

V3 后端 API 不需要因为画布改变；前端只需把 `VariantGroupStatus` 映射为 canvas nodes/edges。

## 与 V4 的衔接

V4 的局部控制仍在主 preview 上画区域，但区域结果需要进入画布：

- `RegionMaskEditor` 挂在 Main Preview，不直接画在 Branch Canvas 上。
- 每个 region constraint 在画布中显示为附属 node，连接到它影响的 checkpoint/run。
- `FineControlPanel` 合入 inspector，而不是独立悬浮面板。
- `PreferencePanel` 可作为全局 inspector tab，preference 事件可在画布上以轻量 chips/annotation 展示。

## 测试计划

### 前端

- `npm run build`
- `buildBranchCanvasModel` 能把 branch tree + active timeline 转成稳定 nodes/edges。
- child run 有 `source_checkpoint_id` 时 edge 连接到 checkpoint node。
- parent timeline 未加载时 edge 回退连接到 parent run node。
- 切换 active run 后 polling 与 canvas 高亮同步。
- 节点拖拽只更新 local layout，不调用 metadata endpoint。
- `Refine from here` 创建 branch draft，提交后调用 `branchRefine` 并刷新 branches。
- `Canvas / List` fallback 能切换。

### 手动验收

- 完成态 root run 可以从 final checkpoint 创建 child。
- running run 可以从已有 iteration checkpoint 创建 child。
- child 创建后画布出现新 run 节点和 branch edge。
- 双击 child run 能切换 active run，主 preview 与状态同步。
- 两个节点可进入 compare mode。
- 画布缩放/fit/auto layout 不遮挡 inspector 和主 preview。

## 实现顺序

1. 安装并接入 `reactflow`，新增基础 canvas 容器。
2. 新增 `branchCanvasModel.ts`，用 mock branch tree/timeline 写单测或轻量 fixtures。
3. 新增 `branchCanvasLayout.ts`，实现 deterministic layered layout。
4. 新增 `BranchCanvasNode` 和 `BranchCanvasInspector`。
5. 接入现有 `fetchBranches` / `fetchTimeline` / `switchRun` / `branchRefine`。
6. 实现 branch draft 和 `Refine from here`。
7. 增加 compare selection 与主 preview 联动。
8. 保留 `Canvas / List` 切换，完成手动验收。

## 风险与边界

| 风险 | 处理 |
|---|---|
| 节点过多导致画布混乱 | 默认折叠非 active run，提供 fit/auto layout/search |
| running run timeline 高频变化 | timeline 仍按 refinement_history 变化拉取，不每秒重建所有 branch 数据 |
| 用户拖拽位置被刷新覆盖 | layout override 优先，auto layout 只补新节点 |
| 缩略图加载慢 | 节点先显示 skeleton/score/status，thumbnail 懒加载 |
| 画布交互学习成本高 | 保留 List fallback 和右侧 inspector 中的明确按钮 |

## 关键文件索引

- [frontend/src/components/PngShaderView.tsx](../frontend/src/components/PngShaderView.tsx) — 工作台挂载点
- [frontend/src/hooks/usePngShader.ts](../frontend/src/hooks/usePngShader.ts) — V2 API 复用
- `frontend/src/components/BranchCanvasWorkspace.tsx` — V2.1 新增主组件
- `frontend/src/components/BranchCanvas.tsx` — React Flow 容器
- `frontend/src/components/BranchCanvasNode.tsx` — 自定义节点
- `frontend/src/components/BranchCanvasInspector.tsx` — 右侧 inspector
- `frontend/src/lib/branchCanvasModel.ts` — branch/timeline/status 到 nodes/edges 的 adapter
- `frontend/src/lib/branchCanvasLayout.ts` — deterministic layout
