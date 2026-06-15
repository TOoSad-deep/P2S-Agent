# V2: Human-in-the-loop Branch Workspace 技术方案

> **状态:** 技术方案，待评审。
>
> **依赖:** V1 `branch-refine` 已完成：checkpoint resolver、child run lineage、用户反馈注入、directed acceptance。
>
> **目标读者:** 后端/前端实现者，以及后续把 V2 拆成 implementation plan 的 agentic worker。

## Goal

V2 的目标是把 V1 的“能从 checkpoint 创建 child run”升级成一个可用的分支工作台：

1. 用 timeline 展示一个 run 内所有可分支 checkpoint。
2. 用 branch tree 展示 parent/child/sibling 关系。
3. 支持在 parent/child run 之间切换、比较、继续分支。
4. 分支历史在 `_run_store` 被淘汰或页面刷新后仍可恢复。
5. 让每个分支具备可读元数据：名称、用户反馈、来源 checkpoint、状态、得分、创建时间。

V2 不新增优化算法；它主要是**工作流和可追溯性**。

## Non-goals

- 不做多 variants 批量生成；这是 V3。
- 不做 mask、局部编辑、属性级强约束；这是 V4。
- 不替换现有 1 秒轮询机制。
- 不引入数据库。v2 采用轻量 JSONL/JSON 文件索引，后续需要多人协作时再迁移 DB。
- 不做复杂画布式节点编辑；branch tree 是可扫描、可切换、可继续的工作台视图。

## 与 V1 的关系

V1 提供能力：

- `GET /png-shader/runs/{run_id}/checkpoints`
- `POST /png-shader/runs/{run_id}/branch-refine`
- child run `lineage`
- checkpoint GLSL seed

V2 在其上补齐：

- run lineage 持久化索引
- checkpoint timeline richer metadata
- branch tree API
- artifact/thumbnail 安全读取
- 前端分支工作台 UI

## 用户体验

### 工作台结构

建议在现有 `PngShaderView` 中新增一个可折叠侧栏或底部 panel：

```text
┌─────────────────────────────────────────────────────────────┐
│ Main Preview / Score / Params / LLM I/O                     │
├─────────────────────────────────────────────────────────────┤
│ Branch Workspace                                            │
│ ├─ Timeline: candidate:selected → iter1 → iter2 → final     │
│ ├─ Branch Tree: root run                                    │
│ │    ├─ run_a final                                         │
│ │    ├─ run_b from iter2 "make reflection stronger"         │
│ │    └─ run_c from run_b iter1 "polish brightness"          │
│ └─ Compare strip: current branch vs parent / sibling        │
└─────────────────────────────────────────────────────────────┘
```

### 核心动作

| 动作 | 行为 |
|---|---|
| 选择 timeline checkpoint | 设置 branch 起点，并驱动 preview 显示该 checkpoint GLSL |
| 从 checkpoint 分支 | 调 V1 `branch-refine`，新 run 成为 active run |
| 切换到 parent/child run | 停止当前 active polling，开始轮询/读取目标 run |
| 复制反馈继续分支 | 用上一次 feedback 预填 branch refine 面板 |
| 重命名分支 | 仅改元数据，不影响 artifacts |
| 标记 favorite | 用于对比和后续 V3 winner promotion |

## 数据模型

### RunLineageRecord

新增持久化索引记录，建议放在 `backend/app/pipeline/run_index.py`。

```python
from dataclasses import dataclass, field

@dataclass
class RunLineageRecord:
    run_id: str
    root_run_id: str
    parent_run_id: str | None
    source_checkpoint_id: str | None
    source_checkpoint_label: str | None
    mode: str | None
    feedback: str | None
    title: str | None
    status: str
    run_dir: str | None
    created_at: float
    completed_at: float | None = None
    final_score: float | None = None
    favorite: bool = False
    tags: list[str] = field(default_factory=list)
```

### BranchTreeNode

API 返回给前端的树节点：

```ts
export interface BranchTreeNode {
  run_id: string;
  root_run_id: string;
  parent_run_id?: string | null;
  source_checkpoint_id?: string | null;
  source_checkpoint_label?: string | null;
  title?: string | null;
  mode?: string | null;
  feedback?: string | null;
  status: string;
  final_score?: number | null;
  created_at: number;
  completed_at?: number | null;
  favorite?: boolean;
  children: BranchTreeNode[];
}
```

### CheckpointTimelineEntry

V1 的 checkpoint metadata 在 V2 扩展为 timeline entry：

```ts
export interface CheckpointTimelineEntry {
  id: string;
  run_id: string;
  kind: "candidate" | "refinement_iter" | "final";
  label: string;
  iteration?: number | null;
  score?: number | null;
  score_before?: number | null;
  delta?: number | null;
  accepted?: boolean | null;
  human_goal_override?: string | null;
  changes_summary?: string | null;
  has_glsl: boolean;
  artifact_ids?: {
    render?: string;
    shader?: string;
    llm_io?: string;
  };
}
```

## 持久化设计

### 为什么需要索引

当前 `_run_store` 是内存结构，且 `_MAX_STORE_SIZE=100`。V1 child run 可以工作，但长期分支树会遇到两个问题：

- 页面刷新或服务重启后，很难从内存恢复 parent/child 关系。
- parent 被 store 淘汰后，child 的上下文仍应可追溯。

### 方案：JSONL Run Index

新增文件：

```text
backend/test_results/run_index.jsonl
```

每个 run 需要一条创建生命周期记录，在 completed/failed 时 append 一条 `updated` 记录。若 submit 时还不知道 `run_dir`，创建记录可以先是 `pending`：

```json
{"event":"created","run_id":"run_a","root_run_id":"run_a","parent_run_id":null,"run_dir":null,"status":"pending","created_at":...}
{"event":"updated","run_id":"run_a","run_dir":"...","status":"running"}
{"event":"created","run_id":"run_b","root_run_id":"run_a","parent_run_id":"run_a","source_checkpoint_id":"refinement:iter:2","feedback":"...","run_dir":null,"status":"pending","created_at":...}
{"event":"updated","run_id":"run_b","run_dir":"...","status":"running"}
{"event":"updated","run_id":"run_b","status":"completed","final_score":0.73,"completed_at":...}
```

读取时 fold JSONL，后写覆盖先写。写入仍用 append，避免频繁重写大 JSON。必要时后续加 compact。

当前 `run_png_shader_pipeline` 在 worker 内部调用 `create_run_dir`，submit 时尚不知道 `run_dir`。实现时有两种安全选择：

1. submit 时写 `status="pending"` / `run_dir=null`，worker 创建 run_dir 后立刻 append `updated(run_dir=...)`。
2. 不在 submit 时写 run index，改为 `run_png_shader_pipeline` 创建 run_dir 后通过回调 append `created`。

推荐选择 1，因为 branch tree 可以立刻显示 queued/pending child；但所有读取 artifact/timeline 的接口必须在 `run_dir is None` 时返回 409 或空 timeline。

### 每个 run 的本地元数据

继续保留 V1 artifacts：

```text
lineage.json
branch_request.json
source_checkpoint.json
```

V2 新增：

```text
run_metadata.json
timeline.json
```

`timeline.json` 在 pipeline 完成时落盘；running 状态下仍从 `_run_store` 构造最新 timeline。

## API 设计

### 1. 获取 run timeline

```http
GET /png-shader/runs/{run_id}/timeline
```

返回：

```json
{
  "run_id": "run_b",
  "status": "completed",
  "timeline": [
    {
      "id": "candidate:selected",
      "kind": "candidate",
      "label": "Selected baseline",
      "score": 0.61,
      "accepted": true,
      "has_glsl": true,
      "artifact_ids": {"shader": "checkpoint:candidate:selected:shader"}
    },
    {
      "id": "refinement:iter:1",
      "kind": "refinement_iter",
      "label": "Iter 1",
      "score_before": 0.61,
      "score": 0.68,
      "delta": 0.07,
      "accepted": true,
      "changes_summary": "12 changed lines; #define REFLECT 0.42",
      "has_glsl": true
    }
  ]
}
```

`GET /checkpoints` 可继续存在，V2 前端优先使用 `/timeline`，因为它包含展示信息。

### 2. 获取 branch tree

```http
GET /png-shader/runs/{run_id}/branches
```

语义：根据 `run_id` 找到 `root_run_id`，返回整棵树。

返回：

```json
{
  "root_run_id": "run_a",
  "active_run_id": "run_b",
  "tree": {
    "run_id": "run_a",
    "root_run_id": "run_a",
    "parent_run_id": null,
    "status": "completed",
    "children": [
      {
        "run_id": "run_b",
        "root_run_id": "run_a",
        "parent_run_id": "run_a",
        "source_checkpoint_id": "refinement:iter:2",
        "feedback": "make reflection stronger",
        "status": "running",
        "children": []
      }
    ]
  }
}
```

### 3. 更新 run metadata

```http
PATCH /png-shader/runs/{run_id}/metadata
Content-Type: application/json
```

请求：

```json
{
  "title": "reflection branch",
  "favorite": true,
  "tags": ["water", "promising"]
}
```

只允许更新展示元数据，不允许改 lineage。

### 4. 安全读取 artifact

```http
GET /png-shader/runs/{run_id}/artifacts/{artifact_id}
```

后端通过 `artifact_id` resolver 映射到 run_dir 内的已知文件，不允许传任意文件路径。

支持的 artifact：

- `selected_shader`
- `selected_render`
- `checkpoint:<checkpoint_id>:shader`
- `checkpoint:<checkpoint_id>:render`
- `checkpoint:<checkpoint_id>:llm_io`

安全规则：

- `run_id` 必须存在于 store 或 run index。
- resolved path 必须在 `run_dir` 内。
- 只返回 allowlist 后缀：`.png`, `.json`, `.glsl`, `.txt`。

## 后端改动

### `backend/app/pipeline/run_index.py` 新增

职责：

- `append_run_created(record)`
- `append_run_updated(run_id, fields)`
- `load_run_index() -> dict[str, RunLineageRecord]`
- `build_branch_tree(records, run_id) -> BranchTreeNode`
- `update_run_metadata(run_id, patch)`

JSON 文件写入使用现有 `save_json`；JSONL 追加使用专门的 append helper。两者都需要模块级 lock，避免多 worker 同时写 run index 时交错。

### `backend/app/pipeline/checkpoints.py` 扩展

新增：

- `build_timeline(result: dict) -> list[CheckpointTimelineEntry]`
- `save_timeline(run_dir, result)`
- `resolve_checkpoint_artifact(result, checkpoint_id, kind)`

timeline 构造逻辑：

1. candidate:selected
2. scoreboard candidates 中可 preview 的候选
3. refinement_history 中每个有 `compile_glsl` 的 iteration
4. final:selected

### `backend/app/routers/png_shader.py` 扩展

改动点：

- `_store_run` 时 append pending created，或在 pipeline 创建 run_dir 后 append created；二选一并保持一致。
- worker 创建 run_dir 后 append run index updated(run_dir, status="running")。
- worker 成功/失败时 append run index updated。
- V1 `branch-refine` 创建 child 时写入 `root_run_id`。
- 新增 `/timeline`, `/branches`, `/metadata`, `/artifacts`。

running timeline 直接从 `_run_store[run_id]` 构造；completed 且 store 缺失时从 run_dir artifacts 构造。

### `_run_post_pipeline` partial 增强

推荐 v2 做法：**endpoint 现算 timeline，不把 timeline 放进 status partial**。

status partial 只需要保证 `/timeline` endpoint 有足够原料：

```python
publish_partial({
    "run_dir": str(run_dir),
    "lineage": state.get("lineage"),
    "scoreboard": build_scoreboard(candidates),
    "refinement_history": current_history,
    "selected_glsl": selected_glsl,
})
```

`GET /timeline` 从 `_run_store[run_id]` 或 run_dir artifacts 现算 timeline。这样避免 status payload 随迭代和候选数量膨胀，也避免同一字段在 partial 与 endpoint 中两套实现漂移。

## 前端改动

### hooks

`usePngShader.ts` 新增：

- `fetchTimeline(runId)`
- `fetchBranches(runId)`
- `updateRunMetadata(runId, patch)`
- `switchRun(runId)`

`switchRun` 逻辑：

1. 停止当前 polling。
2. `GET /status/{runId}`，若 running 则开始 polling。
3. 更新 `result/runId/loading/error`。

### 组件

新增：

- `BranchWorkspacePanel.tsx`
- `CheckpointTimeline.tsx`
- `BranchTree.tsx`
- `BranchCompareStrip.tsx`

`BranchWorkspacePanel` 输入：

```ts
interface Props {
  runId: string | null;
  result: PngShaderResult | null;
  activeCheckpointId: string | null;
  onCheckpointSelect: (id: string) => void;
  onBranchRefine: (request: BranchRefineRequest) => void;
  onSwitchRun: (runId: string) => void;
}
```

### UI 状态

`PngShaderView` 新增：

```ts
const [activeCheckpointId, setActiveCheckpointId] = useState<string | null>(null);
const [timeline, setTimeline] = useState<CheckpointTimelineEntry[]>([]);
const [branchTree, setBranchTree] = useState<BranchTreeNode | null>(null);
```

轮询 status 时不一定每秒拉 branches；建议：

- status 每秒拉。
- timeline 在 `refinement_history.length` 变化时拉。
- branches 在 run 切换、child 创建、run terminal 时拉。

## 测试计划

### 后端

`backend/tests/unit/test_run_index.py`

- created/updated JSONL fold 后字段正确。
- `run_dir=null` 的 pending record 可被后续 updated record 补齐。
- child run 的 `root_run_id` 继承 parent root。
- branch tree 按 parent/child 排列。
- metadata patch 不允许修改 lineage 字段。

`backend/tests/unit/test_checkpoints.py`

- `build_timeline` 包含 candidate、iteration、final。
- rejected iteration 也进入 timeline，但 `accepted=false`。
- `artifact_id` resolver 拒绝路径穿越。

`backend/tests/unit/test_router.py`

- `/timeline` 对 running run 返回实时数据。
- `/branches` 对 child run 返回 root tree。
- `/metadata` 更新 title/favorite/tags。
- `/artifacts` 只能返回 run_dir 内 allowlisted artifact。
- `run_dir is None` 时 `/timeline` / `/artifacts` 返回 409 或空 timeline，不做路径解析。

### 前端

- `npm run build`
- timeline 选择 checkpoint 会更新 active checkpoint。
- branch tree 点击节点会调用 `switchRun`。
- metadata patch 后 UI optimistic update，失败回滚。

## 迁移与兼容

- V1 child run 已经有 `lineage`，V2 run index 可从现有 run_dir 的 `lineage.json` backfill。
- 若某些历史 run 没有 lineage，则把它们视为 root run。
- `/checkpoints` 保留兼容；新 UI 使用 `/timeline`。
- 旧 `/refine/{run_id}` 不纳入 branch tree，除非后续把它改造成 child run。

## 实现顺序

1. 后端 `run_index.py` + 单测。
2. 后端 `build_timeline` + `/timeline`。
3. 后端 `/branches` + `/metadata`。
4. 前端 `switchRun` + BranchWorkspace 基础 UI。
5. artifact 安全读取和 thumbnail。
6. 视觉 polish 与对比 strip。

## 关键文件索引

- [backend/app/routers/png_shader.py](../backend/app/routers/png_shader.py) — run/status/branch/timeline/tree endpoints
- [backend/app/pipeline/checkpoints.py](../backend/app/pipeline/checkpoints.py) — V1 checkpoint resolver，V2 timeline 扩展
- [backend/app/pipeline/artifacts.py](../backend/app/pipeline/artifacts.py) — atomic JSON 和 run_dir 工具
- [frontend/src/hooks/usePngShader.ts](../frontend/src/hooks/usePngShader.ts) — run 状态与新 branch workspace API
- [frontend/src/components/PngShaderView.tsx](../frontend/src/components/PngShaderView.tsx) — 工作台挂载点
- [frontend/src/components/LlmIOPanel.tsx](../frontend/src/components/LlmIOPanel.tsx) — refinement iteration 选择入口
