# V1: Human-in-the-loop 分支定向优化技术方案

> **状态:** 技术方案，待评审。
>
> **版本定位:** V1 基础能力。目标是把“从任意 checkpoint 创建定向优化分支”跑通，并为 V2/V3/V4 保留数据模型与 API 扩展点。
>
> **目标读者:** 后端/前端实现者，以及后续负责把本方案拆成 implementation plan 的 agentic worker。

## Goal

在 PNG-to-Shader 闭环优化过程中或结束后，允许用户从某个已有结果点选择一个起点，输入自然语言反馈，再启动一条新的定向优化分支。新分支不覆盖原 run，而是作为 child run 独立运行、独立评分、独立展示。

核心能力：

1. 用户可以从候选、闭环迭代、最终结果中选择 checkpoint。
2. 用户输入反馈，例如“保持云雾层次，但让水面反射更明显，整体不要变暗”。
3. 系统以该 checkpoint 的 GLSL 作为 seed，结合用户反馈进入闭环精修。
4. 新分支保留 lineage，能追溯 parent run、source checkpoint、用户反馈和优化参数。
5. 支持 running 期间介入，也支持 completed 之后从历史结果继续。

## Non-goals

- v1 不做复杂分支树画布，只提供 parent/child lineage 和当前分支切换的最小 UI。
- v1 不做 mask/区域画刷，不做局部图像编辑式约束。
- v1 不做多分支自动探索；`Explore Variants` 在 V3 用 grouped child runs 扩展。
- v1 不改变主 `/png-shader/run` 的默认行为。
- v1 不让 child run 修改 parent run 的结果。

## 现有基础

项目里已经有几块能力可以直接复用：

| 能力 | 现状 | 对本方案的作用 |
|---|---|---|
| Seed GLSL 入口 | `run_png_shader_pipeline(seed_glsl=...)` 已走 `_run_seed_glsl_path`，跳过候选池，进入 `_run_post_pipeline` | 分支 run 可以把 checkpoint GLSL 当 seed |
| 实时 partial publish | `_run_post_pipeline(..., publish_partial=...)` 已能发布 baseline 和每次 refinement history | running 期间用户能看到可选 checkpoint |
| 前端迭代预览 | `LlmIOPanel` 能展示 `refinement_history`，迭代项含 `compile_glsl` | UI 可以直接从迭代卡片派生 checkpoint |
| 旧 HITL endpoint | `POST /png-shader/refine/{run_id}` 已存在，但只支持 completed 后 DSL 一轮改写 | 不作为新能力核心；保留兼容，新增 checkpoint branch API |

本方案的关键判断：**不要继续扩展旧 `/refine/{run_id}` 为主路径**。它只对 completed run 生效，且依赖 DSL；而当前系统已经大量支持 GLSL seed 与 GLSL 闭环。更稳的方案是新增“从 checkpoint 创建 child run”的能力。

## 用户体验

### v1 交互

1. 主 run 正在优化或已结束。
2. 用户在候选表、闭环迭代卡片或最终结果上选择一个起点。
3. 右侧出现一个轻量的“定向优化”面板：
   - 起点：当前选中的 checkpoint。
   - 模式：`Refine` / `Polish` / `Continue`。
   - 用户反馈输入框。
   - 约束开关：保持构图、保持调色、保持背景、只做细节增强。
   - 可选：同时停止 parent run。
4. 用户点击运行后，系统创建 child run。
5. 前端把 child run 作为当前 active run 轮询展示，同时保留 parent lineage。

### 模式语义

| 模式 | 语义 | 默认策略 |
|---|---|---|
| `continue` | 不额外注入用户目标，只从该 checkpoint 继续自动优化 | `refinement_mode=on` |
| `refine` | 按用户反馈做定向优化 | 强制至少 1 轮，开启 directed acceptance |
| `polish` | 结构尽量不变，只做小幅画面质量提升 | 加强 lock notes，降低允许改动幅度 |

V3 可加 `explore`：同一 checkpoint + 同一反馈创建 N 个 child runs 或一个 grouped branch batch。

## Architecture

新增一条 branch-refine 入口：

```text
parent run + checkpoint_id + user_feedback
   │
   ├─ resolve parent result / run_dir / reference_input.png
   ├─ resolve checkpoint -> seed_glsl + checkpoint metadata
   ├─ build human feedback notes / directed acceptance policy
   ├─ create child run store entry
   └─ run_png_shader_pipeline(
          image_path=parent_run_dir/reference_input.png,
          seed_glsl=checkpoint.glsl,
          human_feedback_notes=...,
          directed_acceptance=...,
          lineage=...,
          publish_partial=...
      )
```

核心原则：

- **分支是新 run**：child run 有自己的 `run_id`、`run_dir`、artifacts、状态和轮询生命周期。
- **checkpoint 统一为 GLSL seed**：v1 不新增 DSL seed pipeline；DSL checkpoint 先使用其 compiled GLSL。
- **用户反馈进入闭环，不只是 UI 注释**：feedback 被注入 LLM prompt，并影响接受策略。
- **原 run 不被覆盖**：parent 继续运行或由用户显式停止。

## Checkpoint 模型

新增内部数据结构 `PipelineCheckpoint`，建议放在 `backend/app/pipeline/checkpoints.py`。

```python
from dataclasses import dataclass, field
from typing import Literal

CheckpointKind = Literal["candidate", "refinement_iter", "final"]
ShaderKind = Literal["glsl"]

@dataclass
class PipelineCheckpoint:
    id: str
    kind: CheckpointKind
    label: str
    shader_kind: ShaderKind
    glsl: str
    score: float | None = None
    metrics: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    iteration: int | None = None
    candidate_id: str | None = None
    accepted: bool | None = None
    changes_summary: str | None = None
    source: str | None = None
```

### checkpoint id 约定

| 来源 | id 示例 | GLSL 来源 |
|---|---|---|
| 候选表 | `candidate:llm_0` | `scoreboard.candidates[].compile_glsl` |
| selected baseline | `candidate:selected` | `selected_glsl` 或 selected candidate |
| 第 N 轮迭代 proposal | `refinement:iter:3` | `refinement_history[2].compile_glsl` |
| 最终结果 | `final:selected` | `selected_glsl` |

v1 允许用户从被回滚的 iteration proposal 分支，因为有时“自动评分不喜欢，但用户喜欢”的方向正是人工介入的价值。

### 需要补强的 history 字段

当前 `RefinementEntry` 已有 `compile_glsl`，可满足 v1 选择 iteration proposal。建议后端同时补充以下可选字段，便于 UI 和审计：

```python
entry["checkpoint_id"] = f"refinement:iter:{i + 1}"
entry["accepted"] = bool(accepted_by_policy)
entry["best_score_after"] = round(best_score, 4)
entry["best_glsl_after"] = best_glsl  # 可选；若担心 payload 过大，先不回传，只落盘
entry["render_path"] = str(actual_render) if actual_render else None
```

其中 v1 前端只依赖 `checkpoint_id`、`compile_glsl`、`accepted`、`score_after` 即可。

## API 设计

### 1. 列出可分支 checkpoint

```http
GET /png-shader/runs/{run_id}/checkpoints
```

返回：

```json
{
  "run_id": "run_abcd1234",
  "status": "running",
  "checkpoints": [
    {
      "id": "candidate:selected",
      "kind": "candidate",
      "label": "Selected baseline",
      "score": 0.61,
      "iteration": null,
      "accepted": true,
      "has_glsl": true
    },
    {
      "id": "refinement:iter:2",
      "kind": "refinement_iter",
      "label": "Iteration 2 proposal",
      "score": 0.67,
      "iteration": 2,
      "accepted": true,
      "has_glsl": true
    },
    {
      "id": "final:selected",
      "kind": "final",
      "label": "Current best",
      "score": 0.72,
      "iteration": null,
      "accepted": true,
      "has_glsl": true
    }
  ]
}
```

列表默认不返回完整 GLSL，避免重复传输大 payload。`branch-refine` 由后端根据 `checkpoint_id` 解析 GLSL。

### 2. 从 checkpoint 创建定向优化分支

```http
POST /png-shader/runs/{run_id}/branch-refine
Content-Type: application/json
```

请求：

```json
{
  "checkpoint_id": "refinement:iter:2",
  "feedback": "保持现在的云雾层次，但让水面反射更明显，整体不要变暗。",
  "mode": "refine",
  "locks": {
    "preserve_layout": true,
    "preserve_palette": false,
    "preserve_background": true,
    "small_edits_only": false
  },
  "stop_parent": false,
  "quality": {
    "max_refinement_iterations": 4
  }
}
```

响应沿用 `/png-shader/run` 的 running 形状：

```json
{
  "run_id": "run_child5678",
  "status": "running",
  "parent_run_id": "run_abcd1234",
  "source_checkpoint_id": "refinement:iter:2",
  "lineage": {
    "parent_run_id": "run_abcd1234",
    "source_checkpoint_id": "refinement:iter:2",
    "mode": "refine",
    "feedback": "保持现在的云雾层次，但让水面反射更明显，整体不要变暗。"
  }
}
```

### 错误处理

| 场景 | HTTP |
|---|---|
| parent run 不存在 | 404 |
| parent 还没有可用 checkpoint | 409 |
| checkpoint_id 不存在或无 GLSL | 422 |
| feedback 为空且 mode 是 `refine`/`polish` | 422 |
| parent run_dir/reference_input.png 不存在 | 409 |
| child 创建成功但 pipeline 失败 | child run status=`failed`，错误写入 store |

## 后端改动

### `backend/app/pipeline/checkpoints.py` 新增

职责：

- `list_checkpoints(result: dict) -> list[dict]`
- `resolve_checkpoint(result: dict, checkpoint_id: str) -> PipelineCheckpoint`
- `checkpoint_metadata(cp: PipelineCheckpoint) -> dict`

解析优先级：

1. `scoreboard.candidates`
2. `refinement_history`
3. `selected_glsl`

注意：running 期间 store 里可能还没有完整 `run_dir`。因此 `_run_post_pipeline` 的 baseline partial 需要带上 `run_dir`，使 branch endpoint 可以找到 `reference_input.png`。

### `backend/app/pipeline/graph.py` 修改

`run_png_shader_pipeline` 增加参数：

```python
human_feedback_notes: list[str] | None = None
directed_acceptance: dict | None = None
force_first_refinement_iteration: bool = False
lineage: dict | None = None
```

写入：

- `input_spec["human_feedback"]`
- `input_spec["lineage"]`
- manifest extras 或 config snapshot
- state 中的 `human_feedback_notes`、`directed_acceptance`、`force_first_refinement_iteration`、`lineage`

`_run_post_pipeline` 修改：

- baseline/pre-refine partial 带 `run_dir`、`lineage`。
- 调用 `run_dsl_refinement_loop` / `run_glsl_refinement_loop` 时传入：

```python
initial_extra_feedback=state.get("human_feedback_notes") or None
directed_acceptance=state.get("directed_acceptance") or None
force_first_iteration=state.get("force_first_refinement_iteration", False) or effective_refinement_mode == "on"
```

branch run 默认覆盖：

```python
quality.refinement_mode = "on"
quality.max_refinement_iterations = max(existing_value, 1)
candidates.llm_enabled = True
candidates.glsl_render_enabled = True
force_first_refinement_iteration = True
```

不要依赖把 `refinement_high_score_stop` 设成 `1.01` 来强制首轮：当前 strategy config 会把该值 clamp 到 `1.0`。实现时需要让 `_should_run_refinement` 在 `force_first_refinement_iteration=True` 时放行，并让 DSL/GLSL loop 的高分早停判断也尊重首轮强制：

```python
if best_score >= high_score_stop and not (force_first_iteration and not history):
    stop_reason = "high_score_stop"
    break
```

这样即使 checkpoint 已经高分，也会至少做一轮定向精修。

### `backend/app/pipeline/glsl_refinement.py` / `refinement.py` 修改

两个 loop 增加：

```python
initial_extra_feedback: list[str] | None = None
directed_acceptance: dict | None = None
```

初始化：

```python
extra_feedback: list[str] = list(initial_extra_feedback or [])
```

每次 LLM 调用仍然组合：

```python
extra_feedback + history_notes + semantic_notes + region_notes
```

### 定向接受策略

现有 loop 的接受逻辑是 `delta > 0.0` 才更新 best。对 human-in-loop 不够，因为用户反馈可能带来小幅像素分下降，但语义上更符合用户目标。

建议新增 directed acceptance：

```python
directed = directed_acceptance or {}
score_drop_tolerance = float(directed.get("score_drop_tolerance", 0.03))
goal_feedback = directed.get("feedback")

# directed_acceptance 只保存可 JSON 序列化的配置，不保存 callable。
# callable 在 _run_post_pipeline 内按当前 run_dir/reference_path/VLM 开关临时构造。
goal_pairwise_judge = make_directed_pairwise_judge(goal_feedback) if directed.get("enabled") else None

accept = delta > 0.0
if (
    not accept
    and goal_pairwise_judge is not None
    and actual_render is not None
    and current_render_path is not None
    and delta >= -score_drop_tolerance
):
    verdict = goal_pairwise_judge(current_render_path, actual_render)
    if verdict == "B":
        accept = True
        entry["human_goal_override"] = "accepted_score_drop"
```

同时保留硬 guardrail：

- 渲染失败不能接受。
- 静态校验失败不能接受。
- 分数下降超过 `score_drop_tolerance` 默认不能接受。
- `polish` 模式把 `score_drop_tolerance` 设为 `0.0` 或非常小。

### `backend/app/llm/vlm_judge.py` 修改

新增目标感知 pairwise judge：

```python
def judge_directed_pairwise(
    reference_path: Path,
    current_render_path: Path,
    candidate_render_path: Path,
    *,
    user_feedback: str,
    work_dir: Path,
) -> Literal["A", "B", "tie"] | None:
    ...
```

Prompt 要求：

- A 是当前 checkpoint/best。
- B 是新候选。
- 判断 B 是否更满足用户反馈，同时不能明显背离 reference。
- 若 B 只是过拟合反馈但破坏主要视觉，应返回 A 或 tie。

如果 VLM 不可用，directed acceptance 自动降级为纯 metric acceptance，但仍保留用户反馈 prompt 注入。

`directed_acceptance` 的推荐可持久化形状：

```json
{
  "enabled": true,
  "feedback": "保持云雾层次，但让水面反射更明显",
  "mode": "refine",
  "score_drop_tolerance": 0.03,
  "require_vlm_for_score_drop": true
}
```

### `backend/app/routers/png_shader.py` 修改

新增：

- `GET /png-shader/runs/{run_id}/checkpoints`
- `POST /png-shader/runs/{run_id}/branch-refine`

建议抽出共用 worker 启动函数，避免 `/run` 和 branch endpoint 复制线程启动逻辑：

```python
def _start_pipeline_worker(
    *,
    run_id: str,
    image_path: Path,
    upload_dir: Path | None,
    pipeline_input_spec: dict | None,
    seed_glsl: str | None,
    human_feedback_notes: list[str] | None = None,
    directed_acceptance: dict | None = None,
    lineage: dict | None = None,
) -> dict:
    ...
```

branch endpoint 不传 `upload_dir`，因为 reference image 来自 parent `run_dir`，不能在 child 完成后删除。

因此 worker cleanup 要改成：

```python
if upload_dir is not None:
    shutil.rmtree(upload_dir, ignore_errors=True)
```

child run 的 `run_dir` 由 `run_png_shader_pipeline` 创建。worker 在 pipeline 创建 run_dir 后必须尽早通过 `publish_partial({"run_dir": str(run_dir), ...})` 或专门回调写回 `_run_store`，否则 running 期间无法继续从 child run 分支。

### 用户反馈 notes 构造

新增 helper，例如 `backend/app/pipeline/human_feedback.py`：

```python
def build_human_feedback_notes(
    *,
    feedback: str,
    mode: str,
    locks: dict,
    checkpoint: PipelineCheckpoint,
) -> list[str]:
    notes = [
        f"[START CHECKPOINT] id={checkpoint.id}; score={checkpoint.score}",
    ]
    if feedback.strip():
        notes.insert(0, "[HUMAN GOAL] " + feedback.strip())
    elif mode == "continue":
        notes.insert(0, "[MODE] Continue automatic optimization from the selected checkpoint.")
    if mode == "polish":
        notes.append("[MODE] Polish only: keep composition and major shader structure stable.")
    if locks.get("preserve_layout"):
        notes.append("[LOCK] Preserve layout/composition; do not move major visual elements.")
    if locks.get("preserve_palette"):
        notes.append("[LOCK] Preserve the current color palette unless required by the human goal.")
    if locks.get("preserve_background"):
        notes.append("[LOCK] Preserve background and large-scale lighting.")
    if locks.get("small_edits_only"):
        notes.append("[LOCK] Make small, targeted edits; avoid rewriting the shader from scratch.")
    return notes
```

### Artifacts

Child run 目录新增：

```text
branch_request.json
lineage.json
human_feedback.txt
source_checkpoint.glsl
source_checkpoint.json
```

`input_spec.json` 也包含：

```json
{
  "lineage": {
    "parent_run_id": "...",
    "source_checkpoint_id": "...",
    "mode": "refine"
  },
  "human_feedback": {
    "feedback": "...",
    "locks": {...}
  }
}
```

## 前端改动

### `frontend/src/hooks/usePngShader.ts`

新增类型：

```ts
export interface PipelineCheckpointMeta {
  id: string;
  kind: "candidate" | "refinement_iter" | "final";
  label: string;
  score?: number | null;
  iteration?: number | null;
  accepted?: boolean | null;
  has_glsl: boolean;
}

export interface BranchRefineRequest {
  checkpoint_id: string;
  feedback: string;
  mode: "continue" | "refine" | "polish";
  locks?: Record<string, boolean>;
  stop_parent?: boolean;
  quality?: Partial<StrategyConfig>;
}
```

新增方法：

- `fetchCheckpoints(runId)`
- `branchRefine(parentRunId, request)`

`branchRefine` 成功后复用现有 `pollStatus(childRunId)`，让 child run 成为 active run。

`PngShaderResult` 扩展：

```ts
lineage?: {
  parent_run_id?: string;
  source_checkpoint_id?: string;
  mode?: string;
  feedback?: string;
} | null;
parent_run_id?: string | null;
source_checkpoint_id?: string | null;
```

### UI 组件

建议新增 `HumanLoopPanel.tsx`，而不是把逻辑塞进 `LlmIOPanel`。

职责：

- 显示当前起点 checkpoint。
- 反馈输入框。
- 模式 segmented control。
- lock toggles。
- 启动 branch refine。

`PngShaderView` 持有：

```ts
const [branchCheckpointId, setBranchCheckpointId] = useState<string | null>(null);
```

checkpoint 选择来源：

- 点击候选表 row：可设置 `candidate:{id}`。
- 点击 refinement iteration：可设置 `refinement:iter:{n}`。
- 默认：`final:selected`。

### 视觉原则

- 分支优化是工作台功能，不做 landing/说明页。
- 控件要紧凑，放在现有右侧/底部工具区。
- 反馈输入框是核心控件，但不要让它遮挡 shader preview。
- 分支 run 开始后，展示 lineage chip，例如 `from run_abcd · iter 2`。

## 运行中介入

running parent 的关键要求：

1. branch 只能在已有 checkpoint 后启动。
2. parent `run_dir` 必须已经发布到 store。
3. child 启动后 parent 默认继续；用户可选择 `stop_parent=true`。
4. child 和 parent 使用不同 run_id，不共享 stop flag / strategy revision。

为支持第 2 点，`publish_partial` 的 baseline partial 加：

```python
{
    "run_dir": str(run_dir),
    "lineage": state.get("lineage"),
}
```

## 测试计划

### 后端单元

`backend/tests/unit/test_checkpoints.py`

- 从 `scoreboard` 构造 candidate checkpoints。
- 从 `refinement_history` 构造 iteration checkpoints。
- 从 `selected_glsl` 构造 final checkpoint。
- `resolve_checkpoint` 对不存在 id 报错。
- 无 GLSL checkpoint 不可分支。

`backend/tests/unit/test_human_feedback.py`

- feedback + locks 生成正确 notes。
- `continue` 模式允许空 feedback。
- `refine` / `polish` 模式拒绝空 feedback。

`backend/tests/unit/test_glsl_refinement.py`

- `initial_extra_feedback` 会进入 LLM 调用。
- directed acceptance 在小幅降分且 judge 选 B 时接受。
- 降分超过 tolerance 不接受。
- VLM 不可用时回退 metric acceptance。
- `force_first_iteration=True` 时，即使初始分数达到 `high_score_stop` 也至少执行一轮。
- `directed_acceptance` 不包含 callable，能写入 `input_spec.json` / manifest。

`backend/tests/unit/test_router.py`

- `/checkpoints` running/completed 均可返回。
- `/branch-refine` 创建 child run，lineage 正确。
- checkpoint 不存在返回 422。
- parent 无 run_dir/reference 图返回 409。
- `stop_parent=true` 会设置 parent `stop_requested`。
- branch worker 不会删除 parent run_dir 下的 `reference_input.png`。

### 集成

`backend/tests/unit/test_graph.py`

- branch seed run 透传 `human_feedback_notes` 到 GLSL loop。
- child result 包含 `lineage`、`input_spec.human_feedback`。
- `publish_partial` 包含 `run_dir`。

### 前端

- `npm run build` 类型通过。
- 从 iteration card 选择 checkpoint 后，branch request body 正确。
- branch accepted 后开始轮询 child run。
- child run result 展示 lineage。

## V1 内部分阶段落地

### V1.1: 最小可用 branch refine

- 后端 checkpoint resolver。
- `POST /branch-refine`。
- `human_feedback_notes` 注入 GLSL/DSL loop。
- 前端 feedback 输入框 + 从当前 preview/final 分支。
- metric-only acceptance，先不加 directed VLM acceptance。

> 这是最快可用版本，但用户反馈可能被纯像素分数回滚。

### V1.2: 定向接受策略

- `judge_directed_pairwise`。
- 小幅降分允许目标裁决。
- history 标记 `human_goal_override`。
- UI 展示“目标裁决接受/回滚”。

## 后续版本

V1 完成后，后续能力拆成三份独立方案：

- V2: checkpoint timeline、branch tree、parent/child 切换与分支管理。
- V3: 多分支探索、批量 variants、候选对比与 winner promotion。
- V4: 局部控制、属性锁定、mask/区域约束、用户偏好学习。

## 推荐 v1 范围

建议一次实现 V1.1 + V1.2 的后端基础，前端先做简洁 UI。

原因：

- 只做 V1.1，用户会遇到“我明确让它更亮，但系统因为像素分下降又回滚”的反直觉体验。
- V1.2 的 directed acceptance 是 human-in-loop 的关键质量点。
- V2/V3/V4 是体验和能力增强，可以在 V1 数据模型稳定后继续做。

## 关键文件索引

- [backend/app/pipeline/graph.py](../backend/app/pipeline/graph.py) — `run_png_shader_pipeline` / `_run_post_pipeline` / `_run_seed_glsl_path`
- [backend/app/pipeline/glsl_refinement.py](../backend/app/pipeline/glsl_refinement.py) — GLSL 闭环主循环
- [backend/app/pipeline/refinement.py](../backend/app/pipeline/refinement.py) — DSL 闭环主循环
- [backend/app/routers/png_shader.py](../backend/app/routers/png_shader.py) — run/status/stop/strategy/refine endpoints
- [frontend/src/hooks/usePngShader.ts](../frontend/src/hooks/usePngShader.ts) — run/poll/stop/strategy hook
- [frontend/src/components/PngShaderView.tsx](../frontend/src/components/PngShaderView.tsx) — 主工作台布局
- [frontend/src/components/LlmIOPanel.tsx](../frontend/src/components/LlmIOPanel.tsx) — refinement iteration 展示与预览
