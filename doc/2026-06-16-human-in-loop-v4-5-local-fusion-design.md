# V4.5: 局部叠加与融合优化技术方案

> **状态:** V4 补充方案，待评审。
>
> **背景:** V3.5 批量抽卡会产生多个各有局部优点的 shader 结果。用户常见需求是“这张整体最好，但那张水面更好，另一张云层更好”。V4.5 的目标是把这些局部优点组合为一个统一 shader，而不是只做静态图片拼贴。
>
> **依赖:** 已完成的 V2.1 Branch Canvas、V3 VariantGroup、V3.5 DrawSession、V4.1 structured constraints、V4.2 region/mask constraints。

## Goal

V4.5 的目标是支持用户从多张抽卡结果中选择局部优点，生成融合目标，并通过 shader 定向优化输出一个统一、可运行、可追溯的 shader。

核心能力：

1. 用户选择一张 base result，作为整体构图和 shader 起点。
2. 用户在 preview 上框选区域，并为每个区域选择 source result。
3. 系统生成 `FusionPlan`，记录 base、source、region、instruction、blend policy。
4. 系统生成 composite target image，用作视觉优化目标。
5. 系统从 base shader 启动 branch refine，结合 composite target、region constraints 和 fusion notes 做闭环优化。
6. 输出一个新的 child run / fusion result，而不是静态拼接图。
7. V4.5 在已完成的 Branch Canvas 基础上扩展 `FusionPlanNode` 和 fusion edges，展示 source cards -> fusion plan -> fusion result 的关系。

## Non-goals

- V4.5 首版不做 shader AST/代码级硬合并。
- 不承诺局部融合像素级完全等同 source card。
- 不把融合结果保存为最终 PNG 后就结束；必须尝试生成统一 shader。
- 不替代 V4.2 region constraints；V4.5 复用其区域选择与局部评分。
- 不要求 fusion source 必须来自同一个 draw session，但首版 UI 优先支持同一 draw session。base run 必须有 selected GLSL；source run 首版只要求有可读取 render。

## 核心判断

局部叠加有两种实现路径：

| 路径 | 说明 | 建议 |
|---|---|---|
| Composite target + shader refine | 先把多张结果按区域合成目标图，再让 base shader 闭环逼近 | V4.5 首版采用 |
| Shader code merge | 尝试把 source shader 的局部函数/参数/逻辑合进 base shader | 后续 V5 或 V4.6 探索 |

原因：

- 当前系统的最终产物是 shader，不能只输出拼贴 PNG。
- 不同 shader 的坐标系、函数、噪声、变量命名可能不兼容，直接合并代码风险高。
- Composite target 可以复用已有 V4 region constraints、metrics、VLM judge 和 branch-refine 闭环。
- 用户体验上仍然像“局部叠加”，技术上则是“有目标图和区域约束的定向优化”。

## 用户体验

### 入口

从 V3.5 draw session 进入：

```text
Draw Session: "增强水面反射"
Card 2: 整体最好
Card 5: 水面最好
Card 1: 云层最好

[Create fusion]
```

也可以从 Branch Canvas 多选结果进入：

```text
Select Card 2 + Card 5 + Card 1
[Create local fusion]
```

### Fusion Builder

```text
Base result:
[Card 2 - overall best]

Regions:
1. water area
   Source: Card 5
   Instruction: use stronger water reflection
   Blend: soft
   Strength: 0.65

2. sky/cloud
   Source: Card 1
   Instruction: preserve better cloud layering
   Blend: protect/source-guided
   Strength: 0.50

[Preview composite target]
[Start fusion refine]
```

用户动作：

| 动作 | 行为 |
|---|---|
| Select base | 设置整体 shader 起点 |
| Add region | 在 Main Preview 上框选区域 |
| Assign source | 为区域选择一张抽卡结果或任意 run |
| Preview composite | 生成或前端预览 composite target |
| Start fusion refine | 从 base shader 创建 fusion child run |
| Compare sources | 查看 base/source/composite/fusion result |
| Adjust strength | 控制区域融合目标影响强度 |

## Branch Canvas 表达

V4.5 不要求 V2.1 预置 fusion 节点。Fusion 相关节点和边由 V4.5 自己扩展 `branchCanvasModel.ts`：

```text
DrawSession draw_123
   ├─ Card 1 "cloud best" ───┐
   ├─ Card 2 "base" ─────────┼─ FusionPlan fusion_456 ── FusionResult run_f
   └─ Card 5 "water best" ───┘
```

节点：

- `FusionPlanNode`
- `FusionSourceNode`，可复用 DrawCard/Run node
- `FusionResultNode`，本质是 child run node

边：

- `fusion_base`: base result -> fusion plan
- `fusion_source`: source result -> fusion plan
- `fusion_output`: fusion plan -> fusion result
- `region_source`: source result -> region constraint

### Canvas model 扩展

```ts
type FusionCanvasNodeType = "fusion_plan";
type FusionCanvasEdgeRelation =
  | "fusion_base"
  | "fusion_source"
  | "fusion_output"
  | "region_source";
```

节点数据：

```ts
interface FusionCanvasNodeData {
  fusion_id: string;
  status: "draft" | "target_ready" | "running" | "completed" | "failed";
  base_run_id: string;
  source_run_ids: string[];
  output_run_id?: string | null;
  composite_target_artifact_id?: string | null;
  region_count: number;
}
```

V4.5 的 canvas adapter 从 `FusionStatus` 生成 fusion nodes/edges；V2.1 的基础 run/checkpoint/branch 视图不需要回改。

## 数据模型

### FusionPlanRecord

新增建议文件：`backend/app/pipeline/fusion_plans.py`。

```python
from dataclasses import dataclass, field

@dataclass
class FusionRegion:
    id: str
    label: str
    source_run_id: str
    instruction: str
    geometry_type: str  # "rect" first, later "polygon" | "mask"
    geometry: dict
    strength: float = 0.5
    blend_mode: str = "soft"  # "soft" | "replace_target" | "protect_base"
    feather: float = 0.08

@dataclass
class FusionPlanRecord:
    fusion_id: str
    root_run_id: str
    parent_run_id: str
    base_run_id: str
    source_run_ids: list[str]
    draw_session_id: str | None
    feedback: str
    status: str  # "draft" | "target_ready" | "running" | "completed" | "failed"
    regions: list[FusionRegion] = field(default_factory=list)
    composite_target_artifact_id: str | None = None
    output_run_id: str | None = None
    created_at: float = 0.0
    updated_at: float | None = None
    metadata: dict = field(default_factory=dict)
```

### JSON shape

```json
{
  "fusion_id": "fusion_456",
  "base_run_id": "run_card_2",
  "draw_session_id": "draw_123",
  "feedback": "combine the best water reflection and cloud layering into one coherent shader",
  "regions": [
    {
      "id": "region_water",
      "label": "water",
      "source_run_id": "run_card_5",
      "instruction": "use stronger water reflection",
      "geometry_type": "rect",
      "geometry": {"x": 0.05, "y": 0.58, "w": 0.90, "h": 0.34},
      "strength": 0.65,
      "blend_mode": "soft",
      "feather": 0.08
    }
  ]
}
```

### FusionStatus

```ts
export interface FusionStatus {
  fusion_id: string;
  status: "draft" | "target_ready" | "running" | "completed" | "failed";
  base_run_id: string;
  source_run_ids: string[];
  output_run_id?: string | null;
  composite_target_url?: string | null;
  regions: FusionRegion[];
  error?: string | null;
}
```

## API 设计

### 1. 创建 fusion draft

```http
POST /png-shader/fusions
Content-Type: application/json
```

请求：

```json
{
  "base_run_id": "run_card_2",
  "draw_session_id": "draw_123",
  "feedback": "融合更好的水面反射和云层层次",
  "regions": [
    {
      "id": "region_water",
      "label": "water",
      "source_run_id": "run_card_5",
      "instruction": "use stronger water reflection",
      "geometry_type": "rect",
      "geometry": {"x": 0.05, "y": 0.58, "w": 0.90, "h": 0.34},
      "strength": 0.65,
      "blend_mode": "soft"
    }
  ]
}
```

响应：

```json
{
  "fusion_id": "fusion_456",
  "status": "draft"
}
```

### 2. 生成 composite target

```http
POST /png-shader/fusions/{fusion_id}/composite-target
```

行为：

- 读取 base render 和 source renders。
- 按 normalized region 生成 feathered mask。
- 合成 `composite_target.png`。
- 保存 `fusion_plan.json`、`composite_target.png`、`region_masks/*.png`。

返回：

```json
{
  "fusion_id": "fusion_456",
  "status": "target_ready",
  "composite_target_url": "/png-shader/fusions/fusion_456/artifacts/composite_target"
}
```

### 3. 启动 fusion refine

```http
POST /png-shader/fusions/{fusion_id}/run
Content-Type: application/json
```

请求：

```json
{
  "quality": {"max_refinement_iterations": 4},
  "directed_acceptance": {
    "score_drop_tolerance": 0.03,
    "require_vlm_for_score_drop": true
  }
}
```

行为：

- 使用 `base_run_id` 的 final/selected GLSL 作为 seed。
- 使用 `composite_target.png` 作为额外 target reference。
- 构造 fusion notes 和 region constraints。
- 创建 child run，lineage 写入 `fusion_id`、`base_run_id`、`source_run_ids`。
- 输出 `output_run_id`。

响应：

```json
{
  "fusion_id": "fusion_456",
  "status": "running",
  "output_run_id": "run_fusion_1"
}
```

### 4. 获取 fusion 状态

```http
GET /png-shader/fusions/{fusion_id}
```

返回 `FusionStatus`。

### 5. Fusion artifact

```http
GET /png-shader/fusions/{fusion_id}/artifacts/{artifact_id}
```

支持：

- `fusion_plan`
- `composite_target`
- `region_mask:<region_id>`
- `source_crop:<region_id>`

安全规则与 V2 artifacts 一致：只允许 fusion dir 内 allowlist 后缀。

## 后端设计

### `fusion_plans.py`

职责：

- 创建/读取/更新 FusionPlanRecord。
- 校验 base run 可解析到 render 和 selected GLSL。
- 校验 source runs 可解析到 render；source shader 可选，首版不依赖 source shader 做代码级合并。
- 校验 region geometry。
- 保存 fusion artifacts。
- 输出 fusion notes。

### `image_composite.py`

新增轻量图像合成 helper：

```python
def build_composite_target(
    *,
    base_render_path: Path,
    source_render_paths: dict[str, Path],
    regions: list[FusionRegion],
    output_dir: Path,
) -> Path:
    ...
```

首版规则：

- 只支持 rect。
- 按 `strength` 做 alpha blend。
- 按 `feather` 做边缘过渡。
- 区域重叠时，后写 region 覆盖先写 region，并在 plan 中记录 order。
- 输出 region masks，便于审计和局部指标。

### Fusion notes

传给 LLM/refinement 的 notes 示例：

```text
[FUSION GOAL] Create one coherent shader that preserves the base composition while borrowing selected local qualities from source renders.
[BASE] Use run_card_2 as the global structure and shader starting point.
[REGION SOURCE region_water] In rect x=0.05 y=0.58 w=0.90 h=0.34, borrow stronger water reflection from run_card_5. Blend softly, strength 0.65.
[REGION SOURCE region_sky] In rect x=0.00 y=0.00 w=1.00 h=0.46, borrow better cloud layering from run_card_1. Keep transitions coherent.
[IMPORTANT] Do not create a pasted collage. Produce a single unified shader.
```

### Pipeline 注入点

`run_png_shader_pipeline` 可扩展：

```python
fusion_context: dict | None = None
target_image_path: str | None = None
```

首版也可以不改函数签名，而是在 fusion run 中：

- `image_path` 仍使用原始 reference input。
- `human_feedback_notes` 增加 fusion notes。
- `human_constraints` 增加 regions。
- `directed_acceptance` 增加 composite target artifact id。

如果当前 metric/VLM judge 只能比较 reference image，则需要补充：

```python
target_reference_paths: list[Path] | None = None
```

或在 fusion run 的 judge 中显式读取 composite target。

### Directed acceptance 扩展

Fusion refine 接受规则：

1. compile/render 失败永不接受。
2. 明显破坏 base 全局结构时拒绝。
3. composite target 区域相似度提升时可接受。
4. protected/base regions 明显退化时拒绝。
5. 整体分数小幅下降但 fusion VLM judge 认为 candidate 更符合 fusion plan 时可接受。

新增 judge：

```python
def judge_fusion_pairwise(
    *,
    base_render_path: Path,
    composite_target_path: Path,
    current_render_path: Path,
    candidate_render_path: Path,
    fusion_plan: FusionPlanRecord,
    work_dir: Path,
) -> Literal["A", "B", "tie"] | None:
    ...
```

## 前端设计

### FusionBuilder

新增组件：

- `FusionBuilderPanel.tsx`
- `FusionRegionList.tsx`
- `FusionSourcePicker.tsx`
- `FusionPreviewPanel.tsx`
- `FusionPlanCanvasNode.tsx`

### 状态

```ts
const [activeFusionId, setActiveFusionId] = useState<string | null>(null);
const [fusionDraft, setFusionDraft] = useState<FusionDraft | null>(null);
const [fusionStatus, setFusionStatus] = useState<FusionStatus | null>(null);
```

### UI 放置

- Branch Canvas：由 V4.5 扩展显示 fusion plan 节点和 source/output 边。
- Main Preview：画 region、预览 composite target、对比 base/source/fusion。
- Inspector：编辑 FusionPlan、启动 composite/run。

## 与 V3.5 的衔接

V3.5 DrawSession 提供候选来源：

- `Use as base` 会创建 fusion draft 的 `base_run_id`。
- `Use region` 会把该 card 加入 source picker。
- DrawSession event 记录用户选择，后续进入 V4 preference profile。

V4.5 不要求来源必须是 DrawSession card；任意 completed run 只要有 render，就可作为视觉 source。只有 base run 必须有 selected GLSL，因为 fusion refine 需要从 base shader 出发。首版 UI 先从同一 DrawSession 选择，减少搜索复杂度。

## 与 V4 的关系

V4.5 复用并扩展：

- V4.1 `HumanConstraintSpec`
- V4.2 `RegionConstraint` 和 region metrics
- V4.3 preference events
- V4.4 preference-assisted ranking

V4 版本拆分新增：

| 版本 | 能力 |
|---|---|
| V4.5 | Local fusion：多结果局部叠加、composite target、fusion refine |

V4.5 之后可考虑：

- V4.6/V5 shader-level merge：函数/参数级 shader 融合。
- Fusion auto-suggest：系统自动识别每张卡的优势区域。

## 测试计划

### 后端

`backend/tests/unit/test_fusion_plans.py`

- 创建 fusion plan，base/source/run ids 校验。
- region geometry 越界返回 422。
- source run 缺 render/GLSL 返回明确错误。
- fusion artifacts 路径安全。

`backend/tests/unit/test_image_composite.py`

- rect region alpha blend 正确。
- feather mask 边缘平滑。
- 重叠 region 顺序可控。
- 输出 composite_target 和 region masks。

`backend/tests/unit/test_glsl_refinement.py`

- fusion notes 注入 LLM prompt。
- fusion directed acceptance 使用 composite target。
- base protected region 退化时拒绝 candidate。

`backend/tests/unit/test_router.py`

- `/fusions` 创建 draft。
- `/composite-target` 生成 artifact。
- `/run` 创建 output child run 并写 lineage。
- `/fusions/{id}` 返回 output_run_id。

### 前端

- `npm run build`
- FusionBuilder 能选择 base 和 source。
- rectangle region 坐标正确写入 FusionPlan。
- composite target 生成后能预览。
- start fusion refine 后切换/高亮 output run。
- Branch Canvas 显示 source -> fusion plan -> fusion result。

## 失败处理

| 场景 | 行为 |
|---|---|
| source render 不存在 | 创建或运行 fusion 前返回 422，提示具体 run_id |
| selected GLSL 不存在 | 该 run 不能作为 base；若仅作为 source 且 render 存在，可以继续 |
| composite 生成失败 | fusion 状态 `failed`，保留错误摘要，不创建 output run |
| fusion refine 失败 | output child run failed，fusion plan 保留 target 和 source |
| 区域重叠 | 按 plan order 处理，并在 UI 提示 |
| VLM judge 不可用 | 回退 region metrics + prompt constraints |

## 指标与日志

新增事件：

- `fusion_plan_created`
- `fusion_region_added`
- `fusion_composite_target_created`
- `fusion_run_started`
- `fusion_run_completed`
- `fusion_source_selected`

统计字段：

- fusion source count
- region count
- composite target score improvement
- output score vs base
- user accepted fusion result

## 实现顺序

1. `fusion_plans.py` + `image_composite.py` + 单测。
2. 新增 `/fusions` create/get/artifacts endpoints。
3. 新增 `/fusions/{id}/composite-target`。
4. 新增 fusion notes 和 fusion directed acceptance。
5. 新增 `/fusions/{id}/run`，创建 fusion child run。
6. run lineage 增加 `fusion_id`、`base_run_id`、`source_run_ids`。
7. 前端 FusionBuilder + source picker。
8. V4.5 扩展 Branch Canvas，显示 FusionPlanNode 和 FusionResultNode。
9. 接入 V3.5 card actions：Use as base / Use region。

## 关键文件索引

- `backend/app/pipeline/fusion_plans.py` — fusion plan 数据与 artifacts
- `backend/app/pipeline/image_composite.py` — composite target 生成
- [backend/app/pipeline/human_constraints.py](../backend/app/pipeline/human_constraints.py) — region constraints 复用
- [backend/app/pipeline/glsl_refinement.py](../backend/app/pipeline/glsl_refinement.py) — fusion directed acceptance
- [backend/app/routers/png_shader.py](../backend/app/routers/png_shader.py) — fusion endpoints
- [frontend/src/components/ImageDiffPanel.tsx](../frontend/src/components/ImageDiffPanel.tsx) — region editor 和 composite preview 挂载点
- `frontend/src/components/FusionBuilderPanel.tsx` — fusion draft 编辑
- `frontend/src/components/FusionPlanCanvasNode.tsx` — canvas fusion 节点
- `frontend/src/hooks/usePngShader.ts` — fusion API
- `frontend/src/lib/branchCanvasModel.ts` — V4.5 扩展 fusion nodes/edges 映射
