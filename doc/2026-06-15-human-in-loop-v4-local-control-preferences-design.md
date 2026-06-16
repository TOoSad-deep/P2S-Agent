# V4: Human-in-the-loop 局部控制与偏好学习技术方案

> **状态:** 技术方案，待评审。
>
> **依赖:** V1 branch refine、V2 branch workspace、V3 variant exploration。若已完成 [V2.1 Branch Canvas Workspace](2026-06-16-human-in-loop-v2-1-branch-canvas-workspace-design.md)，V4 的 controls/preferences 应优先挂入 Branch Canvas inspector，region/preference 以附属节点进入画布。V4 复用用户 feedback、branch lineage、variant ratings/winner events，并新增结构化约束和偏好记忆。
>
> **目标读者:** 后端/前端实现者，以及后续把 V4 拆成 implementation plan 的 agentic worker。

## Goal

V4 的目标是把 human-in-loop 从“自然语言分支优化”升级成“可精细控制的创作工作台”：

1. 用户可以设置结构化约束：保持布局、锁定色彩、保护背景、只调局部。
2. 用户可以在图像上选择区域，并对区域写定向反馈。
3. 系统在 prompt、评分、VLM 裁决中使用这些局部/属性约束。
4. 系统记录用户选择、评分、winner，形成可审计的偏好 profile。
5. 后续分支和 variants 自动带入用户偏好，但用户可随时关闭或编辑。

## Non-goals

- 不训练私有模型，不做不可解释的在线学习。
- 不做像 Photoshop 一样的像素级直接编辑；仍然是 shader 生成/优化。
- 不保证任意 mask 都能精确局部修改；v4 用约束、局部评分和 VLM 裁决逐步提高可控性。
- 不引入多用户权限系统；默认是本地单用户偏好文件。
- 不把用户偏好写死进 prompt；每次 run 都可显式禁用或覆盖。

## 版本拆分

V4 建议拆成 4 个小版本：

| 版本 | 能力 | 说明 |
|---|---|---|
| V4.1 | 结构化全局约束 | locks、属性目标、编辑强度，先不做 mask |
| V4.2 | 区域/Mask 约束 | rectangle/lasso/brush mask，区域 prompt 和局部评分 |
| V4.3 | 偏好事件与 profile | 从 winner/rating/feedback 生成可审计偏好记忆 |
| V4.4 | 偏好辅助排序与生成 | prompt 注入、variant ranking、默认策略推荐 |

## 用户体验

### 精细控制面板

在 V2/V3 的 HumanLoopPanel 中新增 `Controls` 区域。若启用 V2.1，则该区域放入 `BranchCanvasInspector`，根据当前选中的 run/checkpoint/variant node 动态展示：

```text
Directed Refinement
Feedback: [让水面反射更明显，整体不要变暗]

Global locks
[x] Preserve layout      [ ] Preserve palette
[x] Preserve background  [x] Small edits only

Targets
Brightness:  keep / lighter / darker
Contrast:    keep / higher / lower
Detail:      keep / more / less
Reflection:  keep / stronger / softer

Edit strength: [----|-----] 0.35
Use my preferences: [on]
```

### 区域控制

用户在 preview/reference 上画区域：

```text
Region A: water area
Instruction: make reflection clearer
Mode: modify
Strength: 0.45

Region B: sky/cloud
Instruction: preserve current cloud layering
Mode: protect
Strength: 0.80
```

区域可以是：

- rectangle，V4.2 首选，简单可靠。
- polygon/lasso，后续加。
- brush mask，最后加。

### 偏好记忆

用户选择 variant winner 或点赞/点踩时，系统记录偏好事件。

偏好面板展示：

- 喜欢的方向：更亮的反射、保持构图、避免整体变暗。
- 不喜欢的方向：过度蓝紫、背景被改动、纹理噪声过强。
- 默认锁：保持布局、保护背景。
- 当前 run 是否启用偏好。

用户可以编辑、禁用、清空偏好。

### Branch Canvas 集成

V4 不把精细框选直接放在 Branch Canvas 上。画布负责表达约束关系，精细编辑仍在主 preview 完成：

```text
Checkpoint / VariantRun node
   ├─ RegionConstraint: water modify "make reflection clearer"
   ├─ RegionConstraint: sky protect "preserve cloud layering"
   └─ Preference: enabled, prefers conservative + lighting_color
```

交互边界：

- 用户在 Branch Canvas 选择要优化的 checkpoint/run。
- 右侧 inspector 显示 FineControlPanel、PreferencePanel。
- 用户点击 `Add region` 后，在 Main Preview 的 `RegionMaskEditor` 上画 rectangle。
- 保存后，画布在选中节点旁新增 `RegionConstraintNode`，用 `constraint_applies` edge 连接。
- 约束数据仍随 `branchRefine` / `exploreVariants` 请求提交，不独立触发 pipeline。

## 数据模型

### HumanConstraintSpec

新增结构化约束，建议放在 `backend/app/pipeline/human_constraints.py`。

```python
from dataclasses import dataclass, field
from typing import Literal

EditMode = Literal["modify", "protect"]
TargetDirection = Literal["keep", "increase", "decrease"]

@dataclass
class RegionConstraint:
    id: str
    label: str
    mode: EditMode
    instruction: str
    geometry_type: str  # "rect" | "polygon" | "mask"
    geometry: dict      # normalized coordinates, or mask artifact id
    strength: float = 0.5

@dataclass
class HumanConstraintSpec:
    locks: dict = field(default_factory=dict)
    targets: dict[str, TargetDirection] = field(default_factory=dict)
    edit_strength: float = 0.5
    regions: list[RegionConstraint] = field(default_factory=list)
    use_preferences: bool = True
```

### JSON shape

```json
{
  "locks": {
    "preserve_layout": true,
    "preserve_palette": false,
    "preserve_background": true,
    "small_edits_only": true
  },
  "targets": {
    "brightness": "keep",
    "contrast": "increase",
    "detail": "increase",
    "reflection": "increase"
  },
  "edit_strength": 0.35,
  "regions": [
    {
      "id": "region_water",
      "label": "water",
      "mode": "modify",
      "instruction": "make reflection clearer",
      "geometry_type": "rect",
      "geometry": {"x": 0.05, "y": 0.58, "w": 0.90, "h": 0.34},
      "strength": 0.45
    },
    {
      "id": "region_sky",
      "label": "sky",
      "mode": "protect",
      "instruction": "preserve cloud layering",
      "geometry_type": "rect",
      "geometry": {"x": 0.00, "y": 0.00, "w": 1.00, "h": 0.46},
      "strength": 0.80
    }
  ],
  "use_preferences": true
}
```

### PreferenceEvent

```python
@dataclass
class PreferenceEvent:
    event_id: str
    event_type: str  # "winner_selected" | "variant_rated" | "branch_accepted" | "manual_note"
    timestamp: float
    run_id: str | None
    group_id: str | None
    feedback: str | None
    winner_run_id: str | None
    loser_run_ids: list[str]
    rating: int | None
    reason: str | None
    tags: list[str]
    context: dict
```

### PreferenceProfile

可审计、可编辑，不是黑盒模型。

```json
{
  "schema_version": 1,
  "updated_at": 1780000000.0,
  "enabled": true,
  "default_locks": {
    "preserve_layout": true,
    "preserve_background": true
  },
  "positive_preferences": [
    "clearer reflections without darkening the whole image",
    "preserve cloud layering",
    "small targeted changes before full rewrites"
  ],
  "negative_preferences": [
    "avoid strong purple/blue tint",
    "avoid moving major composition elements",
    "avoid noisy texture artifacts"
  ],
  "preferred_variant_labels": ["conservative", "lighting_color"],
  "score_drop_tolerance_hint": 0.02,
  "summary_source_event_count": 18
}
```

## API 设计

### 1. V1/V3 request 扩展

`POST /branch-refine` 和 `POST /explore-variants` 增加：

```json
{
  "constraints": {
    "locks": {...},
    "targets": {...},
    "edit_strength": 0.35,
    "regions": [...],
    "use_preferences": true
  }
}
```

后端向下兼容：不传 `constraints` 时行为不变。

### 2. 预览/保存区域 mask

V4.2 起新增：

```http
POST /png-shader/runs/{run_id}/region-mask
Content-Type: application/json
```

请求：

```json
{
  "region_id": "region_water",
  "geometry_type": "rect",
  "geometry": {"x": 0.05, "y": 0.58, "w": 0.90, "h": 0.34}
}
```

返回：

```json
{
  "region_id": "region_water",
  "mask_artifact_id": "mask:region_water",
  "mask_url": "/png-shader/runs/run_a/artifacts/mask:region_water"
}
```

V4.2 可以先不落真实 PNG mask，只保存 normalized geometry；需要局部指标时再生成 mask PNG。

### 3. 偏好 profile

```http
GET /png-shader/preferences/profile
PATCH /png-shader/preferences/profile
POST /png-shader/preferences/events
POST /png-shader/preferences/rebuild
POST /png-shader/preferences/clear
```

`PATCH /profile` 只允许用户可编辑字段：

- `enabled`
- `default_locks`
- `positive_preferences`
- `negative_preferences`
- `score_drop_tolerance_hint`

`POST /rebuild` 根据 events 重建 profile。实现可以先 deterministic，再可选 LLM summarizer。

## 后端设计

### `human_constraints.py`

职责：

- `parse_constraint_spec(payload) -> HumanConstraintSpec`
- `validate_constraint_spec(spec, image_width, image_height) -> list[str]`
- `build_constraint_notes(spec) -> list[str]`
- `constraint_to_artifacts(run_dir, spec)`

Notes 示例：

```text
[GLOBAL LOCK] Preserve layout and major object positions.
[TARGET] Increase reflection strength; keep brightness stable.
[EDIT STRENGTH] 0.35: make targeted changes, avoid rewriting the shader.
[REGION MODIFY region_water] Make reflection clearer in normalized rect x=0.05 y=0.58 w=0.90 h=0.34.
[REGION PROTECT region_sky] Preserve cloud layering in normalized rect x=0.00 y=0.00 w=1.00 h=0.46.
```

### `region_metrics.py`

V4.2 新增局部指标：

```python
def compute_region_metrics(
    reference_path: Path,
    render_path: Path,
    regions: list[RegionConstraint],
) -> dict:
    ...
```

输出：

```json
{
  "regions": {
    "region_water": {
      "mse": 0.012,
      "ssim": 0.82,
      "mean_delta": [0.02, 0.01, -0.01],
      "edge_delta": 0.04
    },
    "region_sky": {
      "mse": 0.004,
      "ssim": 0.93
    }
  },
  "constraint_score": 0.76
}
```

V4.2 不直接替换 `final_score`，建议先放入：

- `objective_metrics.region_metrics`
- `quality_router.constraint_score`
- `quality_router.constraint_notes`

V4.2 可以用 protected region 的明显退化作为 hard veto，但不把 `constraint_score` 混入总分。V4.4 再考虑将 `constraint_score` 作为排序/ranking 辅助分，避免早期局部指标不稳定时改变主优化目标。

### Directed acceptance 扩展

V1 directed acceptance 看“整体是否更符合用户反馈”。V4 加入约束：

```python
directed_acceptance = {
    "feedback": feedback,
    "constraints": constraint_spec.to_dict(),
    "score_drop_tolerance": profile.score_drop_tolerance_hint,
    "region_score_required": True,
    "protected_region_max_drop": 0.02,
}
```

与 V1 一致，`directed_acceptance` 只保存可 JSON 序列化的数据。`HumanConstraintSpec` dataclass 只在 runtime 内部使用；写入 state/input_spec/artifacts 时使用 plain dict。

接受规则：

1. 渲染/校验失败永不接受。
2. 整体分数提升，且 protected region 没明显变差，可以接受。
3. 整体小幅下降但 modify region 明显符合目标，可交给 VLM region judge。
4. protected region 明显变差，默认 veto，除非用户关闭对应 lock。

### VLM region judge

新增：

```python
def judge_directed_region_pairwise(
    reference_path: Path,
    current_render_path: Path,
    candidate_render_path: Path,
    *,
    feedback: str,
    constraints: HumanConstraintSpec,
    work_dir: Path,
) -> Literal["A", "B", "tie"] | None:
    ...
```

Prompt 输入：

- reference image
- current render
- candidate render
- region descriptions
- user feedback
- global locks

如果能生成 region crops，优先把 crop 图一起传给 VLM；否则以文字坐标描述降级。

### `preferences.py`

新增：

- `append_preference_event(event)`
- `load_preference_events(limit=None)`
- `load_profile()`
- `patch_profile(patch)`
- `rebuild_profile(events)`
- `build_preference_notes(profile) -> list[str]`

持久化：

```text
backend/test_results/preferences/events.jsonl
backend/test_results/preferences/profile.json
```

`rebuild_profile` V4.3 先用确定性规则：

- winner reason/tags 进入 positive。
- dislike reason/tags 进入 negative。
- 高频 locks 进入 default_locks。
- winner variant label 统计进入 preferred_variant_labels。

可选 V4.4 再用 LLM summarizer 生成更自然的 preference summary，但必须保留原始 events。

### Pipeline 注入点

`run_png_shader_pipeline` 增加：

```python
human_constraints: dict | None = None
preference_profile: dict | None = None
```

构建 notes：

```python
notes = []
notes += build_human_feedback_notes(...)
notes += build_constraint_notes(constraint_spec)
if constraint_spec.use_preferences:
    notes += build_preference_notes(profile)
```

落盘：

```text
constraints.json
preference_profile_snapshot.json
region_metrics/*.json
```

## 前端设计

### 组件

新增：

- `FineControlPanel.tsx`
- `RegionMaskEditor.tsx`
- `PreferencePanel.tsx`
- `PreferenceChips.tsx`
- `RegionConstraintCanvasNode.tsx`，V2.1 canvas 模式新增
- `PreferenceCanvasNode.tsx`，V2.1 canvas 模式新增

### RegionMaskEditor

V4.2 首版只做 rectangle：

- 在 `ImageDiffPanel` 的 reference/preview 上覆盖 canvas。
- 鼠标拖拽生成 normalized rect。
- 每个 rect 有 label、mode、instruction、strength。
- 支持删除/重命名。

后续再加 polygon 和 brush。

### FineControlPanel

输出 `HumanConstraintSpec`：

```ts
export interface HumanConstraintSpec {
  locks: Record<string, boolean>;
  targets: Record<string, "keep" | "increase" | "decrease">;
  edit_strength: number;
  regions: RegionConstraint[];
  use_preferences: boolean;
}
```

该 spec 传给：

- `branchRefine`
- `exploreVariants`

### PreferencePanel

功能：

- 显示当前 profile。
- 开关 `enabled`。
- 编辑 positive/negative preference。
- 清空 profile/events。
- 从最近 winner/rating 重建。

### Canvas 模式职责

启用 V2.1 后：

- `FineControlPanel` 嵌入 `BranchCanvasInspector`。
- `PreferencePanel` 作为 inspector tab 或全局 drawer。
- `RegionMaskEditor` 仍挂在 `ImageDiffPanel` / Main Preview。
- `RegionConstraintCanvasNode` 只展示 region label、mode、strength、instruction 摘要，不负责画框。
- `PreferenceCanvasNode` 只展示当前 run 使用的 profile snapshot 或推荐，不允许在节点内复杂编辑。

## 测试计划

### 后端

`backend/tests/unit/test_human_constraints.py`

- locks/targets/regions validation。
- normalized geometry 越界报错。
- constraint notes 正确生成。
- edit_strength clamp 到 `[0, 1]`。

`backend/tests/unit/test_region_metrics.py`

- rect mask metrics 只计算区域内像素。
- protect region 变差会降低 constraint_score。
- 无 region 时返回空 metrics。

`backend/tests/unit/test_preferences.py`

- append events JSONL。
- deterministic rebuild profile。
- patch profile 只允许 editable fields。
- build_preference_notes 尊重 `enabled=false`。

`backend/tests/unit/test_glsl_refinement.py`

- constraints notes 进入 LLM extra feedback。
- protected region veto 会阻止小幅整体提升但局部破坏的 candidate。
- VLM region judge 返回 B 时允许小幅降分但目标区域改善。

### 前端

- `npm run build`
- FineControlPanel 输出 spec 正确。
- rectangle editor normalized 坐标正确。
- PreferencePanel patch/clear/rebuild 调用正确。
- branchRefine/exploreVariants body 包含 constraints。
- V2.1 canvas 模式下，新增/删除 region 后，画布 constraint node 与选中 run/checkpoint 同步。
- 切换 canvas node 时 inspector 中 constraints draft 不串到其他节点。

## 失败处理和安全

| 场景 | 行为 |
|---|---|
| region geometry 越界 | 422，提示具体 region id |
| mask artifact 不存在 | 忽略该 region 或 422，v4.2 推荐 422 |
| preference profile 损坏 | 备份损坏文件，恢复默认空 profile |
| VLM region judge 不可用 | 回退到 metric + prompt constraints |
| 用户关闭 preferences | 不注入 preference notes，不影响 constraints |

隐私与可控性：

- 偏好事件和 profile 只存本地 `backend/test_results/preferences/`。
- 提供 clear endpoint。
- 每次 run 保存 profile snapshot，便于审计“这次为什么这样优化”。

## 实现顺序

1. V4.1: `human_constraints.py` + request schema + prompt notes 注入。
2. V4.1: 前端 FineControlPanel，不含 mask；若启用 V2.1，先嵌入 BranchCanvasInspector。
3. V4.2: rectangle RegionMaskEditor + region constraints。
4. V4.2: V2.1 canvas 模式新增 RegionConstraintCanvasNode，表达约束挂载关系。
5. V4.2: `region_metrics.py` + protected/modify region guardrails。
6. V4.3: `preferences.py` + events/profile endpoints。
7. V4.3: 从 V3 winner/rating group events 镜像或回填 preference events。
8. V4.3: V2.1 canvas 模式新增 PreferenceCanvasNode 或 preference annotation。
9. V4.4: preference notes 注入 branch/variant prompts。
10. V4.4: preference-assisted variant ranking。

## 关键文件索引

- [backend/app/pipeline/glsl_refinement.py](../backend/app/pipeline/glsl_refinement.py) — directed acceptance 与 constraints 注入
- [backend/app/pipeline/refinement.py](../backend/app/pipeline/refinement.py) — DSL loop constraints 注入
- [backend/app/routers/png_shader.py](../backend/app/routers/png_shader.py) — branch/variant/preference endpoints
- [backend/app/metrics/compute.py](../backend/app/metrics/compute.py) — 可复用图像指标，V4 新增局部指标
- [frontend/src/components/ImageDiffPanel.tsx](../frontend/src/components/ImageDiffPanel.tsx) — region editor 挂载点
- [frontend/src/components/PngShaderView.tsx](../frontend/src/components/PngShaderView.tsx) — FineControlPanel / PreferencePanel / canvas 挂载点
- [frontend/src/hooks/usePngShader.ts](../frontend/src/hooks/usePngShader.ts) — constraints/preference API
- `frontend/src/components/BranchCanvasInspector.tsx` — V2.1 canvas 模式下 controls/preferences 的主要挂载点
- `frontend/src/lib/branchCanvasModel.ts` — V2.1 canvas 模式下把 region/preference 映射为附属节点
