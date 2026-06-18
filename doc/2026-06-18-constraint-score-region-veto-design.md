# 设计：constraint_score → protect 区域硬否决 gate

> **日期:** 2026-06-18　**状态:** 待评审　**作者:** 设计协作（brainstorming → spec）
> **背景来源:** `doc/2026-06-18-architecture-deep-audit.md`（审查发现 `constraint_score` 算了、存了、**从无任何接受 gate 读它** —— V4.2「protect 区域硬否决」是个 no-op）。
> **决策摘要:** 在 DSL / GLSL 两条 LLM 精修循环里，以**注入式 veto 回调**实现 protect 区域硬否决：每个候选的渲染与**约束设定时的 seed 渲染**比，protect 区域 SSIM 掉破阈值即**拒绝该候选**（凌驾全局分提升与 directed acceptance）。阈值由区域 `strength` 决定。无 protect 区域时完全 no-op、零回归。

---

## 1. 问题与目标

### 1.1 现状（已用代码核实）
- `region_metrics.compute_region_metrics(reference, render, regions)` 计算每个 protect 区域的 SSIM，并汇总 `constraint_score = mean(protect SSIM)`（[region_metrics.py:121-128](../backend/app/pipeline/region_metrics.py)）。
- 该函数**唯一调用点**是诊断端点 [png_shader.py:3033](../backend/app/routers/png_shader.py)，把结果写进 `region_metrics/<id>.json`。
- `grep constraint_score`（排除定义/测试）在接受逻辑里**零命中**：优化器 / 精修 / selection 没有任何一处读它。
- 约束当前只经 `build_constraint_notes` 转成**文本 notes** 注入 LLM prompt；结构化 `RegionConstraint` 列表**不进精修循环**。
- `RegionConstraint.strength`（[human_constraints.py:50](../backend/app/pipeline/human_constraints.py)）在 protect 模式下**当前未被使用**。

### 1.2 目标
1. 让 protect 区域约束**真正生效**：精修过程中，任何候选若显著破坏 protect 区域，则被**硬否决**——即使全局分提升、即使 directed VLM 判"符合用户目标"。
2. 让否决**可见**（history / summary 记录原因 + 每区域分数），消除审查里"约束无信号"的问题。
3. **零回归**：无 protect 区域时行为与今天逐位一致。

### 1.3 非目标（本次不做）
- 不接入坐标下降优化器 / revision（用户选定仅两条精修循环；优化器只做参数微调、动 protect 区域概率低，留待后续）。
- 不做 polygon/mask 几何（沿用现有 `geometry_type=="rect"`；其他几何记 unsupported 并跳过，不否决）。
- 不改 `modify` 区域行为（只有 `protect` 区域参与否决）。
- 不引入软惩罚（用户选定纯硬否决）。

---

## 2. 关键决策（含用户拍板 + 本设计确定项）

| # | 决策 | 取值 | 依据 |
|---|------|------|------|
| D1 | 否决机制 | **硬否决**，优先级最高（凌驾 Δscore>0、directed、`force_first_iteration`） | 用户选定；落实 V4.2"硬否决凌驾全局分提升" |
| D2 | 接入范围 | **DSL + GLSL 两条精修循环**（`run_dsl_refinement_loop` / `run_glsl_refinement_loop`） | 用户选定 |
| D3 | 保护基准 | **约束设定时的 seed 渲染**（循环已有的初始渲染，全程固定，非滚动 current） | 用户选定；零额外渲染成本 |
| D4 | 否决判据 | **每区域** SSIM(候选, 基准) < 该区域 `strength` 派生阈值 → 否决 | 见 §4；用 per-region 比 mean 更贴"逐区域保护" |
| D5 | 阈值公式 | `min_ssim = FLOOR + (CEIL-FLOOR)·strength`，`FLOOR=0.85`、`CEIL=0.95`（可配） | 见 §4.2；复用现有 `strength` 旋钮，默认 strength 0.5 → 0.90 |
| D6 | 否决后行为 | 当作**未改进回滚**：force `accept=False` + 注入 `[PROTECT VIOLATION]` 反馈 + 计入 `no_improvement_count`，**不立即终止** | 见 §5；与现有 rollback 机制一致，给 LLM 纠错机会，patience 兜底防 churn |
| D7 | 可观测 | history 条目记 `rejected_reason="protect_region_veto"` + per-region `{id,ssim,threshold,violated}` + mean `constraint_score`；run 末写 `region_constraints_result` | 见 §6 |
| D8 | 兼容性 | 无 protect 区域 / 无基准渲染 → `region_veto_fn=None` → 两循环逐位等同今天 | 见 §7 |

---

## 3. 架构：注入式 veto 回调（复用现有 judge 注入模式）

两条循环已经接受注入裁判回调（`pairwise_judge` / `directed_pairwise_judge` / `rubric_judge`）。沿用同一模式新增一个**可选注入**，不破坏既有契约：

```python
# 注入给循环的回调（None 时循环行为不变）
region_veto_fn: Callable[[Path], RegionVetoResult] | None = None
```

```python
@dataclass
class RegionVetoResult:
    vetoed: bool                       # 是否应否决该候选
    constraint_score: float            # protect 区域 mean SSIM（vs 基准），用于记录
    regions: list[dict]                # [{id, label, ssim, threshold, violated}]
    reason: str | None                 # 人类可读否决原因（拼进 LLM 反馈）
    evaluated: bool                    # 是否成功评估（基准/候选缺失时 False → 不否决）
```

- **为什么注入式而非内联进 `evaluate_fn`：** 硬否决用"分数"表达不干净（需哨兵值），且会破坏 `evaluate_fn -> (metrics, quality, score, render)` 契约（多处依赖）。注入式让 region 逻辑留在循环外、可独立纯函数单测，循环只多一个"否决钩子"。
- **谁构造它：** `graph.py:_run_post_pipeline`（regions 与循环都在此作用域）。闭包捕获 `(baseline_render_path, protect_regions, floor, ceil)`，每轮被循环以候选渲染路径调用。

---

## 4. 否决算法

### 4.1 纯函数（加在 `region_metrics.py`）

```python
def protect_region_threshold(strength: float, *, floor=0.85, ceil=0.95) -> float:
    s = min(1.0, max(0.0, strength))
    return floor + (ceil - floor) * s

def evaluate_protect_veto(
    baseline_render: Path, candidate_render: Path,
    protect_regions: list[RegionConstraint],
    *, floor=0.85, ceil=0.95,
) -> RegionVetoResult:
    # 复用 compute_region_metrics（reference=基准, render=候选, regions=protect）
    # 对每个有有效 ssim 的 rect protect 区域：threshold = protect_region_threshold(region.strength)
    #   violated = ssim < threshold
    # vetoed = any(violated)；constraint_score = mean(有效 ssim)
    # 无有效区域（几何 unsupported / empty / 基准缺失）→ evaluated=False, vetoed=False（best-effort 放行）
```

- 基准 = seed 渲染，故 seed-vs-自身 SSIM=1.0；候选若**完全不动** protect 区域，SSIM≈1.0（渲染确定性已由 P1 `frozenTime` 修复保证）。
- 只评估 `mode=="protect"` 且 `geometry_type=="rect"` 且 SSIM 有效的区域。

### 4.2 阈值标定（D5 依据）
SSIM∈[0,1]。把**用户既有的 `strength` 旋钮**映射到阈值：

| strength | 阈值(min_ssim) | 含义 |
|---|---|---|
| 1.0（最强保护） | 0.95 | 只容忍 ~0.05 抖动（抗锯齿/边缘微变），近乎冻结 |
| 0.5（默认） | 0.90 | 容忍 ~0.10 掉幅 |
| 0.0（最弱保护） | 0.85 | 容忍 ~0.15 掉幅 |

- **为何 CEIL=0.95 而非 1.0：** 即使逻辑上不动该区域，候选改动别处也可能让区域边缘抗锯齿轻微变化；留 0.05 余量防 false-veto。
- **为何用 `strength` 而非固定 0.10：** `strength` 是前端 RegionMaskEditor 已暴露的字段、目前 protect 下闲置；接上即给用户一个语义清晰的严格度旋钮，且默认值正好落在原提案的 0.90/0.10。
- `FLOOR`/`CEIL` 经 env 可配（`PROTECT_VETO_SSIM_FLOOR` / `_CEIL`）。

---

## 5. 循环集成

两条循环结构相同（`delta = new_score - best_score`；`accept = delta>0` 经 VLM/directed 调整；接受则更新 best，否则回滚反馈；`no_improvement_count` 计数 + `no_improvement_patience` 终止）。集成点统一：

**在算出候选 `actual_render` 之后、accept 决策之前**插入 veto：

```python
veto = region_veto_fn(actual_render) if (region_veto_fn and actual_render) else None
if veto and veto.vetoed:
    accept = False                      # 凌驾一切：跳过 delta/VLM/directed 接受路径
    entry["rejected_reason"] = "protect_region_veto"
    entry["region_veto"] = veto.regions
    entry["constraint_score"] = veto.constraint_score
    extra_feedback = [
        f"[PROTECT VIOLATION] 你的修改破坏了受保护区域 "
        f"({veto.reason})。必须保持这些区域不变，只在其他地方修改。"
    ]
    no_improvement_count += 1           # 计入 patience，反复违规最终停
    _record(entry); continue            # 回滚并进入下一轮
# …原有 accept 逻辑不变…
```

- **GLSL 循环**（[glsl_refinement.py](../backend/app/pipeline/glsl_refinement.py)）：插在 render-failed 分支之后、`delta` 计算之前。基准 = `initial_render_path`。
- **DSL 循环**（[refinement.py](../backend/app/pipeline/refinement.py)）：同结构插入。基准 = `baseline_render_path`（[refinement.py:172](../backend/app/pipeline/refinement.py) 已渲染）。
- **优先级（D1）**：veto 命中即 `continue`，**根本不进入** directed-acceptance / `force_first_iteration` 的接受路径 → 保证硬否决凌驾一切。
- **D6**：否决=回滚+反馈+计数，不终止；连续违规由现有 `no_improvement_patience` 收敛。

---

## 6. 数据流与 plumbing

```
router(branch-refine / explore-variants)
  parse_constraint_spec → HumanConstraintSpec.regions
  ↓ （新增）筛 mode=="protect" 的 regions，随 run 入 pipeline
run_png_shader_pipeline(..., protect_regions=[...])
  ↓
_run_post_pipeline:
  if protect_regions and baseline_render 可用:
     region_veto_fn = λ cand: evaluate_protect_veto(baseline_render, cand, protect_regions, floor, ceil)
  run_dsl_refinement_loop(...,  region_veto_fn=region_veto_fn)
  run_glsl_refinement_loop(..., region_veto_fn=region_veto_fn)
```

- 约束**当前只转 notes**，本设计新增"把 protect regions 作为结构化数据透传进 pipeline → `_run_post_pipeline` → 循环"这条线（主要 plumbing 工作量）。
- 透传形式（**已定**）：在 pipeline 入口新增**显式参数** `protect_regions: list[RegionConstraint] | None`，而非复用 `spec_to_dict` 文本通道——保持类型清晰、避免与 notes 通道耦合。router 从 `constraint_spec.regions` 筛 `mode=="protect"` 后传入。

---

## 7. 可观测 / 持久化（D7）
- 被否决迭代的 history 条目：`rejected_reason="protect_region_veto"` + `region_veto:[{id,label,ssim,threshold,violated}]` + `constraint_score`。
- run 末在 `refinement_summary.json`（或新增 `region_constraints_result.json`）写：protect 区域数、各区域最终 SSIM、否决次数。
- 前端可据此显示"第 N 轮因保护区域 <label> 退化被拒"。这同时消除审查里"约束算了存了没人读、无信号"。

---

## 8. 兼容性与边界（D8 + 边界）
- **无 protect 区域 / 无基准渲染** → `region_veto_fn=None` → 两循环逐位等同今天；现有 1124 pytest 不受影响。
- **基准渲染失败**（None）→ 无法 veto → `evaluated=False` 放行 + warning（best-effort，不阻断）。
- **候选渲染失败**（actual_render=None）→ 已被现有 render-failed 分支拒绝（在 veto 之前），不进 veto。
- **区域几何非 rect / 空 / SSIM 无效** → 该区域不参与否决（best-effort），仅对有有效 SSIM 的 protect 区域判定。
- **`force_first_iteration`** → 仍被 veto 凌驾：受保护区域被破坏的候选**不能**被强制接受。

---

## 9. 配置旋钮
| env | 默认 | 含义 |
|---|---|---|
| `PROTECT_VETO_SSIM_FLOOR` | 0.85 | strength=0 时的阈值 |
| `PROTECT_VETO_SSIM_CEIL` | 0.95 | strength=1 时的阈值 |
| （region.strength） | 0.5 | 前端 RegionMaskEditor 既有字段，决定该区域严格度 |

---

## 10. 测试计划（TDD）
**纯函数（`test_region_metrics.py`）**
1. `protect_region_threshold`：strength 0/0.5/1.0 → 0.85/0.90/0.95（边界）。
2. `evaluate_protect_veto`：基准与候选 protect 区域一致 → `vetoed=False`、constraint_score≈1.0。
3. 候选把 protect 区域改花（SSIM 低于阈值）→ `vetoed=True`，`regions` 标 violated 与 reason。
4. 多 protect 区域：一个被破坏即 `vetoed=True`（per-region min 语义）。
5. 几何 unsupported / 基准缺失 → `evaluated=False, vetoed=False`（放行）。

**循环集成（`test_glsl_refinement.py` / `test_refinement.py`）**
6. 注入一个"全局分提升但 protect 区域退化"的候选 → **被拒**，best 不变，history 记 `protect_region_veto`。
7. directed_acceptance 想接受 + protect 退化 → **仍被拒**（优先级）。
8. `force_first_iteration` + protect 退化 → **仍被拒**。
9. `region_veto_fn=None`（无 protect 区域）→ 两循环行为与现有测试逐位一致（回归保护，复用现有用例）。
10. 连续 protect 违规 → 经 `no_improvement_patience` 正常收敛终止（不死循环）。

---

## 11. 受影响文件清单
| 文件 | 改动 |
|---|---|
| `backend/app/pipeline/region_metrics.py` | +`protect_region_threshold`、+`evaluate_protect_veto`、+`RegionVetoResult` |
| `backend/app/pipeline/glsl_refinement.py` | +`region_veto_fn` 参数 + veto 集成点 |
| `backend/app/pipeline/refinement.py` | +`region_veto_fn` 参数 + veto 集成点 |
| `backend/app/pipeline/graph.py` | `_run_post_pipeline` 构造 `region_veto_fn` 并传入两循环；run 末写约束结果 |
| `backend/app/routers/png_shader.py` | branch-refine / explore-variants 把 protect regions 透传进 pipeline |
| `backend/app/config.py` | +`PROTECT_VETO_SSIM_FLOOR/_CEIL` |
| 对应 `tests/unit/test_*.py` | §10 全部用例 |

---

## 12. 风险与缓解
| 风险 | 缓解 |
|---|---|
| 改动承重接受逻辑、牵连现有测试 | 集成点是"命中即 continue"的纯增量；`region_veto_fn=None` 路径逐位不变；先跑全量回归 |
| 阈值过严 → 正常精修被频繁误否决 | 默认 CEIL=0.95 留抗锯齿余量；strength 默认 0.5→0.90；env 可调；history 记录便于观测调参 |
| 阈值过松 → 保护形同虚设 | per-region min 语义 + strength 旋钮；高 strength→0.95 接近冻结 |
| 每轮多算区域 SSIM 的成本 | 仅 protect 区域存在时发生；区域裁剪 SSIM 很便宜；基准复用已有渲染、零额外渲染 |
| protect 与"全局更好"长期冲突 → 精修空转 | 计入 `no_improvement_patience`，正常收敛终止 |
