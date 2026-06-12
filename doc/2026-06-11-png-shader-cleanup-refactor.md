# P2S-Agent 迁移收尾与清理实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **适配说明（2026-06-12）：** 本文档适配自 VFX-Agent 的《png-shader 清理重构计划》。原计划的核心目标（删除死代码、提取 `_accept_improvement`、拆分 graph.py）在 P2S-Agent 重构建仓时**已全部落地**——见下方"原计划任务状态核查"。但重构迁移留下了一批**新问题**：3 处生产代码 import 指向不存在的 `app.png_shader.*`、测试套件因旧路径无法收集、`_run_post_pipeline` 丢弃 state 键导致 API 响应缺字段。本适配版改为修复这些迁移遗留，恢复测试安全网——这是《[2026-06-11-png-shader-accuracy-optimization.md](2026-06-11-png-shader-accuracy-optimization.md)》（准确率计划）Phase 2/3 的前置条件。

**Goal:** 修复 P2S-Agent 重构迁移的全部遗留问题：生产代码的 3 处坏 import、测试套件的旧模块路径（当前 `pytest tests/unit/` 无法完成收集）、`_run_post_pipeline` 的 state 键丢失 bug，并补齐文档，使全量单测恢复绿色、为准确率计划扫清地基。

**Architecture:** 纯修复（恢复重构前的预期行为）：① 路由器懒加载 import 改为 P2S 新模块路径；② `_run_post_pipeline` 返回 `{**state, ...}` 保留上游键；③ 测试 import 与 monkeypatch 目标按映射表批量更新；④ 为预置的 `test_decompose.py`（Phase 2 红灯测试）加收集守卫。

**Tech Stack:** Python 3.9+（与 README 一致），无新依赖。

**执行时机：** 在准确率计划 Phase 0/Phase 2 之前执行（准确率计划 Phase 1 已在重构时落地，无需等待）。

---

## VFX → P2S 模块映射表（本表同时供准确率计划使用）

| VFX-Agent 路径 | P2S-Agent 路径 |
|---|---|
| `app/png_shader/metrics.py` | `app/metrics/compute.py` |
| `app/png_shader/quality_router.py` | `app/metrics/quality_router.py` |
| `app/png_shader/dsl_renderer.py` | `app/dsl/renderer.py` |
| `app/png_shader/compiler.py` | `app/dsl/compiler.py` |
| `app/png_shader/dsl_schema.py` | `app/dsl/schema.py` |
| `app/png_shader/dsl_validator.py` | `app/dsl/validator.py` |
| `app/png_shader/graph.py`（单体流水线） | `app/pipeline/graph.py`（**LangGraph StateGraph** + `_run_post_pipeline`） |
| `app/png_shader/pool.py` / `scoring.py` / `refinement.py` | `app/pipeline/pool.py` / `scoring.py` / `refinement.py` |
| `app/png_shader/optimizer.py` / `glsl_optimizer.py` / `revision.py` / `preprocess.py` / `input_spec.py` / `artifacts.py` | `app/pipeline/` 下同名文件 |
| `app/png_shader/candidates/llm_scene_candidate.py` | `app/candidates/llm_scene.py` |
| `app/png_shader/candidates/{baseline,rule,cv,fallback}_candidate.py` | `app/candidates/{baseline,rule,cv,fallback}.py` |
| `app/png_shader/color_utils.py`（原 normalizer） | `app/utils/color.py` |
| `app/png_shader/cv_features.py` | `app/utils/cv_features.py` |
| `app/agents/base.py` 的 `BaseAgent` + `settings.generate` | `app/llm/client.py` 的 `BaseAgent` + `settings.llm` |
| `settings.generate_supports_image` | `settings.llm_supports_image` |
| `backend/tests/unit/png_shader/test_X.py` | `backend/tests/unit/test_X.py`（扁平） |
| `docs/superpowers/plans/` | `doc/` |
| `CLAUDE.md` | `README.md`（P2S 无 CLAUDE.md） |
| E2E 脚手架 `tests/e2e/test_e2e_batch.py` 等 | **不存在**（由准确率计划 Phase 0 新建） |
| 结果目录 `backend/artifacts/` | `backend/test_results/`（`artifacts.py` 的 `DEFAULT_RESULTS_ROOT`） |

## 原计划任务状态核查（C1–C5，已于 2026-06-12 逐项验证）

| # | 原计划任务 | P2S 状态 | 证据 |
|---|---|---|---|
| C1 | 删除 graph.py 死导入 `get_cv_applicability_report` | ✅ 已落地 | `pool.py:23` 只 import `generate_cv_candidate`；`cv_features` 仅被 `candidates/cv.py` 正常使用 |
| C2 | 删除 scene_graph/normalizer 死代码，提取 color_utils | ✅ 已落地 | `app/utils/color.py` + `tests/unit/test_color_utils.py` 存在；normalizer/scene_graph 不存在 |
| C3 | 提取 `_accept_improvement` 统一接受逻辑 | ✅ 已落地 | `scoring.py:63` 定义；`graph.py:246,288` 两处使用 |
| C4 | 拆分 graph.py 为 pool/scoring/refinement | ✅ 已落地（且更进一步：LangGraph 节点化，graph.py 659 行） | `app/pipeline/` 四模块齐备 |
| C5 | 更新 CLAUDE.md | ➡️ 改造为 Task 5 | P2S 无 CLAUDE.md，文档载体为 README.md |

## 新问题 → 方案总览（迁移遗留）

| # | 问题 | 解决方案 | 任务 |
|---|---|---|---|
| N0 | P2S-Agent 不是 git 仓库，本计划与准确率计划的"每任务一 commit"安全网无处落地 | `git init` + 初始提交 | Task 0 |
| N1 | `routers/png_shader.py:293,312,313` 懒加载 import 指向不存在的 `app.png_shader.*` —— `/png-shader/refine/{run_id}` 等路径运行时直接 500 | 改为 P2S 新模块路径 | Task 1 |
| N2 | `_run_post_pipeline` 返回全新 dict 而非合并 state，`run_png_shader_pipeline` 随后读取的 `preprocess`、`selected_candidate_id` 全部丢失（API 响应 `selected_candidate_id` 恒为 None） | 返回 `{**state, ...}` | Task 2 |
| N3 | `test_metrics.py:14`、`test_quality_router.py:10` import 旧路径 `app.pipeline.metrics`/`app.pipeline.quality_router`，**整个测试套件收集失败** | 更新 import | Task 3 |
| N4 | `test_candidates.py`/`test_glsl_optimizer.py`/`test_graph.py` 共 12 处 monkeypatch 目标仍是 `app.png_shader.*`，patch 不生效 | 按映射表更新 | Task 3 |
| N5 | `tests/unit/test_decompose.py` 已预置（准确率计划 Phase 2 的红灯测试）但 `app.pipeline.decompose` 未实现，导致收集错误拖垮全套 | `pytest.importorskip` 守卫，Phase 2 实现后自动失效 | Task 4 |
| N6 | README 未描述模块布局与前端轮询交互（原 C5 的对应物） | 更新 README.md | Task 5 |

## 注意事项

1. **这是行为修复而非重构**：N1/N2 是真 bug（恢复预期行为），N3/N4/N5 只动测试。每个任务结束跑 `cd backend && python -m pytest tests/unit/ -q`，观察失败数单调下降，Task 4 结束应全绿（test_decompose 显示 skipped）。
2. **monkeypatch 目标的判定原理**（与原计划相同）：被打桩的名字在**定义它的模块**（或经懒加载 import 的调用方模块）的全局命名空间解析。Task 3 的映射表已按 P2S 实际调用链逐条验证。
3. 修复 N2 时**不要**顺手改 `_run_post_pipeline` 的其他逻辑——准确率计划 Phase 2/3 会在该函数接线，保持锚点稳定。
4. 系统 Python 为 3.9：代码全部带 `from __future__ import annotations`，无需改语法；新增代码沿用此约定。

---

### Task 0: git init 与基线提交

**Files:**
- Create: `.gitignore`

- [ ] **Step 1: 初始化仓库**

```bash
cd /Users/douwen/Documents/HUAWEl/Shader-Agent/P2S-Agent
git init
```

- [ ] **Step 2: 写 .gitignore**

```text
__pycache__/
*.pyc
.venv/
node_modules/
backend/test_results/
backend/artifacts/
backend/.env
frontend/dist/
.DS_Store
```

- [ ] **Step 3: 基线提交**

```bash
git add -A
git commit -m "chore: initial commit of P2S-Agent (pre-cleanup baseline)"
```

### Task 1: 修复路由器残留的 app.png_shader 导入（生产 bug）

**Files:**
- Modify: `backend/app/routers/png_shader.py:293,312-313`

- [ ] **Step 1: 修改三处懒加载 import**

将 `png_shader.py:293`：

```python
        from app.png_shader.candidates.llm_scene_candidate import generate_llm_refinement
```

改为：

```python
        from app.candidates.llm_scene import generate_llm_refinement
```

将 `png_shader.py:312-313`：

```python
    from app.png_shader.compiler import compile_dsl
    from app.png_shader.dsl_validator import validate_dsl
```

改为：

```python
    from app.dsl.compiler import compile_dsl
    from app.dsl.validator import validate_dsl
```

- [ ] **Step 2: 验证模块可导入**

Run: `cd backend && python -c "from app.routers.png_shader import router; print('ok')" && grep -rn "app.png_shader" app/ | grep -v __pycache__`
Expected: 打印 `ok`；grep 无输出（生产代码零残留）

- [ ] **Step 3: Commit**

```bash
git add backend/app/routers/png_shader.py
git commit -m "fix(router): repair stale app.png_shader imports left from the P2S migration"
```

### Task 2: 修复 _run_post_pipeline 丢弃 state 键

**Files:**
- Modify: `backend/app/pipeline/graph.py:443-454`（`_run_post_pipeline` 的 return）

- [ ] **Step 1: 合并而非替换 state**

将 `_run_post_pipeline` 末尾的：

```python
    return {
        "optimization": optimization_summary,
```

改为：

```python
    return {
        **state,
        "optimization": optimization_summary,
```

（其余键保持不变。`run_png_shader_pipeline` 在 `state = _run_post_pipeline(state)` 之后读取 `state.get("preprocess")`、`state.get("selected_candidate_id")` 等上游键，当前实现下这些键全部丢失。）

- [ ] **Step 2: 运行 graph 相关测试**

Run: `cd backend && python -m pytest tests/unit/test_graph.py tests/unit/test_router.py -q`
Expected: 不新增失败（test_graph 当前的失败属于 N4，Task 3 修复）

- [ ] **Step 3: Commit**

```bash
git add backend/app/pipeline/graph.py
git commit -m "fix(pipeline): preserve upstream state keys in _run_post_pipeline return"
```

### Task 3: 修复测试套件的旧模块路径

**Files:**
- Modify: `backend/tests/unit/test_metrics.py:14`
- Modify: `backend/tests/unit/test_quality_router.py:10`
- Modify: `backend/tests/unit/test_candidates.py`、`test_glsl_optimizer.py`、`test_graph.py`（monkeypatch 目标）

- [ ] **Step 1: 修复两个收集失败的 import**

`test_metrics.py:14`：

```python
from app.pipeline import metrics
```

改为：

```python
from app.metrics import compute as metrics
```

`test_quality_router.py:10`：

```python
from app.pipeline import quality_router
```

改为：

```python
from app.metrics import quality_router
```

（两个文件后续以 `metrics.X` / `quality_router.X` 形式使用模块别名，别名不变则函数体无需改动。）

- [ ] **Step 2: 按映射表更新 monkeypatch 目标**

先定位：

Run: `cd backend && grep -rn "app.png_shader" tests/ | grep -v __pycache__`

| 旧 patch 目标 | 新 patch 目标 | 所在文件 |
|---|---|---|
| `app.png_shader.candidates.llm_scene_candidate.settings.llm_api_key` | `app.candidates.llm_scene.settings.llm_api_key` | test_candidates.py:121,318,344 |
| `app.png_shader.candidates.llm_scene_candidate.settings.llm_supports_image` | `app.candidates.llm_scene.settings.llm_supports_image` | test_candidates.py:319,345 |
| `app.png_shader.glsl_optimizer.score_glsl` | `app.pipeline.glsl_optimizer.score_glsl` | test_glsl_optimizer.py:140,151 |
| `app.png_shader.pool.generate_llm_scene_candidate` | `app.pipeline.pool.generate_llm_scene_candidate` | test_graph.py:411,460,515,647 |
| `app.png_shader.scoring.render_multiple_frames` | `app.pipeline.scoring.render_multiple_frames` | test_graph.py:463,516 |
| `app.png_shader.candidates.llm_scene_candidate.generate_llm_refinement` | `app.candidates.llm_scene.generate_llm_refinement` | test_graph.py:550（refinement.py:102 在函数体内懒 import，patch 源模块） |
| `app.png_shader.graph._score_candidates` | `app.pipeline.graph._score_candidates` | test_graph.py:648 |
| `app.png_shader.graph.run_dsl_refinement_loop` | `app.pipeline.graph.run_dsl_refinement_loop` | test_graph.py:649 |

- [ ] **Step 3: （可选）清理 docstring 中的旧路径**

`test_optimizer.py:1`、`test_input_spec.py:1`、`test_preprocess.py:1` 的 docstring 提到 `app.png_shader.*`，按映射表顺手更正（纯注释，不影响行为）。

- [ ] **Step 4: 运行全量测试**

Run: `cd backend && python -m pytest tests/unit/ -q`
Expected: 仅剩 `test_decompose.py` 一个收集错误（Task 4 处理）；其余全 PASS。如有其他失败，按 Step 2 的解析原理判断该名字现居何模块再修 patch 路径；**禁止改生产代码迁就旧 patch**。

- [ ] **Step 5: Commit**

```bash
git add backend/tests/unit/
git commit -m "test: repair stale module paths and monkeypatch targets from the P2S migration"
```

### Task 4: test_decompose 收集守卫（Phase 2 前置红灯的隔离）

**Files:**
- Modify: `backend/tests/unit/test_decompose.py:9`

- [ ] **Step 1: 把模块级 import 改为 importorskip**

将：

```python
from app.pipeline.decompose import decompose_to_dsl, fit_primitive_layer
```

改为：

```python
_decompose = pytest.importorskip(
    "app.pipeline.decompose", reason="accuracy plan Phase 2 Task 5 not implemented yet"
)
decompose_to_dsl = _decompose.decompose_to_dsl
fit_primitive_layer = _decompose.fit_primitive_layer
```

（准确率计划 Phase 2 Task 5 创建 `app/pipeline/decompose.py` 后，importorskip 自动放行，测试恢复执行，守卫无需回退。）

- [ ] **Step 2: 全量测试确认绿色**

Run: `cd backend && python -m pytest tests/unit/ -q`
Expected: 全 PASS + test_decompose 显示 skipped，0 错误

- [ ] **Step 3: Commit**

```bash
git add backend/tests/unit/test_decompose.py
git commit -m "test: guard pre-staged decompose tests until Phase 2 lands"
```

### Task 5: 文档收尾（原计划 C5 的 P2S 对应物）

**Files:**
- Modify: `README.md`

- [ ] **Step 1: README 增补模块布局与前端交互**

1. 在 "Architecture" 一节后补一段模块布局：候选池在 `backend/app/pipeline/pool.py`，评估/渲染回调在 `scoring.py`，LLM 精修循环在 `refinement.py`，`graph.py` 为 LangGraph 编排（核心四节点 preprocess→candidates→scoring→selection，优化/修订/精修在 `_run_post_pipeline` 同步执行）；客观指标在 `app/metrics/compute.py`（NumPy v2，位置敏感），评分路由在 `app/metrics/quality_router.py`；DSL 编译/渲染/校验在 `app/dsl/`；颜色归一化在 `app/utils/color.py`。
2. 补一句前端交互："`frontend/src/hooks/usePngShader.ts` posts to `POST /png-shader/run`（后台线程执行），随后轮询 `GET /png-shader/status/{run_id}` 直到 `status` 为 `completed`/`failed`。"
3. 运行结果目录说明：每次 run 的 artifacts 写入 `backend/test_results/<date>_png-shader_<label>_<run_id>/`。

- [ ] **Step 2: 最终验证**

Run: `cd backend && python -m pytest tests/unit/ -q`
Expected: 全 PASS

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document P2S module layout, frontend polling, and results directory"
```

---

## Self-Review 记录

- C1–C4 的"已落地"结论均经 2026-06-12 代码核查（grep + 文件清单），非推测；C5 转化为 Task 5。
- N1–N6 全部来自实际运行验证：`pytest tests/unit/` 收集错误 3 个（test_metrics / test_quality_router / test_decompose）、`grep app.png_shader` 在生产代码命中 3 行、`_run_post_pipeline` 返回字面 dict（graph.py:443）。
- Task 3 映射表覆盖 grep 实测的全部 12 处 monkeypatch + 2 处 import；`generate_llm_refinement` 的 patch 目标定为源模块 `app.candidates.llm_scene`，因 `refinement.py:102` 为函数体内懒 import（与 VFX 原计划同理）。
- 显式非目标：`routers/png_shader.py` 的 `run.log` 写入 `Path("artifacts")` 与 pipeline 的 `test_results` 根目录不一致——属于日志落点选择而非 bug，记录于此，不在本计划修改；E2E 脚手架由准确率计划 Phase 0 新建。
