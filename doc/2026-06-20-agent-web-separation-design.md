# Agent ↔ 传统后端 包边界分离（L1）— 设计文档

> **日期：** 2026-06-20　**分支：** `main`　**范围：** 后端**纯重构**(行为不变)，单仓单部署
> **关联：** [深度审计](2026-06-18-architecture-deep-audit.md) · [架构与 SOP 总览](2026-06-16-architecture-and-sop-overview.md) · [数据层 SQLite 设计](2026-06-18-data-layer-sqlite-design.md)
> **决策来源：** 目标形态 = **内部多用户服务**；痛点 = demo 形态、AI 债重、bug 多；选型无约束。结论已定：**不换语言、不推倒重写、架构演进**。本文档是"演进路线"的**第一个、也是基础的子项目**。

---

## 0. 目标与非目标

**目标(可验收)：**
1. 存在一个**不依赖 FastAPI/HTTP** 的 `p2s_agent` 包，可被 `import` / 经 CLI 单独运行。
2. `app/`(web)只做 `parse → 调 p2s_agent → serialize`；**`app` 依赖 `p2s_agent`，反向永不成立**(有测试守这条线)。
3. **纯重构**：997 后端 pytest + 78 vitest + `npm run build` **全程绿**，对外行为**零变化**(端点、URL、响应体不变)。

**非目标(明确排除，YAGNI)：**
- ❌ 不拆进程/微服务(那是 L3，本次不做)。
- ❌ 不换语言、不动前端框架。
- ❌ 不改 store/worker 的**实现**——只是把它们从 HTTP 路由里**搬位置**。真正的硬化(SQLite 化、有界队列、背压)留给后续子项目，本次只建立"可硬化的边界"。

---

## 1. 现状诊断(为什么这么切)

| # | 证据 | 含义 |
|---|---|---|
| 1 | agent 内核(`pipeline/candidates/dsl/metrics/llm`)对 `fastapi/starlette` 依赖 **≈ 0** | **计算内核已经是一个库**，只是住在 `app/` 里 |
| 2 | `routers/png_shader.py` **3664 行**里塞着三层本属于 agent 的逻辑：**状态层 / 任务层 / 编排层** | 真正的耦合在这里——web 壳和 agent 脑子焊死在一个文件 |
| 3 | **13+ 处测试**直接 `from app.routers.png_shader import _run_store / _run_png_shader_background / _variant_preserved / validate_safe_id …` | 逻辑错位的铁证：测试被迫穿透 HTTP 层去测编排/状态逻辑 |
| 4 | `config.py` 一个 `Settings` 类里**混装** agent 配置(`ModelConfig`/`llm_*`/`screenshot_*`/`protect_veto_*`/`langsmith_*`)与 web 配置(`host`/`port`/`frontend_url`)；agent 内核 7+ 处 `from app.config import settings` | 配置也跨在边界上，需沿同一条线切开 |

---

## 2. 目标结构

**唯一不变式：`app/` 依赖 `p2s_agent/`；`p2s_agent/` 永不 import `app.*` / `fastapi` / `starlette`。**
这条线就是"直观可读"的来源——开发者可以**完全不懂 web 也能读完整个 agent**。

```
backend/
  p2s_agent/                 # ★ Agent 本体：纯 Python，无 FastAPI；可 import / CLI 跑
    config.py                # ← 切出 agent 配置：ModelConfig + llm_*/proxy/screenshot_*/
                             #    render_timeout/protect_veto_*/langsmith_*/presets
    strategy.py (+ .json)    # ← app/strategy_config_loader.py + strategy_config.json
    state.py                 # ← app/state.py (P2SPipelineState)
    core/                    # Layer 1+2 计算内核(几乎原样平移)
      pipeline/              # ← graph.py + preprocess/input_spec/scoring/optimizer/glsl_optimizer/
                             #    refinement/glsl_refinement/seed_glsl/residual_layers/decompose/
                             #    artifacts/image_composite
      candidates/            # ← app/candidates/*（6 策略）
      dsl/                   # ← app/dsl/*（schema/validator/compiler）
      metrics/               # ← app/metrics/*（客观指标 + quality_router）
      llm/                   # ← app/llm/*（client/model_resolver/vlm_judge）
      render/                # ← services/browser_render.py + services/shader_validator.py
      utils/                 # ← utils/color.py + cv_features.py + glsl_postprocess.py
    orchestration/           # Layer 3 编排
      # ← pipeline/{variant_groups,draw_sessions,fusion_plans,checkpoints,run_index,
      #    revision,preferences,human_constraints,human_feedback,region_metrics}
      # ← 从 png_shader.py 抽出：_create_variant_group / _create_draw_groups / _fold_draw_overlay /
      #    _resolve_draw_checkpoint / _prevalidate_draw_quality / _finalize_fusion_for_run / _variant_preserved
    store/                   # 运行态状态(搬位置，实现照旧；硬化→子项目2)
      # ← 从 png_shader.py 抽出：_run_store/_run_models(+locks) / _store_run(_locked) / _drop_run /
      #    _evict_one_run_locked / _touch_run / _snapshot_run / _publish_partial_to_store /
      #    _get_run_model / _store_run_model / _evict_one_model_locked / _run_is_live / _index_created/_updated
    workers/                 # 任务/worker(搬位置；队列/背压→子项目3)
      # ← 从 png_shader.py 抽出：_run_png_shader_background / _start_pipeline_worker /
      #    WorkerCapacityError / _env_int
    cli.py                   # ★ 新增：不起 server 也能跑一次 PNG→shader（开发者/批处理/测试用）
  app/                       # 传统后端：薄壳
    main.py                  # FastAPI 入口(保留)
    config.py                # ← 仅 web 配置：host/port/frontend_url(+CORS/上传上限/鉴权)
    api/
      routers/               # 只做 parse → 调 p2s_agent → serialize；png_shader.py 瘦身后
                             #   按域拆：core/branch/variant/draw/fusion + models/strategy_config
      guards.py              # ← 从 png_shader.py 抽出 HTTP 守卫：_guard_upload/_check_content_length/
                             #   _coerce_int/validate_safe_id/_enforce_text_cap/_MAX_*
      schemas/               # （占位）Pydantic 请求/响应——子项目4 OpenAPI→TS codegen 落点
    infra/                   # ← services/logging_config.py + langsmith_tracing.py（横切观测）
  tests/                     # import 路径随迁移更新；新增"边界不变式"测试
```

> **说明：** `langsmith_tracing.py` 若被 agent 内联调用则随 agent 迁入并读 `p2s_agent` 配置；否则留 `app/infra`。属次要判断，迁移时按实际调用点定。

---

## 3. 迁移映射(current → target)总表

| 当前位置 | 去向 | 类型 |
|---|---|---|
| `app/dsl/`、`candidates/`、`metrics/`、`llm/` | `p2s_agent/core/{dsl,candidates,metrics,llm}/` | 平移(已干净) |
| `pipeline/` 计算类(graph/preprocess/input_spec/scoring/optimizer/glsl_optimizer/refinement/glsl_refinement/seed_glsl/residual_layers/decompose/artifacts/image_composite) | `p2s_agent/core/pipeline/` | 平移 |
| `pipeline/` 编排类(variant_groups/draw_sessions/fusion_plans/checkpoints/run_index/revision/preferences/human_constraints/human_feedback/region_metrics) | `p2s_agent/orchestration/` | 平移 |
| `app/state.py` | `p2s_agent/state.py` | 平移(消除 `graph.py → app.state` 反向依赖) |
| `app/strategy_config_loader.py` + `strategy_config.json` | `p2s_agent/strategy.py` | 平移 |
| `services/browser_render.py`、`shader_validator.py` | `p2s_agent/core/render/` | 平移 |
| `utils/color.py`、`cv_features.py`、`glsl_postprocess.py` | `p2s_agent/core/utils/` | 平移 |
| `services/logging_config.py`、`langsmith_tracing.py` | `app/infra/`(或随 agent) | 平移 |
| `config.py` 的 `ModelConfig` + agent 字段 | `p2s_agent/config.py` | **切分** |
| `config.py` 的 `host/port/frontend_url` | `app/config.py` | **切分** |
| `png_shader.py`：`_run_store`/`_run_models`/`_store_run`/`_snapshot_run`/`_publish_partial_to_store`/`_index_*` … | `p2s_agent/store/` | **抽出** |
| `png_shader.py`：`_run_png_shader_background`/`_start_pipeline_worker`/`WorkerCapacityError`/`_env_int` | `p2s_agent/workers/` | **抽出** |
| `png_shader.py`：`_create_variant_group`/`_create_draw_groups`/`_fold_draw_overlay`/`_resolve_draw_checkpoint`/`_prevalidate_draw_quality`/`_finalize_fusion_for_run`/`_variant_preserved` | `p2s_agent/orchestration/` | **抽出** |
| `png_shader.py`：`_guard_upload`/`_check_content_length`/`_coerce_int`/`validate_safe_id`/`_enforce_text_cap`/`_MAX_*` | `app/api/guards.py` | **抽出**(留在 web 侧) |
| `png_shader.py`：33 个 `@router` 路由处理函数 | `app/api/routers/{core,branch,variant,draw,fusion}.py` | 瘦身+按域拆 |

---

## 4. 边界不变式与强制手段

- **规则：** `p2s_agent/**` 不得出现 `import app` / `from app` / `import fastapi` / `import starlette`。
- **强制(零新依赖，贴合 no-venv 文化)：** 新增一个 pytest，遍历 `p2s_agent/` 下所有 `.py`，用 `ast` 解析其 import，断言不含上述前缀。这条测试进门禁，**任何回潮 PR 当场红**。
- **附带修正(本次顺手做掉)：**
  - `graph.py` 的 `from app.state import P2SPipelineState` → `from p2s_agent.state import …`(依赖反转消除)。
  - 7+ 处 `from app.config import settings/ModelConfig` → `from p2s_agent.config import …`。
  - `input_spec.py`/`graph.py` 的 `from app.strategy_config_loader import …` → `from p2s_agent.strategy import …`。

---

## 5. 行为保真的推进顺序(每步门禁全绿、独立可回滚 commit)

> 总原则：**先搬干净的、再切耦合的；每步只动 import 不动逻辑；每步跑全量门禁。** 用 codemod(批量改 import)+ pytest 兜底。

| 步 | 动作 | 验证 |
|---|---|---|
| **S0** | 建 `p2s_agent/` 包骨架(空 `__init__`)+ 占位子包 | `pytest` 绿 |
| **S1** | 平移已干净内核：`dsl/candidates/metrics/llm` + `pipeline` 计算类 + `state.py` + `strategy` + `core/render` + `core/utils`；批量改 import | `pytest` 绿 |
| **S2** | 切分 `config.py`：`ModelConfig`+agent 字段 → `p2s_agent/config.py`；web 字段留 `app/config.py`；改 7+ 引用点 | `pytest` 绿 |
| **S3** | 平移编排类(`variant_groups/draw_sessions/fusion_plans/...`) → `orchestration/` | `pytest` 绿 |
| **S4** | **抽 store**：`_run_store` 等搬入 `p2s_agent/store/`；router 与 13+ 测试 import 改指新位置 | `pytest` 绿 |
| **S5** | **抽 workers**：`_run_png_shader_background`/`_start_pipeline_worker` 搬入 `p2s_agent/workers/` | `pytest` 绿 |
| **S6** | **抽编排辅助**：`_create_variant_group`/`_create_draw_groups`/... 搬入 `orchestration/`；router 改为调用 | `pytest` 绿 |
| **S7** | HTTP 守卫 → `app/api/guards.py`；瘦身后的 33 路由按域拆 `core/branch/variant/draw/fusion`(URL 不变) | `pytest` + `npm run build` 绿 |
| **S8** | 加**边界不变式测试**；加 `cli.py`(跑一次 PNG→shader) | 全门禁绿 |

> **测试 import 策略：** 每步**同步更新**对应测试的 import 到新位置(测试是安全网，要让它指向新家)。不留长期 re-export shim，避免"两个真相"。

---

## 6. 在整体路线图中的位置

L1 是地基，后续子项目都踩在它上面，且**互不重复**：

```
L1 · agent/web 包边界  ← 本文档（纯重构）
  └─ 子项目2 · 数据层：store/ 从内存 dict → SQLite + 事件投影（已有 SQLite 设计文档）
  └─ 子项目3 · 任务层：workers/ → 有界队列(arq)+ 背压 429
  └─ 子项目4 · 契约：app/api/schemas/ 填 Pydantic response_model → OpenAPI → 前端 TS codegen
  └─ 子项目5 · 鉴权：app/ 加内部 SSO + 每用户隔离（多用户上线前）
（横切）前端 god 文件拆分：usePngShader.ts(1381) / BranchCanvasInspector.tsx(1340)
```

**为什么 L1 必须先做：** 子项目2/3 要硬化的 `store` 和 `workers`，**目前根本不是独立模块**——它们是 3664 行路由里的私有函数。L1 把它们变成有边界、可单测的包之后，硬化才有抓手；审计点名"最高风险却没测"的路径(轮询竞态、optimizer 接受/停止)也才能脱离 HTTP+线程层单独测。

---

## 7. 风险与缓解

| 风险 | 缓解 |
|---|---|
| 大范围改 import 引入笔误 | codemod 批量改 + 每步全量 pytest 兜底 + 每步独立 commit 可回滚 |
| 循环 import | 严格单向分层 `core → orchestration → store → workers`；`app` 单向依赖 `p2s_agent` |
| `_run_store` 是全局单例、被多处隐式引用 | S4 集中为单一模块入口；先抽 store 再抽依赖它的 workers/编排 |
| 隐藏的 web 耦合(如某处偷用 `Request`) | 边界不变式测试(S8)兜底；迁移中 grep `Request`/`UploadFile` 复核 |
| 与正在收尾的审计 bug 修复撞车 | L1 是搬运不是改逻辑；建议先合并在途 bug 修复，再起 L1 分支 |

---

## 8. 验收清单

- [ ] `p2s_agent/` 下无 `app.`/`fastapi`/`starlette` import(不变式测试通过)
- [ ] 997 后端 pytest 绿 · `npm run build` 绿 · 78 vitest 绿
- [ ] `png_shader.py` 从 **3664 行**降到"薄路由"(目标各域路由文件 < ~400 行)
- [ ] `config.py` 切分完成，agent 不再读 web 配置
- [ ] `cli.py` 能在**不启动 server** 的情况下跑通一次 PNG→shader
- [ ] 对外端点/URL/响应体**逐一不变**(可用现有 router 测试 + 一次手工 SOP 走查确认)
</content>
</invoke>
