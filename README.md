# P2S-Agent · PNG → GLSL Shader 智能体

把一张 **PNG 图像**,通过「**确定性候选生成 + LLM 改写 + VLM 评审**」的闭环,自动转换成可运行的 **GLSL（Shadertoy）着色器**;并支持**人在环**地分支精修、批量探索、偏好记忆与局部融合。

> **技术栈:** LangGraph（流程编排）· FastAPI（后端）· React 19 + Vite + Tailwind（前端）· WebGL（着色器渲染评分）· LLM / VLM（生成与评审）。
> **定位:** 「Agent 本体」= 一次 PNG→Shader 的单 run 闭环优化器(Layer 1 确定性核心 + Layer 2 LLM/VLM 闭环);其上叠加 Layer 3 人在环编排,把单 run 变成**可分支、可批量、可追溯、可融合的 run 森林**。

---

## 能力全景

| 层 | 能力 | 状态 |
|----|------|------|
| **Layer 1 · 确定性核心** | `preprocess → candidates(6 策略) → scoring → selection`,DSL→GLSL 编译 + 客观指标 + quality_router | ✅ |
| **Layer 2 · LLM/VLM 闭环** | LLM 候选生成 + 逐轮改写(DSL/GLSL 路径);VLM 近分仲裁 / 定向接受 / 最终语义闸门 | ✅ |
| **Seed-GLSL 路径** | 直接从已有 GLSL 进闭环精修(跳过候选池) | ✅ |
| **V1 定向分支** | 从任意 checkpoint + 自然语言 feedback 派生 child run(父 run 不被覆盖) | ✅ |
| **V2 / V2.1 血缘画布** | run 血缘 + checkpoint 时间线 + 分支树;React Flow 自由画布工作台(重启可恢复) | ✅ |
| **V3 变体探索** | 1 checkpoint → N 变体(6 策略模板),比较 / 选 winner / 评分 / 继续 | ✅ |
| **V3.5 批量抽卡** | 一次 2–12 张,自动拆批;追加抽卡 / 单卡重抽 / 收藏淘汰打标签 | ✅ |
| **V4.1 结构化约束** | locks(保布局/色板/背景…)+ targets(亮度增减…)+ edit_strength | ✅ |
| **V4.2 区域 / 蒙版** | 框选区域(modify/protect);protect 区显著退化时**硬否决** | ✅ |
| **V4.3 / V4.4 偏好记忆** | 跨 run 偏好档案(事件→profile),回流 prompt 与辅助排序 | ✅ |
| **V4.5 局部融合** | 多张抽卡 base+source 合成 `composite_target.png` → 融合 run(融成一个连续 shader,非拼贴) | ✅ |
| **Shader Playground** | 独立三栏页面(源码 / 渲染画布 / 可调参数),粘贴任意 GLSL → Render → 实时调参,与 Agent 闭环解耦 | ✅ |

> 完整架构、流程图与设计推导见 [`doc/2026-06-16-architecture-and-sop-overview.md`](doc/2026-06-16-architecture-and-sop-overview.md)。

---

## 快速开始

### 环境要求

- Python **3.9+**(后端直接用系统 `python3`,无需 venv)
- Node.js **18+** + npm

### 安装

```bash
cd P2S-Agent

# 一键安装前后端依赖
./start.sh install

# 或分别安装
pip install -r backend/requirements.txt
cd frontend && npm install
```

### 配置(可选)

```bash
cp backend/.env.example backend/.env   # 填入 LLM/VLM 的 API key
```

> **免 key 也能跑通:** 前端「AI 候选」选 **「关 Off」**,即用纯确定性流程产出着色器——候选池 6 个策略里只有 `llm` 需要 key,其余 5 个策略 + 评分 + 选择全是确定性的。要用 AI 改写/评审,选「自动/开」并配好 key。

### 启动

```bash
./start.sh start        # 同时启动前后端
# 或
./start.sh backend      # 仅后端  → http://localhost:8001
./start.sh frontend     # 仅前端  → http://localhost:5174
./start.sh status       # 查看状态
./start.sh stop         # 停止全部
```

- 前端:**http://localhost:5174**(Vite 代理 `/png-shader` 与 `/api` 到后端 `:8001`)
- 后端:**http://localhost:8001**(FastAPI,`/docs` 有交互式 API 文档)
- Shader Playground(独立页):**http://localhost:5174/playground.html**(无需后端,纯前端 GLSL 渲染 + 参数面板)

---

## 前端结构(双页)

应用是**双页 shell**,顶栏切换;画布页在产生第一次成功 run 后解锁(URL 同步 `?view=canvas`):

### 🅰 工作台 Studio — 单图构建与分析
上传/Seed-GLSL · AI 候选模式(关/自动/开)· 模型选择 · 策略预设 · 偏好配置 · **定向优化分支**;
三栏分析(SceneGraph / 候选 Scoreboard / QualityRouter)+ 四栏工作区(ImageDiff / 可调参数 / DSL 层 / LLM-IO 与精修历史)。

### 🅱 画布 Canvas — 血缘图工作台(全屏 React Flow)
谱系图节点:`input → run → checkpoint → branch / variant_group / draw_session / fusion`;
三个浮层 = 左上**工具栏**(fit / 重置布局 / 状态)· 右上**检查器**(选中节点详情 + 所有分支/变体/抽卡/融合/区域**内联表单**,无弹窗)· 右下**预览坞**(参考图 ↔ 选中节点渲染图)。

> **交互规则:单击 = 非破坏性预览**(更新检查器+预览坞,不改当前 run)· **双击 = 切换 active run / 折叠组**。
> `列表 | 画布` 子标签可切换列表视图(时间线 + 分支树 + 对比条)。

### 🅒 Shader Playground —— 独立 GLSL 试玩页(`/playground.html`)
三栏布局:左**源码编辑**(Render / Clear / Example)· 中**实时画布**(共用主应用的渲染器生命周期与错误上报)· 右**可调参数面板**(沿用 `PngShaderParamPanel`,Reset 还原 baseline)。
不依赖后端、不进 Agent 闭环,适合粘贴已有 shader 做对比 / 调参 / 演示。

---

## 使用流程(SOP)

| # | 流程 | 操作要点 |
|---|------|----------|
| ① | **基础生成** | 上传 PNG → 选模式/模型 → Run → 轮询出候选对比、指标、精修历史、VLM 评审 → 得到 `selected_shader.glsl` |
| ② | **定向分支精修(V1)** | 选 checkpoint → feedback + 模式(continue/refine/polish)+ 锁 → 派生 child run,自动切过去 |
| ③ | **变体探索(V3)** | checkpoint → feedback + 变体数(2–6)+ diversity → 一次产 N 个策略变体 → 比较 → 选 winner → 从 winner 继续 |
| ④ | **批量抽卡(V3.5)** | checkpoint → feedback + 数量(2–12)→ 自动拆成多个 ≤6 的变体组 → 追加抽卡 / 单卡重抽 / 收藏淘汰打标签 |
| ⑤ | **约束 + 区域(V4.1/4.2)** | 给分支/变体加锁 + 目标 + 编辑强度;或框选区域(modify/protect),protect 区退化即硬否决 |
| ⑥ | **偏好记忆(V4.3/4.4)** | winner/评分/卡片事件回填成偏好档案,跨 run 注入 prompt 与辅助排序(不自动替你选 winner) |
| ⑦ | **局部融合(V4.5)** | 抽卡池选 base 卡 + 多张 source 卡 + 区域 → 合成 `composite_target.png` → 融合 run(优化参考改为合成图,`judge_fusion_pairwise` 把关) |

**关键产物(落盘 artifacts):** `selected_shader.glsl` · `scoreboard.json` · `objective_metrics.json` · `refinement_summary.json` · `reference_input.png` · run 血缘 / 区域 / 融合记录。

---

## API 端点

后端基址 `http://localhost:8001`。核心与人在环端点均挂在 `/png-shader` 前缀下;模型/策略配置在 `/api` 下。

### 核心
| 方法 | 路径 | 用途 |
|------|------|------|
| POST | `/png-shader/run` | 初始 PNG→Shader(multipart:`image` + `input_spec_json` + 可选 `seed_glsl`) |
| GET  | `/png-shader/status/{run_id}` | 轮询 run 状态(含 partial 进度) |
| POST | `/png-shader/runs/{run_id}/stop` | 停止运行 |
| POST | `/png-shader/parameterize/{run_id}` | GLSL 补全可调参数 |
| GET  | `/api/models` · `/api/strategy-config` | 可用模型 / 策略配置 |

### V1–V2 分支与血缘
| 方法 | 路径 | 用途 |
|------|------|------|
| GET  | `/png-shader/runs/{run_id}/checkpoints` | 列可分支 checkpoint |
| POST | `/png-shader/runs/{run_id}/branch-refine` | 从 checkpoint 派生 child run |
| GET  | `/png-shader/runs/{run_id}/timeline` · `/branches` | 时间线 / 分支树 |
| PATCH| `/png-shader/runs/{run_id}/metadata` · `/strategy` | 改 title/favorite/tags / 策略 |
| GET  | `/png-shader/runs/{run_id}/artifacts/{id}` | 取 PNG/JSON/GLSL artifact |

### V3 / V3.5 变体与抽卡
| 方法 | 路径 | 用途 |
|------|------|------|
| POST | `/png-shader/runs/{run_id}/explore-variants` | 一次产 N 个变体 |
| GET  | `/png-shader/variant-groups/{group_id}` | 变体组状态 |
| POST | `/png-shader/variant-groups/{id}/winner` · `/ratings` · `/stop` | 选 winner / 评分 / 停止 |
| POST | `/png-shader/runs/{run_id}/draw-session` | 批量抽卡(2–12) |
| GET  | `/png-shader/draw-sessions/{draw_id}` | 抽卡会话状态 |
| POST | `/png-shader/draw-sessions/{id}/draw-more` · `/redraw` · `/cards/{run_id}/event` | 追加 / 重抽 / 卡片事件 |

### V4 约束 / 偏好 / 融合
| 方法 | 路径 | 用途 |
|------|------|------|
| POST | `/png-shader/runs/{run_id}/region-mask` | 区域/蒙版约束(V4.2) |
| GET/PATCH | `/png-shader/preferences/profile` | 偏好档案读取/更新(V4.3) |
| POST | `/png-shader/preferences/events` · `/rebuild` · `/clear` | 偏好事件 / 重建 / 清空 |
| POST | `/png-shader/fusions` | 创建融合计划(V4.5) |
| POST | `/png-shader/fusions/{id}/composite-target` · `/run` | 生成合成目标 / 跑融合 run |
| GET  | `/png-shader/fusions/{id}` · `/artifacts/{id}` | 融合状态 / artifact |

---

## 项目结构

```
P2S-Agent/
├── backend/
│   ├── p2s_agent/              # ★ Agent 本体（不依赖 FastAPI，可 import / CLI 独立运行）
│   │   ├── config.py           # agent 配置（ModelConfig / 模型 / 阈值 / 渲染 / langsmith）
│   │   ├── state.py · strategy.py
│   │   ├── core/               # Layer 1+2 计算内核
│   │   │   ├── pipeline/       # LangGraph 图 + 候选/打分/优化/精修/编译产物
│   │   │   ├── candidates/ · dsl/ · metrics/ · llm/ · render/ · utils/
│   │   │   └── errors.py · validation.py · logging_config.py · tracing.py
│   │   ├── orchestration/      # Layer 3 编排（血缘/checkpoint/变体/抽卡/融合/偏好 + sessions）
│   │   ├── store/              # 运行态 run/model store（LRU + 索引）
│   │   ├── workers/            # 后台 worker + 背压信号量
│   │   └── cli.py              # ★ 无 server 跑一次 PNG→shader
│   ├── app/                    # Web 层薄壳（仅 FastAPI 绑定，不含业务逻辑）
│   │   ├── main.py             # FastAPI 入口 + 域错误→HTTP 翻译
│   │   ├── config.py           # web 配置（host/port）
│   │   ├── api/
│   │   │   ├── routers/        # core / branch / variant / draw / fusion / preferences
│   │   │   └── guards.py       # 上传/请求体守卫
│   │   └── routers/png_shader.py  # 薄聚合器（保持旧 import 路径）
│   ├── tests/                  # pytest 单测
│   └── requirements.txt
├── frontend/                   # React 19 + Vite + Three.js + React Flow
│   ├── src/
│   │   ├── App.tsx             # 单 usePngShader() + PngShaderProvider + 双页 shell
│   │   ├── pages/              # StudioPage / CanvasPage / ShaderPlayground
│   │   ├── context/            # PngShaderContext
│   │   ├── components/         # 面板 / 画布节点 / 检查器 / 表单 / PlaygroundCanvas
│   │   ├── hooks/              # usePngShader / useModels / useStrategyConfig
│   │   ├── lib/                # 候选/布局模型、策略预设、refine 选项、playground 示例(+ vitest)
│   │   └── playground.tsx      # 独立入口(对应 frontend/playground.html)
│   ├── index.html · playground.html · package.json · vite.config.ts  # 多页构建
├── doc/                        # 架构总览 + 各版本设计文档
└── start.sh                    # 启停脚本
```

---

## 测试与门禁

```bash
# 后端单测(系统 python3,无 venv)
cd backend && python3 -m pytest tests/ -q

# 前端构建门禁(tsc + vite build)
cd frontend && npm run build

# 前端纯逻辑单测(vitest,lib 目录)
cd frontend && npx vitest run    # 等价于 npm test
```

> 当前规模(2026-06-24):**1299** pytest + **135** vitest + `npm run build` 全绿。
> 路线图既定原则:**每版可独立验收,不跳测试门禁**——后端单测优先,前端至少 `npm run build` 通过。
> 注:`npm run lint` 当前是空壳(ESLint 未安装),依赖严格 `tsc`(`noUnusedLocals`)兜底。

## 最近更新

- **2026-06-24** `fix(param-panel)` — 「Reset」改为还原**进入调参前的 baseline**,而不是首次渲染值。
- **2026-06-23** `fix(db)` — events `payload=None` 守护;明确 `upsert` 全行写入合同。
- **2026-06-22** Shader Playground 合并主干(独立 3 栏页面,源码 + 渲染 + 参数)。
- **2026-06-22** `fix(db)` 系列 — 四模块 snapshot loader 改为 file-first;修复分支树 / 事件流读一致性残留。
- 详细变更见 `git log` 与 `doc/` 下版本设计文档。

## License

Internal use only.
