# P2S-Agent 架构与使用流程总览

> **用途:** 本文档作为 PPT 生成输入。涵盖 (1) 当前已实现版本架构、(2) 最终方案架构、(3) 当前版本使用流程 (SOP)。
> **重点:** Agent 技术架构（LangGraph 内核 + LLM/VLM 闭环 + 人在环编排）。
> **日期:** 2026-06-16　**对应分支:** `feat/human-in-loop-v3.5-batch-draw`
> **图表说明:** 所有图为 Mermaid（GitHub / 多数 Markdown 工具可直接渲染，亦可导出 PNG/SVG 插入 PPT）。

---

## 0. 建议幻灯片大纲（PPT 目录）

| # | 幻灯片 | 取材章节 | 配图 |
|---|--------|----------|------|
| 1 | 封面：P2S-Agent — PNG → GLSL Shader 智能体 | §1 | — |
| 2 | 一句话定位 + 能力全景 | §1 | — |
| 3 | 三层架构总览（两版共享内核） | §2 | 图 1 |
| 4 | Agent 内核：端到端核心流程图 | §3.1 | 图 2 |
| 5 | Agent 内核：候选池 6 策略 | §3.2 | 图 3 |
| 6 | Agent 内核：LLM 闭环精修决策 | §3.3 | 图 4 |
| 7 | 两条入口路径：候选池 vs Seed-GLSL | §3.4 | 图 2 |
| 8 | 已实现版本：+ 人在环编排层 (V1–V3.5) | §4.1 | 图 5 |
| 9 | 已实现版本：数据模型 ER | §4.2 | 图 6 |
| 10 | 已实现版本：run 血缘森林 | §4.3 | 图 7 |
| 11 | **SOP①：基础单图生成闭环** | §5.1 | 图 8 |
| 12 | **SOP②③④：分支 / 变体 / 抽卡** | §5.2–5.4 | 图 7 |
| 13 | **SOP：端到端时序图** | §5.5 | 图 8 |
| 14 | 最终方案：+ 控制 / 记忆 / 融合 (V4.1–V4.5) | §6 | 图 9 |
| 15 | 最终方案：融合数据流 | §6.5 | 图 10 |
| 16 | 两版对比表 | §7 | — |
| 17 | 演进路线图与现状 | §8 | 图 11 |
| 18 | 附录：端点 / 模块 / 术语 | §9 | — |

---

## 1. 产品定位与能力全景

**一句话定位：** 把一张 PNG 图像，通过「确定性候选生成 + LLM 改写 + VLM 评审」的闭环，自动转换成可运行的 GLSL（Shadertoy）着色器；并支持人在环地分支、批量探索与（最终方案）局部融合。

**技术栈：** LangGraph（流程编排）· FastAPI（后端服务）· React + Vite + Tailwind（前端）· WebGL（着色器渲染评分）· LLM/VLM（生成与评审）。

**能力全景（已实现）：**
- 单图一键生成着色器，多候选并比，客观指标 + 语义评审双重把关。
- LLM 闭环精修（DSL 路径 / GLSL 路径），实时进度可见。
- 人在环：从任意 checkpoint 分支精修、一次产 N 个变体、批量抽卡（API）。
- 全链路可追溯：run 血缘 + checkpoint 时间线 + artifacts 落盘 + 重启可恢复。

---

## 2. 三层架构总览（两版共享内核）

整个系统是**三层**结构。两版差异只在 **第 3 层**及它对第 2 层的「注入内容」；第 1、2 层是两版**共享的 Agent 本体**。

**图 1 · 三层架构**

```mermaid
flowchart TB
    subgraph L3["Layer 3 · 人在环层"]
        direction LR
        L3a["已实现版<br/>分支 / 变体 / 抽卡 + 血缘追溯"]
        L3b["最终方案<br/>+ 控制(约束/区域) + 记忆(偏好) + 融合(局部叠加)"]
    end
    subgraph L2["Layer 2 · LLM/VLM 闭环精修 (agentic)"]
        L2a["LLM：候选生成 + 逐轮改写 DSL/GLSL"]
        L2b["VLM：近分仲裁 / 定向接受 / 语义闸门"]
    end
    subgraph L1["Layer 1 · 确定性核心 (LangGraph)"]
        L1a["preprocess → candidates → scoring → selection"]
        L1b["DSL→GLSL 编译 + 客观指标 + quality_router"]
    end
    L3 -->|"注入：feedback / 约束 / 区域veto / 偏好 / 融合目标"| L2
    L2 -->|"驱动 / 评审"| L1
    L1 -.->|"selected + 指标"| L2

    classDef core fill:#1f6f43,stroke:#0f3,color:#fff
    classDef agent fill:#2a4d8f,stroke:#69f,color:#fff
    classDef human fill:#6b3fa0,stroke:#c9f,color:#fff
    class L1,L1a,L1b core
    class L2,L2a,L2b agent
    class L3,L3a,L3b human
```

**关键认知：** 「Agent」本体 = Layer 1 + Layer 2，即一次 PNG→Shader 的**单 run 闭环优化器**。Layer 3 是把这个优化器反复调度、并喂入不同人类意图的「编排/控制层」。

---

## 3. Agent 内核详解（Layer 1 + Layer 2，两版共享）

统一入口：`run_png_shader_pipeline(...)`（`backend/app/pipeline/graph.py`）。

### 3.1 端到端核心流程

**图 2 · 核心流程图**（与 `graph.py:CORE_PIPELINE_FLOWCHART` 一致）

```mermaid
flowchart TD
    A["run_png_shader_pipeline(image, input_spec)"] --> B{"seed_glsl?"}
    B -->|"有 (Seed-GLSL 路径)"| S["_run_seed_glsl_path<br/>跳过 LangGraph<br/>adapt_seed_glsl → 单 GLSL 候选 → 评分一次"]
    B -->|"无 (候选池路径)"| E["创建 run_dir + 保存 manifest/reference/input_spec"]

    subgraph LG["LangGraph StateGraph · 确定性核心"]
        F["preprocess_step<br/>提取图像特征"]
        G["candidates_step<br/>候选池 6 策略 + DSL编译/GLSL校验"]
        H["scoring_step<br/>渲染 + 客观指标 + quality_router"]
        I["selection_step<br/>选最优"]
        F --> G --> H --> I
    end
    E --> F

    I --> J{"VLM 近分仲裁?"}
    J -->|"近分"| K["judge_pairwise 调整分数"]
    J -->|"否"| M["objective ranking"]
    K --> M

    subgraph POST["Post-Pipeline · 同步, 非 graph"]
        M --> N{"selected 类型?"}
        N -->|"DSL"| O["坐标下降优化 → revision(回滚) → residual 补层"]
        N -->|"GLSL"| W["#define 优化 + WebGL 评分"]
        O --> U["精修判断 _should_run_refinement"]
        W --> U
        U -->|"运行"| Y["LLM 闭环精修循环 (DSL/GLSL)"]
        U -->|"跳过"| AA
        Y --> AA{"VLM final gate?"}
        AA -->|"是"| AB["judge_rubric 融合语义分"]
        AA -->|"否"| AC["同步 selected"]
        AB --> AC
    end
    S --> N
    AC --> OUT["落盘 artifacts<br/>selected_shader.glsl / scoreboard / metrics / refinement_summary / lineage"]
```

### 3.2 候选池 6 策略

**图 3 · 候选池**（仅 `llm` 调 LLM，其余确定性）

```mermaid
flowchart LR
    IN["preprocess 特征 + input_spec"] --> POOL
    subgraph POOL["run_candidate_pool"]
        direction TB
        B1["baseline<br/>确定性兜底形状"]
        B2["rule<br/>几何启发式"]
        B3["decompose<br/>OpenCV 实测几何"]
        B4["cv<br/>视觉形状检测(可选)"]
        B5["llm ★唯一LLM<br/>DSL 或 GLSL"]
        B6["fallback<br/>永远成功"]
    end
    B1 --> VAL
    B2 --> VAL
    B3 --> VAL
    B4 --> VAL
    B5 --> VAL
    B6 --> VAL
    VAL["校验 → DSL→GLSL 编译 / raw GLSL 校验"] --> SCORE["scoring：SSIM / grid_color_sim / mask_iou / edge_iou / rmse → quality_router"]

    classDef llm fill:#2a4d8f,stroke:#69f,color:#fff
    class B5 llm
```

### 3.3 LLM/VLM 闭环精修决策（Layer 2 核心）

> `_run_post_pipeline(...)`，**不在 LangGraph 图里**，作为选最优后的同步后处理。这是人在环 feedback / 定向接受的**唯一注入点**。

**图 4 · 精修循环接受/停止决策**

```mermaid
flowchart TD
    Start["进入精修循环<br/>run_dsl/glsl_refinement_loop"] --> Iter["第 i 轮：LLM 改写 DSL/GLSL"]
    Iter --> Render["渲染 + 计算客观指标"]
    Render --> Cmp{"Δscore > 0 ?"}
    Cmp -->|"提分"| Accept["接受新结果"]
    Cmp -->|"小幅降分"| VLM{"VLM 判定"}
    VLM -->|"directed_pairwise：符合用户目标"| Accept
    VLM -->|"pairwise：新结果确实更好"| Accept
    VLM -->|"否决"| Reject["保留旧结果"]
    Accept --> Stop{"停止条件?<br/>high_score_stop / 阈值 / 停滞 / patience"}
    Reject --> Stop
    Stop -->|"否"| Iter
    Stop -->|"是"| Done["输出 best 结果 + history"]
    Done --> Gate["VLM final gate：judge_rubric → 语义分融合 final_score"]

    classDef vlm fill:#7a4d1f,stroke:#fb6,color:#fff
    class VLM,Gate vlm
```

**LLM 角色：** ① 候选生成（1 次）；② 闭环精修逐轮改写 DSL/GLSL。
**VLM 角色（裁判，可失败降级为纯指标）：** `judge_pairwise`（近分仲裁 + 小幅降分否决）· `judge_directed_pairwise`（带用户目标定向接受）· `judge_rubric`（最终语义闸门）。

### 3.4 两条入口路径

| 路径 | 触发 | 流程 |
|------|------|------|
| **候选池路径** | 普通 `/run` | 走完整 LangGraph 4 节点 → `_run_post_pipeline` |
| **Seed-GLSL 路径** | `/run` 带 `seed_glsl` | **跳过 LangGraph**：`adapt_seed_glsl` 适配 → 合成单候选 → 评分一次 → 直接 `_run_post_pipeline`（强制 GLSL 渲染 + 精修 `on`） |

---

## 4. 已实现版本架构（+ Layer 3 人在环编排，V1 → V3.5）

在内核之上加 **Layer 3 编排层**：把「单 run agent」变成「**可分支、可批量、可追溯的 run 森林**」。
**注意：它不改 Agent 的决策依据，只是反复调度内核，并向 Layer 2 注入文字 feedback / 定向接受配置。**

### 4.1 编排能力一览

**图 5 · 已实现的人在环编排层**

```mermaid
flowchart TB
    subgraph L3["Layer 3 · 人在环编排 (已实现)"]
        V1["V1 branch-refine<br/>从 checkpoint 取 seed GLSL → 派生 child run"]
        V12["V1.2 directed acceptance<br/>force_first_iteration + directed_acceptance"]
        V2["V2 run index<br/>血缘 + timeline + branch tree (重启可恢复)"]
        V21["V2.1 Branch Canvas<br/>reactflow 自由画布工作台"]
        V31["V3.1 variant_groups<br/>1 checkpoint → N 变体 (6 模板, semaphore=2)"]
        V32["V3.2 Variant Canvas<br/>比较 / 选 winner / 评分 / 继续"]
        V35["V3.5 draw_sessions<br/>批量抽卡 2–12 → 多 group (draw-more/redraw/event)"]
    end
    L3 -->|"注入 human_feedback_notes / directed_acceptance / force_first"| K["Agent 内核 Layer 1+2 (未改动)"]

    classDef done fill:#1f6f43,stroke:#0f3,color:#fff
    classDef wip fill:#7a6a1f,stroke:#fd6,color:#fff
    class V1,V12,V2,V21,V31,V32 done
    class V35 wip
```

**实现状态（精确）：**
- ✅ 端到端可用（含前端 UI）：V1 · V2 · V2.1 · V3.1 · V3.2（变体）。
- 🟡 V3.5 批量抽卡：**后端模块 + API 已实现并通过单测**（`draw_sessions.py` / 5 个端点）；**前端尚未接线**（当前分支在做）。
- ⬜ V4.x（约束 / 区域 / 偏好 / 融合）：未实现，属最终方案。

### 4.2 数据模型

**图 6 · 数据模型 ER**（血缘统一记录 + 组/会话）

```mermaid
erDiagram
    RUN ||--o{ RUN : "parent_run_id 分支"
    RUN ||--o{ CHECKPOINT : "包含可分支点"
    RUN ||--o{ VARIANT_GROUP : "explore-variants"
    VARIANT_GROUP ||--o{ RUN : "child 变体"
    RUN ||--o{ DRAW_SESSION : "draw-session"
    DRAW_SESSION ||--o{ VARIANT_GROUP : "plan_draw_batches 拆批"
    DRAW_SESSION ||--o{ RUN : "cards"

    RUN {
        string run_id PK
        string root_run_id
        string parent_run_id FK
        string source_checkpoint_id
        string mode "continue|refine|polish"
        string feedback
        string status
        float final_score
        bool favorite
        string variant_group_id FK
        int variant_index
        string draw_session_id FK
        int draw_card_index
        string replacement_of_run_id
    }
    CHECKPOINT {
        string id "candidate|refinement:iter:n|final:selected"
        string kind
        float score
        bool has_glsl
    }
    VARIANT_GROUP {
        string group_id PK
        int variant_count "2-6"
        string diversity
        string status
        string winner_run_id
    }
    DRAW_SESSION {
        string draw_id PK
        int requested_count "2-12"
        string status
        list group_ids
        list card_run_ids
    }
```

- **持久化：** append-only JSONL（`run_index.jsonl`），`created`/`updated` 事件折叠成最新状态 → **重启可恢复**整棵分支树；内存 store 仅作 LRU 缓存（上限 100）。
- **Checkpoint：** `candidate:{id}` / `refinement:iter:{n}` / `final:selected`；`list_checkpoints` 给元数据，`resolve_checkpoint` 解析出可作 seed 的 GLSL。
- **变体组 / 抽卡会话：** 各自 `<id>.json` + `<id>_events.jsonl`（winner/rating/card-event 写事件）。
- **并发模型：** 后台线程 worker；变体并发 `threading.Semaphore(2)`；排队中的 child 可在 acquire 前被 stop。

### 4.3 run 血缘森林

**图 7 · 一棵典型的 run 森林**（分支 / 变体 / 抽卡如何长出来）

```mermaid
flowchart TD
    ROOT["root run<br/>初始生成 /run"] --> CP["final:selected (checkpoint)"]
    CP -->|"branch-refine + feedback"| CHILD["child run<br/>定向精修"]
    CP -->|"explore-variants (N=4)"| VG["VariantGroup"]
    VG --> Va["variant 1"]
    VG --> Vb["variant 2 ★winner"]
    VG --> Vc["variant 3"]
    VG --> Vd["variant 4"]
    CP -->|"draw-session(8)"| DS["DrawSession"]
    DS --> Ga["group A (≤6)"]
    DS --> Gb["group B (≤6)"]
    Ga --> C1["cards 1..6"]
    Gb --> C2["cards 7..8"]

    classDef win fill:#1f6f43,stroke:#0f3,color:#fff
    class Vb win
```

**前端 Canvas：** `/branches + /timeline + /status` → `buildBranchCanvas` → reactflow 节点/边（input/run/checkpoint/branch_action/variant_group/variant_run），纯**只读视图**，手动拖拽只存本地布局、不污染 run index。

---

## 5. 当前版本使用流程（SOP）

> 以下为**已实现版本**的真实操作流程。基础闭环 + 分支 + 变体已在 UI 可用；批量抽卡当前需经 API。

### 5.1 SOP① 基础单图生成闭环

| 步 | 操作（用户/前端） | 后端 / Agent |
|----|------------------|--------------|
| 1 | 上传 PNG，选模型、LLM 模式、策略预设 | — |
| 2 | 点击 **Run** | `POST /png-shader/run`（multipart：`image` + `input_spec_json`{model/quality/candidates} + 可选 `seed_glsl`） |
| 3 | 拿到 `run_id`，进入 Processing | 创建 run_id → 写 run_index(pending) → 启动后台 worker |
| 4 | **轮询**实时进度 | `GET /png-shader/status/{run_id}` 返回 partial：候选 scoreboard → selected → 精修迭代 → 最终 |
| 5 | 查看结果 | ShaderPreview 实时 WebGL 渲染 + 候选对比 + 指标 + 精修历史 + VLM 评审 |
| 6 | （可选）中途停止 | `POST /runs/{run_id}/stop` |

**关键产物：** `selected_shader.glsl`、`scoreboard.json`、`objective_metrics.json`、`refinement_summary.json`、`reference_input.png`。

### 5.2 SOP② 从 checkpoint 分支精修（V1）

1. 选 checkpoint（`candidate:* / refinement:iter:n / final:selected`）。
2. 填 **feedback**（自然语言目标）+ **mode**（`continue` / `refine` / `polish`）+ 可选 **locks**（保布局/色板/背景/仅小改）。
3. `POST /runs/{run_id}/branch-refine` → 创建 **child run**（seed = 该 checkpoint 的 GLSL，父 run 不被覆盖）。
4. 前端自动切到 child run，复用轮询。
5. **定向接受**：`force_first_refinement_iteration` 强制首轮定向精修；VLM `judge_directed_pairwise` 按用户目标判定，允许「语义更优但小幅降分」被接受。

### 5.3 SOP③ 变体探索（V3.1 / V3.2）

1. 选 checkpoint → 填 feedback + **variant_count(2–6)** + **diversity**（low/medium/high）。
2. `POST /runs/{run_id}/explore-variants` → 一次产 N 个**策略变体**（6 模板：conservative / semantic / lighting_color / detail_texture / structure_form / alt_technique；并发上限 2）。
3. 轮询 `GET /variant-groups/{group_id}`：变体逐个完成，失败变体不阻塞其他。
4. 比较 → **选 winner**（`/winner`）/ 评分（`/ratings`）/ 停止（`/stop`）。
5. 从 winner 继续优化（winner 切为 active run）。

### 5.4 SOP④ 批量抽卡（V3.5，后端 API）

1. 选 checkpoint → feedback + **requested_count(2–12)** + diversity。
2. `POST /runs/{run_id}/draw-session` → `plan_draw_batches` 拆成多个 ≤6 的 VariantGroup（如 12→[6,6]，7→[4,3]）。
3. 轮询 `GET /draw-sessions/{draw_id}`：卡片逐个亮。
4. 卡片操作：`/draw-more` 追加（不覆盖）· `/redraw` 单卡重抽（保留原卡，replacement 可追溯）· `/cards/{run_id}/event` 收藏/淘汰/打标签（为 V4.5 融合预留 `use_as_fusion_base` / `use_as_region_source`）。

### 5.5 端到端时序

**图 8 · 时序图**

```mermaid
sequenceDiagram
    autonumber
    participant U as 用户/前端
    participant API as FastAPI 路由
    participant W as 后台 worker / Agent 内核
    U->>API: POST /run (image, input_spec)
    API->>API: create run_id + index(pending)
    API->>W: _start_pipeline_worker
    API-->>U: { run_id, status: running }
    W->>W: Layer1 preprocess→candidates→scoring→selection
    loop 轮询
        U->>API: GET /status/{run_id}
        W-->>API: publish_partial (候选 / selected / 每轮精修)
        API-->>U: partial (scoreboard / refinement_history)
    end
    W->>W: Layer2 优化 → LLM 精修 → VLM final gate
    W->>API: index(updated) + 落盘 artifacts
    U->>API: GET /status → completed
    Note over U,W: 进阶：/branch-refine · /explore-variants · /draw-session<br/>均复用同一内核, 注入不同人类意图
```

---

## 6. 最终方案架构（补齐 V4.1 → V4.5）

最终方案**不动 Layer 1 的图结构**，而是把 Layer 3 从「重复调度器」升级为「**控制 + 学习 + 融合**」层，并给 Layer 2 增加新的输入通道、裁判与优化参考。

**图 9 · 最终方案分层**

```mermaid
flowchart TB
    WIN["V3 winner/rating · V3.5 card-event"]
    subgraph L3F["Layer 3' · 控制 / 学习 / 融合 (最终方案新增)"]
        C1["V4.1 结构化约束<br/>HumanConstraintSpec → constraint_notes + constraints.json"]
        C2["V4.2 区域/蒙版<br/>RegionConstraint + region_metrics → 硬否决"]
        C3["V4.3/4.4 偏好记忆 ★跨run<br/>events.jsonl → profile.json → preference_notes / 排序"]
        C4["V4.5 局部融合<br/>composite_target.png + fusion run"]
    end
    subgraph L2F["Layer 2 · LLM/VLM (裁判扩容)"]
        J["+ judge_directed_region_pairwise (区域级)<br/>+ judge_fusion_pairwise (匹配 composite_target)"]
    end
    L1F["Layer 1 · 确定性核心 (完全不变)"]

    WIN -.->|"回填偏好事件"| C3
    L3F -->|"约束notes / 区域veto / 偏好notes / 融合notes + composite_target"| L2F
    L2F --> L1F

    classDef new fill:#6b3fa0,stroke:#c9f,color:#fff
    classDef core fill:#1f6f43,stroke:#0f3,color:#fff
    class C1,C2,C3,C4,J new
    class L1F core
```

### 6.1 V4.1 结构化约束
`HumanConstraintSpec`：`locks`（preserve_layout/palette/background…）+ `targets`（brightness keep/increase/decrease…）+ `edit_strength[0,1]`。→ 转 prompt notes，落 `constraints.json`，由 `branch-refine` / `explore-variants` 接收。

### 6.2 V4.2 区域/蒙版约束
`RegionConstraint`（归一化 rect，mode=modify/protect）+ `region_metrics.py`（只算区域内 SSIM/MSE/delta）。**protect 区域显著变差 → 硬否决**，即使全局分提升也拒绝。端点 `POST /runs/{id}/region-mask`。

### 6.3 V4.3 偏好事件 / 档案
`PreferenceEvent` → `events.jsonl`（append-only）→ **确定性 rebuild** → `profile.json`。从 V3 winner/rating、V3.5 card-event **回填**。CRUD/rebuild/clear 端点；`enabled=false` 时不注入。

### 6.4 V4.4 偏好辅助生成 / 排序
`build_preference_notes(profile)` 把正/负偏好、偏好变体标签转 prompt notes；profile 快照随请求落盘；变体可加 `preference_score` 辅助排序（不替用户自动选 winner）；可选 LLM summarizer（保留 raw events）。

### 6.5 V4.5 局部融合（核心跃迁）

**图 10 · 融合数据流**

```mermaid
flowchart LR
    subgraph SRC["抽卡池 (V3.5 DrawSession)"]
        BASE["base 卡<br/>全局结构 + seed shader"]
        S1["source 卡 A<br/>(局部优点: 水面)"]
        S2["source 卡 B<br/>(局部优点: 云)"]
    end
    BASE --> COMP
    S1 -->|"region rect + strength"| COMP
    S2 -->|"region rect + strength"| COMP
    COMP["image_composite.py<br/>→ composite_target.png + region masks"] --> FRUN
    FRUN["fusion run<br/>seed = base GLSL<br/>优化参考 = composite_target<br/>judge_fusion_pairwise 把关"]
    FRUN --> OUT["统一连续 shader (新 child run)<br/>lineage: fusion_id / base_run_id / source_run_ids"]

    classDef tgt fill:#7a4d1f,stroke:#fb6,color:#fff
    class COMP,FRUN tgt
```

从抽卡池选 **base 卡** + 多张 **source 卡** + 区域 → `image_composite.py` 生成 **`composite_target.png`** → **fusion run**：seed = base 的 GLSL，但**优化参考改为 composite_target**，用区域约束 + `judge_fusion_pairwise` 保证「融成一个连续 shader、不是拼贴」。端点：`/fusions` create / `composite-target` / `run`。

**从「选一个最好」到「合成一个更好」** —— 这是最终方案在 Agent 技术上的最大跃迁。

---

## 7. 两版对比表

| 维度 | 已实现版（V1–V3.5） | 最终方案（+V4.1–V4.5） |
|------|---------------------|------------------------|
| Agent 内核（Layer 1+2） | LangGraph 4 节点 + LLM/VLM 闭环 | **不变** |
| 用户控制粒度 | 一句自然语言 feedback（+ 简单 locks） | 结构化约束(锁/目标/强度) + **区域级**指令 |
| 接受判定依据 | 全局指标 + VLM pairwise / directed | + **区域硬否决** + 偏好提示 + 融合裁判 |
| 优化参考(target) | 仅原始 PNG | 原始 PNG **+ 合成 composite_target.png** |
| 记忆 / 学习 | 无（winner 仅记事件） | **偏好 profile 跨 run 回流** prompt 与排序 |
| 编排拓扑 | 分支 / 变体 / 抽卡树 | + **融合 DAG**（多 source → 一个统一 shader） |
| VLM 裁判 | pairwise / directed / rubric | + region pairwise / fusion pairwise |
| 持久化 | run_index.jsonl + 组/会话事件 | + constraints.json / preferences / fusions |
| 前端 | Branch Canvas（run/checkpoint/变体） | + 约束/偏好/融合节点与编辑器 |

---

## 8. 演进路线图与现状

**图 11 · 版本演进与状态**

```mermaid
flowchart LR
    subgraph DONE["已实现 ✅"]
        M0["M0 Preflight"] --> M1["M1 V1.1 单分支"] --> M2["M2 V1.2 定向接受"]
        M2 --> M3["M3 V2 run index"] --> M4["M4 V2 workspace"] --> M5["M5 V2.1 Canvas"]
        M5 --> M6["M6 V3.1 变体后端"] --> M7["M7 V3.2 变体UI"]
    end
    M7 --> M8["M8 V3.5 抽卡<br/>🟡 后端✅/前端进行中"]
    subgraph FINAL["最终方案 ⬜"]
        M9["M9 V4.1 约束"] --> M10["M10 V4.2 区域/mask"]
        M10 --> M11["M11 V4.3 偏好事件"] --> M12["M12 V4.4 偏好辅助"]
        M12 --> M13["M13 V4.5 局部融合"]
    end
    M8 --> M9

    classDef done fill:#1f6f43,stroke:#0f3,color:#fff
    classDef wip fill:#7a6a1f,stroke:#fd6,color:#fff
    classDef todo fill:#444,stroke:#888,color:#ccc
    class M0,M1,M2,M3,M4,M5,M6,M7 done
    class M8 wip
    class M9,M10,M11,M12,M13 todo
```

**核心原则（路线图既定）：** 先可运行闭环再复杂 UI；先数据可追溯再多分支；先结构化输入再 mask 与偏好；每版可独立验收；不跳测试门禁（后端单测优先，前端至少 `npm run build`）。

---

## 9. 附录

### 9.1 端点清单（人在环）

| 方法 | 路径 | 用途 | 版本 |
|------|------|------|------|
| POST | `/png-shader/run` | 初始 PNG→Shader（multipart） | 核心 |
| GET | `/png-shader/status/{run_id}` | 轮询 run 状态（含 partial） | 核心 |
| POST | `/png-shader/runs/{run_id}/stop` | 停止运行 | 核心 |
| GET | `/png-shader/runs/{run_id}/checkpoints` | 列可分支 checkpoint | V1 |
| POST | `/png-shader/runs/{run_id}/branch-refine` | 从 checkpoint 派生 child run | V1 |
| GET | `/png-shader/runs/{run_id}/timeline` | run 时间线（带 artifacts） | V2 |
| GET | `/png-shader/runs/{run_id}/branches` | 分支树 | V2 |
| PATCH | `/png-shader/runs/{run_id}/metadata` | 改 title/favorite/tags | V2 |
| GET | `/png-shader/runs/{run_id}/artifacts/{id}` | 取 PNG/JSON/GLSL artifact | V2 |
| POST | `/png-shader/runs/{run_id}/explore-variants` | 一次产 N 个变体 | V3.1 |
| GET | `/png-shader/variant-groups/{group_id}` | 变体组状态 | V3.1 |
| POST | `/png-shader/variant-groups/{id}/winner` | 设 winner | V3.1 |
| POST | `/png-shader/variant-groups/{id}/ratings` | 评分 | V3.1 |
| POST | `/png-shader/variant-groups/{id}/stop` | 停止变体组 | V3.1 |
| POST | `/png-shader/runs/{run_id}/draw-session` | 批量抽卡(2–12) | V3.5 |
| GET | `/png-shader/draw-sessions/{draw_id}` | 抽卡会话状态 | V3.5 |
| POST | `/png-shader/draw-sessions/{id}/draw-more` | 追加抽卡 | V3.5 |
| POST | `/png-shader/draw-sessions/{id}/redraw` | 单卡重抽 | V3.5 |
| POST | `/png-shader/draw-sessions/{id}/cards/{run_id}/event` | 卡片事件 | V3.5 |

### 9.2 关键模块清单

| 层 | 模块 | 职责 |
|----|------|------|
| 内核 | `pipeline/graph.py` | LangGraph 编排 + post-pipeline 主流程 |
| 内核 | `candidates/*.py` | 6 策略候选生成 |
| 内核 | `dsl/{schema,compiler,renderer,validator}.py` | DSL 定义与确定性 DSL→GLSL 编译 |
| 内核 | `metrics/{compute,quality_router}.py` | 客观指标 + 质量路由 |
| 内核 | `pipeline/{optimizer,glsl_optimizer,revision,residual_layers}.py` | 参数优化 / 修订 / 补层 |
| 内核 | `pipeline/{refinement,glsl_refinement}.py` | LLM 闭环精修（DSL/GLSL） |
| 内核 | `pipeline/seed_glsl.py` | Seed-GLSL 适配路径 |
| 内核 | `llm/{client,vlm_judge,model_resolver}.py` | LLM 调用 / VLM 评审 / 模型解析 |
| 人在环 | `pipeline/run_index.py` | run 血缘记录 + JSONL 持久化 + 分支树 |
| 人在环 | `pipeline/checkpoints.py` | checkpoint 列举/解析 + timeline |
| 人在环 | `pipeline/human_feedback.py` | 人类反馈 → prompt notes |
| 人在环 | `pipeline/variant_groups.py` | 变体策略 + 组持久化/状态/winner |
| 人在环 | `pipeline/draw_sessions.py` | 批量抽卡批次规划 + 会话 |
| 最终方案 | `human_constraints.py` / `region_metrics.py` / `preferences.py` / `fusion_plans.py` / `image_composite.py` | 约束 / 区域 / 偏好 / 融合（待实现） |

### 9.3 术语

- **Checkpoint：** run 内可作为分支起点的状态（候选 / 精修迭代 / 最终）。
- **Variant Group：** 一次从同一 checkpoint 产生的 N 个策略变体集合。
- **Draw Session：** 抽卡会话，把多变体产品化为可追加/重抽/收藏的批次。
- **Directed Acceptance：** 带用户目标的接受策略，允许语义更优但小幅降分被接受。
- **Composite Target：** 由多张抽卡结果局部融合而成的目标图像，作为融合 run 的优化参考。
- **Hard Veto：** protect 区域显著退化时的硬否决，凌驾于全局分提升。
```
