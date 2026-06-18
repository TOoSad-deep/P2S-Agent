# P2S-Agent 架构与实现全面审查报告

> **日期：** 2026-06-18　**对应分支：** `main`（tip `7bf47af`）
> **方法：** 14 路并行深度审查（按子系统 + 并发/安全/前后端契约 3 个横切视角）→ 对每个 high/medium bug 由独立 skeptic 重读真实代码做对抗式验证 → 跨层综合。53 个审查 agent，2.67M tokens。
> **地面真值：** 审查时 **997 个后端单测全绿**（10s），故所有问题都落在测试覆盖之外的边界（并发、安全、资源生命周期、前后端契约、畸形输入）。
> **覆盖范围：** 后端 ~19k 行 Python（62 模块）+ 前端 ~15.5k 行 TS/TSX（54 文件）+ 35 个 API 端点。

---

## 一、总体评估

这是一个**工程素养相当成熟的研究级系统**，绝非原型代码。三层架构（确定性 LangGraph 内核 / 单调最优的 agentic 后处理 / 纯函数式的人在环模块）是真正的优势：每一步 accept 都重渲染重打分，所以没有阶段能让候选回退；防御性模式普遍且大多正确（路径穿越防护有单测、密钥与 `/status` 隔离、VLM 失败降级到客观指标地板、候选失败保留为记录而非丢弃、JSONL append-only + 原子 `os.replace` 支持重启恢复）。纯函数 helper（模型/布局/规划）测得很好。

**系统性脆弱点集中在三个反复出现的模式上**，而非零散的一次性 bug：
1. 内存 run store 名为 LRU 实为 FIFO，且写路径绕过限容 setter；
2. 校验只查结构不查值；
3. 前后端契约手工镜像、生命周期状态词汇漂移。

外加一致的**资源泄漏**与**静默失败**两类模式。对一个本地单用户工具而言这些都不致命，但 store 淘汰 bug、DSL 校验崩溃 cluster、确定性截图竞态是**真实的正确性缺陷**，在任何多用户/规模化使用前应当修复。

---

## 二、架构维度评估

| 维度 | 评级 | 要点 |
|---|---|---|
| 三层分离（内核/agentic/人在环） | 🟢 强 | 分层清晰、可测试、易推理；最大亮点 |
| 错误隔离 / 优雅降级 | 🟢 强 | 逐迭代 try/except、best-effort 持久化、VLM→None 地板、单调 best 追踪、pool 永不丢 fallback |
| 持久化 / 重启恢复 | 🟡 中 | JSONL+原子快照可恢复，但日志无限增长、每次读 O(n) 重折叠、无 schema 版本、读改写有丢更新竞态 |
| 安全 / 输入处理 | 🟡 中 | 路径穿越防护与密钥隔离是真防御并有单测；但 `0.0.0.0` 零鉴权、上传无上限、客户端可控 LLM `base_url`（SSRF）是真实暴露面 |
| 输入校验纪律 | 🔴 弱 | 只校验结构不校验**值**；裸 `int()` 强转返回 500 而非 422 |
| 并发 / 共享状态 | 🔴 弱 | 单 dict 单锁，**文档说 LRU 实际 FIFO**；终态写绕过限容 setter——最严重系统性缺陷 |
| 前后端契约 | 🔴 弱 | 手工镜像 TS 接口、无 codegen；生命周期词汇漂移导致 UI 冻结、轮询不终止 |
| 资源生命周期（FD/GPU/线程/临时文件） | 🔴 弱 | 反复泄漏：`httpx.Client` 不关、截图临时文件不删、每次 GLSL 编辑新建并泄漏 WebGL context、daemon 线程 fire-and-forget |
| 可观测性 | 🔴 弱 | 静默丢弃为主：pool 吞异常、JSONL 跳过坏行无日志、淘汰/模型未命中无信号 |
| 可测试性 | 🟡 中 | 纯函数 helper 测得很好（59 个 canvas 测试、golden 渲染测试），但最高风险路径（轮询竞态、optimizer 接受/停止、畸形 LLM、打分同步不变式）恰恰没测 |

---

## 三、三大系统性风险（跨多个子系统，最值得先修）

### 🔴 风险 1：内存 store 名为 LRU 实为 FIFO，且写路径绕过限容 setter → 运行中的 run 被淘汰

**横跨：** router-core / human-loop-core / xcut-concurrency / xcut-contract

`_store_run`（`backend/app/routers/png_shader.py:180`）按**插入顺序**淘汰且读/更新时不重排序；而终态/queued 写直接 `_run_store[run_id]={...}`（已核实 `png_shader.py:377`），**同时绕过限容和淘汰守卫**。一旦 100 条满（一个 12 卡抽卡 session 就能触达），一个**仍在运行**的早期任务会被淘汰：

- `GET /status` 404、`POST /stop` 404、`/branch-refine` `/explore-variants` `/draw-session` 全部在父查找处 404；
- `stop_requested` 再也送不到 worker（协作式取消通道被切断）；
- `selected_glsl` / `refinement_history` / 控制标志等实时字段丢失；
- `_run_models`（`png_shader.py:166`）独立淘汰，导致活跃 run 静默回退到 `.env` 默认模型（连 API key 都换）。

**一个修复解决五个独立 finding**：改 `OrderedDict` + 读/写 `move_to_end` + 优先淘汰终态 run（豁免 running/queued）+ 所有写都走限容 setter。

### 🔴 风险 2：校验只查结构不查值 → schema 合法的输入要么崩溃 worker、要么静默偏离

**横跨：** dsl / candidates / langgraph-core / optim-refine

`validate_dsl`（`backend/app/dsl/validator.py`）只查键存在和枚举；编译器/渲染器随后无条件信任 `params/transform/stop` 的值。`_generate_param_defines`（`backend/app/dsl/compiler.py:91`）在逐层 try/except **之外**被无保护调用（已核实 `compiler.py:474`）：

- `center=0.5` / `radius='big'` / 非有限浮点 / `sides<=0` / 零 scale → 抛 `TypeError/ValueError` **杀死精修/打分 worker 线程**；
- 或 `success=True` 编译出坏 GLSL（`inf`/`nan` 字面量、除零），**与渲染器的带守卫路径偏离**——打破打分所依赖的"渲染器镜像编译器"契约。
- `run_png_shader_pipeline` 对非 HTTP caller 还跳过 `validate_input_spec`。

**一层值级校验（复用编译器数值 helper）+ 给 `compile_dsl` 包一层保证总返回**即可关闭整个 cluster。

### 🔴 风险 3：生命周期状态词汇在后端发射端与前端终止逻辑之间漂移

**横跨：** xcut-contract / frontend-state / frontend-canvas / router-core

手工镜像 TS 接口、无共享枚举，前端终止判断硬编码 `=== 'running'`：

- 后端给变体/抽卡 child 发 `queued`，前端当**终态** → 选中的卡片永远冻结在 queued 快照（`frontend/src/hooks/usePngShader.ts:783`）；
- 融合 status 永远不离开 `running`（worker 从不关闭 `FusionPlanRecord`，`png_shader.py:3038`）→ 前端 2s 融合轮询**永不终止**（每个融合一个后台请求泄漏）。

**修法：** 前端引入共享 `TERMINAL/NON_TERMINAL` 集合 + worker 在完成时关闭融合生命周期；根因修法是 Pydantic `response_model` + 从 OpenAPI 生成 TS 类型。

### 其余系统性风险（medium）

- **读改写无隔离锁（last-writer-wins 丢数据）：** `draw_more`/`redraw_card` 丢卡片 id；`set_variant_winner`/`run_fusion`/`create_composite_target` 互相覆盖 status；`rebuild_profile` 覆盖手工偏好；`/status` 返回活的可变 dict → 序列化竞态。
- **静默失败 / 丢遥测：** pool 吞异常、JSONL 跳坏行、`constraint_score` 算了存了却从不被读、融合羽化 mask 小区域静默归零、前端侧栏错误污染共享 run-error 通道。
- **无界增长 / 无背压：** 无全局 worker 上限（仅变体 child 共享 `Semaphore(2)`）；上传全量入内存、preprocess 降采样前物化全分辨率像素；JSONL 永不压缩、每次读 O(总 run 数) 重折叠；VLM cache 与 per-call httpx/截图无界泄漏。
- **资源句柄 per-operation 创建从不确定性释放：** `httpx.Client` 无 close/`__del__`/context-manager；截图 mktemp 不 unlink；`ImageDiffPanel` 每次 GLSL/滑块变更新建 `WebGLRenderer`+context+texture，耗尽浏览器 ~16 context 上限、静默变暗。
- **排序分与绝对质量分混淆：** `node_selection` bump（`graph.py:328`）污染 `_should_run_refinement`；optimizer 把渐变方向 clamp 到 [0,1] 致退化——"一个数表两个意思"。
- **网络绑定 server 零鉴权 + 多个放大原语：** 绑 `0.0.0.0:8001`，配合无界上传、全分辨率物化、客户端可控 `base_url`（SSRF）、per-run api_key 计费。

---

## 四、已验证 Bug 清单

> 经对抗式 skeptic 重读真实代码后：**1 个 high + 22 个 medium**（已验证）+ 22 个 low（未单独验证）。

### 唯一的 high 级 bug

- **确定性截图竞态** — `startRendering()`（`frontend/src/lib/shader-renderer.ts:146-166`）的 rAF 循环每帧无条件写 `u_time = clock.getElapsedTime()`，`setTime()`/`__setShaderTime` 最多存活一帧。**Playwright 截图捕获的是时钟随机读数 → 后端 GLSL 候选打分/预览不可复现**，静默损害打分保真度。修法：加 `frozenTime` 标志使 animate 跳过更新 u_time，或截图路径不启动 rAF、仅按需单帧渲染。

### medium bug 按主题归并（均已核实，附 file:line）

**A. 共享状态读改写丢更新 / 序列化竞态**

| Bug | 位置 | 修法 |
|---|---|---|
| `/status` 返回活的可变 dict，worker 中途 `update()` → "dict changed size during iteration" 500 | `png_shader.py:618-625` | 锁内 `copy.deepcopy` |
| `draw_more`/`redraw_card` 丢失追加卡片 id；winner/fusion/composite 互相覆盖 status | human-loop-core/final 多处 | 按 id keyed-lock + 事件日志为真相源 |
| `rebuild_profile` 静默丢弃手工 PATCH 的正/负偏好和 `default_locks` | `preferences.py:295-383` | merge base_profile 而非替换 |

**B. DSL 编译器 / 渲染器偏离与静默数据丢失**

| Bug | 位置 | 说明 |
|---|---|---|
| 多边形 `sides<=0` 编译除零 GLSL 且 `success=True` | `compiler.py:227-228` | 渲染器有 `max(3,...)` 守卫 → 结构性偏离 |
| 非有限数 → `inf`/`nan` GLSL 字面量 | `compiler.py:67-69` (`_float`) | 编译器不清洗不标记 |
| 8 位 RGBA hex（`#112233ff`）→ 编译器与渲染器都静默塌成白色 | `compiler.py:51-58` / `renderer.py:309-318` | 静默颜色数据丢失 |
| 零 scale 变换 `p /= vec2(0.0)`，渲染器有守卫 | `compiler.py:260-261` | 预览/指标偏离 |

**C. 候选生成对畸形输入的脆弱**

| Bug | 位置 | 说明 |
|---|---|---|
| `_merge_similar_colors` 在低色数图上 `IndexError` 崩溃 | `decompose.py:46`（调用自 209） | 被 pool broad except 吞掉，对简单少色图静默丢候选 |
| LLM 渐变归一化可把 stop 降到 2 以下 → 校验失败 | `llm_scene.py:636-640` | 归一化器反而产出非法 DSL |
| `_extract_glsl` 把前导注释当 shader code 捕获 | `llm_scene.py:901-917` | 无效 GLSL，编译失败 |

**D. 优化 / 精修打分语义混淆**

| Bug | 位置 | 说明 |
|---|---|---|
| VLM 近分 bump 抬高 `final_score`，可错误跳过精修 | `graph.py:327-332` | 排序分被当质量门控分，静默质量回退 |
| optimizer 把渐变方向 vec2 clamp 到 [0,1] | `optimizer.py:88-93, 371-372` | `[-1,0]` 变退化 `[0,0]`，静默锁定错误朝向 |
| GLSL 精修在渲染失败但分数 >0 时接受幽灵改进 | `glsl_refinement.py:282-300` | — |

**E. 前端状态 / 画布**

| Bug | 位置 | 说明 |
|---|---|---|
| `stopRun` 在非 ok HTTP 响应时 `stopPending` 永久卡 true | `usePngShader.ts:638-649` | fetch 不对 4xx/5xx 抛错 → Stop 按钮永久禁用 |
| 轮询回写覆盖本地策略编辑（last-writer race） | `usePngShader.ts:476-486` | 滑块编辑被服务器旧值弹回甚至丢失 |
| 切换树时把旧树布局覆写到新树存储键 | `BranchCanvasWorkspace.tsx:201-212` | effect 执行顺序问题 |
| 融合 poller 对 idle draft/target_ready 永不停，每 2s 轮询 | `BranchCanvasWorkspace.tsx:793-818` | TERMINAL 集合漏掉 idle 非 running 态 |

**F. 契约 / 安全**

| Bug | 位置 | 说明 |
|---|---|---|
| `region_mask` 返回的 `mask_url` artifact 端点无法服务 | producer `png_shader.py:2638-2639` vs consumer `2177-2196` | 始终 404，死契约 |
| 非整数 `variant_count`/`card_count` → 未捕获 `ValueError` → 500 而非 422 | `png_shader.py:1073, 1297, 1556` | 违反端点自身校验契约 |
| 上传无大小上限，全量入内存 | `png_shader.py:539` | 无鉴权 server 内存耗尽 DoS |
| preprocess 在降采样前物化全分辨率像素 list | `preprocess.py:48-57` | 配合无界上传 → worker 分配数 GB |
| FE 把 `queued` 当终态，切到 queued 变体/抽卡 child 冻结结果 | `usePngShader.ts:783, 476` | 卡片永远停在 queued 快照 |
| 融合 status 永不离开 `running`，FE 融合轮询永不终止 | `png_shader.py:3038` / FE `BranchCanvasWorkspace.tsx:797` | 每融合后台请求泄漏 |

### 22 个 low bug（一行摘要）

模型淘汰致 child 静默换模型（`png_shader.py:166-175`）· 被淘汰变体 child 以畸形条目复活（`250-252`）· `selected_metrics` 别名可变 dict（`graph.py:437`）· pipeline 入口不校验 caller input_spec（`graph.py:1043`）· `compile_dsl` 违反"总返回"契约崩 worker（`compiler.py:474`）· 零/退化 scale 致预览偏离（`compiler.py:260`）· 两个 optimizer 从 worker 线程 seed **全局** random（`optimizer.py:297`）· 死优化键，radial 渐变 center 从不被优化（`optimizer.py:33`）· 协调下降日志记错 score_before（`optimizer.py:400`）· GLSL 精修幽灵改进（`glsl_refinement.py:282`）· `residual_layers` 无守卫索引 `layers` 致 KeyError（`residual_layers.py:106`）· `httpx.Client` 不关（`client.py:91`）· 截图临时文件不删（`browser_render.py:61`）· `build_branch_tree` 静默丢父环节点（`run_index.py:306`）· 侧栏错误污染共享 run-error 通道（`usePngShader.ts:961+`）· 抽卡根节点 input→root 边悬空（`branchCanvasModel.ts:308`）· `defaultTexture` 泄漏（`shader-renderer.ts:69`）· `updateShaderParam` 正则未锚定致错误重写（`glsl-parser.ts:280`）· PreviewDock 冗余重渲染闪烁（`PreviewDock.tsx:54`）· queued 变体 stop 阻塞在 `semaphore.acquire()` 而非立即取消（`png_shader.py:246`）· `_run_models` 独立淘汰丢模型/API key（`png_shader.py:166`）· 非数字 variant/card_count → 500（`png_shader.py:1073`）

---

## 五、不足（87 项）

**按领域：** robustness 39 · maintainability 12 · observability 11 · testing 10 · performance 9 · architecture 3 · ux 3
**按影响：** high 4 / medium 36 / low 47

**4 个 high-impact 不足：**

1. **所有端点零鉴权**，且 server 绑 `0.0.0.0:8001`——任何能到端口的客户端都能提交 run、写 artifacts/偏好、用你的 API key 计费。CORS 单源不是 authz 控制。
2. **validator 只校验结构不校验值**（同风险 2）。
3. **渲染器无 WebGL context-loss 处理**——三.js 默认丢失后静默黑屏，无 `webglcontextlost/restored` 监听。
4. **静默失败模式遍布**——pool 吞异常、JSONL 跳坏行无日志、`constraint_score`（protect 区域硬否决）算了存了**却从不被任何 gate 读取**，宣传的约束是 no-op。

---

## 六、优化点（67 项：43 低成本 / 22 中 / 2 高）

**高性价比（低成本、立即收益）：**

- 锁内快照 store 条目 → 消除 `/status` 竞态
- 集中 `int/float` 强转 helper → 一类 500 变 422
- **preprocess 边缘/纹理循环前先降采样** → 大图延迟大幅下降
- decompose 调色板长度硬化 → 修崩溃，让其真正能跑简单图标
- preprocess 特征加 coalescing 数值访问器 → 消除一整类 `None` 引发的 `TypeError`
- 编译器集中数值/颜色格式化 → 一处修掉 inf/nan + 8 位 hex
- 多边形 sides 与零 scale 在编译器/渲染器一致处理 → 恢复指标 vs 预览一致
- DSL 协调下降加"无改进早停"（镜像 glsl_optimizer）→ 砍无谓 compile+render+score
- optimizer 用本地 `random.Random(seed)` 而非 seed 全局模块 → 去掉并发 worker 进程级副作用
- 按 `ModelConfig` 缓存复用 `BaseAgent`+`httpx.Client` → 消除连接泄漏、启用 keep-alive
- 黑屏检测改基于方差 → 不再误杀合法暗色渲染
- VLM cache key 加模型身份 + 限容 → 防并发多模型裁判串味
- browser teardown 用 `try/finally` → 止住超时时 Chromium 进程泄漏
- 统一 JSON 抽取 helper（GLSL/DSL 两路）· SDF 库 header 预计算一次 · preprocess.json 在全部派生键设置后再落盘

---

## 七、修复优先级与路线图

### Top 10 优先级（最重要在前）

| # | 类别 | 事项 |
|---|---|---|
| 1 | bug | `_run_store` 改真 LRU + 豁免非终态 + 所有写走限容 setter（一改解决 5 个 finding） |
| 2 | bug | `validate_dsl` 加值级校验 + 包裹 `compile_dsl` 保证总返回 |
| 3 | bug | shader-renderer `frozenTime` 标志（恢复确定性截图，唯一 high 前端 bug） |
| 4 | bug | 生命周期词汇对齐（FE `queued` 当非终态 + worker 关闭融合 status） |
| 5 | bug | VLM 近分 bump 与精修门控分离（读未抬高的 `quality_router` 分） |
| 6 | risk | 上传守卫层（大小 + content-type + magic bytes）+ 分析前降采样 |
| 7 | shortcoming | `int()` 强转加守卫返回 422 而非 500（`_coerce_int` helper） |
| 8 | bug | keyed-lock 注册表治读改写竞态 + `/status` 锁内 snapshot |
| 9 | optimization | 修资源泄漏（复用 httpx.Client / 删截图 / WebGL context 复用编译） |
| 10 | shortcoming | `constraint_score` 接进真实 gate（或删掉）+ 暴露被丢弃的融合区域 |

### 路线图

**Phase 1 — 稳定性与正确性（先修崩溃/冻结/静默打分腐蚀）**
- `_run_store` → `OrderedDict` + `move_to_end` + 优先淘汰终态 + 全写路径走限容 setter
- `validate_dsl` 值级校验（center 双数 list；radius/size/ab/thickness/opacity 有限数；sides int≥3；stop.position/transform 数值）+ 包裹 `compile_dsl`；`_float` 钳非有限值、解析 8 位 hex、钳多边形 sides 与零 scale 对齐渲染器
- shader-renderer 加 `frozenTime`
- FE 引入共享 `TERMINAL/NON_TERMINAL`，`queued`/`acquired` 当非终态；worker 完成时关闭 `FusionPlanRecord`
- 分离 VLM tie-break key 与 `final_score`
- `_coerce_int` helper → 422；`run_png_shader_pipeline` 内调用 `validate_input_spec`
- 修 `stopRun` 非 ok 时 `stopPending` 卡死；修布局覆写键 bug

**Phase 2 — 硬化与隔离（并发竞态、资源泄漏、输入/安全）**
- 跨 load-mutate-save span 加 per-id keyed-lock 注册表；卡片成员/winner 移入 append-only 事件日志为真相源
- `/status` `/checkpoints` `/timeline` 锁内 deepcopy 后再返回/序列化
- 上传守卫（Content-Length 413 + content-type 白名单 + `Image.verify`）+ preprocess 分析前降采样
- 资源生命周期：按 `ModelConfig` 缓存/关闭 `httpx.Client`；截图 `NamedTemporaryFile` 并 unlink；`ImageDiffPanel` 原地重编译而非重建 `WebGLRenderer`；dispose `defaultTexture`；加 `webglcontextlost/restored` handler
- optimizer 参数加 bound-type 标签（方向位保符号、radial center 真正被优化）；用本地 `random.Random(seed)`
- 自定义模型 `base_url` 白名单/scheme 校验（防 SSRF）；集中 `validate_safe_id`；`seed_glsl`/`feedback`/`input_spec_json` 加长度上限
- LLM DSL 归一化后自愈（≥2 渐变 stop、注释安全 `_extract_glsl`、调色板长度安全 decompose、coalescing 数值访问器）

**Phase 3 — 性能、规模与可观测**
- 全局 worker 信号量 / `ThreadPoolExecutor` + 线程注册表（覆盖 `/run` 与 branch-refine，非仅变体 child）；饱和返回 429；关停时标记在途 run
- `load_run_index` 按 mtime/size 缓存（或增量尾折叠）+ append-only 日志周期压缩/轮转 + fold 加终态粘性
- pool 的裸 except 换结构化 per-source 错误上报 scoreboard；记录跳过的坏 JSONL 行；加淘汰活跃 run / 模型未命中告警与指标
- DSL 协调下降加早停/收敛检测 + accept epsilon（或 pairwise judge veto）；边缘/纹理预处理循环前降采样
- `constraint_score` 接进 protect 区域 gate（或删除）；`build_composite_target` 返回 per-region applied/skipped；feather 每 rect 加上限
- memo 化 `PngShaderProvider` context value；拆分 1352 行 god-hook；字段级/revision-gated 策略回写以止住滑块弹回与丢失编辑
- 补齐缺失测试面：VLM 打分同步不变式、optimizer 接受/停止/方向符号、畸形 LLM/平调色板/None preprocess、轮询生命周期竞态、合并模型+多 overlay 布局路径

**Phase 4 — 契约与架构（若要多用户/规模化）**
- 高价值形状采用 Pydantic `response_model` + OpenAPI→TS codegen，根除手工镜像漂移
- 持久化记录与 JSONL 事件加 schema 版本字段
- 部署超出 localhost 前加鉴权/授权层（或绑 loopback），考虑 per-run api_key 计费与 artifact/偏好写面
- 重构 selection 使 output-kind 偏好与 VLM tie-break 集中在一处可审计
- 强化静态 GLSL 闸门（小写未声明符号、多声明 uniform）；文档化 renderer-vs-GPU grain-hash 确定性差距

---

## 附录：审查方法论与计数

- **子系统（14）：** router-core · langgraph-core · candidates · dsl · optim-refine · llm-vlm · human-loop-core · human-loop-final · frontend-state · frontend-canvas · frontend-render · 横切并发 · 横切安全 · 横切前后端契约
- **验证：** 每个 high/medium bug 由独立 skeptic agent 重读真实代码做对抗式验证（默认 is_real=false，除非代码确认），已剔除不成立的 finding
- **计数：** 45 已确认 bug（1 high + 22 medium 已验证 + 22 low 未单独验证）· 87 不足（4 high-impact）· 67 优化点（43 低成本）
- **基线：** 997 backend pytest 全绿（审查时）
