# Shader Playground — 独立着色器试验台设计

日期：2026-06-22
状态：待实现

## 目标与动机

为模型输出的 GLSL 片段着色器提供一个**独立的、与主前端解耦的**渲染 + 调参工具。

核心动机：当主前端（Studio / Canvas）的渲染流程出问题时，仍能把模型输出的一段 GLSL **粘贴**进来直接查看效果，并实时调参。它必须是一条**不依赖后端、不依赖 run pipeline、不依赖主 App** 的独立渲染路径。

## 范围

### 做什么
- 一个三栏页面：源码粘贴/编辑、Canvas 预览、参数面板。
- 输入格式：**纯 GLSL 片段着色器源码**（模型输出的 `selected_glsl` / 候选 `compile_glsl` 形态，含 `#define` 与 `mainImage`）。
- 复用现有库进行渲染与调参，不重写渲染逻辑。
- 编译失败时显示 GLSL 报错而非崩溃/白屏。
- 基本时间控制（播放/暂停/时间滑杆）与分辨率切换。

### 明确不做（YAGNI）
- 不做独立后端服务（"需要时"再说）。
- 不解析完整的 `PngShaderResult` JSON（只吃纯 GLSL）。
- 不做 URL 分享 / 历史记录 / 截图导出 / 纹理通道上传。
- 不动主 App、不动后端、不改现有任何文件的行为。

## 形态与隔离（关键决策）

采用**独立的 Vite HTML 入口**，而非复用主 `index.html`。

- 新增 `frontend/playground.html` + `frontend/src/playground.tsx` 作为独立入口。
- 在 `vite.config.ts` 中通过 `build.rollupOptions.input` 注册为第二个多页入口（保留默认 `index.html`）。
- 该入口的模块图**只**依赖：
  - `src/lib/shader-renderer.ts`（`ShaderRenderer`）
  - `src/lib/shader-format.ts`（`toShaderToyFragment`）
  - `src/lib/glsl-parser.ts`（`parseShader` / `updateShaderParam`，被参数面板内部使用）
  - `src/components/PngShaderParamPanel.tsx`（props `{ glsl, onGlslChange }`，自包含，仅依赖 glsl-parser + lucide-react）
  - `src/index.css`（主题变量 `--bg-secondary` / `--accent-primary` 等定义于 `:root`）
- **绝不** import `App.tsx`、`usePngShader`、`context/`、任何 `/api` 或 `/png-shader` 调用。

### 访问方式
- 开发：`npm run dev` → `http://localhost:5174/playground.html`
- 构建：`npm run build` → 产出独立的 `dist/playground.html`（与 `dist/index.html` 并列）

### 已知耦合与权衡
- 运行期：完全隔离。主 Studio 渲染流程崩溃不影响本页面（不同模块图，互不 import）。
- 构建期：若 `npm run build` 含全局 `tsc` 类型检查，仓库任意位置的类型错误仍会使整体构建失败。这是可接受的构建期耦合，不影响"运行期独立渲染路径"这一核心目标。

### 被否决的替代方案
- 在 `main.tsx` 加 `?playground` 分支（仿 `ShaderPreview` 的 `?shader`）：改动更小，但会把主 `App` 拖入同一 bundle，破坏隔离。否决。
- 单文件零依赖 HTML：最抗故障，但需把渲染逻辑重新内联一份，与主库分叉。用户已选择"复用现有库"，否决。

## 布局

三栏，占满视口：

```
┌─────────────┬──────────────────────┬─────────────┐
│ 源码 / 粘贴   │      Canvas 预览        │  参数面板     │
│ <textarea>  │   (ShaderRenderer)    │ PngShader-   │
│             │                       │ ParamPanel   │
│ [Render]    │  ▶/⏸  时间 ──○──        │ {glsl,       │
│ [清空][示例]  │  分辨率 [自适应/512/…]   │  onGlslChange}│
│ ⚠ 编译错误框  │                       │             │
└─────────────┴──────────────────────┴─────────────┘
```

- **左栏**：可编辑 `<textarea>` 用于粘贴/手改 GLSL；`Render` 按钮显式触发编译；`清空`；`示例`（载入一段内置 demo shader，便于自检）；其下为红色编译错误框。
- **中栏**：`ShaderRenderer` 渲染目标容器 + 时间/分辨率控制条。
- **右栏**：直接挂 `PngShaderParamPanel`。

## 组件划分

新增组件（均在 `frontend/src/` 下，建议放 `pages/` 或新建目录，跟随仓库现有结构）：

| 单元 | 职责 | 依赖 |
|------|------|------|
| `playground.tsx`（入口）| `createRoot` 渲染 `<ShaderPlayground/>`，import `index.css` | React DOM |
| `ShaderPlayground.tsx`（容器）| 持有唯一可信源 `glsl` state；编排三栏；管理 `ShaderRenderer` 生命周期与编译/错误状态；时间与分辨率控制 | renderer, shader-format, ParamPanel |
| `PlaygroundSourcePanel.tsx`（左栏）| textarea + Render/清空/示例按钮 + 错误框 | — |
| `PlaygroundCanvas.tsx`（中栏）| 持有 container ref；播放/暂停、时间滑杆、分辨率下拉 | — |

> 若实现时发现拆分过细，可将左/中栏内联进 `ShaderPlayground.tsx`；右栏务必复用既有 `PngShaderParamPanel`，不复制。

## 数据流

唯一可信源是一段 `glsl: string`（保持模型输出的原始形态，含 `#define`）。

```
粘贴/编辑 textarea ──[点 Render]──┐
                                ▼
参数面板 onGlslChange ─────────► glsl (state)
                                │
                                ├─► 同步回写 textarea 内容
                                │
                                └─► toShaderToyFragment(glsl) ─► renderer.compileShader()
                                                                  ├ success → 正常渲染
                                                                  └ fail → 错误框显示 result.error
```

- 参数面板改值 → 改写对应 `#define` 行 → 新 `glsl` → 立即重编译（重编译单个全屏 quad 很快，无需防抖）。
- 手动编辑 textarea → 点 `Render` → 新 `glsl` → 参数面板自动重新 `parseShader` 刷新 + 重编译。
- 编译走 `toShaderToyFragment(glsl)`（与 `ShaderPreview` 一致），兼容 ES3.0 GLSL 与 Shadertoy `mainImage` 形态。

## 渲染器生命周期

- 挂载：`new ShaderRenderer(containerEl)`；首次有有效 `glsl` 时 `compileShader` 成功后 `startRendering()`。
- `glsl` 变化：`compileShader(toShaderToyFragment(glsl))`；成功保持渲染循环；失败保留上一帧并显示错误。
- 时间控制：播放=`startRendering()` + `unfreezeTime()`；暂停/滑杆=`setTime(t)`（内部冻结时间）。
- 分辨率：下拉切换调用 `resize(w,h)`；"自适应"跟随容器尺寸（监听 resize）。
- 卸载：`dispose()`（释放 GPU 资源、移除 listeners）。
- WebGL context loss/restore 已由 `ShaderRenderer` 内部处理，本页面无需额外逻辑。

## 错误处理

| 情况 | 行为 |
|------|------|
| GLSL 编译失败 | 错误框显示 `compileShader().error`；保留上一帧/黑屏；不崩溃 |
| 空输入 | 提示"粘贴 GLSL 后点 Render"，不渲染、不报错 |
| 缺少 `mainImage` | 编译会失败 → 走编译失败路径，错误框给出 GLSL 报错 |
| 参数面板无可调参数 | 面板自身已显示"No tunable parameters found" |

## 测试策略

- **构建门禁**：`npm run build` 通过（含新入口的多页构建产出 `dist/playground.html`）。
- **单元测试（vitest）**：现有 `shader-renderer` / `glsl-parser` 已有覆盖，复用即可；新增的纯逻辑（如若有 glsl 预处理/示例载入）补少量测试。容器组件主要靠手动 + 构建验证，不强求 DOM 测试。
- **手动验证**：dev 打开 `/playground.html`，粘贴一段含 `#define` 的示例 shader → 渲染成功 → 拖参数面板滑块见画面变化 → 粘贴一段故意写错的 GLSL → 错误框出现且不崩溃 → 播放/暂停/时间滑杆/分辨率切换可用。

## 实现清单（提示，详细顺序留给 plan）

1. `vite.config.ts` 增加多页 `rollupOptions.input`（`index.html` + `playground.html`）。
2. 新增 `frontend/playground.html`（套用 `index.html` 结构，挂载 `src/playground.tsx`）。
3. 新增 `playground.tsx` 入口 + `ShaderPlayground.tsx` 容器。
4. 左栏源码面板 + 中栏 canvas 控制 + 右栏复用 `PngShaderParamPanel`。
5. 接 `ShaderRenderer` 生命周期与编译/错误流。
6. 内置一段示例 shader 供"示例"按钮与自检。
7. 跑 `npm run build` 与手动验证。
