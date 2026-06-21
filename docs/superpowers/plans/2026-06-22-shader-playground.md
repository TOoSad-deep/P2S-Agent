# Shader Playground Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `frontend` 内新增一个独立的 Vite 页面 `playground.html`，复用现有 `ShaderRenderer` 与 `PngShaderParamPanel`，用于粘贴一段 GLSL 片段着色器并渲染 + 调参，完全不依赖主 `App`/后端。

**Architecture:** 独立 HTML 入口（多页 Vite），其模块图只 import 渲染库（`shader-renderer`/`shader-format`）、参数解析与面板（`glsl-parser`/`PngShaderParamPanel`）和 `index.css`。单一可信源是一段 `glsl` 字符串：textarea 手改经 `Render` 按钮提交，参数面板改值实时回写并重渲染；编译失败显示 GLSL 报错而不崩溃。

**Tech Stack:** Vite 多页构建、React 19、Three.js（经 `ShaderRenderer` 封装）、Tailwind（`@tailwindcss/vite`）、vitest。

**Branch:** `feat/shader-playground`（已创建，spec 已提交于此）。所有任务在此分支提交。

**Baseline assumption:** `cd frontend && npm run build`（即 `tsc && vite build`）当前为绿。每个任务结束都跑它作为门禁。

**关键约束（务必遵守）：**
- `frontend` 未安装 `@types/node`。`vite.config.ts` 由 `tsconfig.node.json`（`module: ESNext`）做类型检查，**禁止**引入 `node:path` / `__dirname` / `import.meta` 转路径——Vite 的 `rollupOptions.input` 用相对路径字符串即可。
- TS 严格模式开启（`strict` + `noUnusedLocals` + `noUnusedParameters`）：新文件不得有未用变量/参数。
- 复用 `PngShaderParamPanel`（props `{ glsl, onGlslChange }`）和 `ShaderRenderer`，**不复制**其逻辑。
- CSS 主题变量已存在于 `src/index.css` 的 `:root`：`--bg-primary`/`--bg-secondary`/`--bg-tertiary`/`--border-color`/`--text-primary`/`--text-secondary`/`--text-muted`/`--accent-primary`。

---

## File Structure

| 文件 | 状态 | 职责 |
|------|------|------|
| `frontend/vite.config.ts` | 修改 | 增加 `build.rollupOptions.input` 注册 `index.html` + `playground.html` 两个入口 |
| `frontend/playground.html` | 新建 | 独立页面 HTML，挂载 `src/playground.tsx` |
| `frontend/src/playground.tsx` | 新建 | 入口：`createRoot` 渲染 `<ShaderPlayground/>`，import `index.css` |
| `frontend/src/lib/playground-example.ts` | 新建 | 内置示例 shader 常量（mainImage 形态 + 若干 `#define`） |
| `frontend/src/lib/playground-example.test.ts` | 新建 | 校验示例为合法 mainImage 形态且能解析出参数 |
| `frontend/src/components/PlaygroundCanvas.tsx` | 新建 | 持有 `ShaderRenderer` 生命周期；编译/错误上报；播放/暂停、时间、分辨率控制 |
| `frontend/src/pages/ShaderPlayground.tsx` | 新建 | 容器：唯一 `glsl` 可信源；三栏布局；左栏源码 + Render/清空/示例 + 错误框；右栏复用参数面板 |

所有命令默认在 `frontend/` 目录执行。

---

### Task 1: Vite 多页入口骨架

先把多页构建打通（最高风险点），用一个占位页面验证 `dist/playground.html` 能产出。

**Files:**
- Modify: `frontend/vite.config.ts`
- Create: `frontend/playground.html`
- Create: `frontend/src/playground.tsx`
- Create: `frontend/src/pages/ShaderPlayground.tsx`（本任务为占位，Task 4 替换为完整版）

- [ ] **Step 1: 修改 `vite.config.ts` 增加多页 input**

完整替换文件内容为：

```ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],
  build: {
    rollupOptions: {
      // 多页：主应用 + 独立 shader playground。相对路径相对项目根解析，
      // 避免引入 node:path/__dirname（未安装 @types/node）。
      input: {
        main: 'index.html',
        playground: 'playground.html',
      },
    },
  },
  server: {
    port: 5174,
    proxy: {
      '/api': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
      '/png-shader': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
    },
  },
})
```

- [ ] **Step 2: 创建 `frontend/playground.html`**

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>P2S-Agent | Shader Playground</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/playground.tsx"></script>
  </body>
</html>
```

- [ ] **Step 3: 创建 `frontend/src/playground.tsx`**

```tsx
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import ShaderPlayground from './pages/ShaderPlayground'
import './index.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ShaderPlayground />
  </StrictMode>,
)
```

- [ ] **Step 4: 创建占位 `frontend/src/pages/ShaderPlayground.tsx`**

```tsx
export default function ShaderPlayground() {
  return (
    <div className="flex items-center justify-center h-screen w-screen bg-[var(--bg-primary)] text-[var(--text-primary)]">
      Shader Playground (skeleton)
    </div>
  );
}
```

- [ ] **Step 5: 构建门禁，确认产出 playground.html**

Run: `npm run build && ls dist/playground.html dist/index.html`
Expected: 构建成功（无 tsc/vite 报错），且列出 `dist/playground.html` 与 `dist/index.html` 两个文件。
> 若 `dist/playground.html` 未生成：确认 `playground.html` 位于 `frontend/` 根目录、`input` 值为相对路径字符串。

- [ ] **Step 6: 提交**

```bash
git add vite.config.ts playground.html src/playground.tsx src/pages/ShaderPlayground.tsx
git commit -m "feat(playground): add standalone Vite entry skeleton

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: 示例 shader 模块（TDD）

`Example` 按钮和自检要用的内置 demo shader。先写测试钉住它合法、可解析出参数。

**Files:**
- Create: `frontend/src/lib/playground-example.ts`
- Test: `frontend/src/lib/playground-example.test.ts`

- [ ] **Step 1: 先写失败测试 `playground-example.test.ts`**

```ts
import { describe, it, expect } from "vitest";
import { EXAMPLE_SHADER } from "./playground-example";
import { parseShader } from "./glsl-parser";
import { toShaderToyFragment } from "./shader-format";

describe("EXAMPLE_SHADER", () => {
  it("已是 mainImage 形态（toShaderToyFragment 原样返回）", () => {
    expect(EXAMPLE_SHADER).toContain("void mainImage(");
    expect(toShaderToyFragment(EXAMPLE_SHADER)).toBe(EXAMPLE_SHADER);
  });

  it("暴露可调 #define 参数供参数面板解析", () => {
    const names = parseShader(EXAMPLE_SHADER).parameters.map((p) => p.name);
    expect(names).toContain("SPEED");
    expect(names).toContain("COLOR_A");
    expect(names).toContain("COLOR_B");
    expect(names).toContain("RADIUS");
  });
});
```

- [ ] **Step 2: 跑测试确认失败**

Run: `npx vitest run src/lib/playground-example.test.ts`
Expected: FAIL，报错类似 `Failed to resolve import "./playground-example"`（模块尚不存在）。

- [ ] **Step 3: 创建 `playground-example.ts` 实现**

```ts
/** Playground "Example" 按钮用的自包含示例 shader。
 *  已是 Shadertoy mainImage 形态（故 toShaderToyFragment 原样透传），并暴露
 *  若干 #define 参数，让参数面板能渲染出滑块/颜色控件。 */
export const EXAMPLE_SHADER = `#define SPEED 1.0
#define COLOR_A vec3(0.10, 0.30, 0.80)
#define COLOR_B vec3(0.90, 0.40, 0.20)
#define RADIUS 0.40

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    float d = length(uv) - RADIUS;
    float pulse = 0.5 + 0.5 * sin(iTime * SPEED);
    vec3 col = mix(COLOR_A, COLOR_B, smoothstep(-0.05, 0.05, d));
    col *= 0.6 + 0.4 * pulse;
    fragColor = vec4(col, 1.0);
}
`;
```

- [ ] **Step 4: 跑测试确认通过**

Run: `npx vitest run src/lib/playground-example.test.ts`
Expected: PASS（2 个用例）。

- [ ] **Step 5: 提交**

```bash
git add src/lib/playground-example.ts src/lib/playground-example.test.ts
git commit -m "feat(playground): add example shader + test

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: PlaygroundCanvas 组件

封装 `ShaderRenderer` 的创建/编译/释放，以及播放-暂停、时间滑杆、分辨率控制；编译错误经 `onError` 上报给容器。

**Files:**
- Create: `frontend/src/components/PlaygroundCanvas.tsx`

设计要点（务必照此实现，避免 effect 依赖抖动导致重复编译）：
- 渲染器只在挂载时创建一次，卸载时 `dispose()`。
- 仅在 `glsl` 变化时 `compileShader(toShaderToyFragment(glsl))`；用 `playingRef`/`timeRef` 读取最新播放/时间状态，避免把它们放进编译 effect 的依赖。
- 播放/暂停：`startRendering()` / `stopRendering()`（暂停保留最后一帧）。
- 暂停时拖时间滑杆：`setTime(t)`（内部冻结时间）。
- 分辨率：`fit` 跟随容器并监听 `window.resize`；`512`/`1024` 调 `resize(n, n)`。

- [ ] **Step 1: 创建 `PlaygroundCanvas.tsx`**

```tsx
import { useEffect, useRef, useState } from "react";
import { Play, Pause } from "lucide-react";
import { ShaderRenderer } from "../lib/shader-renderer";
import { toShaderToyFragment } from "../lib/shader-format";

interface PlaygroundCanvasProps {
  /** 原始 shader 源码（模型输出形态，可能含 #define）。 */
  glsl: string;
  /** 上报最近一次编译错误（成功为 null）。 */
  onError: (error: string | null) => void;
}

type ResolutionOption = "fit" | "512" | "1024";

export default function PlaygroundCanvas({ glsl, onError }: PlaygroundCanvasProps) {
  const viewRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<ShaderRenderer | null>(null);
  const [playing, setPlaying] = useState(true);
  const [time, setTime] = useState(0);
  const [resolution, setResolution] = useState<ResolutionOption>("fit");

  // 用 ref 读取最新的播放/时间状态，避免它们进入编译 effect 的依赖。
  const playingRef = useRef(playing);
  const timeRef = useRef(time);
  playingRef.current = playing;
  timeRef.current = time;

  // 仅创建一次渲染器。
  useEffect(() => {
    if (!viewRef.current) return;
    const renderer = new ShaderRenderer(viewRef.current);
    rendererRef.current = renderer;
    renderer.startRendering();
    return () => {
      renderer.dispose();
      rendererRef.current = null;
    };
  }, []);

  // 源码变化时重新编译。
  useEffect(() => {
    const renderer = rendererRef.current;
    if (!renderer) return;
    if (!glsl.trim()) {
      onError(null);
      return;
    }
    const result = renderer.compileShader(toShaderToyFragment(glsl));
    if (!result.success) {
      onError(result.error ?? "Shader compile error");
      return;
    }
    onError(null);
    if (playingRef.current) {
      renderer.unfreezeTime();
      renderer.startRendering();
    } else {
      // 暂停态：渲染一帧到当前时间。
      renderer.setTime(timeRef.current);
    }
  }, [glsl, onError]);

  // 播放 / 暂停。
  useEffect(() => {
    const renderer = rendererRef.current;
    if (!renderer) return;
    if (playing) {
      renderer.unfreezeTime();
      renderer.startRendering();
    } else {
      renderer.stopRendering();
    }
  }, [playing]);

  // 暂停时拖动时间滑杆。
  useEffect(() => {
    const renderer = rendererRef.current;
    if (!renderer || playing) return;
    renderer.setTime(time);
  }, [time, playing]);

  // 应用分辨率。
  useEffect(() => {
    const renderer = rendererRef.current;
    const el = viewRef.current;
    if (!renderer || !el) return;
    const apply = () => {
      if (resolution === "fit") {
        renderer.resize(el.clientWidth, el.clientHeight);
      } else {
        const n = parseInt(resolution, 10);
        renderer.resize(n, n);
      }
    };
    apply();
    if (resolution === "fit") {
      window.addEventListener("resize", apply);
      return () => window.removeEventListener("resize", apply);
    }
  }, [resolution]);

  return (
    <div className="flex flex-col h-full">
      <div ref={viewRef} className="flex-1 min-h-0 bg-black" />
      <div className="flex items-center gap-3 px-3 py-2 border-t border-[var(--border-color)] bg-[var(--bg-secondary)]">
        <button
          onClick={() => setPlaying((p) => !p)}
          className="p-1.5 rounded hover:bg-[var(--bg-tertiary)] text-[var(--text-primary)]"
          title={playing ? "Pause" : "Play"}
        >
          {playing ? <Pause size={14} /> : <Play size={14} />}
        </button>
        <input
          type="range"
          min={0}
          max={10}
          step={0.01}
          value={time}
          disabled={playing}
          onChange={(e) => setTime(parseFloat(e.target.value))}
          className="flex-1 h-1 bg-[var(--border-color)] rounded-lg appearance-none cursor-pointer disabled:opacity-40"
        />
        <span className="text-xs font-mono text-[var(--text-secondary)] w-12 text-right">
          {time.toFixed(2)}s
        </span>
        <select
          value={resolution}
          onChange={(e) => setResolution(e.target.value as ResolutionOption)}
          className="text-xs bg-[var(--bg-tertiary)] border border-[var(--border-color)] rounded px-1 py-0.5 text-[var(--text-primary)]"
        >
          <option value="fit">Fit</option>
          <option value="512">512²</option>
          <option value="1024">1024²</option>
        </select>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: 构建门禁**

Run: `npm run build`
Expected: 成功，无 tsc 报错（特别留意无未用变量/参数）。

- [ ] **Step 3: 提交**

```bash
git add src/components/PlaygroundCanvas.tsx
git commit -m "feat(playground): add PlaygroundCanvas (renderer lifecycle + controls)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: ShaderPlayground 容器（完整版）

用完整三栏实现替换 Task 1 的占位：左栏源码 + Render/清空/示例 + 错误框；中栏 `PlaygroundCanvas`；右栏复用 `PngShaderParamPanel`。

**Files:**
- Modify: `frontend/src/pages/ShaderPlayground.tsx`（整文件替换）

数据流：`draft`（textarea 内容）与 `glsl`（已应用的可信源）分离；点 `Render` 把 `draft` 提交为 `glsl`；参数面板 `onGlslChange` 同时回写 `glsl` 和 `draft`。

- [ ] **Step 1: 整文件替换 `src/pages/ShaderPlayground.tsx`**

```tsx
import { useState } from "react";
import { Play, Trash2, FileCode } from "lucide-react";
import PlaygroundCanvas from "../components/PlaygroundCanvas";
import PngShaderParamPanel from "../components/PngShaderParamPanel";
import { EXAMPLE_SHADER } from "../lib/playground-example";

export default function ShaderPlayground() {
  // draft = 编辑框内容；glsl = 已应用并驱动渲染/参数面板的可信源。
  const [draft, setDraft] = useState("");
  const [glsl, setGlsl] = useState("");
  const [error, setError] = useState<string | null>(null);

  const handleRender = () => setGlsl(draft);
  const handleClear = () => {
    setDraft("");
    setGlsl("");
  };
  const handleExample = () => {
    setDraft(EXAMPLE_SHADER);
    setGlsl(EXAMPLE_SHADER);
  };

  // 参数面板改值：同步回写编辑框与可信源。
  const handleParamChange = (next: string) => {
    setGlsl(next);
    setDraft(next);
  };

  return (
    <div className="flex h-screen w-screen bg-[var(--bg-primary)] text-[var(--text-primary)]">
      {/* 左：源码 */}
      <div className="flex flex-col w-[28rem] border-r border-[var(--border-color)]">
        <div className="flex items-center gap-2 px-3 py-2 border-b border-[var(--border-color)]">
          <FileCode size={14} className="text-[var(--accent-primary)]" />
          <span className="text-sm font-medium">GLSL Source</span>
          <div className="ml-auto flex items-center gap-1">
            <button
              onClick={handleExample}
              className="px-2 py-1 text-xs rounded hover:bg-[var(--bg-tertiary)] text-[var(--text-secondary)]"
              title="Load example shader"
            >
              Example
            </button>
            <button
              onClick={handleClear}
              className="p-1 rounded hover:bg-[var(--bg-tertiary)] text-[var(--text-muted)]"
              title="Clear"
            >
              <Trash2 size={14} />
            </button>
            <button
              onClick={handleRender}
              className="flex items-center gap-1 px-2 py-1 text-xs rounded bg-[var(--accent-primary)] text-black font-medium hover:opacity-90"
              title="Compile & render"
            >
              <Play size={12} /> Render
            </button>
          </div>
        </div>
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Paste a GLSL fragment shader (mainImage or #version 300 es), then click Render."
          spellCheck={false}
          className="flex-1 min-h-0 resize-none bg-[var(--bg-primary)] text-[var(--text-primary)] font-mono text-xs p-3 outline-none"
        />
        {error && (
          <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-words bg-[var(--bg-secondary)] border-t border-[var(--border-color)] p-3 text-xs text-[var(--accent-error)]">
            {error}
          </pre>
        )}
      </div>

      {/* 中：画布 */}
      <div className="flex-1 min-w-0">
        <PlaygroundCanvas glsl={glsl} onError={setError} />
      </div>

      {/* 右：参数面板（复用现有组件） */}
      <div className="w-72 border-l border-[var(--border-color)] p-2">
        <PngShaderParamPanel glsl={glsl || null} onGlslChange={handleParamChange} />
      </div>
    </div>
  );
}
```

- [ ] **Step 2: 构建门禁**

Run: `npm run build && ls dist/playground.html`
Expected: 成功；`dist/playground.html` 存在。

- [ ] **Step 3: 跑全部单测确保未回归**

Run: `npm test`
Expected: 全部通过（含 Task 2 新增的 example 测试）。

- [ ] **Step 4: 提交**

```bash
git add src/pages/ShaderPlayground.tsx
git commit -m "feat(playground): wire 3-pane ShaderPlayground (source + canvas + params)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: 手动验证（验证门禁，非提交）

用 preview 工具实际跑 dev server 验证行为，给出截图证据。

- [ ] **Step 1: 启动 dev server 并打开 playground**

Run: `npm run dev`（端口 5174），用 preview 工具导航到 `http://localhost:5174/playground.html`。

- [ ] **Step 2: 点 `Example` → 验证渲染**

预期：中栏出现脉动的圆形图案；右栏参数面板出现 `SPEED`/`RADIUS` 滑块与 `COLOR_A`/`COLOR_B` 颜色控件；无 console 报错。截图留证。

- [ ] **Step 3: 调参 → 验证实时更新**

拖动 `RADIUS` 滑块或改 `COLOR_A` 颜色：预期画面实时变化，且左栏 textarea 中对应 `#define` 行同步更新。

- [ ] **Step 4: 故意写错 → 验证错误不崩溃**

在 textarea 里把 `mainImage` 改成 `mainImagX`（或删掉一个 `}`），点 `Render`：预期左栏底部出现红色 GLSL 编译错误框，页面不白屏/不崩溃。

- [ ] **Step 5: 时间/分辨率控制**

点暂停（画面冻结）→ 拖时间滑杆（画面随时间变化）→ 切 `512²`/`1024²`（画布尺寸变化）。各项可用。

- [ ] **Step 6: 截图发给用户作为完成证据。**

---

## Self-Review（作者已核对）

- **Spec 覆盖**：独立入口（Task 1）✓；纯 GLSL 粘贴 + Render（Task 4）✓；复用参数面板（Task 4）✓；编译错误显示不崩溃（Task 3 onError + Task 4 错误框 + Task 5 验证）✓；时间/分辨率控制（Task 3）✓；运行期隔离——playground 模块图不 import App/usePngShader/后端（Task 1/3/4 的 import 列表）✓；YAGNI（无后端/无 JSON 解析/无分享）✓。
- **占位扫描**：无 TBD/TODO；所有代码步骤含完整文件内容。
- **类型一致性**：`PlaygroundCanvas` props `{ glsl, onError }` 与 Task 4 调用一致；`PngShaderParamPanel` props `{ glsl, onGlslChange }` 与既有组件签名一致；`EXAMPLE_SHADER` 在 Task 2 定义、Task 4 使用，名称一致。
- **构建约束**：`vite.config.ts` 不引入 node 路径 API（满足无 `@types/node`）；新组件无未用变量/参数（满足严格模式）。
