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
