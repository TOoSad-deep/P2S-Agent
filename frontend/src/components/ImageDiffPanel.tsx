// ImageDiffPanel.tsx
import { useEffect, useRef, useState } from "react";
import { ShaderRenderer } from "../lib/shader-renderer";
import { toShaderToyFragment } from "../lib/shader-format";

interface Props {
  inputImageUrl: string | null;
  selectedGlsl: string | null;
  previewGlsl?: string | null;
  previewLabel?: string | null;
}

export default function ImageDiffPanel({ inputImageUrl, selectedGlsl, previewGlsl, previewLabel }: Props) {
  const activeGlsl = previewGlsl ?? selectedGlsl;
  const canvasContainerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<ShaderRenderer | null>(null);
  const [shaderError, setShaderError] = useState<string | null>(null);

  useEffect(() => {
    setShaderError(null);

    // No GLSL: the canvas container is unmounted from the DOM, so tear down the
    // (now-orphaned) renderer and release its WebGL context.
    if (!activeGlsl || !canvasContainerRef.current) {
      if (rendererRef.current) {
        rendererRef.current.dispose();
        rendererRef.current = null;
      }
      return;
    }

    try {
      // Bug 1: reuse a SINGLE ShaderRenderer (one WebGL context) across
      // shader/param changes. Creating a fresh renderer per change leaks a
      // context + canvas + CanvasTexture each time and exhausts the browser's
      // ~16-context limit, silently darkening older panels. We only construct
      // once per container mount; subsequent changes recompile in place.
      let renderer = rendererRef.current;
      if (!renderer) {
        renderer = new ShaderRenderer(canvasContainerRef.current);
        rendererRef.current = renderer;
      }
      const result = renderer.compileShader(toShaderToyFragment(activeGlsl));
      if (!result.success) {
        setShaderError(result.error ?? "Shader compile error");
        // Keep the renderer alive (context reuse); just stop drawing the
        // broken shader. A subsequent valid GLSL will recompile in place.
        renderer.stopRendering();
        return;
      }
      // Idempotent: startRendering() no-ops if a loop is already running, so
      // recompiles on the reused renderer don't spawn duplicate rAF loops.
      renderer.startRendering();
      setShaderError(null);
    } catch (err) {
      setShaderError(err instanceof Error ? err.message : "Shader error");
    }
    // No per-change cleanup: the renderer persists across activeGlsl value
    // changes and is disposed only by the null-GLSL branch above or the
    // unmount effect below.
  }, [activeGlsl]);

  // Dispose the single renderer exactly once, on unmount.
  useEffect(() => {
    return () => {
      if (rendererRef.current) {
        rendererRef.current.dispose();
        rendererRef.current = null;
      }
    };
  }, []);

  return (
    <div className="bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-xl p-4 h-full flex flex-col overflow-hidden">
      <h3 className="text-sm font-semibold text-[var(--text-primary)] mb-3 flex-shrink-0 leading-tight">
        图像对比
        <span className="block text-[10px] font-normal text-[var(--text-muted)] mt-0.5">Image Diff</span>
      </h3>
      <div className="flex gap-3 flex-1 min-h-0 overflow-hidden">
        {/* Reference */}
        <div className="flex-1 flex flex-col gap-1">
          <span className="text-xs text-[var(--text-muted)]">参考图 <span className="text-[10px]">Reference</span></span>
          <div className="flex-1 bg-[var(--bg-tertiary)] rounded-lg flex items-center justify-center overflow-hidden min-h-0">
            {inputImageUrl ? (
              <img
                src={inputImageUrl}
                alt="Input"
                className="max-w-full max-h-full object-contain"
              />
            ) : (
              <span className="text-xs text-[var(--text-muted)]">暂无图像</span>
            )}
          </div>
        </div>

        {/* Shader Output */}
        <div className="flex-1 flex flex-col gap-1">
          <div className="flex items-center gap-1.5">
            <span className="text-xs text-[var(--text-muted)]">着色器输出 <span className="text-[10px]">Shader Output</span></span>
            {previewGlsl && previewLabel && (
              <span className="px-1.5 py-0.5 rounded text-[9px] bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                预览: {previewLabel}
              </span>
            )}
          </div>
          <div className="flex-1 bg-[var(--bg-tertiary)] rounded-lg flex items-center justify-center overflow-hidden min-h-0 relative">
            {activeGlsl ? (
              <>
                <div
                  ref={canvasContainerRef}
                  style={{ width: 256, height: 256 }}
                  className="rounded overflow-hidden"
                />
                {shaderError && (
                  <div className="absolute inset-0 p-2 text-center flex flex-col items-center justify-center bg-[var(--bg-tertiary)]/95">
                    <p className="text-xs text-red-400">着色器错误</p>
                    <p className="text-[10px] text-[var(--text-muted)] mt-1 break-all">{shaderError.slice(0, 120)}</p>
                  </div>
                )}
              </>
            ) : (
              <span className="text-xs text-[var(--text-muted)]">暂无 GLSL</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
