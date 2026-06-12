// ImageDiffPanel.tsx
import { useEffect, useRef, useState } from "react";
import { ShaderRenderer } from "../lib/shader-renderer";

interface Props {
  inputImageUrl: string | null;
  selectedGlsl: string | null;
  previewGlsl?: string | null;
  previewLabel?: string | null;
}

function toShaderToyFragment(glsl: string): string {
  if (!glsl.trimStart().startsWith("#version 300 es")) {
    return glsl;
  }

  return glsl
    .replace(/^\s*#version\s+300\s+es\s*\n/, "")
    .replace(/^\s*precision\s+\w+\s+float\s*;\s*\n/m, "")
    .replace(/^\s*out\s+vec4\s+fragColor\s*;\s*\n/m, "")
    .replace(/^\s*uniform\s+vec2\s+iResolution\s*;\s*\n/m, "")
    .replace(/^\s*uniform\s+float\s+iTime\s*;\s*\n/m, "")
    .replace(/\bvoid\s+main\s*\(\s*\)\s*\{/, "void mainImage(out vec4 fragColor, in vec2 fragCoord) {")
    .replace(/\bgl_FragCoord\.xy\b/g, "fragCoord");
}

export default function ImageDiffPanel({ inputImageUrl, selectedGlsl, previewGlsl, previewLabel }: Props) {
  const activeGlsl = previewGlsl ?? selectedGlsl;
  const canvasContainerRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<ShaderRenderer | null>(null);
  const [shaderError, setShaderError] = useState<string | null>(null);

  useEffect(() => {
    if (!activeGlsl || !canvasContainerRef.current) {
      if (rendererRef.current) {
        rendererRef.current.dispose();
        rendererRef.current = null;
      }
      return;
    }

    try {
      if (rendererRef.current) {
        rendererRef.current.dispose();
        rendererRef.current = null;
      }
      const container = canvasContainerRef.current;
      const renderer = new ShaderRenderer(container);
      const result = renderer.compileShader(toShaderToyFragment(activeGlsl));
      if (!result.success) {
        setShaderError(result.error ?? "Shader compile error");
        renderer.dispose();
        return;
      }
      renderer.startRendering();
      rendererRef.current = renderer;
      setShaderError(null);
    } catch (err) {
      setShaderError(err instanceof Error ? err.message : "Shader error");
    }

    return () => {
      if (rendererRef.current) {
        rendererRef.current.dispose();
        rendererRef.current = null;
      }
    };
  }, [activeGlsl]);

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
              shaderError ? (
                <div className="p-2 text-center">
                  <p className="text-xs text-red-400">着色器错误</p>
                  <p className="text-[10px] text-[var(--text-muted)] mt-1 break-all">{shaderError.slice(0, 120)}</p>
                </div>
              ) : (
                <div
                  ref={canvasContainerRef}
                  style={{ width: 256, height: 256 }}
                  className="rounded overflow-hidden"
                />
              )
            ) : (
              <span className="text-xs text-[var(--text-muted)]">暂无 GLSL</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
