import { useEffect, useMemo, useRef, useState } from "react";
import { ShaderRenderer } from "../lib/shader-renderer";
import { toShaderToyFragment } from "../lib/shader-format";

function decodeShaderParam(value: string | null): string | null {
  if (!value) return null;
  try {
    const base64 = value.replace(/-/g, "+").replace(/_/g, "/");
    const padded = base64.padEnd(base64.length + ((4 - (base64.length % 4)) % 4), "=");
    const bytes = Uint8Array.from(atob(padded), (char) => char.charCodeAt(0));
    return new TextDecoder().decode(bytes);
  } catch {
    return null;
  }
}

export default function ShaderPreview() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  const shader = useMemo(() => {
    const params = new URLSearchParams(window.location.search);
    return decodeShaderParam(params.get("shader"));
  }, []);
  const initialTime = useMemo(() => {
    const params = new URLSearchParams(window.location.search);
    const parsed = Number(params.get("t") ?? 0);
    return Number.isFinite(parsed) ? parsed : 0;
  }, []);

  useEffect(() => {
    window.__shaderReady = false;
    window.__shaderError = null;

    if (!shader || !containerRef.current) {
      const message = "Missing or invalid shader URL parameter";
      setError(message);
      window.__shaderError = message;
      window.__shaderReady = true;
      return;
    }

    let renderer: ShaderRenderer | null = null;
    try {
      renderer = new ShaderRenderer(containerRef.current);
      const result = renderer.compileShader(toShaderToyFragment(shader));
      if (!result.success) {
        const message = result.error ?? "Shader compile error";
        setError(message);
        window.__shaderError = message;
        window.__shaderReady = true;
        renderer.dispose();
        renderer = null;
        return;
      }

      renderer.startRendering();
      renderer.setTime(initialTime);
      window.__setShaderTime = (timeSeconds: number) => {
        if (!renderer) return false;
        renderer.setTime(Number(timeSeconds) || 0);
        return true;
      };
      window.__shaderReady = true;
    } catch (err) {
      const message = err instanceof Error ? err.message : "Shader preview error";
      setError(message);
      window.__shaderError = message;
      window.__shaderReady = true;
      renderer?.dispose();
      renderer = null;
    }

    return () => {
      renderer?.dispose();
      if (window.__setShaderTime) {
        delete window.__setShaderTime;
      }
    };
  }, [initialTime, shader]);

  return (
    <div className="w-screen h-screen overflow-hidden bg-black">
      <div ref={containerRef} className="w-full h-full" />
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-black p-6">
          <pre className="max-w-full whitespace-pre-wrap break-words text-xs text-red-400">
            {error}
          </pre>
        </div>
      )}
    </div>
  );
}
