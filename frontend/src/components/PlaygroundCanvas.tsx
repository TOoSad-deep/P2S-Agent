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
