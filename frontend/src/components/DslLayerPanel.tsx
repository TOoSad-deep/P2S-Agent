// DslLayerPanel.tsx
import { useState } from "react";
import { ChevronDown, ChevronRight, Cpu, Loader, Sliders } from "lucide-react";
import type { Scoreboard } from "../hooks/usePngShader";

interface Props {
  selectedGlsl: string | null;
  selectedCandidateId: string | null;
  scoreboard: Scoreboard | null;
  onParameterize?: () => void;
  parameterizing?: boolean;
}

const SOURCE_COLORS: Record<string, string> = {
  baseline: "bg-blue-500/20 text-blue-400 border border-blue-500/30",
  rule: "bg-green-500/20 text-green-400 border border-green-500/30",
  cv: "bg-purple-500/20 text-purple-400 border border-purple-500/30",
  fallback: "bg-orange-500/20 text-orange-400 border border-orange-500/30",
  llm: "bg-yellow-500/20 text-yellow-400 border border-yellow-500/30",
};

const PREVIEW_LINES = 30;

export default function DslLayerPanel({ selectedGlsl, selectedCandidateId, scoreboard, onParameterize, parameterizing }: Props) {
  const [expanded, setExpanded] = useState(false);

  const selectedCandidate = scoreboard?.candidates.find((c) => c.id === selectedCandidateId) ?? null;

  const lines = selectedGlsl ? selectedGlsl.split("\n") : [];
  const previewText = expanded ? selectedGlsl : lines.slice(0, PREVIEW_LINES).join("\n");
  const hasMore = lines.length > PREVIEW_LINES;

  return (
    <div className="bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-xl p-4 h-full flex flex-col overflow-hidden">
      <div className="flex items-start justify-between mb-3 flex-shrink-0">
        <h3 className="text-sm font-semibold text-[var(--text-primary)] leading-tight">
          已选候选
          <span className="block text-[10px] font-normal text-[var(--text-muted)] mt-0.5">Selected Candidate</span>
        </h3>
        <div className="flex items-center gap-2">
          {selectedCandidate?.source === "llm" && (
            <span className="flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium bg-yellow-500/15 text-yellow-400 border border-yellow-500/30">
              <Cpu className="w-2.5 h-2.5" /> AI 生成
            </span>
          )}
          {onParameterize && (
            <button
              onClick={onParameterize}
              disabled={!selectedGlsl || parameterizing}
              title="用模型把硬编码常量提升为可调 #define 参数"
              className="flex items-center gap-1 px-2 py-1 rounded-md text-[10px] font-medium bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)] border border-[var(--border-color)] transition-all disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {parameterizing ? <Loader className="w-3 h-3 animate-spin" /> : <Sliders className="w-3 h-3" />}
              补全可调参数
            </button>
          )}
        </div>
      </div>

      {!selectedGlsl ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <p className="text-sm text-[var(--text-muted)]">未选择候选</p>
            <p className="text-xs text-[var(--text-muted)]/60 mt-0.5">No candidate selected</p>
          </div>
        </div>
      ) : (
        <div className="flex-1 flex flex-col overflow-hidden gap-3">
          {/* Candidate info */}
          {selectedCandidate && (
            <div className="flex items-center gap-2 flex-shrink-0">
              <span
                className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                  SOURCE_COLORS[selectedCandidate.source] ?? "bg-gray-500/20 text-gray-400 border border-gray-500/30"
                }`}
              >
                {selectedCandidate.source}
              </span>
              <span className="text-xs text-[var(--text-muted)] font-mono">{selectedCandidate.id}</span>
            </div>
          )}

          {/* GLSL Preview */}
          <div className="flex flex-col flex-1 min-h-0 gap-1">
            <div className="flex items-center justify-between flex-shrink-0">
              <span className="text-xs text-[var(--text-muted)]">GLSL 预览 <span className="text-[10px]">GLSL Preview</span></span>
              <span className="text-[10px] text-[var(--text-muted)]">{selectedGlsl.length} 字符 <span className="text-[9px]">chars</span></span>
            </div>
            <div className="flex-1 overflow-auto min-h-0 bg-[var(--bg-tertiary)] rounded-lg">
              <pre className="p-3 text-[11px] font-mono text-[var(--text-primary)] leading-relaxed whitespace-pre">
                {previewText}
              </pre>
            </div>
            {hasMore && (
              <button
                onClick={() => setExpanded(!expanded)}
                className="flex items-center gap-1 text-xs text-[var(--accent-primary)] hover:text-[var(--text-primary)] transition-colors mt-1 flex-shrink-0"
              >
                {expanded ? (
                  <><ChevronDown className="w-3 h-3" /> 收起 <span className="text-[10px] opacity-70">Show less</span></>
                ) : (
                  <><ChevronRight className="w-3 h-3" /> 展开更多（{lines.length - PREVIEW_LINES} 行）<span className="text-[10px] opacity-70 ml-1">Show more</span></>
                )}
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
