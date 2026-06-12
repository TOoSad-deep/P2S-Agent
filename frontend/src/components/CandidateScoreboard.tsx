// CandidateScoreboard.tsx
import { Check, X, Star, Cpu, Eye } from "lucide-react";
import type { Scoreboard } from "../hooks/usePngShader";

interface Props {
  scoreboard: Scoreboard | null;
  previewId?: string | null;
  onCandidateClick?: (id: string) => void;
}

const SOURCE_COLORS: Record<string, string> = {
  baseline: "bg-blue-500/20 text-blue-400 border border-blue-500/30",
  rule: "bg-green-500/20 text-green-400 border border-green-500/30",
  cv: "bg-purple-500/20 text-purple-400 border border-purple-500/30",
  fallback: "bg-orange-500/20 text-orange-400 border border-orange-500/30",
  llm: "bg-yellow-500/20 text-yellow-400 border border-yellow-500/30",
};

export default function CandidateScoreboard({ scoreboard, previewId, onCandidateClick }: Props) {
  if (!scoreboard) {
    return (
      <div className="bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-xl p-4 h-full flex items-center justify-center">
        <div className="text-center">
          <p className="text-sm text-[var(--text-muted)]">暂无候选</p>
          <p className="text-xs text-[var(--text-muted)]/60 mt-0.5">No candidates yet</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-xl p-4 flex flex-col h-full overflow-hidden">
      <div className="flex items-center justify-between mb-3 flex-shrink-0">
        <h3 className="text-sm font-semibold text-[var(--text-primary)] leading-tight">
          候选评分板
          <span className="block text-[10px] font-normal text-[var(--text-muted)] mt-0.5">Candidate Scoreboard</span>
        </h3>
        <div className="flex gap-3 text-xs text-[var(--text-muted)]">
          <span>总数 <span className="text-[var(--text-primary)]">{scoreboard.total}</span></span>
          <span>启用 <span className="text-[var(--text-primary)]">{scoreboard.enabled}</span></span>
          <span>已编译 <span className="text-[var(--text-primary)]">{scoreboard.compiled}</span></span>
        </div>
      </div>

      {onCandidateClick && (
        <p className="text-[10px] text-[var(--text-muted)] mb-2 flex-shrink-0">
          点击行可预览该候选的着色器输出 · Click a row to preview
        </p>
      )}

      <div className="flex-1 overflow-auto min-h-0">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-[var(--text-muted)] border-b border-[var(--border-color)]">
              <th className="text-left py-1.5 pr-2 font-medium">来源</th>
              <th className="text-right py-1.5 px-2 font-medium">优先级</th>
              <th className="text-center py-1.5 px-2 font-medium">校验</th>
              <th className="text-center py-1.5 px-2 font-medium">编译</th>
              <th className="text-right py-1.5 px-2 font-medium">得分</th>
              <th className="text-center py-1.5 pl-2 font-medium">状态</th>
            </tr>
          </thead>
          <tbody>
            {scoreboard.candidates.map((c) => {
              const isPreviewing = previewId === c.id;
              const isSelected = c.selected;
              const hasPreviewGlsl = c.previewable ?? (c.compile_success && Boolean(c.compile_glsl?.trim()));
              const isClickable = hasPreviewGlsl && !!onCandidateClick;

              return (
                <tr
                  key={c.id}
                  onClick={() => isClickable && onCandidateClick(c.id)}
                  className={`border-b border-[var(--border-color)]/50 transition-colors ${
                    isPreviewing
                      ? "bg-emerald-500/15 border-l-2 border-l-emerald-400"
                    : isSelected
                      ? "bg-blue-500/10"
                    : isClickable
                      ? "hover:bg-[var(--bg-tertiary)] cursor-pointer"
                      : "opacity-70"
                  }`}
                  title={!hasPreviewGlsl && c.compile_success ? "该候选缺少可预览 GLSL" : undefined}
                >
                  <td className="py-1.5 pr-2">
                    <div className="flex items-center gap-1">
                      <span
                        className={`inline-block px-1.5 py-0.5 rounded text-[10px] font-medium ${
                          SOURCE_COLORS[c.source] ?? "bg-gray-500/20 text-gray-400 border border-gray-500/30"
                        }`}
                      >
                        {c.source === "llm" ? (
                          <span className="flex items-center gap-0.5">
                            <Cpu className="w-2.5 h-2.5 inline" /> AI
                          </span>
                        ) : c.source}
                      </span>
                      {c.output_kind === "glsl" && (
                        <span className="px-1 py-0.5 rounded text-[9px] bg-amber-500/20 text-amber-400 border border-amber-500/30">
                          GLSL
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="text-right py-1.5 px-2 text-[var(--text-muted)]">{c.priority}</td>
                  <td className="text-center py-1.5 px-2">
                    {c.validation_valid ? (
                      <Check className="w-3 h-3 text-green-400 inline" />
                    ) : (
                      <X className="w-3 h-3 text-red-400 inline" />
                    )}
                  </td>
                  <td className="text-center py-1.5 px-2">
                    {c.compile_success ? (
                      <Check className="w-3 h-3 text-green-400 inline" />
                    ) : (
                      <X className="w-3 h-3 text-red-400 inline" />
                    )}
                  </td>
                  <td className="text-right py-1.5 px-2 text-[var(--text-primary)]">
                    {c.final_score > 0 ? c.final_score.toFixed(2) : c.output_kind === "glsl" && c.source === "llm" ? (
                      <span className="text-[9px] text-amber-400" title="GLSL 候选无法通过像素指标评分，需 WebGL 渲染">
                        WebGL
                      </span>
                    ) : "—"}
                  </td>
                  <td className="text-center py-1.5 pl-2">
                    {isPreviewing ? (
                      <Eye className="w-3 h-3 text-emerald-400 inline" />
                    ) : isSelected ? (
                      <Star className="w-3 h-3 text-yellow-400 inline fill-yellow-400" />
                    ) : null}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
