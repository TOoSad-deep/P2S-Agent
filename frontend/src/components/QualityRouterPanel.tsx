// QualityRouterPanel.tsx
import { ArrowRight } from "lucide-react";
import type { QualityRouterOutput } from "../hooks/usePngShader";

interface Props {
  qualityRouter: QualityRouterOutput | null;
}

const STATUS_STYLES: Record<string, string> = {
  pass: "bg-green-500/20 text-green-400 border border-green-500/30",
  preview: "bg-yellow-500/20 text-yellow-400 border border-yellow-500/30",
  failed: "bg-red-500/20 text-red-400 border border-red-500/30",
  unsupported: "bg-gray-500/20 text-gray-400 border border-gray-500/30",
};

const BAND_STYLES: Record<string, string> = {
  excellent: "bg-green-500/20 text-green-400 border border-green-500/30",
  good: "bg-teal-500/20 text-teal-400 border border-teal-500/30",
  acceptable: "bg-yellow-500/20 text-yellow-400 border border-yellow-500/30",
  poor: "bg-red-500/20 text-red-400 border border-red-500/30",
};

export default function QualityRouterPanel({ qualityRouter }: Props) {
  return (
    <div className="bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-xl p-4 h-full flex flex-col overflow-hidden">
      <h3 className="text-sm font-semibold text-[var(--text-primary)] mb-3 flex-shrink-0 leading-tight">
        质量路由
        <span className="block text-[10px] font-normal text-[var(--text-muted)] mt-0.5">Quality Router</span>
      </h3>

      {!qualityRouter ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <p className="text-sm text-[var(--text-muted)]">质量分析暂未就绪</p>
            <p className="text-xs text-[var(--text-muted)]/60 mt-0.5">Quality analysis not yet available</p>
          </div>
        </div>
      ) : (
        <div className="flex-1 overflow-auto space-y-3">
          {/* Status + Band */}
          <div className="flex items-center gap-2 flex-wrap">
            <span
              className={`px-2 py-0.5 rounded text-xs font-medium ${
                STATUS_STYLES[qualityRouter.status] ?? STATUS_STYLES["unsupported"]
              }`}
            >
              {qualityRouter.status}
            </span>
            <span
              className={`px-2 py-0.5 rounded text-xs font-medium ${
                BAND_STYLES[qualityRouter.quality_band] ?? BAND_STYLES["acceptable"]
              }`}
            >
              {qualityRouter.quality_band}
            </span>
          </div>

          {/* Score */}
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-bold text-[var(--text-primary)]">
              {qualityRouter.final_score.toFixed(2)}
            </span>
            <span className="text-xs text-[var(--text-muted)]">综合得分 <span className="text-[10px]">final score</span></span>
          </div>

          {/* Next action */}
          <div className="flex items-center gap-1.5">
            <ArrowRight className="w-3 h-3 text-[var(--accent-primary)] flex-shrink-0" />
            <span className="text-xs text-[var(--text-primary)]">{qualityRouter.next_action}</span>
          </div>

          {/* Failure type */}
          {qualityRouter.failure_type && qualityRouter.failure_type !== "none" && (
            <div>
              <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-orange-500/20 text-orange-400 border border-orange-500/30">
                {qualityRouter.failure_type}
              </span>
            </div>
          )}

          {/* Reasons */}
          {qualityRouter.reason.length > 0 && (
            <div>
              <p className="text-xs text-[var(--text-muted)] mb-1">原因 <span className="text-[10px]">Reasons</span></p>
              <ul className="space-y-1">
                {qualityRouter.reason.map((r, i) => (
                  <li key={i} className="text-xs text-[var(--text-primary)] flex gap-1.5">
                    <span className="text-[var(--text-muted)] flex-shrink-0">•</span>
                    <span>{r}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Protected aspects */}
          {qualityRouter.protected_aspects.length > 0 && (
            <div>
              <p className="text-xs text-[var(--text-muted)] mb-1">保护属性 <span className="text-[10px]">Protected Aspects</span></p>
              <div className="flex flex-wrap gap-1">
                {qualityRouter.protected_aspects.map((a, i) => (
                  <span
                    key={i}
                    className="px-1.5 py-0.5 rounded text-[10px] bg-[var(--bg-tertiary)] text-[var(--text-muted)] border border-[var(--border-color)]"
                  >
                    {a}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
