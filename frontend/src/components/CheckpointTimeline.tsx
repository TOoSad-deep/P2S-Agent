// CheckpointTimeline.tsx — scannable left-to-right timeline of a run's branchable checkpoints (V2).
import { GitCommitHorizontal, Target, X } from "lucide-react";
import type { CheckpointTimelineEntry } from "../hooks/usePngShader";

interface Props {
  entries: CheckpointTimelineEntry[];
  activeCheckpointId: string | null;
  onSelect: (id: string) => void;
  disabled?: boolean;
}

function fmtScore(score?: number | null): string {
  return typeof score === "number" ? score.toFixed(3) : "—";
}

export default function CheckpointTimeline({
  entries,
  activeCheckpointId,
  onSelect,
  disabled,
}: Props) {
  return (
    <div className="flex flex-col gap-2 px-3 py-2.5 bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-lg">
      {/* Header */}
      <div className="flex items-center gap-2">
        <GitCommitHorizontal className="w-4 h-4 text-[var(--accent-primary)] flex-shrink-0" />
        <p className="text-xs font-medium text-[var(--text-primary)] leading-tight">
          检查点时间线
          <span className="ml-2 text-[var(--text-muted)] font-normal">Timeline</span>
        </p>
      </div>

      {/* Empty state */}
      {entries.length === 0 ? (
        <p className="text-[11px] text-[var(--text-muted)]">
          运行后显示检查点 / Checkpoints appear after a run
        </p>
      ) : (
        <div className="flex flex-wrap gap-1.5">
          {entries.map((entry) => {
            const isActive = entry.id === activeCheckpointId;
            const hasDelta =
              typeof entry.delta === "number" && entry.delta !== 0;
            const isRejected = entry.accepted === false;

            return (
              <button
                key={entry.id}
                onClick={() => onSelect(entry.id)}
                disabled={disabled}
                title={[
                  `id: ${entry.id}`,
                  `score: ${fmtScore(entry.score)}`,
                  entry.changes_summary ? `changes: ${entry.changes_summary}` : null,
                ]
                  .filter(Boolean)
                  .join("\n")}
                className={`relative flex flex-col items-start px-2 py-1.5 text-[11px] rounded-md transition-all disabled:opacity-40 ${
                  isActive
                    ? "bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-medium"
                    : isRejected
                    ? "bg-[var(--bg-tertiary)] text-[var(--text-muted)] opacity-60 hover:opacity-80 hover:bg-[var(--bg-hover)]"
                    : "bg-[var(--bg-tertiary)] text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)]"
                }`}
              >
                {/* Label row */}
                <span className="flex items-center gap-1">
                  <span className={isActive ? "text-white" : "text-[var(--text-secondary)]"}>
                    {entry.label}
                  </span>

                  {/* Human goal override marker */}
                  {entry.human_goal_override && (
                    <span title={entry.human_goal_override} className="flex-shrink-0">
                      <Target
                        className={`w-3 h-3 ${isActive ? "text-white" : "text-emerald-400"}`}
                      />
                    </span>
                  )}

                  {/* Rejected marker */}
                  {isRejected && (
                    <X
                      className={`w-3 h-3 flex-shrink-0 ${isActive ? "text-white/70" : "text-[var(--text-muted)]"}`}
                    />
                  )}
                </span>

                {/* Score + delta row */}
                <span className="flex items-center gap-1 mt-0.5">
                  <span className={isActive ? "text-white/80" : "text-[var(--text-muted)]"}>
                    {fmtScore(entry.score)}
                  </span>

                  {hasDelta && (
                    <span
                      className={`font-medium ${
                        isActive
                          ? "text-white/90"
                          : entry.delta! > 0
                          ? "text-emerald-400"
                          : "text-red-400"
                      }`}
                    >
                      {entry.delta! > 0 ? "▲" : "▼"}
                      {Math.abs(entry.delta!).toFixed(3)}
                    </span>
                  )}
                </span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
