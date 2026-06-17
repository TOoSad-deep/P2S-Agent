// BranchRefineForm.tsx — shared controlled form body for branch refinement.
// Used by HumanLoopPanel (Studio) and BranchActionView (canvas inspector).
// Does NOT include submit button or explore/draw/constraint extras.
import type { BranchMode, PipelineCheckpointMeta } from "../hooks/usePngShader";
import { MODES, LOCKS } from "../lib/branchRefineOptions";
import { fmtScore } from "../lib/format";

interface BranchRefineFormProps {
  // Optional checkpoint picker — HumanLoopPanel supplies it; canvas omits it.
  checkpoints?: PipelineCheckpointMeta[];
  selectedCheckpointId?: string | null;
  onSelectCheckpoint?: (id: string) => void;
  // Read-only start label (canvas shows the source checkpoint id).
  startLabel?: string | null;
  feedback: string;
  onFeedbackChange: (v: string) => void;
  mode: BranchMode;
  onModeChange: (m: BranchMode) => void;
  locks: Record<string, boolean>;
  onLocksChange: (locks: Record<string, boolean>) => void;
  // Set to false to render only the feedback textarea (used by BranchActionView in explore mode).
  showModeAndLocks?: boolean;
  disabled?: boolean;
  busy?: boolean;
}

export default function BranchRefineForm({
  checkpoints,
  selectedCheckpointId,
  onSelectCheckpoint,
  startLabel,
  feedback,
  onFeedbackChange,
  mode,
  onModeChange,
  locks,
  onLocksChange,
  showModeAndLocks = true,
  disabled,
  busy,
}: BranchRefineFormProps) {
  const toggleLock = (key: string) =>
    onLocksChange({ ...locks, [key]: !locks[key] });

  return (
    <>
      {/* Checkpoint picker (HumanLoopPanel only) */}
      {checkpoints !== undefined && (
        checkpoints.length === 0 ? (
          <p className="text-[11px] text-[var(--text-muted)]">运行完成后可从任意检查点分支。</p>
        ) : (
          <>
            <div className="flex flex-wrap gap-1">
              {checkpoints.map((cp) => (
                <button
                  key={cp.id}
                  onClick={() => onSelectCheckpoint?.(cp.id)}
                  disabled={disabled || busy}
                  title={`${cp.id} · score ${fmtScore(cp.score)}`}
                  className={`px-2 py-1 text-[11px] rounded-md transition-all disabled:opacity-40 ${
                    cp.id === selectedCheckpointId
                      ? "bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-medium"
                      : "bg-[var(--bg-tertiary)] text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)]"
                  }`}
                >
                  {cp.label}
                </button>
              ))}
            </div>
            {selectedCheckpointId && (() => {
              const selected = checkpoints.find((c) => c.id === selectedCheckpointId) ?? null;
              return selected ? (
                <p className="text-[11px] text-[var(--text-muted)]">
                  起点 Start: <span className="font-mono text-[var(--text-secondary)]">{selected.id}</span>
                  {" · "}score {fmtScore(selected.score)}
                </p>
              ) : null;
            })()}
          </>
        )
      )}

      {/* Read-only start label (canvas inspector) */}
      {startLabel !== undefined && startLabel !== null && checkpoints === undefined && (
        <p className="text-[11px] text-[var(--text-muted)]">
          起点 Start:{" "}
          <span className="font-mono text-[var(--text-secondary)]">{startLabel}</span>
        </p>
      )}

      {/* Feedback textarea */}
      <textarea
        value={feedback}
        onChange={(e) => onFeedbackChange(e.target.value)}
        disabled={disabled || busy}
        rows={3}
        placeholder="例如：保持云雾层次，但让水面反射更明显，整体不要变暗。"
        className="w-full text-xs p-2 rounded bg-[var(--bg-tertiary)] border border-[var(--border-color)] text-[var(--text-primary)] resize-y placeholder:text-[var(--text-muted)] disabled:opacity-40"
      />

      {/* Mode segmented control + Locks grid (hidden when parent controls these separately) */}
      {showModeAndLocks && (
        <>
          <div className="flex items-center gap-0.5 bg-[var(--bg-tertiary)] rounded-md p-0.5">
            {MODES.map(({ mode: m, label, sub, desc }) => (
              <button
                key={m}
                onClick={() => onModeChange(m)}
                disabled={disabled || busy}
                title={`${sub} — ${desc}`}
                className={`flex-1 px-2 py-1 text-[11px] rounded-md transition-all disabled:opacity-40 ${
                  mode === m
                    ? "bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-medium"
                    : "text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)]"
                }`}
              >
                {label}
                <span className="ml-1 opacity-70">{sub}</span>
              </button>
            ))}
          </div>

          <div className="grid grid-cols-2 gap-1">
            {LOCKS.map(({ key, label }) => (
              <label key={key} className="flex items-center gap-1.5 text-[11px] text-[var(--text-secondary)] cursor-pointer">
                <input
                  type="checkbox"
                  checked={!!locks[key]}
                  onChange={() => toggleLock(key)}
                  disabled={disabled || busy}
                  className="accent-emerald-500"
                />
                {label}
              </label>
            ))}
          </div>
        </>
      )}
    </>
  );
}
