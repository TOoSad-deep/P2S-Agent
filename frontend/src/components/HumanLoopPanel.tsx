// HumanLoopPanel.tsx — directed branch refinement (human-in-loop V1).
import { useState } from "react";
import { GitBranch, Loader } from "lucide-react";
import type {
  BranchLineage,
  BranchMode,
  BranchRefineRequest,
  PipelineCheckpointMeta,
} from "../hooks/usePngShader";

const MODES: { mode: BranchMode; label: string; sub: string; desc: string }[] = [
  { mode: "refine", label: "定向", sub: "Refine", desc: "按反馈定向优化（强制至少一轮）" },
  { mode: "polish", label: "精修", sub: "Polish", desc: "结构尽量不变，仅小幅画质提升" },
  { mode: "continue", label: "继续", sub: "Continue", desc: "不注入目标，继续自动优化" },
];

const LOCKS: { key: string; label: string }[] = [
  { key: "preserve_layout", label: "保持构图 Layout" },
  { key: "preserve_palette", label: "保持调色 Palette" },
  { key: "preserve_background", label: "保护背景 Background" },
  { key: "small_edits_only", label: "仅小幅改动 Small edits" },
];

interface Props {
  checkpoints: PipelineCheckpointMeta[];
  selectedCheckpointId: string | null;
  onSelectCheckpoint: (id: string) => void;
  onSubmit: (request: BranchRefineRequest) => void;
  lineage?: BranchLineage | null;
  disabled?: boolean;
  busy?: boolean;
}

function fmtScore(score?: number | null): string {
  return typeof score === "number" ? score.toFixed(3) : "—";
}

export default function HumanLoopPanel({
  checkpoints,
  selectedCheckpointId,
  onSelectCheckpoint,
  onSubmit,
  lineage,
  disabled,
  busy,
}: Props) {
  const [feedback, setFeedback] = useState("");
  const [mode, setMode] = useState<BranchMode>("refine");
  const [locks, setLocks] = useState<Record<string, boolean>>({});

  const effectiveId =
    selectedCheckpointId ??
    checkpoints.find((c) => c.id === "final:selected")?.id ??
    checkpoints[checkpoints.length - 1]?.id ??
    null;
  const selected = checkpoints.find((c) => c.id === effectiveId) ?? null;

  const feedbackRequired = mode === "refine" || mode === "polish";
  const canSubmit =
    !!effectiveId && !disabled && !busy && (!feedbackRequired || feedback.trim().length > 0);

  const toggleLock = (key: string) =>
    setLocks((prev) => ({ ...prev, [key]: !prev[key] }));

  const handleSubmit = () => {
    if (!effectiveId || !canSubmit) return;
    onSubmit({
      checkpoint_id: effectiveId,
      feedback: feedback.trim(),
      mode,
      locks,
    });
  };

  return (
    <div className="flex flex-col gap-2.5 px-3 py-2.5 bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-lg">
      <div className="flex items-center gap-2">
        <GitBranch className="w-4 h-4 text-[var(--accent-primary)] flex-shrink-0" />
        <p className="text-xs font-medium text-[var(--text-primary)] leading-tight">
          定向优化分支
          <span className="ml-2 text-[var(--text-muted)] font-normal">Directed Branch</span>
        </p>
      </div>

      {lineage?.parent_run_id && (
        <div className="text-[11px] text-[var(--text-muted)] font-mono truncate">
          from {lineage.parent_run_id}
          {lineage.source_checkpoint_label ? ` · ${lineage.source_checkpoint_label}` : ""}
        </div>
      )}

      {/* Start checkpoint */}
      {checkpoints.length === 0 ? (
        <p className="text-[11px] text-[var(--text-muted)]">运行完成后可从任意检查点分支。</p>
      ) : (
        <div className="flex flex-wrap gap-1">
          {checkpoints.map((cp) => (
            <button
              key={cp.id}
              onClick={() => onSelectCheckpoint(cp.id)}
              disabled={disabled || busy}
              title={`${cp.id} · score ${fmtScore(cp.score)}`}
              className={`px-2 py-1 text-[11px] rounded-md transition-all disabled:opacity-40 ${
                cp.id === effectiveId
                  ? "bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-medium"
                  : "bg-[var(--bg-tertiary)] text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)]"
              }`}
            >
              {cp.label}
            </button>
          ))}
        </div>
      )}

      {selected && (
        <p className="text-[11px] text-[var(--text-muted)]">
          起点 Start: <span className="font-mono text-[var(--text-secondary)]">{selected.id}</span>
          {" · "}score {fmtScore(selected.score)}
        </p>
      )}

      {/* Feedback */}
      <textarea
        value={feedback}
        onChange={(e) => setFeedback(e.target.value)}
        disabled={disabled || busy}
        rows={3}
        placeholder="例如：保持云雾层次，但让水面反射更明显，整体不要变暗。"
        className="w-full text-xs p-2 rounded bg-[var(--bg-tertiary)] border border-[var(--border-color)] text-[var(--text-primary)] resize-y"
      />

      {/* Mode */}
      <div className="flex items-center gap-0.5 bg-[var(--bg-tertiary)] rounded-md p-0.5">
        {MODES.map(({ mode: m, label, sub, desc }) => (
          <button
            key={m}
            onClick={() => setMode(m)}
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

      {/* Locks */}
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

      <button
        onClick={handleSubmit}
        disabled={!canSubmit}
        className="flex items-center justify-center gap-2 px-4 py-2 bg-gradient-to-r from-emerald-500 to-emerald-600 hover:from-emerald-400 hover:to-emerald-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-xs rounded-lg transition-all font-semibold"
      >
        {busy ? (
          <><Loader className="w-3.5 h-3.5 animate-spin" /> 创建分支中...</>
        ) : (
          <><GitBranch className="w-3.5 h-3.5" /> 运行定向优化</>
        )}
      </button>
    </div>
  );
}
