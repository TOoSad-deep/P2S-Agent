// HumanLoopPanel.tsx — directed branch refinement (human-in-loop V1).
import { useState } from "react";
import { GitBranch, Loader } from "lucide-react";
import type {
  BranchLineage,
  BranchMode,
  BranchRefineRequest,
  PipelineCheckpointMeta,
} from "../hooks/usePngShader";
import BranchRefineForm from "./BranchRefineForm";

interface Props {
  checkpoints: PipelineCheckpointMeta[];
  selectedCheckpointId: string | null;
  onSelectCheckpoint: (id: string) => void;
  onSubmit: (request: BranchRefineRequest) => void;
  lineage?: BranchLineage | null;
  disabled?: boolean;
  busy?: boolean;
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

  const feedbackRequired = mode === "refine" || mode === "polish";
  const canSubmit =
    !!effectiveId && !disabled && !busy && (!feedbackRequired || feedback.trim().length > 0);

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

      <BranchRefineForm
        checkpoints={checkpoints}
        selectedCheckpointId={effectiveId}
        onSelectCheckpoint={onSelectCheckpoint}
        feedback={feedback}
        onFeedbackChange={setFeedback}
        mode={mode}
        onModeChange={setMode}
        locks={locks}
        onLocksChange={setLocks}
        disabled={disabled}
        busy={busy}
      />

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
