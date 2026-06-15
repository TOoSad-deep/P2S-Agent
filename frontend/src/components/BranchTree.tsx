// BranchTree.tsx — recursive parent/child run tree (V2).
import { GitBranch, Loader, Star } from "lucide-react";
import type { BranchTreeNode } from "../hooks/usePngShader";

interface Props {
  tree: BranchTreeNode | null;
  activeRunId: string | null;
  onSelectRun: (runId: string) => void;
  disabled?: boolean;
}

function fmtScore(score?: number | null): string {
  return typeof score === "number" ? score.toFixed(3) : "—";
}

function shortId(runId: string): string {
  return runId.length > 8 ? runId.slice(-8) : runId;
}

function nodeLabel(node: BranchTreeNode): string {
  if (node.title) return node.title;
  if (node.feedback) {
    const trimmed = node.feedback.trim();
    return trimmed.length > 40 ? trimmed.slice(0, 38) + "…" : trimmed;
  }
  if (node.source_checkpoint_label) return node.source_checkpoint_label;
  return "root";
}

interface StatusIndicatorProps {
  status: string;
}

function StatusIndicator({ status }: StatusIndicatorProps) {
  if (status === "running") {
    return (
      <span className="flex items-center gap-0.5 text-amber-400">
        <Loader className="w-3 h-3 animate-spin" />
        <span className="text-[10px]">running</span>
      </span>
    );
  }
  if (status === "completed") {
    return (
      <span className="w-2 h-2 rounded-full bg-emerald-500 flex-shrink-0" title="completed" />
    );
  }
  if (status === "failed") {
    return (
      <span className="w-2 h-2 rounded-full bg-red-500 flex-shrink-0" title="failed" />
    );
  }
  // pending / other
  return (
    <span className="w-2 h-2 rounded-full bg-[var(--text-muted)] opacity-50 flex-shrink-0" title={status} />
  );
}

interface BranchNodeProps {
  node: BranchTreeNode;
  activeRunId: string | null;
  onSelectRun: (runId: string) => void;
  disabled?: boolean;
}

function BranchNode({ node, activeRunId, onSelectRun, disabled }: BranchNodeProps) {
  const isActive = node.run_id === activeRunId;
  const label = nodeLabel(node);

  return (
    <div>
      <button
        onClick={() => onSelectRun(node.run_id)}
        disabled={disabled}
        title={[
          `run_id: ${node.run_id}`,
          node.feedback ? `feedback: ${node.feedback}` : null,
          `status: ${node.status}`,
          typeof node.final_score === "number" ? `score: ${fmtScore(node.final_score)}` : null,
        ]
          .filter(Boolean)
          .join("\n")}
        className={`w-full flex items-center gap-1.5 px-2 py-1.5 rounded-md text-left transition-all disabled:opacity-40 ${
          isActive
            ? "bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-medium"
            : "bg-[var(--bg-tertiary)] text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)]"
        }`}
      >
        {/* Short run id */}
        <span className={`font-mono text-[11px] flex-shrink-0 ${isActive ? "text-white/80" : "text-[var(--text-muted)]"}`}>
          {shortId(node.run_id)}
        </span>

        {/* Label */}
        <span className={`flex-1 text-[11px] truncate ${isActive ? "text-white" : "text-[var(--text-secondary)]"}`}>
          {label}
        </span>

        {/* Status indicator */}
        <span className="flex-shrink-0">
          {isActive && node.status === "running" ? (
            <Loader className="w-3 h-3 animate-spin text-white/80 flex-shrink-0" />
          ) : isActive ? (
            <span className="w-2 h-2 rounded-full bg-white/80 flex-shrink-0" title={node.status} />
          ) : (
            <StatusIndicator status={node.status} />
          )}
        </span>

        {/* Final score */}
        {typeof node.final_score === "number" && (
          <span className={`text-[10px] flex-shrink-0 ${isActive ? "text-white/80" : "text-[var(--text-muted)]"}`}>
            {fmtScore(node.final_score)}
          </span>
        )}

        {/* Favorite star */}
        {node.favorite && (
          <Star
            className={`w-3 h-3 flex-shrink-0 ${isActive ? "text-yellow-300" : "text-yellow-400"}`}
            fill="currentColor"
          />
        )}
      </button>

      {/* Children — indented */}
      {node.children.length > 0 && (
        <div className={`ml-3 border-l border-[var(--border-color)] pl-2 mt-1 flex flex-col gap-1`}>
          {node.children.map((child) => (
            <BranchNode
              key={child.run_id}
              node={child}
              activeRunId={activeRunId}
              onSelectRun={onSelectRun}
              disabled={disabled}
            />
          ))}
        </div>
      )}
    </div>
  );
}

export default function BranchTree({
  tree,
  activeRunId,
  onSelectRun,
  disabled,
}: Props) {
  return (
    <div className="flex flex-col gap-2 px-3 py-2.5 bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-lg">
      {/* Header */}
      <div className="flex items-center gap-2">
        <GitBranch className="w-4 h-4 text-[var(--accent-primary)] flex-shrink-0" />
        <p className="text-xs font-medium text-[var(--text-primary)] leading-tight">
          分支树
          <span className="ml-2 text-[var(--text-muted)] font-normal">Branches</span>
        </p>
      </div>

      {/* Empty state */}
      {tree == null ? (
        <p className="text-[11px] text-[var(--text-muted)]">
          运行后显示分支树 / Branch tree appears after a run
        </p>
      ) : (
        <div className="flex flex-col gap-1">
          <BranchNode
            node={tree}
            activeRunId={activeRunId}
            onSelectRun={onSelectRun}
            disabled={disabled}
          />
        </div>
      )}
    </div>
  );
}
