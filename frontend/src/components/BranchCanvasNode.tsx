// BranchCanvasNode.tsx — custom React Flow v12 node components (V2.1-4).
// Pure presentational. No data fetching. No app state.
import { Handle, Position, type NodeProps, type NodeTypes } from "@xyflow/react";
import { ChevronRight, GitBranch, Image, Loader, Star, X } from "lucide-react";
import type { BranchCanvasNode } from "../lib/branchCanvasModel";

// ─── Shared helpers ───────────────────────────────────────────────────────────

function fmtScore(score: number | null | undefined): string {
  return typeof score === "number" ? score.toFixed(3) : "—";
}

function truncate(text: string, max: number): string {
  return text.length > max ? text.slice(0, max - 1) + "…" : text;
}

// ─── Status dot/spinner (reusable within this file) ──────────────────────────

function StatusDot({ status }: { status: string | undefined }) {
  if (status === "running") {
    return <Loader className="w-3 h-3 text-amber-400 animate-spin flex-shrink-0" />;
  }
  if (status === "completed") {
    return <span className="w-2 h-2 rounded-full bg-emerald-500 flex-shrink-0" title="completed" />;
  }
  if (status === "failed") {
    return <span className="w-2 h-2 rounded-full bg-red-500 flex-shrink-0" title="failed" />;
  }
  return (
    <span
      className="w-2 h-2 rounded-full flex-shrink-0 opacity-50"
      style={{ background: "var(--text-muted)" }}
      title={status ?? "pending"}
    />
  );
}

// ─── 1. RunNode ───────────────────────────────────────────────────────────────

export function RunNode({ data, selected }: NodeProps<BranchCanvasNode>) {
  const ringClass = selected ? "ring-2 ring-emerald-500" : "";

  return (
    <div
      className={`rounded-lg border text-[11px] flex flex-col gap-1 px-2.5 py-2 shadow-sm transition-all ${ringClass}`}
      style={{
        width: 180,
        background: "var(--bg-secondary)",
        borderColor: selected ? "var(--accent-primary)" : "var(--border-color)",
        color: "var(--text-primary)",
      }}
    >
      <Handle type="target" position={Position.Top} />

      {/* Header row: status dot + label + favorite star */}
      <div className="flex items-center gap-1.5">
        <StatusDot status={data.status} />
        <span
          className="flex-1 truncate font-medium"
          style={{ color: "var(--text-primary)" }}
          title={data.label}
        >
          {data.label}
        </span>
        {data.favorite && (
          <Star className="w-3 h-3 flex-shrink-0 text-yellow-400" fill="currentColor" />
        )}
        {data.collapsed && (
          <span
            className="flex items-center gap-0.5 px-1 rounded text-[10px] font-medium flex-shrink-0"
            style={{ background: "var(--bg-tertiary)", color: "var(--text-muted)" }}
          >
            <ChevronRight className="w-2.5 h-2.5" />
            collapsed
          </span>
        )}
      </div>

      {/* Score */}
      {typeof data.score === "number" && (
        <div style={{ color: "var(--text-muted)" }}>
          score <span style={{ color: "var(--text-secondary)" }}>{fmtScore(data.score)}</span>
        </div>
      )}

      {/* Source checkpoint */}
      {data.source_checkpoint_id && (
        <div className="font-mono truncate" style={{ color: "var(--text-muted)" }}>
          from {data.source_checkpoint_id}
        </div>
      )}

      {/* Feedback preview */}
      {data.feedback && (
        <div
          className="truncate italic"
          style={{ color: "var(--text-muted)" }}
          title={data.feedback}
        >
          {truncate(data.feedback, 40)}
        </div>
      )}

      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

// ─── 2. CheckpointNode ────────────────────────────────────────────────────────

export function CheckpointNode({ data, selected }: NodeProps<BranchCanvasNode>) {
  const ringClass = selected ? "ring-2 ring-emerald-500" : "";

  const deltaNum = typeof data.delta === "number" ? data.delta : null;
  const deltaPositive = deltaNum !== null && deltaNum > 0;
  const deltaNegative = deltaNum !== null && deltaNum < 0;
  const deltaZero = deltaNum === null || deltaNum === 0;

  return (
    <div
      className={`rounded-lg border text-[11px] flex flex-col gap-1 px-2 py-1.5 shadow-sm transition-all ${ringClass}`}
      style={{
        width: 150,
        background: "var(--bg-secondary)",
        borderColor: selected ? "var(--accent-primary)" : "var(--border-color)",
        color: "var(--text-primary)",
      }}
    >
      <Handle type="target" position={Position.Top} />

      {/* Label row + rejected marker */}
      <div className="flex items-center gap-1">
        <span
          className="flex-1 truncate font-medium"
          style={{ color: "var(--text-primary)" }}
          title={data.label}
        >
          {data.label}
        </span>
        {data.accepted === false && (
          <span title="rejected" className="flex-shrink-0">
            <X className="w-3 h-3 text-red-400" />
          </span>
        )}
      </div>

      {/* Thumbnail skeleton — actual lazy-load is out of scope */}
      <div
        className="w-full h-16 rounded"
        style={{ background: "var(--bg-tertiary)" }}
      />

      {/* Score + delta */}
      <div className="flex items-center gap-1.5">
        {typeof data.score === "number" && (
          <span style={{ color: "var(--text-muted)" }}>
            {fmtScore(data.score)}
          </span>
        )}
        {!deltaZero && deltaNum !== null && (
          <span
            className="font-mono"
            style={{ color: deltaPositive ? "#10b981" : deltaNegative ? "#f87171" : "var(--text-muted)" }}
          >
            {deltaPositive ? "▲" : "▼"}{Math.abs(deltaNum).toFixed(3)}
          </span>
        )}
      </div>

      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

// ─── 3. BranchActionNode ─────────────────────────────────────────────────────

export function BranchActionNode({ data, selected }: NodeProps<BranchCanvasNode>) {
  const ringClass = selected ? "ring-2 ring-emerald-500" : "";

  const sourceRef = data.source_checkpoint_id ?? data.run_id;

  return (
    <div
      className={`rounded-lg text-[11px] flex flex-col gap-1 px-2.5 py-2 transition-all ${ringClass}`}
      style={{
        width: 160,
        background: "color-mix(in srgb, var(--bg-secondary) 70%, transparent)",
        border: "1.5px dashed #10b981",
        color: "var(--text-primary)",
        opacity: 0.88,
      }}
    >
      <Handle type="target" position={Position.Top} />

      {/* Header */}
      <div className="flex items-center gap-1.5">
        <GitBranch className="w-3.5 h-3.5 text-emerald-400 flex-shrink-0" />
        <span className="font-semibold text-emerald-400">
          新分支
          <span className="ml-1 font-normal" style={{ color: "var(--text-muted)" }}>
            New branch
          </span>
        </span>
      </div>

      {/* Source reference */}
      {sourceRef && (
        <div className="font-mono truncate" style={{ color: "var(--text-muted)" }}>
          from {sourceRef}
        </div>
      )}

      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

// ─── 4. InputNode ─────────────────────────────────────────────────────────────

export function InputNode({ data, selected }: NodeProps<BranchCanvasNode>) {
  const ringClass = selected ? "ring-2 ring-emerald-500" : "";

  return (
    <div
      className={`rounded-lg border text-[11px] flex items-center gap-1.5 px-2.5 py-2 shadow-sm transition-all ${ringClass}`}
      style={{
        width: 120,
        background: "var(--bg-secondary)",
        borderColor: selected ? "var(--accent-primary)" : "var(--border-color)",
        color: "var(--text-primary)",
      }}
    >
      <Handle type="target" position={Position.Top} />

      <Image
        className="w-3.5 h-3.5 flex-shrink-0"
        style={{ color: "var(--accent-primary)" }}
      />
      <span className="truncate font-medium" style={{ color: "var(--text-primary)" }}>
        {data.label}
      </span>

      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

// ─── Stable nodeTypes export ──────────────────────────────────────────────────
// Module constant — satisfies React Flow's nodeTypes stability contract.

export const branchCanvasNodeTypes: NodeTypes = {
  input: InputNode,
  run: RunNode,
  checkpoint: CheckpointNode,
  branch_action: BranchActionNode,
};
