// BranchCanvasNode.tsx — custom React Flow v12 node components (V2.1-4).
// Pure presentational. No data fetching. No app state.
import { memo } from "react";
import { Handle, Position, type NodeProps, type NodeTypes } from "@xyflow/react";
import { ChevronDown, ChevronRight, Crop, Dices, GitBranch, GitMerge, Image, Layers, Loader, Sparkles, Star, X } from "lucide-react";
import type { BranchCanvasNode } from "../lib/branchCanvasModel";
import { fmtScore, truncate } from "../lib/format";

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

export const RunNode = memo(function RunNode({ data, selected }: NodeProps<BranchCanvasNode>) {
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
});

// ─── 2. CheckpointNode ────────────────────────────────────────────────────────

export const CheckpointNode = memo(function CheckpointNode({ data, selected }: NodeProps<BranchCanvasNode>) {
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
            className={`font-mono ${deltaPositive ? "text-emerald-400" : deltaNegative ? "text-red-400" : ""}`}
          >
            {deltaPositive ? "▲" : "▼"}{Math.abs(deltaNum).toFixed(3)}
          </span>
        )}
      </div>

      <Handle type="source" position={Position.Bottom} />
    </div>
  );
});

// ─── 3. BranchActionNode ─────────────────────────────────────────────────────

export const BranchActionNode = memo(function BranchActionNode({ data, selected }: NodeProps<BranchCanvasNode>) {
  const ringClass = selected ? "ring-2 ring-emerald-500" : "";

  const sourceRef = data.source_checkpoint_id ?? data.run_id;

  return (
    <div
      className={`rounded-lg text-[11px] flex flex-col gap-1 px-2.5 py-2 transition-all ${ringClass}`}
      style={{
        width: 160,
        background: "color-mix(in srgb, var(--bg-secondary) 70%, transparent)",
        border: "1.5px dashed var(--accent-primary)",
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
});

// ─── 4. InputNode ─────────────────────────────────────────────────────────────

export const InputNode = memo(function InputNode({ data, selected }: NodeProps<BranchCanvasNode>) {
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
});

// ─── 5. VariantGroupNode ──────────────────────────────────────────────────────

export const VariantGroupNode = memo(function VariantGroupNode({ data, selected }: NodeProps<BranchCanvasNode>) {
  const ringClass = selected ? "ring-2 ring-emerald-500" : "";

  return (
    <div
      className={`rounded-lg border text-[11px] flex flex-col gap-1 px-2.5 py-2 shadow-sm transition-all ${ringClass}`}
      style={{
        width: 190,
        background: "var(--bg-secondary)",
        borderColor: selected ? "var(--accent-primary)" : "var(--border-color)",
        color: "var(--text-primary)",
      }}
    >
      <Handle type="target" position={Position.Top} />

      {/* Header row: icon + label + collapse chevron + status */}
      <div className="flex items-center gap-1.5">
        <span title="variant group" className="flex-shrink-0">
          <Layers className="w-3.5 h-3.5 text-emerald-400" />
        </span>
        <span
          className="flex-1 truncate font-medium"
          style={{ color: "var(--text-primary)" }}
          title={data.label}
        >
          {data.label}
        </span>
        {/* Collapse affordance — purely visual */}
        {data.collapsed ? (
          <span title="collapsed" className="flex-shrink-0">
            <ChevronRight className="w-3.5 h-3.5" style={{ color: "var(--text-muted)" }} />
          </span>
        ) : (
          <span title="expanded" className="flex-shrink-0">
            <ChevronDown className="w-3.5 h-3.5" style={{ color: "var(--text-muted)" }} />
          </span>
        )}
        <StatusDot status={data.status} />
      </div>

      <Handle type="source" position={Position.Bottom} />
    </div>
  );
});

// ─── 6. VariantRunNode ────────────────────────────────────────────────────────

export const VariantRunNode = memo(function VariantRunNode({ data, selected }: NodeProps<BranchCanvasNode>) {
  const ringClass = selected ? "ring-2 ring-emerald-500" : "";

  const indexLabel =
    typeof data.variant_index === "number" ? `#${data.variant_index}` : null;

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

      {/* Header row: index badge + label + favorite star */}
      <div className="flex items-center gap-1.5">
        {indexLabel !== null && (
          <span
            className="font-mono rounded px-0.5 flex-shrink-0 text-[10px]"
            style={{ background: "var(--bg-tertiary)", color: "var(--text-muted)" }}
          >
            {indexLabel}
          </span>
        )}
        <span
          className="flex-1 truncate font-medium"
          style={{ color: "var(--text-primary)" }}
          title={data.label}
        >
          {data.label}
        </span>
        {data.favorite && (
          <span title="winner / favorite" className="flex-shrink-0">
            <Star className="w-3 h-3 text-emerald-400" fill="currentColor" />
          </span>
        )}
      </div>

      {/* Score + status dot */}
      <div className="flex items-center gap-1.5">
        <StatusDot status={data.status} />
        <span style={{ color: "var(--text-muted)" }}>
          {fmtScore(data.score)}
        </span>
      </div>

      <Handle type="source" position={Position.Bottom} />
    </div>
  );
});

// ─── 7. DrawSessionNode ───────────────────────────────────────────────────────

export const DrawSessionNode = memo(function DrawSessionNode({ data, selected }: NodeProps<BranchCanvasNode>) {
  const ringClass = selected ? "ring-2 ring-emerald-500" : "";

  const completedCount = (data.completed_count as number | undefined) ?? 0;
  const cardCount = (data.card_count as number | undefined) ?? 0;
  const runningCount = (data.running_count as number | undefined) ?? 0;
  const failedCount = (data.failed_count as number | undefined) ?? 0;

  return (
    <div
      className={`rounded-lg border text-[11px] flex flex-col gap-1 px-2.5 py-2 shadow-sm transition-all ${ringClass}`}
      style={{
        width: 190,
        background: "var(--bg-secondary)",
        borderColor: selected ? "var(--accent-primary)" : "var(--border-color)",
        color: "var(--text-primary)",
      }}
    >
      <Handle type="target" position={Position.Top} />

      {/* Header row: gacha icon + label + collapse chevron + status */}
      <div className="flex items-center gap-1.5">
        <span title="draw session" className="flex-shrink-0">
          <Dices className="w-3.5 h-3.5 text-emerald-400" />
        </span>
        <span
          className="flex-1 truncate font-medium"
          style={{ color: "var(--text-primary)" }}
          title={data.label}
        >
          {data.label}
        </span>
        {/* Collapse affordance — purely visual */}
        {data.collapsed ? (
          <span title="collapsed" className="flex-shrink-0">
            <ChevronRight className="w-3.5 h-3.5" style={{ color: "var(--text-muted)" }} />
          </span>
        ) : (
          <span title="expanded" className="flex-shrink-0">
            <ChevronDown className="w-3.5 h-3.5" style={{ color: "var(--text-muted)" }} />
          </span>
        )}
        <StatusDot status={data.status} />
      </div>

      {/* Progress line */}
      <div className="flex items-center gap-1 flex-wrap" style={{ color: "var(--text-muted)" }}>
        <span>{completedCount}/{cardCount} done</span>
        {runningCount > 0 && <span>· {runningCount} running</span>}
        {failedCount > 0 && <span>· {failedCount} failed</span>}
      </div>

      {/* Winner star */}
      {!!data.winner_run_id && (
        <div className="flex items-center gap-1">
          <Star className="w-3 h-3 text-emerald-400 flex-shrink-0" fill="currentColor" />
          <span style={{ color: "var(--text-muted)" }}>winner</span>
        </div>
      )}

      <Handle type="source" position={Position.Bottom} />
    </div>
  );
});

// ─── 8. DrawCardNode ──────────────────────────────────────────────────────────

export const DrawCardNode = memo(function DrawCardNode({ data, selected }: NodeProps<BranchCanvasNode>) {
  const ringClass = selected ? "ring-2 ring-emerald-500" : "";

  const indexLabel = typeof data.index === "number" ? `#${data.index}` : null;
  const isWinnerOrFavorite = data.is_winner || data.favorite;
  const isEliminated = data.eliminated === true;

  return (
    <div
      className={`rounded-lg border text-[11px] flex flex-col gap-1 px-2 py-1.5 shadow-sm transition-all ${ringClass} ${isEliminated ? "opacity-50" : ""}`}
      style={{
        width: 150,
        background: "var(--bg-secondary)",
        borderColor: selected ? "var(--accent-primary)" : "var(--border-color)",
        color: "var(--text-primary)",
      }}
    >
      <Handle type="target" position={Position.Top} />

      {/* Header row: index badge + label + winner/favorite star */}
      <div className="flex items-center gap-1.5">
        {indexLabel !== null && (
          <span
            className="font-mono rounded px-0.5 flex-shrink-0 text-[10px]"
            style={{ background: "var(--bg-tertiary)", color: "var(--text-muted)" }}
          >
            {indexLabel}
          </span>
        )}
        <span
          className={`flex-1 truncate font-medium ${isEliminated ? "line-through" : ""}`}
          style={{ color: "var(--text-primary)" }}
          title={data.label}
        >
          {data.label}
        </span>
        {isWinnerOrFavorite && (
          <span title={data.is_winner ? "winner" : "favorite"} className="flex-shrink-0">
            <Star className="w-3 h-3 text-emerald-400" fill="currentColor" />
          </span>
        )}
      </div>

      {/* Status dot + score */}
      <div className="flex items-center gap-1.5">
        <StatusDot status={data.status} />
        {typeof data.final_score === "number" && (
          <span style={{ color: "var(--text-muted)" }}>
            {fmtScore(data.final_score)}
          </span>
        )}
        {/* Fusion affordance — visual placeholder for V4.5 */}
        {!!data.can_use_for_fusion && (
          <span title="can be used as fusion source" className="flex-shrink-0 ml-auto">
            <Sparkles className="w-3 h-3" style={{ color: "var(--text-muted)" }} />
          </span>
        )}
      </div>

      <Handle type="source" position={Position.Bottom} />
    </div>
  );
});

// ─── 9. RegionConstraintNode ─────────────────────────────────────────────────

export const RegionConstraintNode = memo(function RegionConstraintNode({ data, selected }: NodeProps<BranchCanvasNode>) {
  const ringClass = selected ? "ring-2 ring-emerald-500" : "";
  const mode = data.mode as string | undefined;
  const isProtect = mode === "protect";
  const modeBgClass = isProtect ? "bg-amber-500/20 text-amber-400" : "bg-emerald-500/20 text-emerald-400";
  const strength = typeof data.strength === "number" ? data.strength : null;
  const instruction = typeof data.instruction === "string" ? data.instruction : "";

  return (
    <div
      className={`rounded-lg border text-[11px] flex flex-col gap-1 px-2 py-1.5 shadow-sm transition-all ${ringClass}`}
      style={{
        width: 160,
        background: "var(--bg-secondary)",
        borderColor: selected ? "var(--accent-primary)" : "var(--border-color)",
        color: "var(--text-primary)",
      }}
    >
      <Handle type="target" position={Position.Top} />

      {/* Header row: icon + label + mode badge */}
      <div className="flex items-center gap-1.5">
        <Crop
          className="w-3 h-3 flex-shrink-0"
          style={{ color: "var(--accent-primary)" }}
        />
        <span
          className="flex-1 truncate font-medium"
          style={{ color: "var(--text-primary)" }}
          title={data.label}
        >
          {data.label}
        </span>
        {mode && (
          <span className={`text-[10px] px-1 rounded font-medium flex-shrink-0 ${modeBgClass}`}>
            {mode}
          </span>
        )}
      </div>

      {/* Strength */}
      {strength !== null && (
        <div className="flex items-center gap-1" style={{ color: "var(--text-muted)" }}>
          <span>strength</span>
          <span className="font-mono" style={{ color: "var(--text-secondary)" }}>
            {strength.toFixed(2)}
          </span>
        </div>
      )}

      {/* Instruction (truncated) */}
      {instruction && (
        <div
          className="truncate italic text-[10px]"
          style={{ color: "var(--text-muted)" }}
          title={instruction}
        >
          {truncate(instruction, 36)}
        </div>
      )}
    </div>
  );
});

// ─── 10. FusionPlanNode ───────────────────────────────────────────────────────

export const FusionPlanNode = memo(function FusionPlanNode({ data, selected }: NodeProps<BranchCanvasNode>) {
  const ringClass = selected ? "ring-2 ring-emerald-500" : "";

  const status = data.status as string | undefined;
  const regionCount = (data.region_count as number | undefined) ?? 0;
  const outputRunId = data.output_run_id as string | null | undefined;

  // Status badge color
  let statusBgClass = "bg-[var(--bg-tertiary)] text-[color:var(--text-muted)]";
  if (status === "completed" || status === "target_ready") {
    statusBgClass = "bg-emerald-500/20 text-emerald-400";
  } else if (status === "running") {
    statusBgClass = "bg-amber-500/20 text-amber-400";
  } else if (status === "failed") {
    statusBgClass = "bg-red-500/20 text-red-400";
  }

  return (
    <div
      className={`rounded-lg border text-[11px] flex flex-col gap-1 px-2.5 py-2 shadow-sm transition-all ${ringClass}`}
      style={{
        width: 190,
        background: "var(--bg-secondary)",
        borderColor: selected ? "var(--accent-primary)" : "var(--border-color)",
        color: "var(--text-primary)",
      }}
    >
      <Handle type="target" position={Position.Top} />

      {/* Header row: merge icon + label + status badge */}
      <div className="flex items-center gap-1.5">
        <span title="fusion plan" className="flex-shrink-0">
          <GitMerge className="w-3.5 h-3.5 text-emerald-400" />
        </span>
        <span
          className="flex-1 truncate font-medium"
          style={{ color: "var(--text-primary)" }}
          title={data.label}
        >
          {data.label}
        </span>
        {status && (
          <span className={`text-[10px] px-1 rounded font-medium flex-shrink-0 ${statusBgClass}`}>
            {status === "running" ? (
              <Loader className="w-2.5 h-2.5 animate-spin inline mr-0.5" />
            ) : null}
            {status}
          </span>
        )}
      </div>

      {/* Region count */}
      <div className="flex items-center gap-1" style={{ color: "var(--text-muted)" }}>
        <Crop className="w-3 h-3 flex-shrink-0" />
        <span>{regionCount} region{regionCount !== 1 ? "s" : ""}</span>
      </div>

      {/* Output hint */}
      {outputRunId && (
        <div
          className="truncate text-[10px]"
          style={{ color: "var(--text-muted)" }}
        >
          → output
        </div>
      )}

      <Handle type="source" position={Position.Bottom} />
    </div>
  );
});

// ─── Stable nodeTypes export ──────────────────────────────────────────────────
// Module constant — satisfies React Flow's nodeTypes stability contract.

export const branchCanvasNodeTypes: NodeTypes = {
  input: InputNode,
  run: RunNode,
  checkpoint: CheckpointNode,
  branch_action: BranchActionNode,
  variant_group: VariantGroupNode,
  variant_run: VariantRunNode,
  draw_session: DrawSessionNode,
  draw_card: DrawCardNode,
  region_constraint: RegionConstraintNode,
  fusion_plan: FusionPlanNode,
};
