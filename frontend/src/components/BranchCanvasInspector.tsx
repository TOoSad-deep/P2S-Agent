// BranchCanvasInspector.tsx — right-side inspector for the Branch Canvas Workspace (V2.1-5).
// Switches content by selected node type: null / input / run / checkpoint / branch_action / variant_group / variant_run / reserved.
// Presentational + callbacks only; branch-draft form owns local state. No data fetching.
import { useState, useEffect } from "react";
import { Search, Star, GitBranch, CheckCircle, XCircle, GitMerge, ThumbsUp, ThumbsDown, StopCircle, Layers2, ChevronDown, ChevronRight } from "lucide-react";
import type { BranchCanvasNode } from "../lib/branchCanvasModel";
import type {
  BranchMode,
  BranchRefineRequest,
  ExploreVariantsRequest,
  VariantGroupStatus,
  VariantStatusEntry,
  DrawSessionStatus,
  CreateDrawSessionRequest,
  DrawMoreRequest,
  DrawCardEventType,
  HumanConstraintSpec,
} from "../hooks/usePngShader";
import { fmtScore } from "../lib/format";
import DrawSessionInspector from "./DrawSessionInspector";
import FineControlPanel, { DEFAULT_CONSTRAINT_SPEC, isMeaningfulConstraint } from "./FineControlPanel";

// ─── Modes & Locks (mirrors HumanLoopPanel) ──────────────────────────────────

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

// ─── Props ────────────────────────────────────────────────────────────────────

interface Props {
  node: BranchCanvasNode | null;
  activeRunId: string | null;
  onSwitchRun: (runId: string) => void;
  onUpdateMetadata: (runId: string, patch: { title?: string; favorite?: boolean }) => void;
  onRefineFromCheckpoint: (runId: string, checkpointId: string) => void;
  onContinueFromRun: (runId: string) => void;
  onSubmitBranch: (runId: string, request: BranchRefineRequest) => void;
  onCancelBranch: () => void;
  // V3 variant props (optional — used by BranchActionView now; others wired for next task)
  onExploreVariants?: (parentRunId: string, request: ExploreVariantsRequest) => void;
  variantGroup?: VariantGroupStatus | null;
  onSelectWinner?: (groupId: string, winnerRunId: string, reason?: string) => void;
  onStopGroup?: (groupId: string) => void;
  onRateVariant?: (groupId: string, runId: string, rating: number) => void;
  // V3.5 draw-session props
  drawSession?: DrawSessionStatus | null;
  onStartDraw?: (parentRunId: string, request: CreateDrawSessionRequest) => void;
  onDrawMore?: (drawId: string, request: DrawMoreRequest) => void;
  onRedrawCard?: (drawId: string, runId: string) => void;
  onCardEvent?: (
    drawId: string,
    runId: string,
    eventType: DrawCardEventType,
    opts?: { value?: unknown; reason?: string; tags?: string[] },
  ) => void;
  onPreviewCard?: (runId: string) => void;
  onContinueCard?: (runId: string) => void;
  onSelectDrawWinner?: (drawId: string, runId: string) => void;
  onStopDraw?: (drawId: string) => void;
  fusionEnabled?: boolean;
  submitError?: string | null;
  disabled?: boolean;
}

// ─── Panel wrapper ────────────────────────────────────────────────────────────

function PanelShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-2.5 px-3 py-2.5 bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-lg h-full overflow-y-auto">
      {/* Header */}
      <div className="flex items-center gap-2 shrink-0">
        <Search className="w-4 h-4 text-[var(--accent-primary)] flex-shrink-0" />
        <p className="text-xs font-medium text-[var(--text-primary)] leading-tight">
          检查器
          <span className="ml-2 text-[var(--text-muted)] font-normal">Inspector</span>
        </p>
      </div>
      {children}
    </div>
  );
}

// ─── Sub-views ────────────────────────────────────────────────────────────────

function NoNodeView() {
  return (
    <p className="text-[11px] text-[var(--text-muted)] py-4 text-center">
      选择一个节点查看详情
      <br />
      <span className="opacity-70">Select a node</span>
    </p>
  );
}

function InputNodeView({ label }: { label: string }) {
  return (
    <div className="flex flex-col gap-1.5">
      <p className="text-[11px] font-medium text-[var(--text-secondary)]">
        输入图像 <span className="text-[var(--text-muted)] font-normal">Input image</span>
      </p>
      <p className="text-[11px] text-[var(--text-primary)] font-mono truncate">{label}</p>
    </div>
  );
}

interface RunNodeViewProps {
  node: BranchCanvasNode;
  activeRunId: string | null;
  onSwitchRun: (runId: string) => void;
  onUpdateMetadata: (runId: string, patch: { title?: string; favorite?: boolean }) => void;
  onContinueFromRun: (runId: string) => void;
  disabled?: boolean;
}

function RunNodeView({
  node,
  activeRunId,
  onSwitchRun,
  onUpdateMetadata,
  onContinueFromRun,
  disabled,
}: RunNodeViewProps) {
  const data = node.data;
  const runId = data.run_id!;

  // Local state for the editable title
  const [titleDraft, setTitleDraft] = useState(data.title ?? "");

  // Sync seed when node changes (parent effect resets too, but be defensive)
  useEffect(() => {
    setTitleDraft(data.title ?? "");
  }, [node.id, data.title]);

  const commitTitle = () => {
    const trimmed = titleDraft.trim();
    onUpdateMetadata(runId, { title: trimmed || undefined });
  };

  const handleTitleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.currentTarget.blur();
    }
  };

  const isCurrent = runId === activeRunId;

  return (
    <div className="flex flex-col gap-2">
      {/* Status + score row */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-[11px] px-1.5 py-0.5 rounded bg-[var(--bg-tertiary)] text-[var(--text-secondary)] font-mono">
          {data.status ?? "—"}
        </span>
        <span className="text-[11px] text-[var(--text-muted)]">
          score{" "}
          <span className="text-[var(--text-primary)] font-mono">{fmtScore(data.score)}</span>
        </span>
        {isCurrent && (
          <span className="text-[11px] px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400 font-medium">
            活跃 Active
          </span>
        )}
      </div>

      {/* Source checkpoint */}
      {data.source_checkpoint_id && (
        <p className="text-[11px] text-[var(--text-muted)]">
          来自检查点 <span className="font-mono text-[var(--text-secondary)]">{data.source_checkpoint_id}</span>
        </p>
      )}

      {/* Feedback */}
      {data.feedback && (
        <div className="text-[11px] text-[var(--text-secondary)] bg-[var(--bg-tertiary)] rounded p-2 leading-relaxed">
          <span className="text-[var(--text-muted)]">反馈 Feedback: </span>
          {data.feedback}
        </div>
      )}

      {/* Editable title */}
      <div className="flex flex-col gap-1">
        <label className="text-[11px] text-[var(--text-muted)]">标题 Title</label>
        <div className="flex items-center gap-1.5">
          <input
            type="text"
            value={titleDraft}
            onChange={(e) => setTitleDraft(e.target.value)}
            onBlur={commitTitle}
            onKeyDown={handleTitleKeyDown}
            disabled={disabled}
            placeholder={runId.slice(-8)}
            className="flex-1 text-xs px-2 py-1 rounded bg-[var(--bg-tertiary)] border border-[var(--border-color)] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] disabled:opacity-40"
          />
          {/* Favorite star */}
          <button
            onClick={() => onUpdateMetadata(runId, { favorite: !data.favorite })}
            disabled={disabled}
            title={data.favorite ? "取消收藏 Unfavorite" : "收藏 Favorite"}
            className={`p-1 rounded transition-all disabled:opacity-40 ${
              data.favorite
                ? "text-yellow-400 hover:text-yellow-300"
                : "text-[var(--text-muted)] hover:text-[var(--text-primary)]"
            }`}
          >
            <Star className={`w-3.5 h-3.5 ${data.favorite ? "fill-current" : ""}`} />
          </button>
        </div>
      </div>

      {/* Run ID (readonly, small) */}
      <p className="text-[10px] text-[var(--text-muted)] font-mono truncate" title={runId}>
        {runId}
      </p>

      {/* Actions */}
      <div className="flex flex-col gap-1.5 pt-1">
        <button
          onClick={() => onSwitchRun(runId)}
          disabled={disabled || isCurrent}
          className="flex items-center justify-center gap-1.5 px-3 py-1.5 text-[11px] rounded-md transition-all disabled:opacity-40 disabled:cursor-not-allowed bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]"
        >
          <GitBranch className="w-3 h-3" />
          切换到此分支 / Switch to this run
        </button>
        <button
          onClick={() => onContinueFromRun(runId)}
          disabled={disabled}
          className="flex items-center justify-center gap-1.5 px-3 py-1.5 text-[11px] rounded-md transition-all disabled:opacity-40 disabled:cursor-not-allowed bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]"
        >
          <GitMerge className="w-3 h-3" />
          从最终结果继续 / Continue from final
        </button>
      </div>
    </div>
  );
}

interface CheckpointNodeViewProps {
  node: BranchCanvasNode;
  onRefineFromCheckpoint: (runId: string, checkpointId: string) => void;
  disabled?: boolean;
}

function CheckpointNodeView({ node, onRefineFromCheckpoint, disabled }: CheckpointNodeViewProps) {
  const data = node.data;
  const runId = data.run_id!;
  const checkpointId = data.checkpoint_id!;

  const deltaPositive = typeof data.delta === "number" && data.delta > 0;
  const deltaNegative = typeof data.delta === "number" && data.delta < 0;
  const deltaZero = typeof data.delta === "number" && data.delta === 0;

  return (
    <div className="flex flex-col gap-2">
      {/* Label + score */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-[11px] font-medium text-[var(--text-primary)]">{data.label}</span>
        <span className="text-[11px] text-[var(--text-muted)]">
          score <span className="font-mono text-[var(--text-primary)]">{fmtScore(data.score)}</span>
        </span>
      </div>

      {/* Delta */}
      {typeof data.delta === "number" && !deltaZero && (
        <p className={`text-[11px] font-mono ${deltaPositive ? "text-emerald-400" : deltaNegative ? "text-red-400" : "text-[var(--text-muted)]"}`}>
          {deltaPositive ? "▲" : "▼"} {Math.abs(data.delta).toFixed(3)}
        </p>
      )}

      {/* Accepted / rejected */}
      {data.accepted !== null && data.accepted !== undefined && (
        <div className={`flex items-center gap-1.5 text-[11px] ${data.accepted ? "text-emerald-400" : "text-red-400"}`}>
          {data.accepted ? (
            <><CheckCircle className="w-3 h-3" /> 已接受 Accepted</>
          ) : (
            <><XCircle className="w-3 h-3" /> 已拒绝 Rejected</>
          )}
        </div>
      )}

      {/* Changes summary */}
      {data.changes_summary && (
        <div className="text-[11px] text-[var(--text-secondary)] bg-[var(--bg-tertiary)] rounded p-2 leading-relaxed">
          <span className="text-[var(--text-muted)]">变更摘要 Summary: </span>
          {data.changes_summary}
        </div>
      )}

      {/* Checkpoint ID */}
      <p className="text-[10px] text-[var(--text-muted)] font-mono truncate" title={checkpointId}>
        {checkpointId}
      </p>

      {/* Action */}
      <button
        onClick={() => onRefineFromCheckpoint(runId, checkpointId)}
        disabled={disabled}
        className="flex items-center justify-center gap-1.5 px-3 py-1.5 text-[11px] rounded-md transition-all disabled:opacity-40 disabled:cursor-not-allowed bg-gradient-to-r from-emerald-500 to-emerald-600 hover:from-emerald-400 hover:to-emerald-500 text-white font-medium"
      >
        <GitBranch className="w-3 h-3" />
        从此处定向优化 / Refine from here
      </button>
    </div>
  );
}

const VARIANT_COUNTS = [2, 3, 4, 6] as const;
type VariantCount = typeof VARIANT_COUNTS[number];
type Diversity = "low" | "medium" | "high";

interface BranchActionViewProps {
  node: BranchCanvasNode;
  onSubmitBranch: (runId: string, request: BranchRefineRequest) => void;
  onCancelBranch: () => void;
  onExploreVariants?: (parentRunId: string, request: ExploreVariantsRequest) => void;
  // V3.5 batch-draw entry (start form rendered when drawMode is active)
  onStartDraw?: (parentRunId: string, request: CreateDrawSessionRequest) => void;
  onDrawMore?: (drawId: string, request: DrawMoreRequest) => void;
  onRedrawCard?: (drawId: string, runId: string) => void;
  onCardEvent?: (
    drawId: string,
    runId: string,
    eventType: DrawCardEventType,
    opts?: { value?: unknown; reason?: string; tags?: string[] },
  ) => void;
  onPreviewCard?: (runId: string) => void;
  onContinueCard?: (runId: string) => void;
  fusionEnabled?: boolean;
  submitError?: string | null;
  disabled?: boolean;
}

function BranchActionView({
  node,
  onSubmitBranch,
  onCancelBranch,
  onExploreVariants,
  onStartDraw,
  onDrawMore,
  onRedrawCard,
  onCardEvent,
  onPreviewCard,
  onContinueCard,
  fusionEnabled,
  submitError,
  disabled,
}: BranchActionViewProps) {
  const data = node.data;
  const runId = data.run_id!;

  const [feedback, setFeedback] = useState("");
  const [mode, setMode] = useState<BranchMode>("refine");
  const [locks, setLocks] = useState<Record<string, boolean>>({});
  const [exploreMode, setExploreMode] = useState(false);
  const [drawMode, setDrawMode] = useState(false);
  const [variantCount, setVariantCount] = useState<VariantCount>(4);
  const [diversity, setDiversity] = useState<Diversity>("medium");
  const [constraintSpec, setConstraintSpec] = useState<HumanConstraintSpec>(DEFAULT_CONSTRAINT_SPEC);
  const [fineControlOpen, setFineControlOpen] = useState(false);

  const sourceCheckpointId = data.source_checkpoint_id ?? data.checkpoint_id ?? "final:selected";

  // Batch-draw mode hands the whole start form to DrawSessionInspector (session=null).
  if (drawMode && onStartDraw) {
    return (
      <div className="flex flex-col gap-2">
        <DrawSessionInspector
          parentRunId={runId}
          checkpointId={sourceCheckpointId}
          session={null}
          fusionEnabled={fusionEnabled}
          onStartDraw={onStartDraw}
          onDrawMore={onDrawMore ?? (() => {})}
          onRedrawCard={onRedrawCard ?? (() => {})}
          onCardEvent={onCardEvent ?? (() => {})}
          onPreviewCard={onPreviewCard ?? (() => {})}
          onContinueCard={onContinueCard ?? (() => {})}
          disabled={disabled}
          error={submitError}
        />
        <button
          onClick={() => setDrawMode(false)}
          disabled={disabled}
          className="px-3 py-1.5 text-[11px] rounded-md transition-all disabled:opacity-40 bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]"
        >
          返回分支 / Back to branch
        </button>
      </div>
    );
  }

  const feedbackRequired = exploreMode || mode === "refine" || mode === "polish";
  const canSubmit = !disabled && (!feedbackRequired || feedback.trim().length > 0);

  const toggleLock = (key: string) =>
    setLocks((prev) => ({ ...prev, [key]: !prev[key] }));

  const handleSubmit = () => {
    if (!canSubmit) return;
    const checkpointId = data.source_checkpoint_id ?? data.checkpoint_id ?? "final:selected";
    const constraintsPayload = isMeaningfulConstraint(constraintSpec) ? constraintSpec : undefined;
    if (exploreMode && onExploreVariants) {
      onExploreVariants(runId, {
        checkpoint_id: checkpointId,
        feedback: feedback.trim(),
        variant_count: variantCount,
        diversity,
        mode: "explore",
        ...(constraintsPayload !== undefined ? { constraints: constraintsPayload } : {}),
      });
    } else {
      onSubmitBranch(runId, {
        checkpoint_id: checkpointId,
        feedback: feedback.trim(),
        mode,
        locks,
        ...(constraintsPayload !== undefined ? { constraints: constraintsPayload } : {}),
      });
    }
  };

  return (
    <div className="flex flex-col gap-2">
      {/* Source info */}
      {data.source_checkpoint_id && (
        <p className="text-[11px] text-[var(--text-muted)]">
          起点 Start:{" "}
          <span className="font-mono text-[var(--text-secondary)]">{data.source_checkpoint_id}</span>
        </p>
      )}

      {/* Feedback textarea */}
      <textarea
        value={feedback}
        onChange={(e) => setFeedback(e.target.value)}
        disabled={disabled}
        rows={3}
        placeholder="例如：保持云雾层次，但让水面反射更明显，整体不要变暗。"
        className="w-full text-xs p-2 rounded bg-[var(--bg-tertiary)] border border-[var(--border-color)] text-[var(--text-primary)] resize-y placeholder:text-[var(--text-muted)] disabled:opacity-40"
      />

      {/* Explore variants toggle */}
      <label className="flex items-center gap-1.5 text-[11px] text-[var(--text-secondary)] cursor-pointer">
        <input
          type="checkbox"
          checked={exploreMode}
          onChange={(e) => setExploreMode(e.target.checked)}
          disabled={disabled}
          className="accent-emerald-500"
        />
        探索多个变体 / Explore variants
      </label>

      {/* Batch draw entry */}
      {onStartDraw && (
        <button
          onClick={() => setDrawMode(true)}
          disabled={disabled}
          className="flex items-center justify-center gap-1.5 px-3 py-1.5 text-[11px] rounded-md transition-all disabled:opacity-40 disabled:cursor-not-allowed bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]"
        >
          <Layers2 className="w-3 h-3" />
          批量抽卡 / Batch Draw
        </button>
      )}

      {exploreMode ? (
        /* Explore controls */
        <div className="flex flex-col gap-1.5">
          {/* Variant count */}
          <div className="flex items-center gap-1.5">
            <span className="text-[11px] text-[var(--text-muted)] shrink-0">数量 Count</span>
            <div className="flex items-center gap-0.5 bg-[var(--bg-tertiary)] rounded-md p-0.5">
              {VARIANT_COUNTS.map((n) => (
                <button
                  key={n}
                  onClick={() => setVariantCount(n)}
                  disabled={disabled}
                  className={`px-2 py-0.5 text-[11px] rounded-md transition-all disabled:opacity-40 ${
                    variantCount === n
                      ? "bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-medium"
                      : "text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)]"
                  }`}
                >
                  {n}
                </button>
              ))}
            </div>
          </div>
          {/* Diversity */}
          <div className="flex items-center gap-1.5">
            <span className="text-[11px] text-[var(--text-muted)] shrink-0">多样性 Diversity</span>
            <div className="flex items-center gap-0.5 bg-[var(--bg-tertiary)] rounded-md p-0.5">
              {(["low", "medium", "high"] as const).map((d) => (
                <button
                  key={d}
                  onClick={() => setDiversity(d)}
                  disabled={disabled}
                  className={`px-2 py-0.5 text-[11px] rounded-md transition-all disabled:opacity-40 ${
                    diversity === d
                      ? "bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-medium"
                      : "text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)]"
                  }`}
                >
                  {d}
                </button>
              ))}
            </div>
          </div>
        </div>
      ) : (
        /* Refine controls (unchanged) */
        <>
          {/* Mode segmented */}
          <div className="flex items-center gap-0.5 bg-[var(--bg-tertiary)] rounded-md p-0.5">
            {MODES.map(({ mode: m, label, sub, desc }) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                disabled={disabled}
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
              <label
                key={key}
                className="flex items-center gap-1.5 text-[11px] text-[var(--text-secondary)] cursor-pointer"
              >
                <input
                  type="checkbox"
                  checked={!!locks[key]}
                  onChange={() => toggleLock(key)}
                  disabled={disabled}
                  className="accent-emerald-500"
                />
                {label}
              </label>
            ))}
          </div>
        </>
      )}

      {/* Fine controls (collapsible) */}
      <div className="border border-[var(--border-color)] rounded-md overflow-hidden">
        <button
          onClick={() => setFineControlOpen((prev) => !prev)}
          disabled={disabled}
          className="flex items-center gap-1.5 w-full px-2 py-1.5 text-[11px] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)] transition-all disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {fineControlOpen ? (
            <ChevronDown className="w-3 h-3 shrink-0" />
          ) : (
            <ChevronRight className="w-3 h-3 shrink-0" />
          )}
          精细控制 / Fine controls
        </button>
        {fineControlOpen && (
          <div className="px-2 pb-2 pt-1 border-t border-[var(--border-color)] bg-[var(--bg-tertiary)]">
            <FineControlPanel
              value={constraintSpec}
              onChange={setConstraintSpec}
              disabled={disabled}
            />
          </div>
        )}
      </div>

      {/* Submit error */}
      {submitError && (
        <p className="text-[11px] text-red-400">{submitError}</p>
      )}

      {/* Submit / Cancel */}
      <div className="flex gap-1.5 pt-1">
        <button
          onClick={onCancelBranch}
          disabled={disabled}
          className="flex-1 px-3 py-1.5 text-[11px] rounded-md transition-all disabled:opacity-40 bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]"
        >
          取消 Cancel
        </button>
        <button
          onClick={handleSubmit}
          disabled={!canSubmit}
          className="flex-1 flex items-center justify-center gap-1.5 px-3 py-1.5 text-[11px] rounded-md transition-all disabled:opacity-40 disabled:cursor-not-allowed bg-gradient-to-r from-emerald-500 to-emerald-600 hover:from-emerald-400 hover:to-emerald-500 text-white font-semibold"
        >
          <GitBranch className="w-3 h-3" />
          {exploreMode ? "运行变体探索 / Explore" : "运行分支 Submit"}
        </button>
      </div>
    </div>
  );
}

// ─── Variant helpers ──────────────────────────────────────────────────────────

const TERMINAL_STATUSES = new Set(["completed", "failed", "partial_failed", "cancelled"]);

function statusDot(status: string): React.ReactElement {
  const color =
    status === "completed"
      ? "bg-emerald-500"
      : status === "running" || status === "queued"
      ? "bg-yellow-400 animate-pulse"
      : status === "failed"
      ? "bg-red-500"
      : status === "partial_failed"
      ? "bg-orange-400"
      : status === "cancelled"
      ? "bg-[var(--text-muted)]"
      : "bg-[var(--text-muted)]";
  return <span className={`inline-block w-2 h-2 rounded-full shrink-0 ${color}`} />;
}

// ─── Per-variant action row ────────────────────────────────────────────────

interface VariantActionsProps {
  v: VariantStatusEntry;
  groupId: string;
  winnerRunId?: string | null;
  activeRunId: string | null;
  onSwitchRun: (runId: string) => void;
  onRefineFromCheckpoint?: (runId: string, checkpointId: string) => void;
  onSelectWinner?: (groupId: string, winnerRunId: string, reason?: string) => void;
  onRateVariant?: (groupId: string, runId: string, rating: number) => void;
  disabled?: boolean;
}

function VariantActions({
  v,
  groupId,
  winnerRunId,
  activeRunId,
  onSwitchRun,
  onRefineFromCheckpoint,
  onSelectWinner,
  onRateVariant,
  disabled,
}: VariantActionsProps) {
  const isActive = v.run_id === activeRunId;
  const isWinner = v.run_id === winnerRunId;

  return (
    <div className="flex flex-wrap items-center gap-1 pt-1">
      {/* Preview */}
      <button
        onClick={() => onSwitchRun(v.run_id)}
        disabled={disabled || isActive}
        title="预览此变体 Preview this variant"
        className="px-2 py-0.5 text-[11px] rounded transition-all disabled:opacity-40 disabled:cursor-not-allowed bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]"
      >
        预览 Preview
      </button>

      {/* Select winner */}
      <button
        onClick={() => onSelectWinner?.(groupId, v.run_id)}
        disabled={disabled || isWinner || !onSelectWinner}
        title="设为胜出 Select as winner"
        className="px-2 py-0.5 text-[11px] rounded transition-all disabled:opacity-40 disabled:cursor-not-allowed bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]"
      >
        设为胜出 Winner
      </button>

      {/* Continue (only for completed variants) */}
      {v.status === "completed" && onRefineFromCheckpoint && (
        <button
          onClick={() => onRefineFromCheckpoint(v.run_id, "final:selected")}
          disabled={disabled}
          title="从此变体的最终结果继续 Continue from final"
          className="px-2 py-0.5 text-[11px] rounded transition-all disabled:opacity-40 disabled:cursor-not-allowed bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]"
        >
          继续 Continue
        </button>
      )}

      {/* Rate */}
      <button
        onClick={() => onRateVariant?.(groupId, v.run_id, 1)}
        disabled={disabled || !onRateVariant}
        title="好评 Thumbs up"
        className="p-1 rounded transition-all disabled:opacity-40 disabled:cursor-not-allowed text-[var(--text-muted)] hover:text-emerald-400"
      >
        <ThumbsUp className="w-3 h-3" />
      </button>
      <button
        onClick={() => onRateVariant?.(groupId, v.run_id, -1)}
        disabled={disabled || !onRateVariant}
        title="差评 Thumbs down"
        className="p-1 rounded transition-all disabled:opacity-40 disabled:cursor-not-allowed text-[var(--text-muted)] hover:text-red-400"
      >
        <ThumbsDown className="w-3 h-3" />
      </button>
    </div>
  );
}

// ─── Single variant card ──────────────────────────────────────────────────────

interface VariantCardProps {
  v: VariantStatusEntry;
  groupId: string;
  winnerRunId?: string | null;
  activeRunId: string | null;
  onSwitchRun: (runId: string) => void;
  onRefineFromCheckpoint?: (runId: string, checkpointId: string) => void;
  onSelectWinner?: (groupId: string, winnerRunId: string, reason?: string) => void;
  onRateVariant?: (groupId: string, runId: string, rating: number) => void;
  disabled?: boolean;
}

function VariantCard({
  v,
  groupId,
  winnerRunId,
  activeRunId,
  onSwitchRun,
  onRefineFromCheckpoint,
  onSelectWinner,
  onRateVariant,
  disabled,
}: VariantCardProps) {
  const isWinner = v.run_id === winnerRunId;

  return (
    <div
      className={`flex flex-col gap-1 px-2 py-1.5 rounded-md border ${
        isWinner
          ? "border-emerald-500/40 bg-emerald-500/5"
          : "border-[var(--border-color)] bg-[var(--bg-tertiary)]"
      }`}
    >
      {/* Header row: index badge, label, status dot, score, winner star */}
      <div className="flex items-center gap-1.5 flex-wrap">
        <span className="text-[10px] font-mono px-1 py-0.5 rounded bg-[var(--bg-secondary)] text-[var(--text-muted)]">
          #{v.variant_index}
        </span>
        {statusDot(v.status)}
        <span className="text-[11px] font-medium text-[var(--text-primary)] truncate flex-1 min-w-0">
          {v.label}
        </span>
        <span className="text-[11px] font-mono text-[var(--text-muted)] shrink-0">
          {fmtScore(v.final_score ?? v.current_score)}
        </span>
        {isWinner && (
          <span title="胜出 Winner" className="shrink-0">
            <Star className="w-3.5 h-3.5 text-emerald-400 fill-current" />
          </span>
        )}
      </div>

      {/* Changes summary */}
      {v.changes_summary && (
        <p className="text-[11px] text-[var(--text-secondary)] leading-snug">
          {v.changes_summary}
        </p>
      )}

      {/* Error */}
      {v.error && (
        <p className="text-[11px] text-red-400 leading-snug">{v.error}</p>
      )}

      {/* Actions */}
      <VariantActions
        v={v}
        groupId={groupId}
        winnerRunId={winnerRunId}
        activeRunId={activeRunId}
        onSwitchRun={onSwitchRun}
        onRefineFromCheckpoint={onRefineFromCheckpoint}
        onSelectWinner={onSelectWinner}
        onRateVariant={onRateVariant}
        disabled={disabled}
      />
    </div>
  );
}

// ─── VariantGroupView ────────────────────────────────────────────────────────

interface VariantGroupViewProps {
  node: BranchCanvasNode;
  variantGroup: VariantGroupStatus | null | undefined;
  activeRunId: string | null;
  onSwitchRun: (runId: string) => void;
  onRefineFromCheckpoint?: (runId: string, checkpointId: string) => void;
  onSelectWinner?: (groupId: string, winnerRunId: string, reason?: string) => void;
  onStopGroup?: (groupId: string) => void;
  onRateVariant?: (groupId: string, runId: string, rating: number) => void;
  disabled?: boolean;
}

function VariantGroupView({
  node,
  variantGroup,
  activeRunId,
  onSwitchRun,
  onRefineFromCheckpoint,
  onSelectWinner,
  onStopGroup,
  onRateVariant,
  disabled,
}: VariantGroupViewProps) {
  const groupId = node.data.group_id as string;

  // Use the live group only when its id matches the selected node
  const group = variantGroup?.group_id === groupId ? variantGroup : null;

  if (!group) {
    return (
      <p className="text-[11px] text-[var(--text-muted)] py-4 text-center">
        加载分组中…
        <br />
        <span className="opacity-70">Loading group…</span>
      </p>
    );
  }

  const isTerminal = TERMINAL_STATUSES.has(group.status);

  return (
    <div className="flex flex-col gap-2">
      {/* Header: status badge + feedback + stop button */}
      <div className="flex items-start gap-2">
        <div className="flex items-center gap-1.5 flex-1 min-w-0 flex-wrap">
          {statusDot(group.status)}
          <span className="text-[11px] px-1.5 py-0.5 rounded bg-[var(--bg-tertiary)] text-[var(--text-secondary)] font-mono shrink-0">
            {group.status}
          </span>
          {group.feedback && (
            <span className="text-[11px] text-[var(--text-secondary)] truncate">{group.feedback}</span>
          )}
        </div>
        {/* Stop group button */}
        <button
          onClick={() => onStopGroup?.(groupId)}
          disabled={disabled || isTerminal || !onStopGroup}
          title="停止分组 Stop group"
          className="flex items-center gap-1 px-2 py-0.5 text-[11px] rounded transition-all shrink-0 disabled:opacity-40 disabled:cursor-not-allowed bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:bg-red-500/20 hover:text-red-400"
        >
          <StopCircle className="w-3 h-3" />
          停止 Stop
        </button>
      </div>

      {/* Feedback block (full text, if truncated above) */}
      {group.feedback && (
        <div className="text-[11px] text-[var(--text-secondary)] bg-[var(--bg-tertiary)] rounded p-2 leading-relaxed">
          <span className="text-[var(--text-muted)]">反馈 Feedback: </span>
          {group.feedback}
        </div>
      )}

      {/* Variant cards */}
      {group.variants.length > 0 ? (
        <div className="flex flex-col gap-1.5">
          <p className="text-[11px] text-[var(--text-muted)]">
            变体 Variants ({group.variants.length})
          </p>
          {group.variants.map((v) => (
            <VariantCard
              key={v.run_id}
              v={v}
              groupId={groupId}
              winnerRunId={group.winner_run_id}
              activeRunId={activeRunId}
              onSwitchRun={onSwitchRun}
              onRefineFromCheckpoint={onRefineFromCheckpoint}
              onSelectWinner={onSelectWinner}
              onRateVariant={onRateVariant}
              disabled={disabled}
            />
          ))}
        </div>
      ) : (
        <p className="text-[11px] text-[var(--text-muted)]">暂无变体 No variants yet</p>
      )}
    </div>
  );
}

// ─── VariantRunDetailView ────────────────────────────────────────────────────

interface VariantRunDetailViewProps {
  node: BranchCanvasNode;
  variantGroup: VariantGroupStatus | null | undefined;
  activeRunId: string | null;
  onSwitchRun: (runId: string) => void;
  onRefineFromCheckpoint?: (runId: string, checkpointId: string) => void;
  onSelectWinner?: (groupId: string, winnerRunId: string, reason?: string) => void;
  onRateVariant?: (groupId: string, runId: string, rating: number) => void;
  disabled?: boolean;
}

function VariantRunDetailView({
  node,
  variantGroup,
  activeRunId,
  onSwitchRun,
  onRefineFromCheckpoint,
  onSelectWinner,
  onRateVariant,
  disabled,
}: VariantRunDetailViewProps) {
  const runId = node.data.run_id as string;
  const variantGroupId = node.data.variant_group_id as string | undefined;

  // Match the group only when its id aligns with this variant run's group
  const group =
    variantGroup?.group_id === variantGroupId ? variantGroup : null;
  const entry = group?.variants.find((v) => v.run_id === runId) ?? null;

  if (!group || !entry) {
    return (
      <p className="text-[11px] text-[var(--text-muted)] py-4 text-center">
        加载变体中…
        <br />
        <span className="opacity-70">Loading variant…</span>
      </p>
    );
  }

  const groupId = group.group_id;

  return (
    <div className="flex flex-col gap-2">
      {/* Back-ref: which group */}
      <p className="text-[10px] text-[var(--text-muted)] font-mono truncate" title={groupId}>
        分组 Group: {groupId.slice(-8)}
      </p>

      {/* Single variant card (reuse the same card component) */}
      <VariantCard
        key={node.id}
        v={entry}
        groupId={groupId}
        winnerRunId={group.winner_run_id}
        activeRunId={activeRunId}
        onSwitchRun={onSwitchRun}
        onRefineFromCheckpoint={onRefineFromCheckpoint}
        onSelectWinner={onSelectWinner}
        onRateVariant={onRateVariant}
        disabled={disabled}
      />
    </div>
  );
}

// ─── DrawSessionDetailView (V3.5) ─────────────────────────────────────────────

interface DrawSessionDetailViewProps {
  drawSession: DrawSessionStatus | null | undefined;
  fusionEnabled?: boolean;
  onStartDraw?: (parentRunId: string, request: CreateDrawSessionRequest) => void;
  onDrawMore?: (drawId: string, request: DrawMoreRequest) => void;
  onRedrawCard?: (drawId: string, runId: string) => void;
  onCardEvent?: (
    drawId: string,
    runId: string,
    eventType: DrawCardEventType,
    opts?: { value?: unknown; reason?: string; tags?: string[] },
  ) => void;
  onPreviewCard?: (runId: string) => void;
  onContinueCard?: (runId: string) => void;
  onSelectDrawWinner?: (drawId: string, runId: string) => void;
  onStopDraw?: (drawId: string) => void;
  disabled?: boolean;
}

function DrawSessionDetailView({
  drawSession,
  fusionEnabled,
  onStartDraw,
  onDrawMore,
  onRedrawCard,
  onCardEvent,
  onPreviewCard,
  onContinueCard,
  onSelectDrawWinner,
  onStopDraw,
  disabled,
}: DrawSessionDetailViewProps) {
  if (!drawSession) {
    return (
      <p className="text-[11px] text-[var(--text-muted)] py-4 text-center">
        加载抽卡中…
        <br />
        <span className="opacity-70">Loading draw session…</span>
      </p>
    );
  }

  return (
    <DrawSessionInspector
      parentRunId={drawSession.parent_run_id ?? ""}
      checkpointId={drawSession.source_checkpoint_id}
      session={drawSession}
      fusionEnabled={fusionEnabled}
      onStartDraw={onStartDraw ?? (() => {})}
      onDrawMore={onDrawMore ?? (() => {})}
      onRedrawCard={onRedrawCard ?? (() => {})}
      onCardEvent={onCardEvent ?? (() => {})}
      onPreviewCard={onPreviewCard ?? (() => {})}
      onContinueCard={onContinueCard ?? (() => {})}
      onSelectWinner={onSelectDrawWinner}
      onStopDraw={onStopDraw}
      disabled={disabled}
    />
  );
}

function ReservedNodeView() {
  return (
    <p className="text-[11px] text-[var(--text-muted)] py-4 text-center">
      暂未实现
      <br />
      <span className="opacity-70">Not yet available</span>
    </p>
  );
}

// ─── Main export ──────────────────────────────────────────────────────────────

export default function BranchCanvasInspector({
  node,
  activeRunId,
  onSwitchRun,
  onUpdateMetadata,
  onRefineFromCheckpoint,
  onContinueFromRun,
  onSubmitBranch,
  onCancelBranch,
  onExploreVariants,
  variantGroup,
  onSelectWinner,
  onStopGroup,
  onRateVariant,
  drawSession,
  onStartDraw,
  onDrawMore,
  onRedrawCard,
  onCardEvent,
  onPreviewCard,
  onContinueCard,
  onSelectDrawWinner,
  onStopDraw,
  fusionEnabled,
  disabled,
  submitError,
}: Props) {
  // Reset run-node title draft when node identity changes (handled inside RunNodeView too,
  // but the key prop on RunNodeView ensures full remount on node switch).

  const renderContent = () => {
    if (!node) return <NoNodeView />;

    const { type } = node.data;

    switch (type) {
      case "input":
        return <InputNodeView label={node.data.label} />;

      case "run":
        return (
          <RunNodeView
            key={node.id}
            node={node}
            activeRunId={activeRunId}
            onSwitchRun={onSwitchRun}
            onUpdateMetadata={onUpdateMetadata}
            onContinueFromRun={onContinueFromRun}
            disabled={disabled}
          />
        );

      case "checkpoint":
        return (
          <CheckpointNodeView
            key={node.id}
            node={node}
            onRefineFromCheckpoint={onRefineFromCheckpoint}
            disabled={disabled}
          />
        );

      case "branch_action":
        return (
          <BranchActionView
            key={node.id}
            node={node}
            onSubmitBranch={onSubmitBranch}
            onCancelBranch={onCancelBranch}
            onExploreVariants={onExploreVariants}
            onStartDraw={onStartDraw}
            onDrawMore={onDrawMore}
            onRedrawCard={onRedrawCard}
            onCardEvent={onCardEvent}
            onPreviewCard={onPreviewCard}
            onContinueCard={onContinueCard}
            fusionEnabled={fusionEnabled}
            submitError={submitError}
            disabled={disabled}
          />
        );

      case "variant_group":
        return (
          <VariantGroupView
            key={node.id}
            node={node}
            variantGroup={variantGroup}
            activeRunId={activeRunId}
            onSwitchRun={onSwitchRun}
            onRefineFromCheckpoint={onRefineFromCheckpoint}
            onSelectWinner={onSelectWinner}
            onStopGroup={onStopGroup}
            onRateVariant={onRateVariant}
            disabled={disabled}
          />
        );

      case "variant_run":
        return (
          <VariantRunDetailView
            key={node.id}
            node={node}
            variantGroup={variantGroup}
            activeRunId={activeRunId}
            onSwitchRun={onSwitchRun}
            onRefineFromCheckpoint={onRefineFromCheckpoint}
            onSelectWinner={onSelectWinner}
            onRateVariant={onRateVariant}
            disabled={disabled}
          />
        );

      // V3.5 draw-session detail (a selected draw_card shows the whole session grid)
      case "draw_session":
      case "draw_card": {
        const nodeDrawId = (node.data as { draw_id?: string }).draw_id;
        const guardedSession =
          drawSession != null && drawSession.draw_id === nodeDrawId ? drawSession : null;
        if (drawSession != null && guardedSession == null) {
          // drawSession exists but belongs to a different node — show placeholder
          return (
            <p className="text-[11px] text-[var(--text-muted)] py-4 text-center">
              加载抽卡批次…
              <br />
              <span className="opacity-70">Loading draw session…</span>
            </p>
          );
        }
        return (
          <DrawSessionDetailView
            key={node.id}
            drawSession={guardedSession}
            fusionEnabled={fusionEnabled}
            onStartDraw={onStartDraw}
            onDrawMore={onDrawMore}
            onRedrawCard={onRedrawCard}
            onCardEvent={onCardEvent}
            onPreviewCard={onPreviewCard}
            onContinueCard={onContinueCard}
            onSelectDrawWinner={onSelectDrawWinner}
            onStopDraw={onStopDraw}
            disabled={disabled}
          />
        );
      }

      // Reserved V4 types
      case "region_constraint":
      case "preference":
        return <ReservedNodeView />;

      default:
        return <ReservedNodeView />;
    }
  };

  return <PanelShell>{renderContent()}</PanelShell>;
}
