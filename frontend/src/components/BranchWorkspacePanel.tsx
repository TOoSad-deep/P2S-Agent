// BranchWorkspacePanel.tsx — orchestrates CheckpointTimeline + BranchTree + BranchCompareStrip (V2).
import { useState, useEffect, useCallback, useRef } from "react";
import { LayoutDashboard, Star } from "lucide-react";
import type {
  PngShaderResult,
  CheckpointTimelineEntry,
  BranchTreeResponse,
  RunMetadataPatch,
  RunMetadataRecord,
} from "../hooks/usePngShader";
import { findNode } from "../lib/branchTree";
import CheckpointTimeline from "./CheckpointTimeline";
import BranchTree from "./BranchTree";
import BranchCompareStrip from "./BranchCompareStrip";

interface Props {
  runId: string | null;
  result: PngShaderResult | null;
  activeCheckpointId: string | null;
  onCheckpointSelect: (id: string) => void;
  onSwitchRun: (runId: string) => void;
  fetchTimeline: (id: string) => Promise<CheckpointTimelineEntry[]>;
  fetchBranches: (id: string) => Promise<BranchTreeResponse>;
  updateRunMetadata: (id: string, patch: RunMetadataPatch) => Promise<RunMetadataRecord>;
  disabled?: boolean;
}

export default function BranchWorkspacePanel({
  runId,
  result,
  activeCheckpointId,
  onCheckpointSelect,
  onSwitchRun,
  fetchTimeline,
  fetchBranches,
  updateRunMetadata,
  disabled,
}: Props) {
  const [timeline, setTimeline] = useState<CheckpointTimelineEntry[]>([]);
  const [branchInfo, setBranchInfo] = useState<BranchTreeResponse | null>(null);

  // Local metadata edit state — seeded from the active node when branchInfo loads.
  const [localFavorite, setLocalFavorite] = useState(false);
  const [localTitle, setLocalTitle] = useState("");

  // M-1: track input focus so a refetch doesn't clobber an in-progress title edit.
  const titleFocusedRef = useRef(false);
  // M-2: track the last-seeded/committed title so we skip no-op PATCHes.
  const seededTitleRef = useRef("");

  // Fetch timeline when run or its iteration count/status changes.
  useEffect(() => {
    if (!runId) {
      setTimeline([]);
      return;
    }
    let alive = true;
    fetchTimeline(runId)
      .then((t) => { if (alive) setTimeline(t); })
      .catch(() => {});
    return () => { alive = false; };
  }, [runId, result?.refinement_history?.length, result?.status, fetchTimeline]);

  // Fetch branches when run switches or reaches a terminal status.
  useEffect(() => {
    if (!runId) {
      setBranchInfo(null);
      return;
    }
    let alive = true;
    fetchBranches(runId)
      .then((b) => { if (alive) setBranchInfo(b); })
      .catch(() => {});
    return () => { alive = false; };
  }, [runId, result?.status, fetchBranches]);

  // Seed local metadata state from the active node whenever branchInfo updates.
  // M-1: skip seeding localTitle while the input is focused to avoid clobbering.
  useEffect(() => {
    if (!branchInfo) return;
    const activeId = branchInfo.active_run_id ?? runId;
    const node = activeId ? findNode(branchInfo.tree, activeId) : null;
    if (node && !titleFocusedRef.current) {
      setLocalFavorite(node.favorite ?? false);
      setLocalTitle(node.title ?? "");
      seededTitleRef.current = node.title ?? "";  // M-2
    }
  }, [branchInfo, runId]);

  const handleFavoriteToggle = useCallback(() => {
    if (!runId) return;
    const next = !localFavorite;
    setLocalFavorite(next);
    updateRunMetadata(runId, { favorite: next })
      .then(() => fetchBranches(runId))
      .then((b) => setBranchInfo(b))
      .catch(() => {});
  }, [runId, localFavorite, updateRunMetadata, fetchBranches]);

  // M-2: skip PATCH when the title is unchanged from the last seeded/committed value.
  const commitTitle = useCallback(() => {
    if (!runId) return;
    const trimmed = localTitle.trim();
    if (trimmed === seededTitleRef.current) return;  // no change, skip PATCH
    updateRunMetadata(runId, { title: trimmed })
      .then(() => fetchBranches(runId))
      .then((b) => {
        setBranchInfo(b);
        seededTitleRef.current = trimmed;  // update so second blur doesn't re-send
      })
      .catch(() => {});
  }, [runId, localTitle, updateRunMetadata, fetchBranches]);

  // I-2: blur triggers onBlur → commitTitle, so don't call commitTitle directly here.
  const handleTitleKeyDown = useCallback((e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.currentTarget.blur();   // onBlur fires commitTitle — one call only
    }
  }, []);

  const controlsDisabled = disabled || !runId;

  return (
    <div className="flex flex-col gap-2.5 px-3 py-2.5 bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-lg">
      {/* Header */}
      <div className="flex items-center gap-2">
        <LayoutDashboard className="w-4 h-4 text-[var(--accent-primary)] flex-shrink-0" />
        <p className="text-xs font-medium text-[var(--text-primary)] leading-tight">
          分支工作台
          <span className="ml-2 text-[var(--text-muted)] font-normal">Branch Workspace</span>
        </p>
      </div>

      {/* Metadata editor row */}
      <div className="flex items-center gap-2">
        <button
          onClick={handleFavoriteToggle}
          disabled={controlsDisabled}
          title={localFavorite ? "取消收藏 Unfavorite" : "收藏 Favorite"}
          className="flex-shrink-0 p-1 rounded transition-all disabled:opacity-40 hover:bg-[var(--bg-hover)]"
        >
          <Star
            className={`w-4 h-4 transition-colors ${
              localFavorite ? "text-yellow-400" : "text-[var(--text-muted)]"
            }`}
            fill={localFavorite ? "currentColor" : "none"}
          />
        </button>
        <input
          type="text"
          value={localTitle}
          onChange={(e) => setLocalTitle(e.target.value)}
          onFocus={() => { titleFocusedRef.current = true; }}   // M-1
          onBlur={() => { titleFocusedRef.current = false; commitTitle(); }}  // M-1 + existing
          onKeyDown={handleTitleKeyDown}
          disabled={controlsDisabled}
          placeholder="分支标题 / Branch title"
          className="flex-1 text-xs px-2 py-1 rounded bg-[var(--bg-tertiary)] border border-[var(--border-color)] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] disabled:opacity-40 focus:outline-none focus:border-[var(--accent-primary)]"
        />
      </div>

      {/* Checkpoint timeline */}
      <CheckpointTimeline
        entries={timeline}
        activeCheckpointId={activeCheckpointId}
        onSelect={onCheckpointSelect}
        disabled={disabled}
      />

      {/* Branch tree */}
      <BranchTree
        tree={branchInfo?.tree ?? null}
        activeRunId={branchInfo?.active_run_id ?? runId}
        onSelectRun={onSwitchRun}
        disabled={disabled}
      />

      {/* Compare strip */}
      <BranchCompareStrip
        activeRunId={branchInfo?.active_run_id ?? runId}
        tree={branchInfo?.tree ?? null}
      />
    </div>
  );
}
