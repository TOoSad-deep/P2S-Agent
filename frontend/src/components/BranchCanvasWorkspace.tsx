// BranchCanvasWorkspace.tsx — orchestrator for the Branch Canvas Workspace (V2.1-6).
// Fetches V2 data, builds the canvas model + layout, and renders the canvas + inspector
// with selection / run-switch / drag-persistence.
// Branch-draft submit, compare mode, Canvas/List toggle, and mounting into PngShaderView
// are deferred to V2.1-7 — those props are no-op stubs here.
import { useState, useEffect, useMemo, useCallback, useRef } from "react";
import { RotateCcw } from "lucide-react";
import type {
  PngShaderResult,
  CheckpointTimelineEntry,
  BranchTreeResponse,
  RunMetadataPatch,
} from "../hooks/usePngShader";
import {
  buildBranchCanvasModel,
  type BranchCanvasNode,
} from "../lib/branchCanvasModel";
import { layoutBranchCanvas } from "../lib/branchCanvasLayout";
import BranchCanvas from "./BranchCanvas";
import { branchCanvasNodeTypes } from "./BranchCanvasNode";
import BranchCanvasInspector from "./BranchCanvasInspector";

// ─── Props ────────────────────────────────────────────────────────────────────

interface Props {
  runId: string | null;
  result: PngShaderResult | null;               // active run's live status/score
  fetchBranches: (id: string) => Promise<BranchTreeResponse>;
  fetchTimeline: (id: string) => Promise<CheckpointTimelineEntry[]>;
  switchRun: (id: string) => void;
  updateRunMetadata: (id: string, patch: RunMetadataPatch) => Promise<unknown>;
  // delegated to parent (wired in V2.1-7); optional no-op defaults:
  onPreviewNode?: (node: BranchCanvasNode | null) => void;
  onRefineFromCheckpoint?: (runId: string, checkpointId: string) => void;
  onContinueFromRun?: (runId: string) => void;
  disabled?: boolean;
}

// ─── localStorage helpers (SSR/quota safe) ───────────────────────────────────

function loadLayoutOverrides(key: string): Record<string, { x: number; y: number }> {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return {};
    const parsed = JSON.parse(raw) as unknown;
    if (parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, { x: number; y: number }>;
    }
    return {};
  } catch {
    return {};
  }
}

function saveLayoutOverrides(
  key: string,
  overrides: Record<string, { x: number; y: number }>,
): void {
  try {
    localStorage.setItem(key, JSON.stringify(overrides));
  } catch {
    // quota exceeded or SSR — ignore
  }
}

function removeLayoutOverrides(key: string): void {
  try {
    localStorage.removeItem(key);
  } catch {
    // ignore
  }
}

// ─── BranchCanvasWorkspace ────────────────────────────────────────────────────

export default function BranchCanvasWorkspace({
  runId,
  result,
  fetchBranches,
  fetchTimeline,
  switchRun,
  updateRunMetadata,
  onPreviewNode,
  onRefineFromCheckpoint,
  onContinueFromRun,
  disabled,
}: Props) {
  // ── State ──────────────────────────────────────────────────────────────────
  const [branchInfo, setBranchInfo] = useState<BranchTreeResponse | null>(null);
  const [activeTimeline, setActiveTimeline] = useState<CheckpointTimelineEntry[]>([]);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  // TODO(V2.1-7): make stateful when the collapse toggle UI is added
  const collapsedRunIds = useMemo(() => new Set<string>(), []);
  const [layoutOverrides, setLayoutOverrides] = useState<Record<string, { x: number; y: number }>>({});
  const [loadError, setLoadError] = useState<string | null>(null);

  // ── runId ref: kept in sync so async callbacks read the latest value ──────
  const runIdRef = useRef(runId);
  useEffect(() => { runIdRef.current = runId; }, [runId]);

  // ── Clear selection on run-switch (I3) ────────────────────────────────────
  useEffect(() => { setSelectedNodeId(null); }, [runId]);

  // ── localStorage key (keyed by root run) ──────────────────────────────────
  const layoutStorageKey = `branchCanvasLayout:${branchInfo?.root_run_id ?? runId ?? "none"}`;

  // ── Load overrides from localStorage when the key changes ─────────────────
  useEffect(() => {
    setLayoutOverrides(loadLayoutOverrides(layoutStorageKey));
  }, [layoutStorageKey]);

  // ── Persist layoutOverrides to localStorage when they change (M3) ─────────
  // Skipped when overrides are empty to avoid clobbering on initial mount.
  useEffect(() => {
    if (Object.keys(layoutOverrides).length > 0) {
      saveLayoutOverrides(layoutStorageKey, layoutOverrides);
    }
  }, [layoutOverrides, layoutStorageKey]);

  // ── Fetch branchInfo (alive-guarded) ──────────────────────────────────────
  useEffect(() => {
    if (!runId) {
      setBranchInfo(null);
      return;
    }
    let alive = true;
    fetchBranches(runId)
      .then((b) => {
        if (alive) {
          setBranchInfo(b);
          setLoadError(null);
        }
      })
      .catch(() => {
        if (alive) setLoadError("分支信息加载失败 / Failed to load branches");
      });
    return () => { alive = false; };
  }, [runId, result?.status, fetchBranches]);

  // ── Fetch activeTimeline (alive-guarded) ───────────────────────────────────
  useEffect(() => {
    if (!runId) {
      setActiveTimeline([]);
      return;
    }
    let alive = true;
    fetchTimeline(runId)
      .then((t) => { if (alive) setActiveTimeline(t); })
      .catch(() => {});
    return () => { alive = false; };
  }, [runId, result?.refinement_history?.length, result?.status, fetchTimeline]);

  // ── Derived: active run id ─────────────────────────────────────────────────
  const activeRunId = useMemo(
    () => branchInfo?.active_run_id ?? runId,
    [branchInfo?.active_run_id, runId],
  );

  // ── Derived: favoriteRunIds ────────────────────────────────────────────────
  const favoriteRunIds = useMemo<Set<string>>(() => {
    if (!branchInfo) return new Set();
    const favSet = new Set<string>();
    function walk(node: BranchTreeResponse["tree"]): void {
      if (node.favorite) favSet.add(node.run_id);
      for (const child of node.children) walk(child);
    }
    walk(branchInfo.tree);
    return favSet;
  }, [branchInfo]);

  // ── Derived: timelinesByRunId (only active run fetched in V2.1-6) ──────────
  const timelinesByRunId = useMemo<Record<string, CheckpointTimelineEntry[]>>(
    () => (activeRunId ? { [activeRunId]: activeTimeline } : {}),
    [activeRunId, activeTimeline],
  );

  // ── Derived: statusesByRunId ───────────────────────────────────────────────
  const statusesByRunId = useMemo(
    () =>
      activeRunId
        ? {
            [activeRunId]: {
              status: result?.status,
              final_score: result?.quality_router?.final_score ?? null,
            },
          }
        : {},
    [activeRunId, result?.status, result?.quality_router?.final_score],
  );

  // ── Derived: canvas model ──────────────────────────────────────────────────
  const model = useMemo(
    () =>
      buildBranchCanvasModel({
        activeRunId,
        branchTree: branchInfo?.tree ?? null,
        timelinesByRunId,
        statusesByRunId,
        collapsedRunIds,
        favoriteRunIds,
      }),
    [activeRunId, branchInfo?.tree, timelinesByRunId, statusesByRunId, collapsedRunIds, favoriteRunIds],
  );

  // ── Derived: positioned nodes (layout only; no selection) ─────────────────
  const positionedNodes = useMemo(
    () => layoutBranchCanvas(model.nodes, model.edges, layoutOverrides),
    [model.nodes, model.edges, layoutOverrides],
  );

  // ── Derived: display nodes (inject selected flag; unchanged nodes are stable)
  const displayNodes = useMemo(
    () =>
      positionedNodes.map((n) => {
        const sel = n.id === selectedNodeId;
        return n.selected === sel ? n : { ...n, selected: sel };
      }),
    [positionedNodes, selectedNodeId],
  );

  // ── Derived: selected node ─────────────────────────────────────────────────
  const selectedNode = useMemo(
    () => positionedNodes.find((n) => n.id === selectedNodeId) ?? null,
    [positionedNodes, selectedNodeId],
  );

  // ── Handlers ───────────────────────────────────────────────────────────────

  const handleNodeClick = useCallback(
    (id: string) => {
      setSelectedNodeId(id);
      const node = displayNodes.find((n) => n.id === id) ?? null;
      onPreviewNode?.(node);
    },
    [displayNodes, onPreviewNode],
  );

  const handleNodeDoubleClick = useCallback(
    (id: string) => {
      const node = displayNodes.find((n) => n.id === id);
      if (
        node &&
        node.data.type === "run" &&
        node.data.run_id &&
        node.data.run_id !== activeRunId
      ) {
        switchRun(node.data.run_id);
      }
    },
    [displayNodes, activeRunId, switchRun],
  );

  const handleDragStop = useCallback(
    (id: string, pos: { x: number; y: number }) => {
      setLayoutOverrides((prev) => ({ ...prev, [id]: pos }));
    },
    [],
  );

  const handleResetLayout = useCallback(() => {
    setLayoutOverrides({});
    removeLayoutOverrides(layoutStorageKey);
  }, [layoutStorageKey]);

  const handleUpdateMetadata = useCallback(
    (rid: string, patch: RunMetadataPatch) => {
      updateRunMetadata(rid, patch)
        .then(() => {
          const id = runIdRef.current;
          if (id) fetchBranches(id).then(setBranchInfo).catch(() => {});
        })
        .catch(() => {});
    },
    [updateRunMetadata, fetchBranches],
  );

  // ── Placeholder when no run ────────────────────────────────────────────────
  if (!runId || !branchInfo) {
    return (
      <div
        className="flex items-center justify-center rounded-lg border"
        style={{
          minHeight: 200,
          borderColor: "var(--border-color)",
          background: "var(--bg-secondary)",
        }}
      >
        {loadError ? (
          <p
            className="text-[13px] text-center leading-relaxed text-red-400"
          >
            {loadError}
          </p>
        ) : (
          <p
            className="text-[13px] text-center leading-relaxed"
            style={{ color: "var(--text-muted)" }}
          >
            运行后显示分支画布
            <br />
            <span className="text-[11px] opacity-70">Branch canvas appears after a run</span>
          </p>
        )}
      </div>
    );
  }

  // ── Short active-run label for toolbar ────────────────────────────────────
  const activeRunShort = activeRunId ? activeRunId.slice(-8) : "—";
  const statusLabel = result?.status ?? "—";

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div
      className="flex flex-col gap-2 rounded-lg border overflow-hidden"
      style={{
        borderColor: "var(--border-color)",
        background: "var(--bg-primary)",
      }}
    >
      {/* Toolbar */}
      <div
        className="flex items-center gap-2 px-3 py-2 border-b shrink-0"
        style={{ borderColor: "var(--border-color)" }}
      >
        <span
          className="text-[11px] font-mono px-1.5 py-0.5 rounded"
          style={{ background: "var(--bg-tertiary)", color: "var(--text-secondary)" }}
          title={activeRunId ?? ""}
        >
          {activeRunShort}
        </span>
        <span
          className="text-[11px]"
          style={{ color: "var(--text-muted)" }}
        >
          {statusLabel}
        </span>

        <div className="flex-1" />

        <button
          onClick={handleResetLayout}
          title="重置布局 / Reset layout"
          className="flex items-center gap-1 text-[11px] px-2 py-1 rounded transition-all hover:bg-[var(--bg-hover)]"
          style={{ color: "var(--text-muted)" }}
        >
          <RotateCcw className="w-3 h-3" />
          <span>重置布局 / Reset layout</span>
        </button>
      </div>

      {/* Canvas + Inspector split */}
      <div className="flex gap-2 px-2 pb-2" style={{ minHeight: 520 }}>
        {/* Canvas — flex-1 */}
        <div className="flex-1 min-w-0">
          <BranchCanvas
            nodes={displayNodes}
            edges={model.edges}
            nodeTypes={branchCanvasNodeTypes}
            selectedNodeId={selectedNodeId}
            onNodeClick={handleNodeClick}
            onNodeDoubleClick={handleNodeDoubleClick}
            onNodeDragStop={handleDragStop}
          />
        </div>

        {/* Inspector — fixed width */}
        <div className="shrink-0" style={{ width: 280 }}>
          <BranchCanvasInspector
            node={selectedNode}
            activeRunId={activeRunId}
            onSwitchRun={switchRun}
            onUpdateMetadata={handleUpdateMetadata}
            onRefineFromCheckpoint={(rid, cpId) => onRefineFromCheckpoint?.(rid, cpId)}
            onContinueFromRun={(rid) => onContinueFromRun?.(rid)}
            // TODO(V2.1-7): wire real branch draft submit; branch_action nodes are not emitted until then
            onSubmitBranch={() => {}}
            onCancelBranch={() => {}}
            disabled={disabled}
          />
        </div>
      </div>
    </div>
  );
}
