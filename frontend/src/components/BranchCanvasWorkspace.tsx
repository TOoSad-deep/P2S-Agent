// BranchCanvasWorkspace.tsx — orchestrator for the Branch Canvas Workspace (V2.1-7).
// Fetches V2 data, builds the canvas model + layout, and renders the canvas + inspector
// with selection / run-switch / drag-persistence / branch-draft submit flow.
import { useState, useEffect, useMemo, useCallback, useRef } from "react";
import { RotateCcw } from "lucide-react";
import type {
  PngShaderResult,
  CheckpointTimelineEntry,
  BranchTreeResponse,
  RunMetadataPatch,
  BranchRefineRequest,
  ExploreVariantsRequest,
  ExploreVariantsResponse,
  VariantGroupStatus,
  CreateDrawSessionRequest,
  CreateDrawSessionResponse,
  DrawSessionStatus,
  DrawMoreRequest,
  DrawMoreResponse,
  RedrawCardResponse,
  DrawCardEventType,
  RegionConstraint,
} from "../hooks/usePngShader";
import {
  buildBranchCanvasModel,
  buildDrawSessionModel,
  buildRegionConstraintModel,
  type BranchCanvasNode,
  type BranchCanvasEdge,
} from "../lib/branchCanvasModel";
import { layoutBranchCanvas, COLUMN_WIDTH } from "../lib/branchCanvasLayout";
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
  branchRefine: (parentRunId: string, request: BranchRefineRequest) => Promise<string | null>;
  exploreVariants: (parentRunId: string, request: ExploreVariantsRequest) => Promise<ExploreVariantsResponse | null>;
  fetchVariantGroup: (groupId: string) => Promise<VariantGroupStatus>;
  stopVariantGroup: (groupId: string) => Promise<void>;
  selectVariantWinner: (groupId: string, runId: string, reason?: string) => Promise<void>;
  rateVariant: (groupId: string, runId: string, rating: number, reason?: string, tags?: string[]) => Promise<void>;
  createDrawSession: (parentRunId: string, request: CreateDrawSessionRequest) => Promise<CreateDrawSessionResponse | null>;
  fetchDrawSession: (drawId: string) => Promise<DrawSessionStatus>;
  drawMore: (drawId: string, request: DrawMoreRequest) => Promise<DrawMoreResponse | null>;
  redrawCard: (drawId: string, runId: string, opts?: { reason?: string; diversity?: string }) => Promise<RedrawCardResponse | null>;
  cardEvent: (drawId: string, runId: string, eventType: DrawCardEventType, opts?: { value?: unknown; reason?: string; tags?: string[] }) => Promise<void>;
  // optional preview callback (wired by parent if desired)
  onPreviewNode?: (node: BranchCanvasNode | null) => void;
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
  branchRefine,
  exploreVariants,
  fetchVariantGroup,
  stopVariantGroup,
  selectVariantWinner,
  rateVariant,
  createDrawSession,
  fetchDrawSession,
  drawMore,
  redrawCard,
  cardEvent,
  onPreviewNode,
  disabled,
}: Props) {
  // ── State ──────────────────────────────────────────────────────────────────
  const [branchInfo, setBranchInfo] = useState<BranchTreeResponse | null>(null);
  const [activeTimeline, setActiveTimeline] = useState<CheckpointTimelineEntry[]>([]);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [branchDraft, setBranchDraft] = useState<{ sourceRunId: string; sourceCheckpointId: string } | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  // TODO(V2.1-8): make stateful when the collapse toggle UI is added
  const collapsedRunIds = useMemo(() => new Set<string>(), []);
  const [collapsedGroupIds, setCollapsedGroupIds] = useState<Set<string>>(new Set());
  const [activeVariantGroupId, setActiveVariantGroupId] = useState<string | null>(null);
  const [variantGroup, setVariantGroup] = useState<VariantGroupStatus | null>(null);
  const [activeDrawId, setActiveDrawId] = useState<string | null>(null);
  const [drawSession, setDrawSession] = useState<DrawSessionStatus | null>(null);
  const [collapsedDrawIds, setCollapsedDrawIds] = useState<Set<string>>(new Set());
  const [layoutOverrides, setLayoutOverrides] = useState<Record<string, { x: number; y: number }>>({});
  const [loadError, setLoadError] = useState<string | null>(null);
  const [regionDraft, setRegionDraft] = useState<{ anchorNodeId: string; regions: RegionConstraint[] } | null>(null);

  // ── runId ref: kept in sync so async callbacks read the latest value ──────
  const runIdRef = useRef(runId);
  // ── Double-submit guard ────────────────────────────────────────────────────
  const submittingRef = useRef(false);
  useEffect(() => { runIdRef.current = runId; }, [runId]);

  // ── Clear selection AND draft on run-switch (I3 + draft cleanup) ──────────
  useEffect(() => {
    setSelectedNodeId(null);
    setBranchDraft(null);
  }, [runId]);

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

  // ── Derived: canvas model (base branch model + optional draw-session + region merge) ─
  const model = useMemo(() => {
    const base = buildBranchCanvasModel({
      activeRunId,
      branchTree: branchInfo?.tree ?? null,
      timelinesByRunId,
      statusesByRunId,
      collapsedRunIds,
      favoriteRunIds,
      collapsedGroupIds,
    });
    let merged = base;
    if (drawSession) {
      const drawModel = buildDrawSessionModel(drawSession, {
        anchorNodeId: `run:${drawSession.parent_run_id}`,
        collapsed: collapsedDrawIds.has(drawSession.draw_id),
      });
      merged = {
        nodes: [...merged.nodes, ...drawModel.nodes],
        edges: [...merged.edges, ...drawModel.edges],
      };
    }
    if (regionDraft) {
      const regionModel = buildRegionConstraintModel(regionDraft.regions, {
        anchorNodeId: regionDraft.anchorNodeId,
      });
      merged = {
        nodes: [...merged.nodes, ...regionModel.nodes],
        edges: [...merged.edges, ...regionModel.edges],
      };
    }
    return merged;
  }, [activeRunId, branchInfo?.tree, timelinesByRunId, statusesByRunId, collapsedRunIds, favoriteRunIds, collapsedGroupIds, drawSession, collapsedDrawIds, regionDraft]);

  // ── Derived: positioned nodes (layout only; no selection) ─────────────────
  const positionedNodes = useMemo(
    () => layoutBranchCanvas(model.nodes, model.edges, layoutOverrides),
    [model.nodes, model.edges, layoutOverrides],
  );

  // ── Derived: draft node source (shared between displayNodes + displayEdges)─
  const draftSrcNode = useMemo<BranchCanvasNode | null>(() => {
    if (!branchDraft) return null;
    const cpId = `cp:${branchDraft.sourceRunId}:${branchDraft.sourceCheckpointId}`;
    return (
      positionedNodes.find((n) => n.id === cpId) ??
      positionedNodes.find((n) => n.id === `run:${branchDraft.sourceRunId}`) ??
      null
    );
  }, [branchDraft, positionedNodes]);

  // ── Derived: display nodes (inject selected flag + optional draft node) ────
  const displayNodes = useMemo<BranchCanvasNode[]>(() => {
    const base = positionedNodes.map((n) => {
      const sel = n.id === selectedNodeId;
      return n.selected === sel ? n : { ...n, selected: sel };
    });

    if (!branchDraft || !draftSrcNode) return base;

    const pos = { x: draftSrcNode.position.x + COLUMN_WIDTH, y: draftSrcNode.position.y + 40 };

    const draftNode: BranchCanvasNode = {
      id: "draft",
      type: "branch_action",
      position: pos,
      selected: selectedNodeId === "draft",
      data: {
        type: "branch_action",
        run_id: branchDraft.sourceRunId,
        source_checkpoint_id: branchDraft.sourceCheckpointId,
        label: "新分支 / New branch",
      },
    };

    return [...base, draftNode];
  }, [positionedNodes, selectedNodeId, branchDraft, draftSrcNode]);

  // ── Derived: display edges (add draft edge when draft is active) ───────────
  const displayEdges = useMemo<BranchCanvasEdge[]>(
    () =>
      branchDraft && draftSrcNode
        ? [
            ...model.edges,
            {
              id: "draft-edge",
              source: draftSrcNode.id,
              target: "draft",
              data: { relation: "branch_from" as const },
            },
          ]
        : model.edges,
    [branchDraft, draftSrcNode, model.edges],
  );

  // ── Derived: selected node (must derive from displayNodes to resolve "draft")
  const selectedNode = useMemo(
    () => displayNodes.find((n) => n.id === selectedNodeId) ?? null,
    [displayNodes, selectedNodeId],
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
      if (!node) return;
      // Double-click a variant_group: toggle collapse (do NOT also switchRun)
      if (node.data.type === "variant_group" && node.data.group_id) {
        const gid = node.data.group_id;
        setCollapsedGroupIds((prev) => {
          const next = new Set(prev);
          if (next.has(gid)) { next.delete(gid); } else { next.add(gid); }
          return next;
        });
        return;
      }
      // Double-click a draw_session: toggle collapse (do NOT also switchRun)
      if (node.data.type === "draw_session" && typeof node.data.draw_id === "string") {
        const did = node.data.draw_id;
        setCollapsedDrawIds((prev) => {
          const next = new Set(prev);
          if (next.has(did)) { next.delete(did); } else { next.add(did); }
          return next;
        });
        return;
      }
      // Double-click a run or variant_run: switch active run
      if (
        (node.data.type === "run" || node.data.type === "variant_run") &&
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

  // ── Branch draft internal handlers ─────────────────────────────────────────

  const handleRefineFromCheckpoint = useCallback(
    (rid: string, checkpointId: string) => {
      setSubmitError(null);
      setBranchDraft({ sourceRunId: rid, sourceCheckpointId: checkpointId });
      setSelectedNodeId("draft");
    },
    [],
  );

  const handleContinueFromRun = useCallback(
    (rid: string) => {
      setSubmitError(null);
      setBranchDraft({ sourceRunId: rid, sourceCheckpointId: "final:selected" });
      setSelectedNodeId("draft");
    },
    [],
  );

  const handleSubmitBranch = useCallback(
    (_runId: string, request: BranchRefineRequest) => {
      if (!branchDraft || submittingRef.current) return;
      submittingRef.current = true;
      setSubmitError(null);
      branchRefine(branchDraft.sourceRunId, request)
        .then((newRunId) => {
          if (newRunId) {
            setBranchDraft(null);
            setSelectedNodeId(null);
          } else {
            setSubmitError("提交失败 / Submit failed");
          }
        })
        .catch(() => { setSubmitError("提交失败 / Submit failed"); })  // belt-and-suspenders if it ever throws
        .finally(() => { submittingRef.current = false; });
    },
    [branchDraft, branchRefine],
  );

  const handleCancelBranch = useCallback(() => {
    setSubmitError(null);
    setBranchDraft(null);
    setSelectedNodeId(null);
  }, []);

  // ── Region draft callback (V4.2) ───────────────────────────────────────────
  const handleRegionsChange = useCallback(
    (anchorNodeId: string, regions: RegionConstraint[]) => {
      setRegionDraft(regions.length > 0 ? { anchorNodeId, regions } : null);
    },
    [],
  );

  // ── Variant exploration handlers ───────────────────────────────────────────

  const handleExploreVariants = useCallback(
    async (parentRunId: string, request: ExploreVariantsRequest) => {
      try {
        const resp = await exploreVariants(parentRunId, request);
        setBranchDraft(null);
        setSelectedNodeId(null);
        if (resp?.group_id) {
          setActiveVariantGroupId(resp.group_id);
          if (runIdRef.current) fetchBranches(runIdRef.current).then(setBranchInfo).catch(() => {});
        }
      } catch {
        setSubmitError("变体探索失败 / Explore variants failed");
      }
    },
    [exploreVariants, fetchBranches],
  );

  const handleSelectWinner = useCallback(
    async (groupId: string, winnerRunId: string, reason?: string) => {
      try { await selectVariantWinner(groupId, winnerRunId, reason); } catch { return; }
      switchRun(winnerRunId);
      setActiveVariantGroupId(null);
      if (runIdRef.current) fetchBranches(runIdRef.current).then(setBranchInfo).catch(() => {});
      fetchVariantGroup(groupId).then(setVariantGroup).catch(() => {});
    },
    [selectVariantWinner, switchRun, fetchBranches, fetchVariantGroup],
  );

  const handleStopGroup = useCallback(
    (groupId: string) => { stopVariantGroup(groupId); },
    [stopVariantGroup],
  );

  // reason/tags forwarding added in V3-8 when the rating UI lands
  const handleRateVariant = useCallback(
    (groupId: string, rid: string, rating: number) => { rateVariant(groupId, rid, rating); },
    [rateVariant],
  );

  // ── Draw-session handlers (V3.5) ───────────────────────────────────────────

  const handleStartDraw = useCallback(
    async (parentRunId: string, request: CreateDrawSessionRequest) => {
      const resp = await createDrawSession(parentRunId, request);
      if (resp) {
        setDrawSession(null);
        setActiveDrawId(resp.draw_id);
        setBranchDraft(null);
        setSelectedNodeId(null);
        if (runIdRef.current) fetchBranches(runIdRef.current).then(setBranchInfo).catch(() => {});
      }
    },
    [createDrawSession, fetchBranches],
  );

  const handleDrawMore = useCallback(
    async (drawId: string, request: DrawMoreRequest) => {
      await drawMore(drawId, request);
      fetchDrawSession(drawId).then(setDrawSession).catch(() => {});
    },
    [drawMore, fetchDrawSession],
  );

  const handleRedrawCard = useCallback(
    async (drawId: string, rid: string) => {
      await redrawCard(drawId, rid);
      fetchDrawSession(drawId).then(setDrawSession).catch(() => {});
      if (runIdRef.current) fetchBranches(runIdRef.current).then(setBranchInfo).catch(() => {});
    },
    [redrawCard, fetchDrawSession, fetchBranches],
  );

  const handleCardEvent = useCallback(
    async (
      drawId: string,
      rid: string,
      eventType: DrawCardEventType,
      opts?: { value?: unknown; reason?: string; tags?: string[] },
    ) => {
      await cardEvent(drawId, rid, eventType, opts);
      fetchDrawSession(drawId).then(setDrawSession).catch(() => {});
    },
    [cardEvent, fetchDrawSession],
  );

  const handlePreviewCard = useCallback(
    (rid: string) => { switchRun(rid); },
    [switchRun],
  );

  // Continue from a card: open a branch draft on its final result (mirrors the
  // variant Continue, which calls onRefineFromCheckpoint(run_id, "final:selected")).
  const handleContinueCard = useCallback(
    (rid: string) => { handleRefineFromCheckpoint(rid, "final:selected"); },
    [handleRefineFromCheckpoint],
  );

  const handleSelectDrawWinner = useCallback(
    async (drawId: string, rid: string) => {
      const card = drawSession?.cards.find((c) => c.run_id === rid);
      if (card?.group_id) {
        await selectVariantWinner(card.group_id, rid);
        fetchDrawSession(drawId).then(setDrawSession).catch(() => {});
        if (runIdRef.current) fetchBranches(runIdRef.current).then(setBranchInfo).catch(() => {});
      }
    },
    [drawSession, selectVariantWinner, fetchDrawSession, fetchBranches],
  );

  const handleStopDraw = useCallback(
    (_drawId: string) => {
      (drawSession?.group_ids ?? []).forEach((gid) => { stopVariantGroup(gid); });
    },
    [drawSession, stopVariantGroup],
  );

  // ── Group polling (2s, stops on terminal status) ───────────────────────────
  useEffect(() => {
    if (!activeVariantGroupId) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const TERMINAL = new Set(["completed", "failed", "partial_failed", "cancelled"]);
    const poll = async () => {
      if (!alive) return;
      try {
        const g = await fetchVariantGroup(activeVariantGroupId);
        if (!alive) return;
        setVariantGroup(g);
        if (runIdRef.current) fetchBranches(runIdRef.current).then((b) => { if (alive) setBranchInfo(b); }).catch(() => {});
        if (TERMINAL.has(g.status)) return; // stop polling
      } catch {
        // best-effort: keep trying
      }
      timer = setTimeout(poll, 2000);
    };
    poll();
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
    };
  }, [activeVariantGroupId, fetchVariantGroup, fetchBranches]);

  // ── Draw-session polling (2s, stops on terminal status) ────────────────────
  useEffect(() => {
    if (!activeDrawId) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | null = null;
    // partial_failed IS terminal for a draw session.
    const TERMINAL = new Set(["completed", "failed", "partial_failed", "cancelled"]);
    const poll = async () => {
      if (!alive) return;
      try {
        const s = await fetchDrawSession(activeDrawId);
        if (!alive) return;
        setDrawSession(s);
        if (runIdRef.current) fetchBranches(runIdRef.current).then((b) => { if (alive) setBranchInfo(b); }).catch(() => {});
        if (TERMINAL.has(s.status)) return; // stop polling
      } catch {
        // best-effort: keep trying
      }
      timer = setTimeout(poll, 2000);
    };
    poll();
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
    };
  }, [activeDrawId, fetchDrawSession, fetchBranches]);

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
            edges={displayEdges}
            nodeTypes={branchCanvasNodeTypes}
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
            onRefineFromCheckpoint={handleRefineFromCheckpoint}
            onContinueFromRun={handleContinueFromRun}
            onSubmitBranch={handleSubmitBranch}
            onCancelBranch={handleCancelBranch}
            onExploreVariants={handleExploreVariants}
            variantGroup={variantGroup}
            onSelectWinner={handleSelectWinner}
            onStopGroup={handleStopGroup}
            onRateVariant={handleRateVariant}
            drawSession={drawSession}
            onStartDraw={handleStartDraw}
            onDrawMore={handleDrawMore}
            onRedrawCard={handleRedrawCard}
            onCardEvent={handleCardEvent}
            onPreviewCard={handlePreviewCard}
            onContinueCard={handleContinueCard}
            onSelectDrawWinner={handleSelectDrawWinner}
            onStopDraw={handleStopDraw}
            fusionEnabled={false}
            submitError={submitError}
            disabled={disabled}
            onRegionsChange={handleRegionsChange}
          />
        </div>
      </div>
    </div>
  );
}
