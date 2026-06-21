// BranchCanvasWorkspace.tsx — orchestrator for the Branch Canvas Workspace (V2.1-7).
// Fetches V2 data, builds the canvas model + layout, and renders the canvas + inspector
// with selection / run-switch / drag-persistence / branch-draft submit flow.
import { useState, useEffect, useMemo, useCallback, useRef } from "react";
import { Panel } from "@xyflow/react";
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
  FusionStatus,
  CreateFusionRequest,
} from "../hooks/usePngShader";
import type { FusionDraft } from "./FusionBuilderPanel";
import {
  buildBranchCanvasModel,
  buildDrawSessionModel,
  buildRegionConstraintModel,
  buildFusionModel,
  selectionFetchTargets,
  type BranchCanvasNode,
  type BranchCanvasEdge,
} from "../lib/branchCanvasModel";
import { layoutBranchCanvas, COLUMN_WIDTH } from "../lib/branchCanvasLayout";
import { shouldPollFusion } from "../lib/fusionPolling";
import BranchCanvas from "./BranchCanvas";
import CanvasToolRail from "./CanvasToolRail";
import { branchCanvasNodeTypes } from "./BranchCanvasNode";
import BranchCanvasInspector from "./BranchCanvasInspector";
import PreviewDock from "./PreviewDock";

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
  createFusion: (request: CreateFusionRequest) => Promise<{ fusion_id: string; status: string } | null>;
  fetchFusion: (fusionId: string) => Promise<FusionStatus>;
  generateCompositeTarget: (fusionId: string) => Promise<FusionStatus | null>;
  runFusion: (fusionId: string, body?: { quality?: Record<string, unknown>; directed_acceptance?: Record<string, unknown> }) => Promise<{ fusion_id: string; status: string; output_run_id: string } | null>;
  // optional preview callback (wired by parent if desired)
  onPreviewNode?: (node: BranchCanvasNode | null) => void;
  inputImageUrl?: string | null;
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
  createFusion,
  fetchFusion,
  generateCompositeTarget,
  runFusion,
  onPreviewNode,
  inputImageUrl,
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
  const [fusionDraft, setFusionDraft] = useState<FusionDraft | null>(null);
  const [activeFusionId, setActiveFusionId] = useState<string | null>(null);
  const [fusionStatus, setFusionStatus] = useState<FusionStatus | null>(null);
  // ── Inspector collapse (P4 layout; persisted to localStorage in P6) ─────────
  const [inspectorOpen, setInspectorOpen] = useState(() => {
    try {
      const raw = localStorage.getItem("p2s.canvas.inspectorOpen");
      if (raw === "0") return false;
      return true; // default true when absent or invalid
    } catch {
      return true;
    }
  });

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

  // ── Reset per-tree detail state when the ROOT run changes ──────────────────
  // variantGroup / drawSession / fusionStatus and their polling pointers belong
  // to one branch tree. drawSession + fusionStatus are merged into the canvas
  // model unconditionally, so without this they leak onto a newly-loaded tree's
  // canvas (anchored at run ids that aren't in it) and the polling effects keep
  // hitting stale ids. A within-tree active-run switch keeps the same root, so
  // this does NOT fire on those — only on loading a genuinely different tree.
  const rootRunId = branchInfo?.root_run_id ?? null;
  useEffect(() => {
    setActiveVariantGroupId(null);
    setVariantGroup(null);
    setActiveDrawId(null);
    setDrawSession(null);
    setActiveFusionId(null);
    setFusionStatus(null);
    setRegionDraft(null);
    setFusionDraft(null);
  }, [rootRunId]);

  // ── localStorage key (keyed by root run) ──────────────────────────────────
  const layoutStorageKey = `branchCanvasLayout:${branchInfo?.root_run_id ?? runId ?? "none"}`;

  // ── Load overrides from localStorage when the key changes ─────────────────
  // Persistence is NOT done via a save-effect keyed on layoutStorageKey: that
  // effect would run in the same commit as this load (which only SCHEDULES the
  // setLayoutOverrides), so on a tree switch it would still see the PRIOR tree's
  // layoutOverrides and write them under the NEW tree's key. Since every tree
  // shares the "input" node id (and re-visited trees share run:/cp: ids), those
  // leaked positions later get applied to the WRONG tree on reload. Instead we
  // persist only from the user-action paths (handleDragStop / handleResetLayout),
  // which always carry the key that was current when the user acted.
  useEffect(() => {
    setLayoutOverrides(loadLayoutOverrides(layoutStorageKey));
  }, [layoutStorageKey]);

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

  // ── Derive activeDrawId from the active run's draw_session_id (BUG-006) ─────
  // Draw-card runs (those carrying a draw_session_id) are excluded from the
  // regular graph pass and only reappear via buildDrawSessionModel once
  // `drawSession` is loaded. A live draw action sets activeDrawId directly, but
  // switching to a draw-card run from the branch list (or after a reload) does
  // not — leaving the active run invisible in the graph. Derive the pointer
  // from the branch tree so the run's draw session is fetched and its cards
  // render. Only acts when the active run IS a draw card; otherwise it leaves
  // a live action's pointer untouched.
  useEffect(() => {
    const tree = branchInfo?.tree;
    if (!tree) return;
    let drawId: string | null = null;
    (function walk(node: BranchTreeResponse["tree"]): void {
      if (drawId) return;
      if (node.run_id === activeRunId) {
        drawId = node.draw_session_id ?? null;
        return;
      }
      for (const child of node.children) walk(child);
    })(tree);
    if (drawId && drawId !== activeDrawId) setActiveDrawId(drawId);
  }, [activeRunId, branchInfo?.tree, activeDrawId]);

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

  // ── Derived: canvas model (base branch model + optional draw-session + region + fusion merge) ─
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
    if (fusionStatus) {
      const fusionModel = buildFusionModel(fusionStatus, {
        baseAnchorNodeId: `run:${fusionStatus.base_run_id}`,
        sourceAnchorNodeIds: Object.fromEntries(
          fusionStatus.source_run_ids.map((r) => [r, `run:${r}`]),
        ),
        outputAnchorNodeId: fusionStatus.output_run_id
          ? `run:${fusionStatus.output_run_id}`
          : undefined,
      });
      merged = {
        nodes: [...merged.nodes, ...fusionModel.nodes],
        edges: [...merged.edges, ...fusionModel.edges],
      };
    }
    return merged;
  }, [activeRunId, branchInfo?.tree, timelinesByRunId, statusesByRunId, collapsedRunIds, favoriteRunIds, collapsedGroupIds, drawSession, collapsedDrawIds, regionDraft, fusionStatus]);

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
      // Selecting a variant/draw node must drive the corresponding fetch so the
      // inspector can render it — not just nodes a live action created. The
      // polling effects key off these ids. Only set (never clear) so an
      // in-progress explore/draw keeps polling when the user clicks elsewhere.
      const { variantGroupId, drawId } = selectionFetchTargets(node);
      if (variantGroupId) setActiveVariantGroupId(variantGroupId);
      if (drawId) setActiveDrawId(drawId);
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
      // Persist on the user-action path (not via a key-driven effect — see the
      // load effect above) so the write always targets the key that is current
      // for the tree the user is actually dragging on.
      setLayoutOverrides((prev) => {
        const next = { ...prev, [id]: pos };
        saveLayoutOverrides(layoutStorageKey, next);
        return next;
      });
    },
    [layoutStorageKey],
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

  // ── Fusion handlers (V4.5) ─────────────────────────────────────────────────

  const handleUseAsBase = useCallback(
    (runId: string) => {
      setFusionDraft((prev) =>
        prev
          ? { ...prev, base_run_id: runId }
          : {
              base_run_id: runId,
              draw_session_id: drawSession?.draw_id ?? null,
              feedback: "",
              regions: [],
            },
      );
    },
    [drawSession],
  );

  const handleUseRegion = useCallback(
    (runId: string) => {
      setFusionDraft((prev) => {
        const base = prev ?? {
          base_run_id: null,
          draw_session_id: drawSession?.draw_id ?? null,
          feedback: "",
          regions: [],
        };
        const n = base.regions.length;
        const newRegion = {
          id: `region_${n}_${Date.now().toString(36)}`,
          label: `Region ${n + 1}`,
          source_run_id: runId,
          instruction: "",
          geometry_type: "rect" as const,
          geometry: { x: 0.25, y: 0.25, w: 0.5, h: 0.5 },
          strength: 0.5,
          blend_mode: "soft" as const,
          feather: 0.08,
        };
        return { ...base, regions: [...base.regions, newRegion] };
      });
    },
    [drawSession],
  );

  const handleFusionDraftChange = useCallback((next: FusionDraft) => {
    setFusionDraft(next);
  }, []);

  const handleCreateFusion = useCallback(async () => {
    if (!fusionDraft?.base_run_id) return;
    const r = await createFusion({
      base_run_id: fusionDraft.base_run_id,
      draw_session_id: fusionDraft.draw_session_id,
      feedback: fusionDraft.feedback,
      regions: fusionDraft.regions,
    });
    if (r) {
      setActiveFusionId(r.fusion_id);
      setFusionStatus(null);
    }
  }, [fusionDraft, createFusion]);

  const handleComposite = useCallback(async () => {
    if (!activeFusionId) return;
    await generateCompositeTarget(activeFusionId);
    fetchFusion(activeFusionId).then(setFusionStatus).catch(() => {});
  }, [activeFusionId, generateCompositeTarget, fetchFusion]);

  const handleRunFusion = useCallback(async () => {
    if (!activeFusionId) return;
    const r = await runFusion(activeFusionId);
    if (r) {
      switchRun(r.output_run_id);
      fetchFusion(activeFusionId).then(setFusionStatus).catch(() => {});
      if (runIdRef.current) fetchBranches(runIdRef.current).then(setBranchInfo).catch(() => {});
    }
  }, [activeFusionId, runFusion, switchRun, fetchFusion, fetchBranches]);

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
      // Intercept fusion-specific events before posting to backend
      if (eventType === "use_as_fusion_base") {
        handleUseAsBase(rid);
        return;
      }
      if (eventType === "use_as_region_source") {
        handleUseRegion(rid);
        return;
      }
      await cardEvent(drawId, rid, eventType, opts);
      fetchDrawSession(drawId).then(setDrawSession).catch(() => {});
    },
    [cardEvent, fetchDrawSession, handleUseAsBase, handleUseRegion],
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

  // ── Fusion polling (2s, runs only while ACTIVELY advancing) ───────────────
  // A fusion only advances on its own while status is "running". draft /
  // target_ready are idle (wait for the user to composite/run) and completed /
  // failed are terminal, so polling any of those just hits the backend every 2s
  // forever. shouldPollFusion() gates re-scheduling on status === "running".
  // The effect is re-keyed on fusionStatus?.status so that when an action
  // handler (handleComposite / handleRunFusion) refreshes an idle fusion into
  // "running", the effect re-runs and RESUMES polling; idle states are kept
  // fresh by those handlers' own fetchFusion calls.
  const activeFusionStatusValue = fusionStatus?.status ?? null;
  useEffect(() => {
    if (!activeFusionId) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const poll = async () => {
      if (!alive) return;
      try {
        const s = await fetchFusion(activeFusionId);
        if (!alive) return;
        setFusionStatus(s);
        if (s.output_run_id && runIdRef.current) {
          fetchBranches(runIdRef.current).then((b) => { if (alive) setBranchInfo(b); }).catch(() => {});
        }
        if (!shouldPollFusion(s.status)) return; // stop: idle (draft/target_ready) or terminal
      } catch {
        // best-effort
      }
      timer = setTimeout(poll, 2000);
    };
    poll();
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
    };
    // activeFusionStatusValue is a key, not read inside: re-running the effect
    // when the status transitions (e.g. idle → running) is exactly how polling
    // resumes after a user action.
  }, [activeFusionId, activeFusionStatusValue, fetchFusion, fetchBranches]);

  // ── Placeholder when no run ────────────────────────────────────────────────
  if (!runId || !branchInfo) {
    return (
      <div
        className="flex items-center justify-center w-full h-full"
        style={{
          minHeight: 200,
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

  // ── Render — full-bleed canvas with floating <Panel> overlays (P4) ─────────
  return (
    <div
      className="relative w-full h-full min-h-0"
      style={{ background: "var(--bg-primary)" }}
    >
      <BranchCanvas
        nodes={displayNodes}
        edges={displayEdges}
        nodeTypes={branchCanvasNodeTypes}
        onNodeClick={handleNodeClick}
        onNodeDoubleClick={handleNodeDoubleClick}
        onNodeDragStop={handleDragStop}
      >
        {/* Left floating tool rail (fit-view / reset-layout / run badge + status) */}
        <CanvasToolRail
          onResetLayout={handleResetLayout}
          activeRunShort={activeRunShort}
          statusLabel={statusLabel}
        />

        {/* Right floating inspector (collapsible) */}
        <Panel position="top-right" style={{ margin: 8 }}>
          {inspectorOpen ? (
            <div
              className="canvas-panel"
              style={{
                width: 380,
                maxHeight: "calc(100% - 16px)",
                overflowY: "auto",
              }}
            >
              <div className="flex justify-end p-1">
                <button
                  onClick={() => {
                    setInspectorOpen(false);
                    try { localStorage.setItem("p2s.canvas.inspectorOpen", "0"); } catch { /* ignore */ }
                  }}
                  title="折叠 / Collapse"
                  className="flex items-center justify-center w-6 h-6 rounded transition-all hover:bg-[var(--bg-hover)]"
                  style={{ color: "var(--text-muted)" }}
                >
                  «
                </button>
              </div>
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
                fusionEnabled={true}
                fusionDraft={fusionDraft}
                fusionStatus={fusionStatus}
                fusionCandidates={
                  drawSession?.cards?.map((c) => ({
                    run_id: c.run_id,
                    label: c.label,
                    thumbnail_url: c.thumbnail_url,
                  })) ?? []
                }
                fusionBaseImageUrl={
                  fusionDraft?.base_run_id
                    ? `/png-shader/runs/${fusionDraft.base_run_id}/artifacts/selected_render`
                    : null
                }
                onFusionDraftChange={handleFusionDraftChange}
                onCreateFusion={handleCreateFusion}
                onComposite={handleComposite}
                onRunFusion={handleRunFusion}
                submitError={submitError}
                disabled={disabled}
                onRegionsChange={handleRegionsChange}
              />
            </div>
          ) : (
            <button
              onClick={() => {
                setInspectorOpen(true);
                try { localStorage.setItem("p2s.canvas.inspectorOpen", "1"); } catch { /* ignore */ }
              }}
              title="展开检查器 / Inspector"
              className="canvas-panel flex items-center justify-center w-7 py-3 transition-all hover:bg-[var(--bg-hover)]"
              style={{ color: "var(--text-secondary)" }}
            >
              »
            </button>
          )}
        </Panel>

        {/* Bottom-right preview dock (single-click shows selected node render vs reference) */}
        <PreviewDock referenceUrl={inputImageUrl} node={selectedNode} />
      </BranchCanvas>
    </div>
  );
}
