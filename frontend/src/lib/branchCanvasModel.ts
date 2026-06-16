import type { Node, Edge } from "@xyflow/react";
import type { BranchTreeNode, CheckpointTimelineEntry } from "../hooks/usePngShader";

export type BranchCanvasNodeType =
  | "input"
  | "run"
  | "checkpoint"
  | "branch_action"
  | "variant_group"
  | "variant_run"
  | "region_constraint"
  | "preference";

// React Flow v12 requires node data to extend Record<string, unknown>.
export interface BranchCanvasNodeData extends Record<string, unknown> {
  type: BranchCanvasNodeType;
  run_id?: string;
  checkpoint_id?: string;
  parent_run_id?: string | null;
  source_checkpoint_id?: string | null;
  title?: string | null;
  label: string;
  status?: string;
  score?: number | null;
  delta?: number | null;
  accepted?: boolean | null;
  favorite?: boolean;
  feedback?: string | null;
  thumbnail_artifact_id?: string | null;
  shader_artifact_id?: string | null;
  group_id?: string | null;
  collapsed?: boolean;
}

export type BranchCanvasEdgeRelation =
  | "timeline_next"
  | "branch_from"
  | "active_run"
  | "variant_child"
  | "constraint_applies"
  | "preference_influences";

export interface BranchCanvasEdgeData extends Record<string, unknown> {
  relation: BranchCanvasEdgeRelation;
  label?: string;
}

export type BranchCanvasNode = Node<BranchCanvasNodeData>;
export type BranchCanvasEdge = Edge<BranchCanvasEdgeData>;

// ─── Adapter (V2.1-2) ────────────────────────────────────────────────────────

export interface BuildBranchCanvasInput {
  activeRunId: string | null;
  branchTree: BranchTreeNode | null;
  timelinesByRunId: Record<string, CheckpointTimelineEntry[]>;
  statusesByRunId: Record<string, { status?: string; final_score?: number | null } | null>;
  collapsedRunIds: Set<string>;
  favoriteRunIds?: Set<string>;
}

export interface BuildBranchCanvasOutput {
  nodes: BranchCanvasNode[];
  edges: BranchCanvasEdge[];
}

export const MAX_EXPANDED_RUNS = 3;
export const MAX_VISIBLE_CHECKPOINTS_PER_RUN = 8;

/** Stable DFS flattening of BranchTreeNode into an ordered array. */
function flattenTree(root: BranchTreeNode): BranchTreeNode[] {
  const result: BranchTreeNode[] = [];
  const stack: BranchTreeNode[] = [root];
  while (stack.length > 0) {
    const node = stack.pop()!;
    result.push(node);
    // Push children in reverse so DFS pops left-to-right (created_at-sorted by backend)
    for (let i = node.children.length - 1; i >= 0; i--) {
      stack.push(node.children[i]);
    }
  }
  return result;
}

/**
 * Determine which run_ids are "referenced" as a source_checkpoint_id by some
 * child run, keyed by parent run_id → Set<source_checkpoint_id>.
 */
function buildReferencedCpMap(allNodes: BranchTreeNode[]): Map<string, Set<string>> {
  const map = new Map<string, Set<string>>();
  for (const node of allNodes) {
    if (node.parent_run_id && node.source_checkpoint_id) {
      let set = map.get(node.parent_run_id);
      if (!set) {
        set = new Set();
        map.set(node.parent_run_id, set);
      }
      set.add(node.source_checkpoint_id);
    }
  }
  return map;
}

/**
 * Cap timeline entries to MAX_VISIBLE_CHECKPOINTS_PER_RUN.
 * Priority keep: candidate:selected, accepted===true, final:selected, referenced by child.
 * Then trim to last MAX_VISIBLE_CHECKPOINTS_PER_RUN.
 */
function capTimeline(
  entries: CheckpointTimelineEntry[],
  referencedCpIds: Set<string>,
): CheckpointTimelineEntry[] {
  if (entries.length <= MAX_VISIBLE_CHECKPOINTS_PER_RUN) return entries;

  // Filter to priority entries, preserving timeline order
  const priority = entries.filter(
    (e) =>
      e.id === "candidate:selected" ||
      e.accepted === true ||
      e.id === "final:selected" ||
      referencedCpIds.has(e.id),
  );

  const filtered = priority.length <= MAX_VISIBLE_CHECKPOINTS_PER_RUN
    ? priority
    : priority.slice(-MAX_VISIBLE_CHECKPOINTS_PER_RUN);

  return filtered;
}

/**
 * Pure adapter: converts V2 branch/timeline/status data into React-Flow
 * canvas nodes/edges. All nodes get position {x:0,y:0} — layout is V2.1-3.
 * No Date/Math.random — fully deterministic.
 */
export function buildBranchCanvasModel(
  input: BuildBranchCanvasInput,
): BuildBranchCanvasOutput {
  const {
    activeRunId,
    branchTree,
    timelinesByRunId,
    statusesByRunId,
    collapsedRunIds,
    favoriteRunIds = new Set(),
  } = input;

  if (branchTree === null) return { nodes: [], edges: [] };

  const allTreeNodes = flattenTree(branchTree);
  const treeNodeMap = new Map<string, BranchTreeNode>(
    allTreeNodes.map((n) => [n.run_id, n]),
  );
  const rootRunId = branchTree.run_id;

  // ── Compute expandedRunIds ──────────────────────────────────────────────────
  // Priority: active first, then favorites in DFS order. Cap at MAX_EXPANDED_RUNS.
  // Then remove any id in collapsedRunIds (explicit collapse always wins).
  const candidateIds: string[] = [];
  if (activeRunId !== null && treeNodeMap.has(activeRunId)) {
    candidateIds.push(activeRunId);
  }
  for (const node of allTreeNodes) {
    if (
      favoriteRunIds.has(node.run_id) &&
      !candidateIds.includes(node.run_id)
    ) {
      candidateIds.push(node.run_id);
    }
  }
  const expandedRunIds = new Set(
    candidateIds
      .slice(0, MAX_EXPANDED_RUNS)
      .filter((id) => !collapsedRunIds.has(id)),
  );

  // ── Build referenced-cp map for cap logic ──────────────────────────────────
  const referencedCpByRunId = buildReferencedCpMap(allTreeNodes);

  // ── Track which cp: nodes were actually emitted ────────────────────────────
  // key = "cp:{run_id}:{checkpoint_id}" → true
  const emittedCpNodeIds = new Set<string>();

  const nodes: BranchCanvasNode[] = [];
  const edges: BranchCanvasEdge[] = [];

  // ── Input node ─────────────────────────────────────────────────────────────
  nodes.push({
    id: "input",
    type: "input",
    position: { x: 0, y: 0 },
    data: { type: "input", label: "Input PNG" },
  });

  // ── Input → root run edge ──────────────────────────────────────────────────
  edges.push({
    id: `input->${rootRunId}`,
    source: "input",
    target: `run:${rootRunId}`,
    data: { relation: "timeline_next" },
  });

  // ── Run nodes + checkpoint nodes (DFS order) ───────────────────────────────
  for (const treeNode of allTreeNodes) {
    const { run_id } = treeNode;
    const statusOverride = statusesByRunId[run_id];
    const status =
      statusOverride?.status ?? treeNode.status;
    const score =
      statusOverride !== undefined && statusOverride !== null && "final_score" in statusOverride
        ? (statusOverride.final_score ?? null)
        : (treeNode.final_score ?? null);

    const title = treeNode.title ?? null;
    const label = title ?? run_id.slice(-8);
    const isExpanded = expandedRunIds.has(run_id);

    nodes.push({
      id: `run:${run_id}`,
      type: "run",
      position: { x: 0, y: 0 },
      data: {
        type: "run",
        run_id,
        parent_run_id: treeNode.parent_run_id ?? null,
        source_checkpoint_id: treeNode.source_checkpoint_id ?? null,
        title,
        label,
        status,
        score,
        favorite: treeNode.favorite ?? false,
        feedback: treeNode.feedback ?? null,
        collapsed: !isExpanded,
      },
    });

    // ── Checkpoint nodes for expanded runs ──────────────────────────────────
    if (isExpanded) {
      const rawTimeline = timelinesByRunId[run_id] ?? [];
      const referencedCpIds = referencedCpByRunId.get(run_id) ?? new Set<string>();
      const kept = capTimeline(rawTimeline, referencedCpIds);

      // chain: run node → cp[0] → cp[1] → …
      let prevId = `run:${run_id}`;
      for (let idx = 0; idx < kept.length; idx++) {
        const entry = kept[idx];
        const cpNodeId = `cp:${run_id}:${entry.id}`;
        emittedCpNodeIds.add(cpNodeId);

        nodes.push({
          id: cpNodeId,
          type: "checkpoint",
          position: { x: 0, y: 0 },
          data: {
            type: "checkpoint",
            run_id,
            checkpoint_id: entry.id,
            label: entry.label,
            score: entry.score ?? null,
            delta: entry.delta ?? null,
            accepted: entry.accepted ?? null,
            thumbnail_artifact_id: entry.artifact_ids?.render ?? null,
            shader_artifact_id: entry.artifact_ids?.shader ?? null,
          },
        });

        edges.push({
          id: `tl:${run_id}:${idx}`,
          source: prevId,
          target: cpNodeId,
          data: { relation: "timeline_next" },
        });

        prevId = cpNodeId;
      }
    }
  }

  // ── branch_from edges for non-root runs ────────────────────────────────────
  for (const treeNode of allTreeNodes) {
    if (!treeNode.parent_run_id) continue; // root — no branch edge

    const { run_id, parent_run_id, source_checkpoint_id } = treeNode;
    const cpNodeId =
      source_checkpoint_id
        ? `cp:${parent_run_id}:${source_checkpoint_id}`
        : null;

    const sourceId =
      cpNodeId && emittedCpNodeIds.has(cpNodeId)
        ? cpNodeId
        : `run:${parent_run_id}`;

    // Resolve the source checkpoint label for the edge, if we know it
    const sourceCpLabel = treeNode.source_checkpoint_label ?? undefined;

    edges.push({
      id: `branch:${run_id}`,
      source: sourceId,
      target: `run:${run_id}`,
      data: {
        relation: "branch_from",
        ...(sourceCpLabel !== undefined ? { label: sourceCpLabel } : {}),
      },
    });
  }

  return { nodes, edges };
}
