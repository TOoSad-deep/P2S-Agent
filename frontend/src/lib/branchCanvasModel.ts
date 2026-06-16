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
  changes_summary?: string | null;
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
  collapsedGroupIds?: Set<string>;
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
 * Must-keep: candidate:selected, accepted===true, final:selected, referenced by child.
 * When must-keep entries alone meet or exceed the cap, keep exactly those (no eviction).
 * Remaining slots are filled from the most-recent non-priority entries.
 * Output preserves original timeline order.
 */
function capTimeline(
  entries: CheckpointTimelineEntry[],
  referencedCpIds: Set<string>,
): CheckpointTimelineEntry[] {
  if (entries.length <= MAX_VISIBLE_CHECKPOINTS_PER_RUN) return entries;

  const mustKeepIds = new Set<string>();
  for (const e of entries) {
    if (
      e.id === "candidate:selected" ||
      e.accepted === true ||
      e.id === "final:selected" ||
      referencedCpIds.has(e.id)
    ) {
      mustKeepIds.add(e.id);
    }
  }

  // All slots consumed by must-keep entries → keep exactly those (timeline order).
  if (mustKeepIds.size >= MAX_VISIBLE_CHECKPOINTS_PER_RUN) {
    return entries.filter((e) => mustKeepIds.has(e.id));
  }

  // Fill remaining slots from the most recent non-priority entries.
  const freeSlots = MAX_VISIBLE_CHECKPOINTS_PER_RUN - mustKeepIds.size;
  const fillIds = new Set(
    entries.filter((e) => !mustKeepIds.has(e.id)).slice(-freeSlots).map((e) => e.id),
  );
  return entries.filter((e) => mustKeepIds.has(e.id) || fillIds.has(e.id));
}

// ─── Variant group helpers ───────────────────────────────────────────────────

/**
 * Aggregate status for a variant group from its member statuses.
 * Rules:
 *   - all "completed"            → "completed"
 *   - any "running" or "queued"  → "running"
 *   - any "completed" but not all → "partial_failed"
 *   - else                       → "failed"
 */
function aggregateVariantStatus(statuses: string[]): string {
  if (statuses.length === 0) return "failed";
  const hasRunning = statuses.some((s) => s === "running" || s === "queued");
  if (hasRunning) return "running";
  const allCompleted = statuses.every((s) => s === "completed");
  if (allCompleted) return "completed";
  const anyCompleted = statuses.some((s) => s === "completed");
  if (anyCompleted) return "partial_failed";
  return "failed";
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
    collapsedGroupIds = new Set(),
  } = input;

  if (branchTree === null) return { nodes: [], edges: [] };

  const allTreeNodes = flattenTree(branchTree);
  const rootRunId = branchTree.run_id;

  // ── Identify variant run_ids (those with a variant_group_id) ──────────────
  const variantRunIds = new Set<string>();
  // Map from group_id → ordered list of tree nodes (DFS order, first-encounter)
  const variantGroupMap = new Map<string, BranchTreeNode[]>();
  for (const node of allTreeNodes) {
    if (node.variant_group_id) {
      variantRunIds.add(node.run_id);
      let members = variantGroupMap.get(node.variant_group_id);
      if (!members) {
        members = [];
        variantGroupMap.set(node.variant_group_id, members);
      }
      members.push(node);
    }
  }

  // ── Node lookup for ancestor-chain expansion ──────────────────────────────
  const nodeById = new Map<string, BranchTreeNode>(
    allTreeNodes.map((n) => [n.run_id, n]),
  );

  // ── Compute expandedRunIds (only non-variant runs participate) ─────────────
  // Priority: active first, then the active run's ancestor chain (so a child's
  // branch_from edge can connect to the real source checkpoint on its parent
  // rather than degrading to the parent run node), then favorites in DFS order.
  // Cap at MAX_EXPANDED_RUNS. Then remove any id in collapsedRunIds (explicit
  // collapse always wins). Variant runs are excluded — they have no checkpoint
  // nodes and must not consume a budget slot that a real run could use.
  const candidateIds: string[] = [];
  if (activeRunId !== null
      && !variantRunIds.has(activeRunId)
      && allTreeNodes.some((n) => n.run_id === activeRunId)) {
    candidateIds.push(activeRunId);
  }
  // Walk up from the active run, immediate parent first. Skipped for variant
  // active runs: their branch_from edge targets the variant group node, not a
  // checkpoint, so expanding ancestors would only burn budget for favorites.
  if (activeRunId !== null && !variantRunIds.has(activeRunId)) {
    const seen = new Set<string>();
    let cursor = nodeById.get(activeRunId)?.parent_run_id ?? null;
    while (cursor && !seen.has(cursor)) {
      seen.add(cursor);
      const pnode = nodeById.get(cursor);
      if (pnode && !variantRunIds.has(cursor) && !candidateIds.includes(cursor)) {
        candidateIds.push(cursor);
      }
      cursor = pnode?.parent_run_id ?? null;
    }
  }
  for (const node of allTreeNodes) {
    if (
      !variantRunIds.has(node.run_id) &&
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

  // ── Regular run nodes + checkpoint nodes (DFS order, skipping variant runs) ─
  for (const treeNode of allTreeNodes) {
    const { run_id } = treeNode;

    // Skip variant runs — they are handled in the variant group pass below
    if (variantRunIds.has(run_id)) continue;

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
            changes_summary: entry.changes_summary ?? null,
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

  // ── Variant group nodes + variant_run child nodes (in first-encounter order) ─
  for (const [groupId, members] of variantGroupMap) {
    // Take parent info from the first member (all share the same parent + source checkpoint)
    const firstMember = members[0];
    const parentRunId = firstMember.parent_run_id ?? null;
    const sourceCheckpointId = firstMember.source_checkpoint_id ?? null;

    // Aggregate status from member tree nodes
    const memberStatuses = members.map((m) => {
      const ov = statusesByRunId[m.run_id];
      return ov?.status ?? m.status;
    });
    const groupStatus = aggregateVariantStatus(memberStatuses);

    const isGroupCollapsed = collapsedGroupIds.has(groupId);
    const groupNodeId = `vg:${groupId}`;

    nodes.push({
      id: groupNodeId,
      type: "variant_group",
      position: { x: 0, y: 0 },
      data: {
        type: "variant_group",
        group_id: groupId,
        label: `Variants (${members.length})`,
        parent_run_id: parentRunId,
        source_checkpoint_id: sourceCheckpointId,
        status: groupStatus,
        collapsed: isGroupCollapsed,
      },
    });

    // branch_from edge: source = cp: node if emitted, else run: node
    const cpNodeId =
      sourceCheckpointId && parentRunId
        ? `cp:${parentRunId}:${sourceCheckpointId}`
        : null;

    const groupSourceId =
      cpNodeId && emittedCpNodeIds.has(cpNodeId)
        ? cpNodeId
        : parentRunId
          ? `run:${parentRunId}`
          : "input";

    edges.push({
      id: `vg-branch:${groupId}`,
      source: groupSourceId,
      target: groupNodeId,
      data: { relation: "branch_from" },
    });

    // If not collapsed: emit variant_run child nodes + variant_child edges
    if (!isGroupCollapsed) {
      // Sort by variant_index then id for stable ordering
      const sortedMembers = [...members].sort((a, b) => {
        const ia = a.variant_index ?? 0;
        const ib = b.variant_index ?? 0;
        if (ia !== ib) return ia - ib;
        return a.run_id < b.run_id ? -1 : a.run_id > b.run_id ? 1 : 0;
      });

      for (const member of sortedMembers) {
        const { run_id: memberId } = member;
        const memberStatusOverride = statusesByRunId[memberId];
        const memberStatus = memberStatusOverride?.status ?? member.status;
        const memberScore =
          memberStatusOverride !== undefined && memberStatusOverride !== null && "final_score" in memberStatusOverride
            ? (memberStatusOverride.final_score ?? null)
            : (member.final_score ?? null);

        nodes.push({
          id: `run:${memberId}`,
          type: "variant_run",
          position: { x: 0, y: 0 },
          data: {
            type: "variant_run",
            run_id: memberId,
            variant_group_id: groupId,
            variant_index: member.variant_index ?? null,
            variant_label: member.variant_label ?? null,
            label: member.variant_label ?? memberId.slice(-8),
            status: memberStatus,
            score: memberScore,
            favorite: member.favorite ?? false,
          },
        });

        edges.push({
          id: `vc:${memberId}`,
          source: groupNodeId,
          target: `run:${memberId}`,
          data: { relation: "variant_child" },
        });
      }
    }
  }

  // ── branch_from edges for non-root, non-variant runs ──────────────────────
  for (const treeNode of allTreeNodes) {
    if (!treeNode.parent_run_id) continue; // root — no branch edge
    if (variantRunIds.has(treeNode.run_id)) continue; // variant runs handled above

    const { run_id, parent_run_id, source_checkpoint_id } = treeNode;

    // Parent might be a variant run (no cp nodes) — fallback to run: node always works
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
