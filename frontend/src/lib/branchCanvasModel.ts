import type { Node, Edge } from "@xyflow/react";
import type { BranchTreeNode, CheckpointTimelineEntry, DrawSessionStatus, RegionConstraint, FusionStatus } from "../hooks/usePngShader";
import { truncate } from "./format";

export type BranchCanvasNodeType =
  | "input"
  | "run"
  | "checkpoint"
  | "branch_action"
  | "variant_group"
  | "variant_run"
  | "region_constraint"
  | "preference"
  | "draw_session"
  | "draw_card"
  | "fusion_plan";

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
  | "preference_influences"
  | "draw_from"
  | "draw_card"
  | "replacement_of"
  | "fusion_base"
  | "fusion_source"
  | "fusion_output"
  | "region_source";

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
 * Aligned with backend aggregate_group_status rules:
 *   - []                                       → "queued"
 *   - all "queued"                             → "queued"
 *   - NOT all terminal (terminal = completed|failed|cancelled) → "running"
 *   - all terminal, all completed              → "completed"
 *   - all terminal, some completed             → "partial_failed"
 *   - all terminal, no completed, any cancelled→ "cancelled"
 *   - else                                     → "failed"
 */
function aggregateVariantStatus(statuses: string[]): string {
  if (statuses.length === 0) return "queued";
  const TERMINAL = new Set(["completed", "failed", "cancelled"]);
  if (statuses.every((s) => s === "queued")) return "queued";
  if (!statuses.every((s) => TERMINAL.has(s))) return "running";
  // All terminal from here
  const anyCompleted = statuses.some((s) => s === "completed");
  const allCompleted = statuses.every((s) => s === "completed");
  if (allCompleted) return "completed";
  if (anyCompleted) return "partial_failed";
  if (statuses.some((s) => s === "cancelled")) return "cancelled";
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
  // Runs that also have draw_session_id are draw cards — they are handled by
  // the draw model and must NOT participate in variant_group/variant_run aggregation.
  const variantRunIds = new Set<string>();
  // Map from group_id → ordered list of tree nodes (DFS order, first-encounter)
  const variantGroupMap = new Map<string, BranchTreeNode[]>();
  for (const node of allTreeNodes) {
    if (node.variant_group_id && !node.draw_session_id) {
      variantRunIds.add(node.run_id);
      let members = variantGroupMap.get(node.variant_group_id);
      if (!members) {
        members = [];
        variantGroupMap.set(node.variant_group_id, members);
      }
      members.push(node);
    }
  }

  // ── Compute expandedRunIds (only non-variant runs participate) ─────────────
  // Priority: active first, then favorites in DFS order. Cap at MAX_EXPANDED_RUNS.
  // Then remove any id in collapsedRunIds (explicit collapse always wins).
  // Variant runs are excluded — they have no checkpoint nodes and must not
  // consume a budget slot that a real run could use.
  const candidateIds: string[] = [];
  if (activeRunId !== null
      && !variantRunIds.has(activeRunId)
      && allTreeNodes.some((n) => n.run_id === activeRunId)) {
    candidateIds.push(activeRunId);
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

  // ── Regular run nodes + checkpoint nodes (DFS order, skipping variant/draw runs) ─
  for (const treeNode of allTreeNodes) {
    const { run_id } = treeNode;

    // Skip variant runs — they are handled in the variant group pass below
    if (variantRunIds.has(run_id)) continue;
    // Skip draw card runs — they are handled by the draw model (buildDrawSessionModel)
    if (treeNode.draw_session_id) continue;

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
    if (treeNode.draw_session_id) continue; // draw cards handled by draw model

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

// ─── Draw Session Model ───────────────────────────────────────────────────────

export interface BuildDrawSessionOptions {
  anchorNodeId: string;   // canvas node id the "draw_from" edge originates from
  collapsed?: boolean;    // when true, emit only the draw_session node (no cards)
}

/**
 * Pure function: converts a DrawSessionStatus into React-Flow canvas
 * nodes/edges for the draw_session and its draw_card children.
 * All nodes get position {x:0, y:0} — layout assigns real coordinates.
 */
export function buildDrawSessionModel(
  session: DrawSessionStatus,
  opts: BuildDrawSessionOptions,
): { nodes: BranchCanvasNode[]; edges: BranchCanvasEdge[] } {
  const { draw_id, status, feedback, requested_count, completed_count, running_count, failed_count, winner_run_id, cards } = session;

  const nodes: BranchCanvasNode[] = [];
  const edges: BranchCanvasEdge[] = [];

  // ── draw_session node ──────────────────────────────────────────────────────
  const dsNodeId = `draw:${draw_id}`;
  const rawLabel = feedback ? truncate(feedback, 24) : `Draw ${draw_id.slice(-4)}`;
  nodes.push({
    id: dsNodeId,
    type: "draw_session",
    position: { x: 0, y: 0 },
    data: {
      type: "draw_session",
      draw_id,
      status,
      label: rawLabel,
      requested_count,
      completed_count,
      running_count,
      failed_count,
      winner_run_id: winner_run_id ?? null,
      card_count: cards.length,
    },
  });

  // ── draw_from edge: anchor → draw_session ──────────────────────────────────
  edges.push({
    id: `drawfrom:${draw_id}`,
    source: opts.anchorNodeId,
    target: dsNodeId,
    data: { relation: "draw_from" },
  });

  // Collapsed: stop here — no card nodes/edges.
  if (opts.collapsed) {
    return { nodes, edges };
  }

  // ── Collect run_ids in this session for replacement-link guard ────────────
  const sessionRunIds = new Set(cards.map((c) => c.run_id));

  // ── draw_card nodes + draw_card edges ─────────────────────────────────────
  for (const card of cards) {
    const dcNodeId = `drawcard:${card.run_id}`;

    nodes.push({
      id: dcNodeId,
      type: "draw_card",
      position: { x: 0, y: 0 },
      data: {
        type: "draw_card",
        draw_id,
        run_id: card.run_id,
        card_id: card.card_id,
        group_id: card.group_id ?? null,
        index: card.index,
        status: card.status,
        label: card.label,
        strategy_label: card.strategy_label ?? null,
        favorite: !!card.favorite,
        eliminated: !!card.eliminated,
        final_score: card.final_score ?? null,
        replacement_of_run_id: card.replacement_of_run_id ?? null,
        can_use_for_fusion: !!card.can_use_for_fusion,
        is_winner: winner_run_id === card.run_id,
      },
    });

    edges.push({
      id: `drawcard:${card.run_id}`,
      source: dsNodeId,
      target: dcNodeId,
      data: { relation: "draw_card" },
    });

    // ── replacement_of edge (only when target is also in this session) ───────
    if (card.replacement_of_run_id && sessionRunIds.has(card.replacement_of_run_id)) {
      edges.push({
        id: `repl:${card.run_id}`,
        source: dcNodeId,
        target: `drawcard:${card.replacement_of_run_id}`,
        data: { relation: "replacement_of" },
      });
    }
  }

  return { nodes, edges };
}

// ─── Region Constraint Model ──────────────────────────────────────────────────

export interface BuildRegionConstraintOptions {
  anchorNodeId: string;  // canvas node id the "constraint_applies" edge originates from
}

/**
 * Pure function: converts an array of RegionConstraints into React-Flow canvas
 * nodes/edges. Each region emits one region_constraint node + one constraint_applies
 * edge from anchorNodeId → region node. Deterministic (array order). No Date/random.
 */
export function buildRegionConstraintModel(
  regions: RegionConstraint[],
  opts: BuildRegionConstraintOptions,
): { nodes: BranchCanvasNode[]; edges: BranchCanvasEdge[] } {
  if (regions.length === 0) return { nodes: [], edges: [] };

  const nodes: BranchCanvasNode[] = [];
  const edges: BranchCanvasEdge[] = [];

  for (const region of regions) {
    const nodeId = `region:${region.id}`;

    nodes.push({
      id: nodeId,
      type: "region_constraint",
      position: { x: 0, y: 0 },
      data: {
        type: "region_constraint",
        region_id: region.id,
        label: region.label,
        mode: region.mode,
        instruction: region.instruction,
        strength: region.strength,
        geometry: region.geometry,
      },
    });

    edges.push({
      id: `applies:${region.id}`,
      source: opts.anchorNodeId,
      target: nodeId,
      data: { relation: "constraint_applies" },
    });
  }

  return { nodes, edges };
}

// ─── Fusion Plan Model ────────────────────────────────────────────────────────

export interface BuildFusionModelOpts {
  /** Canvas node id of the base run (e.g. `run:${base_run_id}`). */
  baseAnchorNodeId?: string;
  /** Map from source_run_id → canvas node id for each fusion source. */
  sourceAnchorNodeIds?: Record<string, string>;
  /** Canvas node id of the output run (e.g. `run:${output_run_id}`). */
  outputAnchorNodeId?: string;
}

/**
 * Pure function: converts a FusionStatus into React-Flow canvas nodes/edges
 * representing the fusion plan. Emits exactly one `fusion_plan` node and up to
 * (1 + N + 1) edges depending on which anchor node ids are provided.
 * All nodes get position {x:0, y:0} — layout assigns real coordinates.
 * Deterministic (source_run_ids array order). No Date/Math.random.
 */
export function buildFusionModel(
  fusion: FusionStatus,
  opts: BuildFusionModelOpts,
): { nodes: BranchCanvasNode[]; edges: BranchCanvasEdge[] } {
  const { fusion_id, status, base_run_id, source_run_ids, output_run_id, regions } = fusion;
  const { baseAnchorNodeId, sourceAnchorNodeIds, outputAnchorNodeId } = opts;

  const nodes: BranchCanvasNode[] = [];
  const edges: BranchCanvasEdge[] = [];

  const fusionNodeId = `fusion:${fusion_id}`;

  // ── Fusion plan node ───────────────────────────────────────────────────────
  nodes.push({
    id: fusionNodeId,
    type: "fusion_plan",
    position: { x: 0, y: 0 },
    data: {
      type: "fusion_plan",
      fusion_id,
      status,
      base_run_id,
      source_run_ids,
      output_run_id: output_run_id ?? null,
      region_count: regions.length,
      label: `Fusion ${fusion_id.slice(-4)}`,
    },
  });

  // ── fusion_base edge: baseAnchorNodeId → fusionNodeId ─────────────────────
  if (baseAnchorNodeId !== undefined) {
    edges.push({
      id: `fbase:${fusion_id}`,
      source: baseAnchorNodeId,
      target: fusionNodeId,
      data: { relation: "fusion_base" },
    });
  }

  // ── fusion_source edges: sourceAnchorNodeId → fusionNodeId ────────────────
  for (const runId of source_run_ids) {
    const anchorId = sourceAnchorNodeIds?.[runId];
    if (anchorId !== undefined) {
      edges.push({
        id: `fsrc:${fusion_id}:${runId}`,
        source: anchorId,
        target: fusionNodeId,
        data: { relation: "fusion_source" },
      });
    }
  }

  // ── fusion_output edge: fusionNodeId → outputAnchorNodeId ─────────────────
  if (outputAnchorNodeId !== undefined && output_run_id != null) {
    edges.push({
      id: `fout:${fusion_id}`,
      source: fusionNodeId,
      target: outputAnchorNodeId,
      data: { relation: "fusion_output" },
    });
  }

  return { nodes, edges };
}
