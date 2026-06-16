import type { BranchCanvasNode, BranchCanvasEdge } from "./branchCanvasModel";

// ─── Layout constants ────────────────────────────────────────────────────────

export const COLUMN_WIDTH = 280;
export const ROW_HEIGHT = 90;
export const RUN_GAP = 40;

// ─── Internal types ───────────────────────────────────────────────────────────

interface RunEntry {
  node: BranchCanvasNode;
  runId: string;
  children: RunEntry[];
}

// ─── layoutBranchCanvas ──────────────────────────────────────────────────────

/**
 * Pure deterministic layered layout for BranchCanvas nodes.
 *
 * Algorithm:
 * - Partition nodes into: one input node, run nodes, checkpoint nodes grouped by run_id.
 * - Build run tree from parent_run_id links (root = null/undefined parent).
 * - DFS traverse the run tree with a shared nextY cursor:
 *   - Place run at {x: (depth+1)*COLUMN_WIDTH, y: nextY}.
 *   - Stack its checkpoints directly below at the same x, each ROW_HEIGHT apart.
 *   - Advance nextY by band height + RUN_GAP, then recurse into children.
 * - Place the input node at {x:0, y: rootRun.y} (or {0,0} if no runs).
 * - Apply layoutOverrides last: any node id in overrides gets that exact position.
 * - Unknown-type nodes keep their incoming position.
 * - Inputs are not mutated; every returned node is a shallow clone with a new position object.
 * - `_edges` is accepted but currently unused; reserved for future constraint-aware layout.
 * - Checkpoint nodes whose run_id is absent from the node list keep their incoming position (typically {0,0}).
 */
export function layoutBranchCanvas(
  nodes: BranchCanvasNode[],
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  _edges: BranchCanvasEdge[],
  layoutOverrides: Record<string, { x: number; y: number }>,
): BranchCanvasNode[] {
  // ── 1. Partition nodes ───────────────────────────────────────────────────
  let inputNode: BranchCanvasNode | null = null;
  const runNodes: BranchCanvasNode[] = [];
  // Map from run_id → ordered list of checkpoint nodes (preserving input order)
  const cpByRunId = new Map<string, BranchCanvasNode[]>();
  const variantGroupNodes: BranchCanvasNode[] = [];
  const variantRunNodes: BranchCanvasNode[] = [];

  for (const node of nodes) {
    const nodeType = (node.data as { type?: string }).type;
    if (nodeType === "input") {
      inputNode = node;
    } else if (nodeType === "run") {
      runNodes.push(node);
    } else if (nodeType === "checkpoint") {
      const runId = (node.data as { run_id?: string }).run_id;
      if (runId !== undefined) {
        let list = cpByRunId.get(runId);
        if (!list) {
          list = [];
          cpByRunId.set(runId, list);
        }
        list.push(node);
      }
    } else if (nodeType === "variant_group") {
      variantGroupNodes.push(node);
    } else if (nodeType === "variant_run") {
      variantRunNodes.push(node);
    }
  }

  // ── 2. Build run tree ────────────────────────────────────────────────────
  const runEntryMap = new Map<string, RunEntry>();

  // First pass: create entries (preserving input array order for children)
  for (const rn of runNodes) {
    const runId = (rn.data as { run_id?: string }).run_id ?? rn.id;
    runEntryMap.set(runId, { node: rn, runId, children: [] });
  }

  // Second pass: link children to parents, preserving input array order
  const roots: RunEntry[] = [];
  for (const rn of runNodes) {
    const runId = (rn.data as { run_id?: string }).run_id ?? rn.id;
    const parentRunId = (rn.data as { parent_run_id?: string | null }).parent_run_id;
    const entry = runEntryMap.get(runId)!;

    if (parentRunId != null && runEntryMap.has(parentRunId)) {
      runEntryMap.get(parentRunId)!.children.push(entry);
    } else {
      roots.push(entry);
    }
  }

  // ── 3. DFS placement ─────────────────────────────────────────────────────
  // positions: id → {x, y}
  const computedPositions = new Map<string, { x: number; y: number }>();
  let nextY = 0;
  let firstRootY: number | null = null;

  function placeRun(entry: RunEntry, depth: number): void {
    const runX = (depth + 1) * COLUMN_WIDTH;
    const runY = nextY;

    if (firstRootY === null && depth === 0) {
      firstRootY = runY;
    }

    computedPositions.set(entry.node.id, { x: runX, y: runY });

    // Place checkpoints for this run
    const cpList = cpByRunId.get(entry.runId) ?? [];
    for (let i = 0; i < cpList.length; i++) {
      computedPositions.set(cpList[i].id, {
        x: runX,
        y: runY + (i + 1) * ROW_HEIGHT,
      });
    }

    // Advance the cursor past this run's band
    const bandHeight = (1 + cpList.length) * ROW_HEIGHT;
    nextY = runY + bandHeight + RUN_GAP;

    // Recurse into children
    for (const child of entry.children) {
      placeRun(child, depth + 1);
    }
  }

  for (const root of roots) {
    placeRun(root, 0);
  }

  // ── 4. Build output nodes ─────────────────────────────────────────────────
  // Process in original input array order.
  const resultMap = new Map<string, BranchCanvasNode>();

  // Place all known nodes
  for (const node of nodes) {
    const autoPos = computedPositions.get(node.id);
    const pos = autoPos ?? { ...node.position };
    resultMap.set(node.id, { ...node, position: { ...pos } });
  }

  // Apply input node's y alignment with first root run
  if (inputNode !== null) {
    const inputResult = resultMap.get(inputNode.id)!;
    const inputX = 0;
    const inputY = firstRootY ?? 0;
    resultMap.set(inputNode.id, {
      ...inputResult,
      position: { x: inputX, y: inputY },
    });
  }

  // ── 4b. Variant post-pass: place variant_group and variant_run nodes ───────
  // Group nodes: one COLUMN_WIDTH right of their parent run.
  for (const vgNode of variantGroupNodes) {
    const parentRunId = (vgNode.data as { parent_run_id?: string | null }).parent_run_id;
    const parentRunNodeId = parentRunId ? `run:${parentRunId}` : null;
    const parentPos = parentRunNodeId ? computedPositions.get(parentRunNodeId) : null;
    const gx = parentPos ? parentPos.x + COLUMN_WIDTH : 0;
    const gy = parentPos ? parentPos.y : 0;
    computedPositions.set(vgNode.id, { x: gx, y: gy });
    const existing = resultMap.get(vgNode.id);
    if (existing !== undefined) {
      resultMap.set(vgNode.id, { ...existing, position: { x: gx, y: gy } });
    }
  }

  // Variant run nodes: COLUMN_WIDTH right of their group, stacked by variant_index then id.
  // Build a stable ordering within each group.
  const variantRunsByGroup = new Map<string, BranchCanvasNode[]>();
  for (const vrNode of variantRunNodes) {
    const gid = (vrNode.data as { variant_group_id?: string | null }).variant_group_id;
    if (!gid) continue;
    let list = variantRunsByGroup.get(gid);
    if (!list) {
      list = [];
      variantRunsByGroup.set(gid, list);
    }
    list.push(vrNode);
  }

  for (const [gid, members] of variantRunsByGroup) {
    const groupNodeId = `vg:${gid}`;
    const groupPos = computedPositions.get(groupNodeId);

    // Sort by variant_index then id (stable)
    const sorted = [...members].sort((a, b) => {
      const ia = (a.data as { variant_index?: number | null }).variant_index ?? 0;
      const ib = (b.data as { variant_index?: number | null }).variant_index ?? 0;
      if (ia !== ib) return ia - ib;
      return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
    });

    for (let i = 0; i < sorted.length; i++) {
      const vrNode = sorted[i];
      const vx = groupPos ? groupPos.x + COLUMN_WIDTH : 0;
      const vy = groupPos ? groupPos.y + i * ROW_HEIGHT : 0;
      computedPositions.set(vrNode.id, { x: vx, y: vy });
      const existing = resultMap.get(vrNode.id);
      if (existing !== undefined) {
        resultMap.set(vrNode.id, { ...existing, position: { x: vx, y: vy } });
      }
    }
  }

  // ── 5. Apply layoutOverrides (wins over everything) ───────────────────────
  for (const [id, overridePos] of Object.entries(layoutOverrides)) {
    const existing = resultMap.get(id);
    if (existing !== undefined) {
      resultMap.set(id, { ...existing, position: { ...overridePos } });
    }
  }

  // ── 6. Return in original input array order ───────────────────────────────
  return nodes.map((n) => resultMap.get(n.id)!);
}
