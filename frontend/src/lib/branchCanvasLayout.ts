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
