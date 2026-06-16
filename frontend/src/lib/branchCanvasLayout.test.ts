import { describe, it, expect } from "vitest";
import {
  layoutBranchCanvas,
  COLUMN_WIDTH,
  ROW_HEIGHT,
  RUN_GAP,
} from "./branchCanvasLayout";
import type { BranchCanvasNode, BranchCanvasEdge } from "./branchCanvasModel";

// ─── Fixtures ────────────────────────────────────────────────────────────────

function makeRunNode(
  runId: string,
  parentRunId: string | null = null,
  overrides: Partial<BranchCanvasNode> = {},
): BranchCanvasNode {
  return {
    id: `run:${runId}`,
    type: "run",
    position: { x: 0, y: 0 },
    data: {
      type: "run",
      label: runId,
      run_id: runId,
      parent_run_id: parentRunId,
    },
    ...overrides,
  };
}

function makeCpNode(
  runId: string,
  cpId: string,
  overrides: Partial<BranchCanvasNode> = {},
): BranchCanvasNode {
  return {
    id: `cp:${runId}:${cpId}`,
    type: "checkpoint",
    position: { x: 0, y: 0 },
    data: {
      type: "checkpoint",
      label: cpId,
      run_id: runId,
      checkpoint_id: cpId,
    },
    ...overrides,
  };
}

function makeInputNode(): BranchCanvasNode {
  return {
    id: "input",
    type: "input",
    position: { x: 0, y: 0 },
    data: { type: "input", label: "Input PNG" },
  };
}

const NO_EDGES: BranchCanvasEdge[] = [];
const NO_OVERRIDES: Record<string, { x: number; y: number }> = {};

// ─── Tests ───────────────────────────────────────────────────────────────────

describe("layoutBranchCanvas", () => {
  // ── 1. Child run is one column to the right of its parent ─────────────────
  it("places child run one COLUMN_WIDTH to the right of root run", () => {
    const ROOT = "root-001";
    const CHILD = "child-001";

    const nodes: BranchCanvasNode[] = [
      makeInputNode(),
      makeRunNode(ROOT),           // root
      makeRunNode(CHILD, ROOT),    // child of root
    ];

    const result = layoutBranchCanvas(nodes, NO_EDGES, NO_OVERRIDES);

    const rootNode = result.find((n) => n.id === `run:${ROOT}`)!;
    const childNode = result.find((n) => n.id === `run:${CHILD}`)!;

    expect(rootNode).toBeDefined();
    expect(childNode).toBeDefined();

    // Root is at depth 0 → x = 1 * COLUMN_WIDTH
    expect(rootNode.position.x).toBe(COLUMN_WIDTH);
    // Child is at depth 1 → x = 2 * COLUMN_WIDTH
    expect(childNode.position.x).toBe(rootNode.position.x + COLUMN_WIDTH);
  });

  // ── 2. Checkpoints share their run's x and stack below ────────────────────
  it("places checkpoints at same x as their run, stacked below with increasing y", () => {
    const ROOT = "root-002";
    const CP1 = "cp-a";
    const CP2 = "cp-b";

    const nodes: BranchCanvasNode[] = [
      makeInputNode(),
      makeRunNode(ROOT),
      makeCpNode(ROOT, CP1),
      makeCpNode(ROOT, CP2),
    ];

    const result = layoutBranchCanvas(nodes, NO_EDGES, NO_OVERRIDES);

    const runNode = result.find((n) => n.id === `run:${ROOT}`)!;
    const cp1Node = result.find((n) => n.id === `cp:${ROOT}:${CP1}`)!;
    const cp2Node = result.find((n) => n.id === `cp:${ROOT}:${CP2}`)!;

    // Checkpoints share the run's x column
    expect(cp1Node.position.x).toBe(runNode.position.x);
    expect(cp2Node.position.x).toBe(runNode.position.x);

    // First cp is directly below the run node
    expect(cp1Node.position.y).toBe(runNode.position.y + ROW_HEIGHT);
    // Second cp is below the first
    expect(cp2Node.position.y).toBe(runNode.position.y + 2 * ROW_HEIGHT);
    // Strictly increasing y
    expect(cp1Node.position.y).toBeLessThan(cp2Node.position.y);
  });

  // ── 3. Input node at x=0, aligned with root run's y ──────────────────────
  it("places the input node at x=0, same y as the root run", () => {
    const ROOT = "root-003";
    const nodes: BranchCanvasNode[] = [
      makeInputNode(),
      makeRunNode(ROOT),
    ];

    const result = layoutBranchCanvas(nodes, NO_EDGES, NO_OVERRIDES);

    const inputNode = result.find((n) => n.id === "input")!;
    const rootNode = result.find((n) => n.id === `run:${ROOT}`)!;

    expect(inputNode.position.x).toBe(0);
    expect(inputNode.position.y).toBe(rootNode.position.y);
  });

  // ── 4. layoutOverrides win — exact position override ─────────────────────
  it("applies layoutOverrides exactly, discarding auto position for that node", () => {
    const ROOT = "root-004";
    const nodes: BranchCanvasNode[] = [
      makeInputNode(),
      makeRunNode(ROOT),
    ];

    const overrides = { [`run:${ROOT}`]: { x: 999, y: 888 } };
    const result = layoutBranchCanvas(nodes, NO_EDGES, overrides);

    const rootNode = result.find((n) => n.id === `run:${ROOT}`)!;
    expect(rootNode.position).toEqual({ x: 999, y: 888 });
  });

  it("applies layoutOverrides to the input node", () => {
    const ROOT = "root-004b";
    const nodes: BranchCanvasNode[] = [
      makeInputNode(),
      makeRunNode(ROOT),
    ];

    const overrides = { input: { x: 50, y: 75 } };
    const result = layoutBranchCanvas(nodes, NO_EDGES, overrides);

    const inputNode = result.find((n) => n.id === "input")!;
    expect(inputNode.position).toEqual({ x: 50, y: 75 });
  });

  // ── 5. Determinism ────────────────────────────────────────────────────────
  it("is deterministic: two calls with identical inputs return deeply-equal arrays", () => {
    const ROOT = "root-005";
    const CHILD = "child-005";
    const nodes: BranchCanvasNode[] = [
      makeInputNode(),
      makeRunNode(ROOT),
      makeCpNode(ROOT, "cp-x"),
      makeCpNode(ROOT, "cp-y"),
      makeRunNode(CHILD, ROOT),
    ];

    const a = layoutBranchCanvas(nodes, NO_EDGES, NO_OVERRIDES);
    const b = layoutBranchCanvas(nodes, NO_EDGES, NO_OVERRIDES);
    expect(a).toEqual(b);
  });

  // ── 6. No two RUN nodes share the same (x, y) ─────────────────────────────
  it("places three run nodes in a tree without x,y collisions", () => {
    const ROOT = "root-006";
    const C1 = "child-006a";
    const C2 = "child-006b";

    // Tree: ROOT → [C1, C2]
    const nodes: BranchCanvasNode[] = [
      makeInputNode(),
      makeRunNode(ROOT),
      makeRunNode(C1, ROOT),
      makeRunNode(C2, ROOT),
    ];

    const result = layoutBranchCanvas(nodes, NO_EDGES, NO_OVERRIDES);

    const runPositions = result
      .filter((n) => (n.data as { type: string }).type === "run")
      .map((n) => `${n.position.x},${n.position.y}`);

    const uniquePositions = new Set(runPositions);
    expect(uniquePositions.size).toBe(runPositions.length);
  });

  // ── 7. Input mutation guard ───────────────────────────────────────────────
  it("does not mutate input nodes — originals retain their original positions", () => {
    const ROOT = "root-007";
    const original: BranchCanvasNode[] = [
      makeInputNode(),
      makeRunNode(ROOT),
      makeCpNode(ROOT, "cp-z"),
    ];

    // Capture original positions
    const before = original.map((n) => ({ id: n.id, pos: { ...n.position } }));

    layoutBranchCanvas(original, NO_EDGES, NO_OVERRIDES);

    // Originals must be unchanged
    for (const b of before) {
      const orig = original.find((n) => n.id === b.id)!;
      expect(orig.position).toEqual(b.pos);
    }
  });

  // ── 8. Returned nodes are new objects ────────────────────────────────────
  it("returns new node objects — not the same references as inputs", () => {
    const ROOT = "root-008";
    const nodes: BranchCanvasNode[] = [
      makeInputNode(),
      makeRunNode(ROOT),
    ];

    const result = layoutBranchCanvas(nodes, NO_EDGES, NO_OVERRIDES);

    for (const r of result) {
      const original = nodes.find((n) => n.id === r.id);
      if (original) {
        expect(r).not.toBe(original); // different reference
      }
    }
  });

  // ── 9. No-run-nodes edge case — input at (0,0) ─────────────────────────
  it("places input at (0,0) when there are no run nodes", () => {
    const nodes: BranchCanvasNode[] = [makeInputNode()];
    const result = layoutBranchCanvas(nodes, NO_EDGES, NO_OVERRIDES);

    const inputNode = result.find((n) => n.id === "input")!;
    expect(inputNode.position).toEqual({ x: 0, y: 0 });
  });

  // ── 10. Root+child+checkpoints — child after root's band ──────────────────
  it("places child run after the root's full band (run + checkpoints + gap)", () => {
    const ROOT = "root-010";
    const CHILD = "child-010";
    const CP1 = "cp-1";
    const CP2 = "cp-2";

    const nodes: BranchCanvasNode[] = [
      makeInputNode(),
      makeRunNode(ROOT),
      makeCpNode(ROOT, CP1),
      makeCpNode(ROOT, CP2),
      makeRunNode(CHILD, ROOT),
    ];

    const result = layoutBranchCanvas(nodes, NO_EDGES, NO_OVERRIDES);

    const rootNode = result.find((n) => n.id === `run:${ROOT}`)!;
    const childNode = result.find((n) => n.id === `run:${CHILD}`)!;
    const cp2Node = result.find((n) => n.id === `cp:${ROOT}:${CP2}`)!;

    // The band = run + 2 checkpoints = 3 * ROW_HEIGHT; gap = RUN_GAP
    // But child is at depth 1, so it's in the next vertical band below root's band
    // root starts at y=0, band height = (1+2)*ROW_HEIGHT = 3*ROW_HEIGHT
    // child starts at y = rootY + band + RUN_GAP
    const expectedChildY = rootNode.position.y + (1 + 2) * ROW_HEIGHT + RUN_GAP;
    expect(childNode.position.y).toBe(expectedChildY);

    // Child's column: COLUMN_WIDTH to the right of root
    expect(childNode.position.x).toBe(rootNode.position.x + COLUMN_WIDTH);

    // Ensure child is below all root checkpoints
    expect(childNode.position.y).toBeGreaterThan(cp2Node.position.y);
  });

  // ── 11. Sibling runs don't overlap ────────────────────────────────────────
  it("places two sibling runs (same parent) at different y positions", () => {
    const ROOT = "root-011";
    const C1 = "child-011a";
    const C2 = "child-011b";
    const CP_C1 = "cp-c1";

    // C1 has a checkpoint; C2 comes after C1 in DFS
    const nodes: BranchCanvasNode[] = [
      makeInputNode(),
      makeRunNode(ROOT),
      makeRunNode(C1, ROOT),
      makeCpNode(C1, CP_C1),
      makeRunNode(C2, ROOT),
    ];

    const result = layoutBranchCanvas(nodes, NO_EDGES, NO_OVERRIDES);

    const c1Node = result.find((n) => n.id === `run:${C1}`)!;
    const c2Node = result.find((n) => n.id === `run:${C2}`)!;

    // Same x (same depth)
    expect(c1Node.position.x).toBe(c2Node.position.x);
    // Different y — no overlap
    expect(c1Node.position.y).not.toBe(c2Node.position.y);
    // C2 is below C1 (in DFS order: C1 visited first, then C2)
    expect(c2Node.position.y).toBeGreaterThan(c1Node.position.y);
  });

  // ── 12. Unknown node type keeps incoming position ─────────────────────────
  it("preserves existing position for unknown-type nodes", () => {
    const ROOT = "root-012";
    const unknownNode: BranchCanvasNode = {
      id: "mystery-node",
      type: "unknown_type" as never,
      position: { x: 42, y: 100 },
      data: { type: "preference" as never, label: "??" },
    };

    const nodes: BranchCanvasNode[] = [
      makeInputNode(),
      makeRunNode(ROOT),
      unknownNode,
    ];

    const result = layoutBranchCanvas(nodes, NO_EDGES, NO_OVERRIDES);

    const mystery = result.find((n) => n.id === "mystery-node")!;
    expect(mystery).toBeDefined();
    expect(mystery.position).toEqual({ x: 42, y: 100 });
  });

  // ── 13. Orphan run (unknown parent_run_id) treated as root ────────────────
  it("places an orphan run (unknown parent_run_id) as a root without crashing", () => {
    const nodes: BranchCanvasNode[] = [
      makeInputNode(),
      makeRunNode("orphan", "does-not-exist"),
    ];
    expect(() => layoutBranchCanvas(nodes, NO_EDGES, NO_OVERRIDES)).not.toThrow();
    const result = layoutBranchCanvas(nodes, NO_EDGES, NO_OVERRIDES);
    const orphan = result.find((n) => n.id === "run:orphan")!;
    expect(orphan.position.x).toBe(COLUMN_WIDTH); // treated as a depth-0 root
  });

  // ── 14. No infinite-loop on a parent cycle (A.parent=B, B.parent=A) ───────
  it("does not infinite-loop on a parent cycle (A.parent=B, B.parent=A)", () => {
    const nodes: BranchCanvasNode[] = [
      makeInputNode(),
      makeRunNode("A", "B"),
      makeRunNode("B", "A"),
    ];
    expect(() => layoutBranchCanvas(nodes, NO_EDGES, NO_OVERRIDES)).not.toThrow();
  });
});
