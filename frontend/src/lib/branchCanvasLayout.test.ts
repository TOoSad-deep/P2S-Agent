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

// ─── V3 variant node fixtures ────────────────────────────────────────────────

function makeVariantGroupNode(
  groupId: string,
  parentRunId: string | null,
  overrides: Partial<BranchCanvasNode> = {},
): BranchCanvasNode {
  return {
    id: `vg:${groupId}`,
    type: "variant_group",
    position: { x: 0, y: 0 },
    data: {
      type: "variant_group",
      label: "Variants (2)",
      group_id: groupId,
      parent_run_id: parentRunId,
      source_checkpoint_id: null,
      status: "completed",
      collapsed: false,
    },
    ...overrides,
  };
}

function makeVariantRunNode(
  runId: string,
  groupId: string,
  variantIndex: number,
  overrides: Partial<BranchCanvasNode> = {},
): BranchCanvasNode {
  return {
    id: `run:${runId}`,
    type: "variant_run",
    position: { x: 0, y: 0 },
    data: {
      type: "variant_run",
      label: runId,
      run_id: runId,
      variant_group_id: groupId,
      variant_index: variantIndex,
      variant_label: null,
      status: "completed",
      score: null,
      favorite: false,
    },
    ...overrides,
  };
}

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

  // ─── V3 variant group layout tests ────────────────────────────────────────

  // ── V3-L1. variant_group node placed one COLUMN_WIDTH right of parent run ──
  it("places a variant_group node one COLUMN_WIDTH right of its parent run, same y", () => {
    const ROOT = "root-vg-001";
    const GROUP_ID = "grp-layout-001";
    const VAR_A = "var-a-001";
    const VAR_B = "var-b-001";

    const nodes: BranchCanvasNode[] = [
      makeInputNode(),
      makeRunNode(ROOT),
      makeVariantGroupNode(GROUP_ID, ROOT),
      makeVariantRunNode(VAR_A, GROUP_ID, 0),
      makeVariantRunNode(VAR_B, GROUP_ID, 1),
    ];

    const result = layoutBranchCanvas(nodes, NO_EDGES, NO_OVERRIDES);

    const rootNode = result.find((n) => n.id === `run:${ROOT}`)!;
    const vgNode = result.find((n) => n.id === `vg:${GROUP_ID}`)!;

    expect(vgNode).toBeDefined();
    // One column right of parent run
    expect(vgNode.position.x).toBe(rootNode.position.x + COLUMN_WIDTH);
    // Same y as parent run
    expect(vgNode.position.y).toBe(rootNode.position.y);
  });

  // ── V3-L2. variant_run nodes stacked right of group, separated by ROW_HEIGHT
  it("places variant_run nodes one COLUMN_WIDTH right of group, stacked by ROW_HEIGHT", () => {
    const ROOT = "root-vg-002";
    const GROUP_ID = "grp-layout-002";
    const VAR_A = "var-a-002";
    const VAR_B = "var-b-002";

    const nodes: BranchCanvasNode[] = [
      makeInputNode(),
      makeRunNode(ROOT),
      makeVariantGroupNode(GROUP_ID, ROOT),
      makeVariantRunNode(VAR_A, GROUP_ID, 0),
      makeVariantRunNode(VAR_B, GROUP_ID, 1),
    ];

    const result = layoutBranchCanvas(nodes, NO_EDGES, NO_OVERRIDES);

    const vgNode = result.find((n) => n.id === `vg:${GROUP_ID}`)!;
    const vrA = result.find((n) => n.id === `run:${VAR_A}`)!;
    const vrB = result.find((n) => n.id === `run:${VAR_B}`)!;

    // Both variant runs are COLUMN_WIDTH right of the group
    expect(vrA.position.x).toBe(vgNode.position.x + COLUMN_WIDTH);
    expect(vrB.position.x).toBe(vgNode.position.x + COLUMN_WIDTH);

    // First at group.y + 0 * ROW_HEIGHT, second at group.y + 1 * ROW_HEIGHT
    expect(vrA.position.y).toBe(vgNode.position.y + 0 * ROW_HEIGHT);
    expect(vrB.position.y).toBe(vgNode.position.y + 1 * ROW_HEIGHT);

    // Strictly increasing y
    expect(vrB.position.y).toBeGreaterThan(vrA.position.y);
  });

  // ── V3-L3. layoutOverrides win for a variant_group node ───────────────────
  it("applies layoutOverrides to variant_group nodes, overriding the computed position", () => {
    const ROOT = "root-vg-003";
    const GROUP_ID = "grp-layout-003";

    const nodes: BranchCanvasNode[] = [
      makeInputNode(),
      makeRunNode(ROOT),
      makeVariantGroupNode(GROUP_ID, ROOT),
    ];

    const overrides = { [`vg:${GROUP_ID}`]: { x: 1234, y: 5678 } };
    const result = layoutBranchCanvas(nodes, NO_EDGES, overrides);

    const vgNode = result.find((n) => n.id === `vg:${GROUP_ID}`)!;
    expect(vgNode.position).toEqual({ x: 1234, y: 5678 });
  });

  // ── V3-L4. layoutOverrides win for a variant_run node ─────────────────────
  it("applies layoutOverrides to variant_run nodes", () => {
    const ROOT = "root-vg-004";
    const GROUP_ID = "grp-layout-004";
    const VAR_A = "var-a-004";

    const nodes: BranchCanvasNode[] = [
      makeInputNode(),
      makeRunNode(ROOT),
      makeVariantGroupNode(GROUP_ID, ROOT),
      makeVariantRunNode(VAR_A, GROUP_ID, 0),
    ];

    const overrides = { [`run:${VAR_A}`]: { x: 999, y: 888 } };
    const result = layoutBranchCanvas(nodes, NO_EDGES, overrides);

    const vrA = result.find((n) => n.id === `run:${VAR_A}`)!;
    expect(vrA.position).toEqual({ x: 999, y: 888 });
  });

  // ── V3-L5. Run placement is not disturbed by variant nodes ────────────────
  it("does not disturb regular run placement when variant nodes are present", () => {
    const ROOT = "root-vg-005";
    const CHILD = "child-vg-005";
    const GROUP_ID = "grp-layout-005";
    const VAR_A = "var-a-005";

    // Build with only regular nodes first to get expected positions
    const regularNodes: BranchCanvasNode[] = [
      makeInputNode(),
      makeRunNode(ROOT),
      makeRunNode(CHILD, ROOT),
    ];
    const regularResult = layoutBranchCanvas(regularNodes, NO_EDGES, NO_OVERRIDES);
    const expectedRoot = regularResult.find((n) => n.id === `run:${ROOT}`)!;
    const expectedChild = regularResult.find((n) => n.id === `run:${CHILD}`)!;

    // Now add variant nodes
    const mixedNodes: BranchCanvasNode[] = [
      makeInputNode(),
      makeRunNode(ROOT),
      makeRunNode(CHILD, ROOT),
      makeVariantGroupNode(GROUP_ID, ROOT),
      makeVariantRunNode(VAR_A, GROUP_ID, 0),
    ];
    const mixedResult = layoutBranchCanvas(mixedNodes, NO_EDGES, NO_OVERRIDES);

    const rootPos = mixedResult.find((n) => n.id === `run:${ROOT}`)!;
    const childPos = mixedResult.find((n) => n.id === `run:${CHILD}`)!;

    expect(rootPos.position).toEqual(expectedRoot.position);
    expect(childPos.position).toEqual(expectedChild.position);
  });

  // ─── V3.5 draw post-pass layout tests ─────────────────────────────────────

  function makeDrawSessionNode(
    drawId: string,
    overrides: Partial<BranchCanvasNode> = {},
  ): BranchCanvasNode {
    return {
      id: `draw:${drawId}`,
      type: "draw_session",
      position: { x: 0, y: 0 },
      data: {
        type: "draw_session",
        label: `Draw ${drawId}`,
        draw_id: drawId,
        status: "running",
        requested_count: 4,
        completed_count: 2,
        running_count: 1,
        failed_count: 0,
        winner_run_id: null,
        card_count: 4,
      },
      ...overrides,
    };
  }

  function makeDrawCardNode(
    runId: string,
    drawId: string,
    index: number,
    overrides: Partial<BranchCanvasNode> = {},
  ): BranchCanvasNode {
    return {
      id: `drawcard:${runId}`,
      type: "draw_card",
      position: { x: 0, y: 0 },
      data: {
        type: "draw_card",
        label: runId,
        draw_id: drawId,
        run_id: runId,
        card_id: `card-${runId}`,
        group_id: null,
        index,
        status: "completed",
        favorite: false,
        eliminated: false,
        final_score: null,
        replacement_of_run_id: null,
        can_use_for_fusion: false,
        is_winner: false,
      },
      ...overrides,
    };
  }

  function makeDrawFromEdge(
    anchorNodeId: string,
    drawId: string,
  ): BranchCanvasEdge {
    return {
      id: `drawfrom:${drawId}`,
      source: anchorNodeId,
      target: `draw:${drawId}`,
      data: { relation: "draw_from" },
    };
  }

  // ── D1. draw_session placed one COLUMN_WIDTH right of its anchor ──────────
  it("places a draw_session node one COLUMN_WIDTH right of its anchor at the anchor's y", () => {
    const ROOT = "root-draw-001";
    const DRAW_ID = "draw-001";

    const nodes: BranchCanvasNode[] = [
      makeInputNode(),
      makeRunNode(ROOT),
      makeDrawSessionNode(DRAW_ID),
    ];

    const edges: BranchCanvasEdge[] = [
      makeDrawFromEdge(`run:${ROOT}`, DRAW_ID),
    ];

    const result = layoutBranchCanvas(nodes, edges, NO_OVERRIDES);

    const rootNode = result.find((n) => n.id === `run:${ROOT}`)!;
    const dsNode = result.find((n) => n.id === `draw:${DRAW_ID}`)!;

    expect(dsNode).toBeDefined();
    expect(dsNode.position.x).toBe(rootNode.position.x + COLUMN_WIDTH);
    expect(dsNode.position.y).toBe(rootNode.position.y);
  });

  // ── D2. draw_card nodes placed right of session, stacked by index ─────────
  it("places two draw_card nodes one COLUMN_WIDTH right of the session, stacked ROW_HEIGHT apart by index", () => {
    const ROOT = "root-draw-002";
    const DRAW_ID = "draw-002";
    const CARD_A = "card-run-a";
    const CARD_B = "card-run-b";

    const nodes: BranchCanvasNode[] = [
      makeInputNode(),
      makeRunNode(ROOT),
      makeDrawSessionNode(DRAW_ID),
      makeDrawCardNode(CARD_A, DRAW_ID, 0),
      makeDrawCardNode(CARD_B, DRAW_ID, 1),
    ];

    const edges: BranchCanvasEdge[] = [
      makeDrawFromEdge(`run:${ROOT}`, DRAW_ID),
    ];

    const result = layoutBranchCanvas(nodes, edges, NO_OVERRIDES);

    const dsNode = result.find((n) => n.id === `draw:${DRAW_ID}`)!;
    const cardA = result.find((n) => n.id === `drawcard:${CARD_A}`)!;
    const cardB = result.find((n) => n.id === `drawcard:${CARD_B}`)!;

    expect(cardA).toBeDefined();
    expect(cardB).toBeDefined();

    // Both cards are COLUMN_WIDTH right of the session
    expect(cardA.position.x).toBe(dsNode.position.x + COLUMN_WIDTH);
    expect(cardB.position.x).toBe(dsNode.position.x + COLUMN_WIDTH);

    // Stacked by index: index 0 first, index 1 ROW_HEIGHT below
    expect(cardA.position.y).toBe(dsNode.position.y + 0 * ROW_HEIGHT);
    expect(cardB.position.y).toBe(dsNode.position.y + 1 * ROW_HEIGHT);

    // Strictly increasing y
    expect(cardB.position.y).toBeGreaterThan(cardA.position.y);
  });

  // ── D3. layoutOverrides wins over draw_card computed position ─────────────
  it("applies layoutOverrides to a draw_card node, overriding the computed position", () => {
    const ROOT = "root-draw-003";
    const DRAW_ID = "draw-003";
    const CARD_A = "card-run-003a";

    const nodes: BranchCanvasNode[] = [
      makeInputNode(),
      makeRunNode(ROOT),
      makeDrawSessionNode(DRAW_ID),
      makeDrawCardNode(CARD_A, DRAW_ID, 0),
    ];

    const edges: BranchCanvasEdge[] = [
      makeDrawFromEdge(`run:${ROOT}`, DRAW_ID),
    ];

    const overrides = { [`drawcard:${CARD_A}`]: { x: 7777, y: 8888 } };
    const result = layoutBranchCanvas(nodes, edges, overrides);

    const cardA = result.find((n) => n.id === `drawcard:${CARD_A}`)!;
    expect(cardA.position).toEqual({ x: 7777, y: 8888 });
  });

  // ── D4. draw_session with missing anchor keeps incoming position (no crash) ─
  it("keeps the incoming position of a draw_session whose anchor is absent (no crash)", () => {
    const DRAW_ID = "draw-004";

    const nodes: BranchCanvasNode[] = [
      makeInputNode(),
      makeDrawSessionNode(DRAW_ID, { position: { x: 42, y: 99 } }),
    ];

    // No draw_from edge → anchor unknown
    const result = layoutBranchCanvas(nodes, NO_EDGES, NO_OVERRIDES);

    const dsNode = result.find((n) => n.id === `draw:${DRAW_ID}`)!;
    expect(dsNode).toBeDefined();
    // Should not crash; keeps incoming position
    expect(dsNode.position).toEqual({ x: 42, y: 99 });
  });

  // ── D5. Two sessions sharing one anchor don't get identical y ─────────────
  it("offsets two draw_sessions sharing the same anchor so they don't overlap", () => {
    const ROOT = "root-draw-005";
    const DRAW_A = "draw-005a";
    const DRAW_B = "draw-005b";

    const nodes: BranchCanvasNode[] = [
      makeInputNode(),
      makeRunNode(ROOT),
      makeDrawSessionNode(DRAW_A),
      makeDrawSessionNode(DRAW_B),
    ];

    const edges: BranchCanvasEdge[] = [
      makeDrawFromEdge(`run:${ROOT}`, DRAW_A),
      makeDrawFromEdge(`run:${ROOT}`, DRAW_B),
    ];

    const result = layoutBranchCanvas(nodes, edges, NO_OVERRIDES);

    const dsA = result.find((n) => n.id === `draw:${DRAW_A}`)!;
    const dsB = result.find((n) => n.id === `draw:${DRAW_B}`)!;

    expect(dsA).toBeDefined();
    expect(dsB).toBeDefined();

    // Both are COLUMN_WIDTH right of their anchor
    const runNode = result.find((n) => n.id === `run:${ROOT}`)!;
    expect(dsA.position.x).toBe(runNode.position.x + COLUMN_WIDTH);
    expect(dsB.position.x).toBe(runNode.position.x + COLUMN_WIDTH);

    // They must NOT be at the same y
    expect(dsA.position.y).not.toBe(dsB.position.y);
  });
});
