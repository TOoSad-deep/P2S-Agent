import { describe, it, expect } from "vitest";
import {
  buildBranchCanvasModel,
  buildDrawSessionModel,
  buildRegionConstraintModel,
  buildFusionModel,
  MAX_EXPANDED_RUNS,
  MAX_VISIBLE_CHECKPOINTS_PER_RUN,
  type BuildBranchCanvasInput,
} from "./branchCanvasModel";
import type { BranchTreeNode, CheckpointTimelineEntry, DrawSessionStatus, DrawCardStatus, RegionConstraint, FusionStatus, FusionRegion } from "../hooks/usePngShader";

// ─── Fixtures ────────────────────────────────────────────────────────────────

function makeTreeNode(overrides: Partial<BranchTreeNode> & { run_id: string }): BranchTreeNode {
  return {
    root_run_id: overrides.run_id,
    parent_run_id: null,
    source_checkpoint_id: null,
    source_checkpoint_label: null,
    title: null,
    mode: null,
    feedback: null,
    status: "completed",
    final_score: null,
    created_at: null,
    completed_at: null,
    favorite: false,
    children: [],
    variant_group_id: null,
    variant_index: null,
    variant_label: null,
    draw_session_id: null,
    draw_card_index: null,
    replacement_of_run_id: null,
    ...overrides,
  };
}

function makeDrawCard(overrides: Partial<DrawCardStatus> & { card_id: string; run_id: string; index: number }): DrawCardStatus {
  return {
    group_id: null,
    status: "completed",
    label: `Card ${overrides.index}`,
    strategy_label: null,
    final_score: null,
    favorite: false,
    eliminated: false,
    replacement_of_run_id: null,
    can_use_for_fusion: false,
    ...overrides,
  };
}

function makeDrawSession(overrides: Partial<DrawSessionStatus> & { draw_id: string }): DrawSessionStatus {
  return {
    parent_run_id: "run-parent-0001",
    source_checkpoint_id: "candidate:selected",
    feedback: "Make it more vibrant",
    status: "completed",
    requested_count: 3,
    completed_count: 3,
    running_count: 0,
    failed_count: 0,
    winner_run_id: null,
    group_ids: [],
    cards: [],
    ...overrides,
  };
}

function makeTimeline(
  entries: Array<{ id: string; accepted?: boolean; score?: number; delta?: number }>,
  runId: string,
): CheckpointTimelineEntry[] {
  return entries.map((e) => ({
    id: e.id,
    run_id: runId,
    kind: "candidate" as const,
    label: `Label(${e.id})`,
    score: e.score ?? null,
    delta: e.delta ?? null,
    accepted: e.accepted ?? null,
    has_glsl: false,
  }));
}

function baseInput(overrides: Partial<BuildBranchCanvasInput> = {}): BuildBranchCanvasInput {
  return {
    activeRunId: null,
    branchTree: null,
    timelinesByRunId: {},
    statusesByRunId: {},
    collapsedRunIds: new Set(),
    favoriteRunIds: new Set(),
    ...overrides,
  };
}

const ROOT_ID = "run-root-0001";

// ─── Tests ───────────────────────────────────────────────────────────────────

describe("buildBranchCanvasModel", () => {
  // ── 1. null branchTree → empty ──────────────────────────────────────────
  it("returns empty nodes and edges when branchTree is null", () => {
    const out = buildBranchCanvasModel(baseInput());
    expect(out.nodes).toEqual([]);
    expect(out.edges).toEqual([]);
  });

  // ── 2. single root run, NOT active/expanded ──────────────────────────────
  it("emits input node + run node + input-> edge for a non-expanded root run", () => {
    const tree = makeTreeNode({ run_id: ROOT_ID });
    const out = buildBranchCanvasModel(
      baseInput({ branchTree: tree, activeRunId: null }),
    );

    const nodeIds = out.nodes.map((n) => n.id);
    expect(nodeIds).toContain("input");
    expect(nodeIds).toContain(`run:${ROOT_ID}`);

    // No checkpoint nodes
    expect(out.nodes.filter((n) => n.id.startsWith("cp:"))).toHaveLength(0);

    // Edge: input → root run
    const inputEdge = out.edges.find((e) => e.id === `input->${ROOT_ID}`);
    expect(inputEdge).toBeDefined();
    expect(inputEdge!.source).toBe("input");
    expect(inputEdge!.target).toBe(`run:${ROOT_ID}`);
    expect(inputEdge!.data?.relation).toBe("timeline_next");
  });

  // ── 3. active root run → checkpoint nodes chained by timeline_next ───────
  it("emits checkpoint nodes chained run→cp0→cp1 when root run is active", () => {
    const tree = makeTreeNode({ run_id: ROOT_ID });
    const timeline = makeTimeline(
      [{ id: "candidate:selected" }, { id: "cp2" }, { id: "final:selected" }],
      ROOT_ID,
    );
    const out = buildBranchCanvasModel(
      baseInput({
        branchTree: tree,
        activeRunId: ROOT_ID,
        timelinesByRunId: { [ROOT_ID]: timeline },
      }),
    );

    const cpNodes = out.nodes.filter((n) => n.id.startsWith("cp:"));
    expect(cpNodes).toHaveLength(3);

    // Chain: run node → first cp → second cp → third cp
    const chainEdge0 = out.edges.find((e) => e.id === `tl:${ROOT_ID}:0`);
    expect(chainEdge0).toBeDefined();
    expect(chainEdge0!.source).toBe(`run:${ROOT_ID}`);
    expect(chainEdge0!.target).toBe(`cp:${ROOT_ID}:candidate:selected`);
    expect(chainEdge0!.data?.relation).toBe("timeline_next");

    const chainEdge1 = out.edges.find((e) => e.id === `tl:${ROOT_ID}:1`);
    expect(chainEdge1).toBeDefined();
    expect(chainEdge1!.source).toBe(`cp:${ROOT_ID}:candidate:selected`);
    expect(chainEdge1!.target).toBe(`cp:${ROOT_ID}:cp2`);

    const chainEdge2 = out.edges.find((e) => e.id === `tl:${ROOT_ID}:2`);
    expect(chainEdge2).toBeDefined();
    expect(chainEdge2!.source).toBe(`cp:${ROOT_ID}:cp2`);
    expect(chainEdge2!.target).toBe(`cp:${ROOT_ID}:final:selected`);
  });

  // ── 4. branch_from edge source = cp: node when parent IS expanded ─────────
  it("uses cp: node as branch_from source when parent is expanded and source_cp is kept", () => {
    const CHILD_ID = "run-child-0002";
    const SOURCE_CP = "candidate:selected";

    const child = makeTreeNode({
      run_id: CHILD_ID,
      root_run_id: ROOT_ID,
      parent_run_id: ROOT_ID,
      source_checkpoint_id: SOURCE_CP,
    });
    const tree = makeTreeNode({ run_id: ROOT_ID, children: [child] });

    const parentTimeline = makeTimeline(
      [{ id: "candidate:selected" }, { id: "final:selected" }],
      ROOT_ID,
    );

    const out = buildBranchCanvasModel(
      baseInput({
        branchTree: tree,
        activeRunId: ROOT_ID,        // makes parent expanded
        timelinesByRunId: { [ROOT_ID]: parentTimeline },
      }),
    );

    const branchEdge = out.edges.find((e) => e.id === `branch:${CHILD_ID}`);
    expect(branchEdge).toBeDefined();
    expect(branchEdge!.source).toBe(`cp:${ROOT_ID}:${SOURCE_CP}`);
    expect(branchEdge!.target).toBe(`run:${CHILD_ID}`);
    expect(branchEdge!.data?.relation).toBe("branch_from");
  });

  // ── 5. branch_from edge falls back to run: node when parent NOT expanded ──
  it("falls back to run: node as branch_from source when parent is not expanded", () => {
    const CHILD_ID = "run-child-0003";
    const SOURCE_CP = "candidate:selected";

    const child = makeTreeNode({
      run_id: CHILD_ID,
      root_run_id: ROOT_ID,
      parent_run_id: ROOT_ID,
      source_checkpoint_id: SOURCE_CP,
    });
    const tree = makeTreeNode({ run_id: ROOT_ID, children: [child] });

    // Neither root nor child is active → parent not expanded
    const out = buildBranchCanvasModel(
      baseInput({
        branchTree: tree,
        activeRunId: null,
        timelinesByRunId: { [ROOT_ID]: makeTimeline([{ id: SOURCE_CP }], ROOT_ID) },
      }),
    );

    const branchEdge = out.edges.find((e) => e.id === `branch:${CHILD_ID}`);
    expect(branchEdge).toBeDefined();
    expect(branchEdge!.source).toBe(`run:${ROOT_ID}`);
    expect(branchEdge!.target).toBe(`run:${CHILD_ID}`);
  });

  // ── 6. collapsedRunIds overrides even the active run ─────────────────────
  it("emits no checkpoint nodes for the active run if it's in collapsedRunIds", () => {
    const tree = makeTreeNode({ run_id: ROOT_ID });
    const timeline = makeTimeline(
      [{ id: "candidate:selected" }, { id: "final:selected" }],
      ROOT_ID,
    );
    const out = buildBranchCanvasModel(
      baseInput({
        branchTree: tree,
        activeRunId: ROOT_ID,
        timelinesByRunId: { [ROOT_ID]: timeline },
        collapsedRunIds: new Set([ROOT_ID]),   // active BUT explicitly collapsed
      }),
    );

    expect(out.nodes.filter((n) => n.id.startsWith("cp:"))).toHaveLength(0);
  });

  // ── 7. MAX_VISIBLE_CHECKPOINTS_PER_RUN cap logic ─────────────────────────
  it(`caps timeline at ${MAX_VISIBLE_CHECKPOINTS_PER_RUN} and keeps candidate:selected, accepted, final:selected`, () => {
    // Build 12 entries: mix of plain, accepted, candidate:selected, final:selected
    const entries: Array<{ id: string; accepted?: boolean }> = [];
    for (let i = 0; i < 8; i++) {
      entries.push({ id: `plain-${i}` });
    }
    entries.push({ id: "candidate:selected" });     // must keep
    entries.push({ id: "plain-acc-1", accepted: true }); // must keep (accepted)
    entries.push({ id: "plain-acc-2", accepted: true }); // must keep (accepted)
    entries.push({ id: "final:selected" });         // must keep

    // Total 12 entries > MAX_VISIBLE_CHECKPOINTS_PER_RUN (8)
    expect(entries.length).toBeGreaterThan(MAX_VISIBLE_CHECKPOINTS_PER_RUN);

    const timeline = makeTimeline(entries, ROOT_ID);
    const tree = makeTreeNode({ run_id: ROOT_ID });

    const out = buildBranchCanvasModel(
      baseInput({
        branchTree: tree,
        activeRunId: ROOT_ID,
        timelinesByRunId: { [ROOT_ID]: timeline },
      }),
    );

    const cpIds = out.nodes
      .filter((n) => n.id.startsWith("cp:"))
      .map((n) => n.id.replace(`cp:${ROOT_ID}:`, ""));

    expect(cpIds.length).toBeLessThanOrEqual(MAX_VISIBLE_CHECKPOINTS_PER_RUN);
    expect(cpIds).toContain("candidate:selected");
    expect(cpIds).toContain("plain-acc-1");
    expect(cpIds).toContain("plain-acc-2");
    expect(cpIds).toContain("final:selected");
  });

  // ── 8. determinism: identical input → deeply-equal output ─────────────────
  it("is deterministic: two calls with identical input produce deeply-equal output", () => {
    const CHILD_ID = "run-child-0004";
    const child = makeTreeNode({
      run_id: CHILD_ID,
      root_run_id: ROOT_ID,
      parent_run_id: ROOT_ID,
      source_checkpoint_id: "candidate:selected",
    });
    const tree = makeTreeNode({ run_id: ROOT_ID, children: [child] });
    const tl = makeTimeline(
      [{ id: "candidate:selected" }, { id: "cp-x" }, { id: "final:selected" }],
      ROOT_ID,
    );
    const input = baseInput({
      branchTree: tree,
      activeRunId: ROOT_ID,
      timelinesByRunId: { [ROOT_ID]: tl },
    });

    const a = buildBranchCanvasModel(input);
    const b = buildBranchCanvasModel(input);
    expect(a).toEqual(b);
  });

  // ── 9. MAX_EXPANDED_RUNS cap: active + favorites limited ──────────────────
  it(`expands at most ${MAX_EXPANDED_RUNS} runs even with many favorites`, () => {
    // Build a root with 4 child runs, all favorites. Active = root.
    const children: BranchTreeNode[] = [];
    for (let i = 1; i <= 4; i++) {
      children.push(
        makeTreeNode({
          run_id: `run-child-${i}`,
          root_run_id: ROOT_ID,
          parent_run_id: ROOT_ID,
          source_checkpoint_id: "candidate:selected",
          favorite: true,
        }),
      );
    }
    const tree = makeTreeNode({ run_id: ROOT_ID, children });
    const tl = makeTimeline([{ id: "candidate:selected" }], ROOT_ID);
    const tlByRunId: Record<string, CheckpointTimelineEntry[]> = {
      [ROOT_ID]: tl,
    };
    for (const c of children) {
      tlByRunId[c.run_id] = makeTimeline([{ id: "candidate:selected" }], c.run_id);
    }

    const out = buildBranchCanvasModel(
      baseInput({
        branchTree: tree,
        activeRunId: ROOT_ID,
        timelinesByRunId: tlByRunId,
        favoriteRunIds: new Set(children.map((c) => c.run_id)),
      }),
    );

    // Count distinct run_ids that have checkpoint nodes.
    // cp id format is cp:{run_id}:{cp_id}; cp_id may contain colons, so extract
    // run_id as the segment between the first and second colon.
    const expandedRunIds = new Set(
      out.nodes
        .filter((n) => n.id.startsWith("cp:"))
        .map((n) => n.id.slice(3, n.id.indexOf(":", 3))),
    );
    expect(expandedRunIds.size).toBeLessThanOrEqual(MAX_EXPANDED_RUNS);
  });

  // ── 10. run node data shape ────────────────────────────────────────────────
  it("populates run node data fields correctly", () => {
    const tree = makeTreeNode({
      run_id: ROOT_ID,
      title: "My Root Run",
      status: "completed",
      final_score: 0.87,
      favorite: true,
      feedback: "looks great",
    });
    const out = buildBranchCanvasModel(
      baseInput({ branchTree: tree }),
    );

    const runNode = out.nodes.find((n) => n.id === `run:${ROOT_ID}`);
    expect(runNode).toBeDefined();
    expect(runNode!.type).toBe("run");
    expect(runNode!.data.run_id).toBe(ROOT_ID);
    expect(runNode!.data.title).toBe("My Root Run");
    expect(runNode!.data.label).toBe("My Root Run");
    expect(runNode!.data.status).toBe("completed");
    expect(runNode!.data.score).toBe(0.87);
    expect(runNode!.data.favorite).toBe(true);
    expect(runNode!.data.feedback).toBe("looks great");
    expect(runNode!.position).toEqual({ x: 0, y: 0 });
  });

  // ── 11. checkpoint node data shape ────────────────────────────────────────
  it("populates checkpoint node data fields correctly", () => {
    const tree = makeTreeNode({ run_id: ROOT_ID });
    const entry: CheckpointTimelineEntry = {
      id: "candidate:selected",
      run_id: ROOT_ID,
      kind: "candidate",
      label: "Best candidate",
      score: 0.92,
      delta: 0.05,
      accepted: true,
      has_glsl: true,
      artifact_ids: { render: "art-render-001", shader: "art-shader-001" },
      changes_summary: "Added fog layer and brightened water reflections",
    };
    const out = buildBranchCanvasModel(
      baseInput({
        branchTree: tree,
        activeRunId: ROOT_ID,
        timelinesByRunId: { [ROOT_ID]: [entry] },
      }),
    );

    const cpNode = out.nodes.find((n) => n.id === `cp:${ROOT_ID}:candidate:selected`);
    expect(cpNode).toBeDefined();
    expect(cpNode!.type).toBe("checkpoint");
    expect(cpNode!.data.run_id).toBe(ROOT_ID);
    expect(cpNode!.data.checkpoint_id).toBe("candidate:selected");
    expect(cpNode!.data.label).toBe("Best candidate");
    expect(cpNode!.data.score).toBe(0.92);
    expect(cpNode!.data.delta).toBe(0.05);
    expect(cpNode!.data.accepted).toBe(true);
    expect(cpNode!.data.thumbnail_artifact_id).toBe("art-render-001");
    expect(cpNode!.data.shader_artifact_id).toBe("art-shader-001");
    expect(cpNode!.data.changes_summary).toBe("Added fog layer and brightened water reflections");
    expect(cpNode!.position).toEqual({ x: 0, y: 0 });
  });

  // ── 11b. changes_summary is null when not provided ─────────────────────────
  it("sets changes_summary to null when timeline entry omits it", () => {
    const tree = makeTreeNode({ run_id: ROOT_ID });
    const entry: CheckpointTimelineEntry = {
      id: "final:selected",
      run_id: ROOT_ID,
      kind: "candidate",
      label: "Final",
      score: 0.80,
      delta: null,
      accepted: null,
      has_glsl: false,
    };
    const out = buildBranchCanvasModel(
      baseInput({
        branchTree: tree,
        activeRunId: ROOT_ID,
        timelinesByRunId: { [ROOT_ID]: [entry] },
      }),
    );

    const cpNode = out.nodes.find((n) => n.id === `cp:${ROOT_ID}:final:selected`);
    expect(cpNode).toBeDefined();
    expect(cpNode!.data.changes_summary).toBeNull();
  });

  // ── 12. source_checkpoint_id that is a child reference is kept during cap ─
  it("keeps a source_checkpoint_id entry even if it would otherwise be pruned by cap", () => {
    const CHILD_ID = "run-child-ref";
    const SOURCE_CP = "plain-3"; // would normally be pruned (plain entry in middle)

    const child = makeTreeNode({
      run_id: CHILD_ID,
      root_run_id: ROOT_ID,
      parent_run_id: ROOT_ID,
      source_checkpoint_id: SOURCE_CP,
    });
    const tree = makeTreeNode({ run_id: ROOT_ID, children: [child] });

    // 12 plain entries, referenced entry is plain-3 (index 3)
    const entries: Array<{ id: string }> = [];
    for (let i = 0; i < 10; i++) entries.push({ id: `plain-${i}` });
    entries.push({ id: "candidate:selected" });
    entries.push({ id: "final:selected" });

    const timeline = makeTimeline(entries, ROOT_ID);
    const out = buildBranchCanvasModel(
      baseInput({
        branchTree: tree,
        activeRunId: ROOT_ID,
        timelinesByRunId: { [ROOT_ID]: timeline },
      }),
    );

    const cpIds = out.nodes
      .filter((n) => n.id.startsWith(`cp:${ROOT_ID}:`))
      .map((n) => n.id.replace(`cp:${ROOT_ID}:`, ""));

    expect(cpIds).toContain(SOURCE_CP);

    // And the branch edge should use the cp: node (not fallback)
    const branchEdge = out.edges.find((e) => e.id === `branch:${CHILD_ID}`);
    expect(branchEdge!.source).toBe(`cp:${ROOT_ID}:${SOURCE_CP}`);
  });

  // ── 15. Secondary cap keeps ALL must-keep entries (mustKeep.size ≥ 8 path) ──
  it("never evicts must-keep entries when the priority subset itself exceeds 8", () => {
    // 10 entries, all must-keep: candidate:selected, 8 accepted, final:selected
    const entries: Array<{ id: string; accepted?: boolean }> = [];
    entries.push({ id: "candidate:selected" });
    for (let i = 0; i < 8; i++) {
      entries.push({ id: `acc-${i}`, accepted: true });
    }
    entries.push({ id: "final:selected" });

    expect(entries.length).toBe(10);
    expect(entries.length).toBeGreaterThan(MAX_VISIBLE_CHECKPOINTS_PER_RUN);

    const timeline = makeTimeline(entries, ROOT_ID);
    const tree = makeTreeNode({ run_id: ROOT_ID });

    const out = buildBranchCanvasModel(
      baseInput({
        branchTree: tree,
        activeRunId: ROOT_ID,
        timelinesByRunId: { [ROOT_ID]: timeline },
      }),
    );

    const cpIds = out.nodes
      .filter((n) => n.id.startsWith(`cp:${ROOT_ID}:`))
      .map((n) => n.id.slice(`cp:${ROOT_ID}:`.length));

    // candidate:selected and final:selected must always be present
    expect(cpIds).toContain("candidate:selected");
    expect(cpIds).toContain("final:selected");

    // All 8 accepted entries must be kept
    for (let i = 0; i < 8; i++) {
      expect(cpIds).toContain(`acc-${i}`);
    }

    // All 10 must-keep entries are emitted (none evicted)
    expect(cpIds).toHaveLength(10);
  });

  // ── 16. Referenced cp survives when priority set overflows the cap ─────────
  it("keeps a referenced cp node even when ≥8 accepted entries would push it out with old slice logic", () => {
    const CHILD_ID = "run-child-refoverflow";
    const SOURCE_CP = "early-ref"; // referenced early; old slice(-8) would evict it

    const child = makeTreeNode({
      run_id: CHILD_ID,
      root_run_id: ROOT_ID,
      parent_run_id: ROOT_ID,
      source_checkpoint_id: SOURCE_CP,
    });
    const tree = makeTreeNode({ run_id: ROOT_ID, children: [child] });

    // Timeline: early-ref (referenced), then 8 accepted entries → 9 priority entries total
    // Old code: priority.slice(-8) → early-ref (index 0) gets evicted
    const entries: Array<{ id: string; accepted?: boolean }> = [];
    entries.push({ id: SOURCE_CP }); // referenced — must keep
    for (let i = 0; i < 8; i++) {
      entries.push({ id: `acc-${i}`, accepted: true }); // also priority
    }

    expect(entries.length).toBe(9);
    expect(entries.length).toBeGreaterThan(MAX_VISIBLE_CHECKPOINTS_PER_RUN);

    const timeline = makeTimeline(entries, ROOT_ID);

    const out = buildBranchCanvasModel(
      baseInput({
        branchTree: tree,
        activeRunId: ROOT_ID,
        timelinesByRunId: { [ROOT_ID]: timeline },
      }),
    );

    const cpIds = out.nodes
      .filter((n) => n.id.startsWith(`cp:${ROOT_ID}:`))
      .map((n) => n.id.slice(`cp:${ROOT_ID}:`.length));

    // The referenced checkpoint must be kept
    expect(cpIds).toContain(SOURCE_CP);

    // The branch edge must point to the cp: node, not the fallback run: node
    const branchEdge = out.edges.find((e) => e.id === `branch:${CHILD_ID}`);
    expect(branchEdge).toBeDefined();
    expect(branchEdge!.source).toBe(`cp:${ROOT_ID}:${SOURCE_CP}`);
    expect(branchEdge!.target).toBe(`run:${CHILD_ID}`);
  });

  // ── 13. input node type/data ───────────────────────────────────────────────
  it("emits an input node with correct type and data", () => {
    const tree = makeTreeNode({ run_id: ROOT_ID });
    const out = buildBranchCanvasModel(baseInput({ branchTree: tree }));

    const inputNode = out.nodes.find((n) => n.id === "input");
    expect(inputNode).toBeDefined();
    expect(inputNode!.type).toBe("input");
    expect(inputNode!.data.type).toBe("input");
    expect(inputNode!.data.label).toBe("Input PNG");
    expect(inputNode!.position).toEqual({ x: 0, y: 0 });
  });

  // ── 14. statusesByRunId overrides tree status/score ───────────────────────
  it("uses statusesByRunId values when present, overriding tree node values", () => {
    const tree = makeTreeNode({
      run_id: ROOT_ID,
      status: "running",
      final_score: 0.5,
    });
    const out = buildBranchCanvasModel(
      baseInput({
        branchTree: tree,
        statusesByRunId: {
          [ROOT_ID]: { status: "completed", final_score: 0.99 },
        },
      }),
    );

    const runNode = out.nodes.find((n) => n.id === `run:${ROOT_ID}`);
    expect(runNode!.data.status).toBe("completed");
    expect(runNode!.data.score).toBe(0.99);
  });

  // ─── V3 Variant Group tests ───────────────────────────────────────────────

  // ── V3-1. Two sibling variant runs → one vg: node + two variant_run nodes ──
  it("groups two sibling variant runs into a vg: node and emits variant_run children (expanded)", () => {
    const GROUP_ID = "grp-001";
    const VAR_A = "run-var-aaaa";
    const VAR_B = "run-var-bbbb";

    const varA = makeTreeNode({
      run_id: VAR_A,
      root_run_id: ROOT_ID,
      parent_run_id: ROOT_ID,
      source_checkpoint_id: null,
      variant_group_id: GROUP_ID,
      variant_index: 0,
      variant_label: "Variant A",
      status: "completed",
    });
    const varB = makeTreeNode({
      run_id: VAR_B,
      root_run_id: ROOT_ID,
      parent_run_id: ROOT_ID,
      source_checkpoint_id: null,
      variant_group_id: GROUP_ID,
      variant_index: 1,
      variant_label: "Variant B",
      status: "completed",
    });

    const tree = makeTreeNode({ run_id: ROOT_ID, children: [varA, varB] });
    const out = buildBranchCanvasModel(baseInput({ branchTree: tree }));

    // Must have a vg: node
    const vgNode = out.nodes.find((n) => n.id === `vg:${GROUP_ID}`);
    expect(vgNode).toBeDefined();
    expect(vgNode!.type).toBe("variant_group");
    expect(vgNode!.data.label).toBe("Variants (2)");
    expect(vgNode!.data.group_id).toBe(GROUP_ID);

    // Must NOT have regular run: nodes for the two variant runs
    expect(out.nodes.find((n) => n.id === `run:${VAR_A}` && n.type === "run")).toBeUndefined();
    expect(out.nodes.find((n) => n.id === `run:${VAR_B}` && n.type === "run")).toBeUndefined();

    // Must have variant_run nodes using run: prefix
    const vrA = out.nodes.find((n) => n.id === `run:${VAR_A}`);
    const vrB = out.nodes.find((n) => n.id === `run:${VAR_B}`);
    expect(vrA).toBeDefined();
    expect(vrA!.type).toBe("variant_run");
    expect(vrA!.data.variant_group_id).toBe(GROUP_ID);
    expect(vrB).toBeDefined();
    expect(vrB!.type).toBe("variant_run");

    // Must have a branch_from edge to the vg: node
    const branchEdge = out.edges.find((e) => e.id === `vg-branch:${GROUP_ID}`);
    expect(branchEdge).toBeDefined();
    expect(branchEdge!.target).toBe(`vg:${GROUP_ID}`);
    expect(branchEdge!.data?.relation).toBe("branch_from");

    // Must have variant_child edges
    const vcA = out.edges.find((e) => e.id === `vc:${VAR_A}`);
    const vcB = out.edges.find((e) => e.id === `vc:${VAR_B}`);
    expect(vcA).toBeDefined();
    expect(vcA!.source).toBe(`vg:${GROUP_ID}`);
    expect(vcA!.target).toBe(`run:${VAR_A}`);
    expect(vcA!.data?.relation).toBe("variant_child");
    expect(vcB).toBeDefined();
  });

  // ── V3-2. collapsedGroupIds → only vg: node, no variant_run nodes/edges ───
  it("collapses a variant group when its group_id is in collapsedGroupIds", () => {
    const GROUP_ID = "grp-002";
    const VAR_A = "run-var-cccc";
    const VAR_B = "run-var-dddd";

    const varA = makeTreeNode({
      run_id: VAR_A,
      root_run_id: ROOT_ID,
      parent_run_id: ROOT_ID,
      variant_group_id: GROUP_ID,
      variant_index: 0,
    });
    const varB = makeTreeNode({
      run_id: VAR_B,
      root_run_id: ROOT_ID,
      parent_run_id: ROOT_ID,
      variant_group_id: GROUP_ID,
      variant_index: 1,
    });

    const tree = makeTreeNode({ run_id: ROOT_ID, children: [varA, varB] });
    const out = buildBranchCanvasModel(
      baseInput({ branchTree: tree, collapsedGroupIds: new Set([GROUP_ID]) }),
    );

    // vg: node is present
    const vgNode = out.nodes.find((n) => n.id === `vg:${GROUP_ID}`);
    expect(vgNode).toBeDefined();
    expect(vgNode!.data.collapsed).toBe(true);

    // No variant_run nodes
    expect(out.nodes.find((n) => n.id === `run:${VAR_A}`)).toBeUndefined();
    expect(out.nodes.find((n) => n.id === `run:${VAR_B}`)).toBeUndefined();

    // No variant_child edges
    expect(out.edges.find((e) => e.id === `vc:${VAR_A}`)).toBeUndefined();
    expect(out.edges.find((e) => e.id === `vc:${VAR_B}`)).toBeUndefined();
  });

  // ── V3-3. Non-variant child still emits a regular run node ─────────────────
  it("still emits a regular run node for a non-variant child", () => {
    const REGULAR_CHILD = "run-child-regular";
    const child = makeTreeNode({
      run_id: REGULAR_CHILD,
      root_run_id: ROOT_ID,
      parent_run_id: ROOT_ID,
    });
    const tree = makeTreeNode({ run_id: ROOT_ID, children: [child] });
    const out = buildBranchCanvasModel(baseInput({ branchTree: tree }));

    const runNode = out.nodes.find((n) => n.id === `run:${REGULAR_CHILD}`);
    expect(runNode).toBeDefined();
    expect(runNode!.type).toBe("run");

    const branchEdge = out.edges.find((e) => e.id === `branch:${REGULAR_CHILD}`);
    expect(branchEdge).toBeDefined();
    expect(branchEdge!.target).toBe(`run:${REGULAR_CHILD}`);
  });

  // ── V3-4. Descendant of a variant run gets fallback branch_from edge ────────
  it("emits a branch_from edge from run:{winnerId} for a child of a variant run", () => {
    const GROUP_ID = "grp-003";
    const WINNER_ID = "run-var-winner";
    const CONTINUE_ID = "run-continue-from-winner";

    const varWinner = makeTreeNode({
      run_id: WINNER_ID,
      root_run_id: ROOT_ID,
      parent_run_id: ROOT_ID,
      variant_group_id: GROUP_ID,
      variant_index: 0,
      status: "completed",
    });

    // A regular child of the winner variant run
    const continueRun = makeTreeNode({
      run_id: CONTINUE_ID,
      root_run_id: ROOT_ID,
      parent_run_id: WINNER_ID,
      source_checkpoint_id: "cp-winner-final",
    });

    // Attach continueRun as child of varWinner
    varWinner.children = [continueRun];
    const tree = makeTreeNode({ run_id: ROOT_ID, children: [varWinner] });

    const out = buildBranchCanvasModel(baseInput({ branchTree: tree }));

    // continue run must be a regular run node
    const runNode = out.nodes.find((n) => n.id === `run:${CONTINUE_ID}`);
    expect(runNode).toBeDefined();
    expect(runNode!.type).toBe("run");

    // branch_from edge must fall back to run:{WINNER_ID} (no cp: nodes for variant runs)
    const branchEdge = out.edges.find((e) => e.id === `branch:${CONTINUE_ID}`);
    expect(branchEdge).toBeDefined();
    expect(branchEdge!.source).toBe(`run:${WINNER_ID}`);
    expect(branchEdge!.target).toBe(`run:${CONTINUE_ID}`);
    expect(branchEdge!.data?.relation).toBe("branch_from");
  });

  // ── V3-5. Group status aggregate rules ────────────────────────────────────
  it("aggregates variant group status: [completed, completed] → completed", () => {
    const GROUP_ID = "grp-status-all-done";
    const VA = "run-var-s1";
    const VB = "run-var-s2";

    const tree = makeTreeNode({
      run_id: ROOT_ID,
      children: [
        makeTreeNode({ run_id: VA, root_run_id: ROOT_ID, parent_run_id: ROOT_ID, variant_group_id: GROUP_ID, variant_index: 0, status: "completed" }),
        makeTreeNode({ run_id: VB, root_run_id: ROOT_ID, parent_run_id: ROOT_ID, variant_group_id: GROUP_ID, variant_index: 1, status: "completed" }),
      ],
    });

    const out = buildBranchCanvasModel(baseInput({ branchTree: tree }));
    const vgNode = out.nodes.find((n) => n.id === `vg:${GROUP_ID}`);
    expect(vgNode!.data.status).toBe("completed");
  });

  it("aggregates variant group status: [completed, running] → running", () => {
    const GROUP_ID = "grp-status-running";
    const VA = "run-var-r1";
    const VB = "run-var-r2";

    const tree = makeTreeNode({
      run_id: ROOT_ID,
      children: [
        makeTreeNode({ run_id: VA, root_run_id: ROOT_ID, parent_run_id: ROOT_ID, variant_group_id: GROUP_ID, variant_index: 0, status: "completed" }),
        makeTreeNode({ run_id: VB, root_run_id: ROOT_ID, parent_run_id: ROOT_ID, variant_group_id: GROUP_ID, variant_index: 1, status: "running" }),
      ],
    });

    const out = buildBranchCanvasModel(baseInput({ branchTree: tree }));
    const vgNode = out.nodes.find((n) => n.id === `vg:${GROUP_ID}`);
    expect(vgNode!.data.status).toBe("running");
  });

  it("aggregates variant group status: [completed, failed] → partial_failed", () => {
    const GROUP_ID = "grp-status-partial";
    const VA = "run-var-p1";
    const VB = "run-var-p2";

    const tree = makeTreeNode({
      run_id: ROOT_ID,
      children: [
        makeTreeNode({ run_id: VA, root_run_id: ROOT_ID, parent_run_id: ROOT_ID, variant_group_id: GROUP_ID, variant_index: 0, status: "completed" }),
        makeTreeNode({ run_id: VB, root_run_id: ROOT_ID, parent_run_id: ROOT_ID, variant_group_id: GROUP_ID, variant_index: 1, status: "failed" }),
      ],
    });

    const out = buildBranchCanvasModel(baseInput({ branchTree: tree }));
    const vgNode = out.nodes.find((n) => n.id === `vg:${GROUP_ID}`);
    expect(vgNode!.data.status).toBe("partial_failed");
  });

  // ── V3-5b. [] → "queued" (empty member list) ─────────────────────────────
  // This case should not arise in normal operation but the helper must be safe.
  // We test it indirectly via a group whose only member has been stripped from
  // statusesByRunId so memberStatuses resolves to an empty-ish scenario; instead
  // we expose the helper directly by building a one-member group with status
  // "queued" and verifying the group node status, plus a dedicated 2-queued case.

  it("aggregates variant group status: [] (no members in group) → queued via empty statusesByRunId override", () => {
    // We simulate an empty-like statuses list by verifying with a single "queued" member.
    // Direct [] path: covered by next test with ["queued","queued"] to also hit all-queued branch.
    // For the true empty case we use a workaround: build with one member but intercept via
    // the exported helper indirectly. Since the helper is not exported, we rely on a
    // single-member group with status=queued as a proxy and add the all-queued 2-member test.

    const GROUP_ID = "grp-status-queued-single";
    const VA = "run-var-q1";

    const tree = makeTreeNode({
      run_id: ROOT_ID,
      children: [
        makeTreeNode({ run_id: VA, root_run_id: ROOT_ID, parent_run_id: ROOT_ID, variant_group_id: GROUP_ID, variant_index: 0, status: "queued" }),
      ],
    });

    const out = buildBranchCanvasModel(baseInput({ branchTree: tree }));
    const vgNode = out.nodes.find((n) => n.id === `vg:${GROUP_ID}`);
    expect(vgNode).toBeDefined();
    expect(vgNode!.data.status).toBe("queued");
  });

  it("aggregates variant group status: [queued, queued] → queued", () => {
    const GROUP_ID = "grp-status-all-queued";
    const VA = "run-var-q2";
    const VB = "run-var-q3";

    const tree = makeTreeNode({
      run_id: ROOT_ID,
      children: [
        makeTreeNode({ run_id: VA, root_run_id: ROOT_ID, parent_run_id: ROOT_ID, variant_group_id: GROUP_ID, variant_index: 0, status: "queued" }),
        makeTreeNode({ run_id: VB, root_run_id: ROOT_ID, parent_run_id: ROOT_ID, variant_group_id: GROUP_ID, variant_index: 1, status: "queued" }),
      ],
    });

    const out = buildBranchCanvasModel(baseInput({ branchTree: tree }));
    const vgNode = out.nodes.find((n) => n.id === `vg:${GROUP_ID}`);
    expect(vgNode).toBeDefined();
    expect(vgNode!.data.status).toBe("queued");
  });

  it("aggregates variant group status: [cancelled, failed] → cancelled", () => {
    const GROUP_ID = "grp-status-cancelled-failed";
    const VA = "run-var-cf1";
    const VB = "run-var-cf2";

    const tree = makeTreeNode({
      run_id: ROOT_ID,
      children: [
        makeTreeNode({ run_id: VA, root_run_id: ROOT_ID, parent_run_id: ROOT_ID, variant_group_id: GROUP_ID, variant_index: 0, status: "cancelled" }),
        makeTreeNode({ run_id: VB, root_run_id: ROOT_ID, parent_run_id: ROOT_ID, variant_group_id: GROUP_ID, variant_index: 1, status: "failed" }),
      ],
    });

    const out = buildBranchCanvasModel(baseInput({ branchTree: tree }));
    const vgNode = out.nodes.find((n) => n.id === `vg:${GROUP_ID}`);
    expect(vgNode).toBeDefined();
    expect(vgNode!.data.status).toBe("cancelled");
  });

  it("aggregates variant group status: [completed, cancelled] → partial_failed", () => {
    const GROUP_ID = "grp-status-completed-cancelled";
    const VA = "run-var-cc1";
    const VB = "run-var-cc2";

    const tree = makeTreeNode({
      run_id: ROOT_ID,
      children: [
        makeTreeNode({ run_id: VA, root_run_id: ROOT_ID, parent_run_id: ROOT_ID, variant_group_id: GROUP_ID, variant_index: 0, status: "completed" }),
        makeTreeNode({ run_id: VB, root_run_id: ROOT_ID, parent_run_id: ROOT_ID, variant_group_id: GROUP_ID, variant_index: 1, status: "cancelled" }),
      ],
    });

    const out = buildBranchCanvasModel(baseInput({ branchTree: tree }));
    const vgNode = out.nodes.find((n) => n.id === `vg:${GROUP_ID}`);
    expect(vgNode).toBeDefined();
    expect(vgNode!.data.status).toBe("partial_failed");
  });

  // ── V3-6. [failed, failed] → "failed" ────────────────────────────────────
  it("aggregates variant group status: [failed, failed] → failed", () => {
    const GROUP_ID = "grp-status-all-failed";
    const VA = "run-var-f1";
    const VB = "run-var-f2";

    const tree = makeTreeNode({
      run_id: ROOT_ID,
      children: [
        makeTreeNode({ run_id: VA, root_run_id: ROOT_ID, parent_run_id: ROOT_ID, variant_group_id: GROUP_ID, variant_index: 0, status: "failed" }),
        makeTreeNode({ run_id: VB, root_run_id: ROOT_ID, parent_run_id: ROOT_ID, variant_group_id: GROUP_ID, variant_index: 1, status: "failed" }),
      ],
    });

    const out = buildBranchCanvasModel(baseInput({ branchTree: tree }));
    const vgNode = out.nodes.find((n) => n.id === `vg:${GROUP_ID}`);
    expect(vgNode).toBeDefined();
    expect(vgNode!.data.status).toBe("failed");
  });

  // ── V3.5 Exclusion: run with variant_group_id + draw_session_id is excluded ──
  it("excludes a run with both variant_group_id and draw_session_id from variant_group/variant_run nodes", () => {
    const GROUP_ID = "grp-draw-excl";
    // Draw card: has BOTH variant_group_id and draw_session_id — must NOT produce variant nodes
    const DRAW_RUN = "run-var-draw-card";
    // Pure variant: has variant_group_id ONLY — must still produce variant nodes
    const PURE_VAR = "run-var-pure";

    const drawNode = makeTreeNode({
      run_id: DRAW_RUN,
      root_run_id: ROOT_ID,
      parent_run_id: ROOT_ID,
      variant_group_id: GROUP_ID,
      variant_index: 0,
      draw_session_id: "ds-excl-001",
    });
    const pureVarNode = makeTreeNode({
      run_id: PURE_VAR,
      root_run_id: ROOT_ID,
      parent_run_id: ROOT_ID,
      variant_group_id: GROUP_ID,
      variant_index: 1,
    });
    const tree = makeTreeNode({ run_id: ROOT_ID, children: [drawNode, pureVarNode] });
    const out = buildBranchCanvasModel(baseInput({ branchTree: tree }));

    // Must NOT produce variant_run or variant_group node for the draw-session run
    expect(out.nodes.find((n) => n.id === `run:${DRAW_RUN}` && n.type === "variant_run")).toBeUndefined();

    // Control: pure variant run is handled by variant pass → variant_run node
    const pureVarRunNode = out.nodes.find((n) => n.id === `run:${PURE_VAR}`);
    expect(pureVarRunNode).toBeDefined();
    expect(pureVarRunNode!.type).toBe("variant_run");
  });

  // ── V3-7. Active variant run doesn't waste an expansion budget slot ────────
  it("active variant run produces no checkpoint nodes and a favorited non-variant run still expands", () => {
    // MAX_EXPANDED_RUNS = 3. Build 3 non-variant runs that should all expand
    // if budget allows, plus one variant run set as activeRunId.
    // The variant run must NOT consume a slot; all 3 non-variant runs should expand.
    const GROUP_ID = "grp-budget-test";
    const VAR_ID = "run-var-budget";
    const NON_VAR_A = "run-nv-budget-a";
    const NON_VAR_B = "run-nv-budget-b";
    const NON_VAR_C = "run-nv-budget-c";

    const variantRun = makeTreeNode({
      run_id: VAR_ID,
      root_run_id: ROOT_ID,
      parent_run_id: ROOT_ID,
      variant_group_id: GROUP_ID,
      variant_index: 0,
    });

    const nonVarA = makeTreeNode({ run_id: NON_VAR_A, root_run_id: ROOT_ID, parent_run_id: ROOT_ID, favorite: true });
    const nonVarB = makeTreeNode({ run_id: NON_VAR_B, root_run_id: ROOT_ID, parent_run_id: ROOT_ID, favorite: true });
    const nonVarC = makeTreeNode({ run_id: NON_VAR_C, root_run_id: ROOT_ID, parent_run_id: ROOT_ID, favorite: true });

    const tree = makeTreeNode({ run_id: ROOT_ID, children: [variantRun, nonVarA, nonVarB, nonVarC] });

    const cpEntry = [{ id: "candidate:selected" }];
    const out = buildBranchCanvasModel(
      baseInput({
        branchTree: tree,
        // Set the variant run as the active run — it must NOT burn a budget slot
        activeRunId: VAR_ID,
        favoriteRunIds: new Set([NON_VAR_A, NON_VAR_B, NON_VAR_C]),
        timelinesByRunId: {
          [VAR_ID]: makeTimeline(cpEntry, VAR_ID),
          [NON_VAR_A]: makeTimeline(cpEntry, NON_VAR_A),
          [NON_VAR_B]: makeTimeline(cpEntry, NON_VAR_B),
          [NON_VAR_C]: makeTimeline(cpEntry, NON_VAR_C),
        },
      }),
    );

    // The active variant run must have NO checkpoint nodes (variant runs never expand)
    const varCpNodes = out.nodes.filter((n) => n.id.startsWith(`cp:${VAR_ID}:`));
    expect(varCpNodes).toHaveLength(0);

    // All 3 favorited non-variant runs should be expanded (budget = 3, variant didn't consume any)
    const expandedRunIds = new Set(
      out.nodes
        .filter((n) => n.id.startsWith("cp:"))
        .map((n) => n.id.slice(3, n.id.indexOf(":", 3))),
    );
    expect(expandedRunIds.has(NON_VAR_A)).toBe(true);
    expect(expandedRunIds.has(NON_VAR_B)).toBe(true);
    expect(expandedRunIds.has(NON_VAR_C)).toBe(true);
    expect(expandedRunIds.size).toBe(3);
  });
});

// ─── buildDrawSessionModel tests ─────────────────────────────────────────────

describe("buildDrawSessionModel", () => {
  const ANCHOR = "run:run-root-0001";
  const DRAW_ID = "ds-test-0001";

  // ── DS-1. 3 cards → correct node/edge counts and id scheme ────────────────
  it("3 cards → 1 draw_session node + 3 draw_card nodes + 1 draw_from edge + 3 draw_card edges", () => {
    const cardA = makeDrawCard({ card_id: "c1", run_id: "run-c1", index: 0 });
    const cardB = makeDrawCard({ card_id: "c2", run_id: "run-c2", index: 1 });
    const cardC = makeDrawCard({ card_id: "c3", run_id: "run-c3", index: 2 });
    const session = makeDrawSession({ draw_id: DRAW_ID, cards: [cardA, cardB, cardC] });

    const { nodes, edges } = buildDrawSessionModel(session, { anchorNodeId: ANCHOR });

    // Nodes: 1 draw_session + 3 draw_card
    expect(nodes).toHaveLength(4);
    const dsNode = nodes.find((n) => n.id === `draw:${DRAW_ID}`);
    expect(dsNode).toBeDefined();
    expect(dsNode!.type).toBe("draw_session");
    expect(dsNode!.data.type).toBe("draw_session");

    const cardIds = nodes.filter((n) => n.type === "draw_card").map((n) => n.id);
    expect(cardIds).toContain("drawcard:run-c1");
    expect(cardIds).toContain("drawcard:run-c2");
    expect(cardIds).toContain("drawcard:run-c3");

    // Edges: 1 draw_from + 3 draw_card
    expect(edges).toHaveLength(4);
    const dfEdge = edges.find((e) => e.id === `drawfrom:${DRAW_ID}`);
    expect(dfEdge).toBeDefined();
    expect(dfEdge!.source).toBe(ANCHOR);
    expect(dfEdge!.target).toBe(`draw:${DRAW_ID}`);
    expect(dfEdge!.data?.relation).toBe("draw_from");

    for (const card of [cardA, cardB, cardC]) {
      const dcEdge = edges.find((e) => e.id === `drawcard:${card.run_id}`);
      expect(dcEdge).toBeDefined();
      expect(dcEdge!.source).toBe(`draw:${DRAW_ID}`);
      expect(dcEdge!.target).toBe(`drawcard:${card.run_id}`);
      expect(dcEdge!.data?.relation).toBe("draw_card");
    }
  });

  // ── DS-2. collapsed:true → only draw_session node + draw_from edge ─────────
  it("collapsed:true emits only the draw_session node + draw_from edge", () => {
    const cardA = makeDrawCard({ card_id: "c1", run_id: "run-c1", index: 0 });
    const cardB = makeDrawCard({ card_id: "c2", run_id: "run-c2", index: 1 });
    const session = makeDrawSession({ draw_id: DRAW_ID, cards: [cardA, cardB] });

    const { nodes, edges } = buildDrawSessionModel(session, { anchorNodeId: ANCHOR, collapsed: true });

    expect(nodes).toHaveLength(1);
    expect(nodes[0].id).toBe(`draw:${DRAW_ID}`);
    expect(edges).toHaveLength(1);
    expect(edges[0].id).toBe(`drawfrom:${DRAW_ID}`);
  });

  // ── DS-3. replacement_of edge: in-session target → edge emitted ─────────────
  it("emits replacement_of edge when card B replaces card A (both in session)", () => {
    const cardA = makeDrawCard({ card_id: "c-a", run_id: "run-a", index: 0 });
    const cardB = makeDrawCard({ card_id: "c-b", run_id: "run-b", index: 1, replacement_of_run_id: "run-a" });
    const session = makeDrawSession({ draw_id: DRAW_ID, cards: [cardA, cardB] });

    const { edges } = buildDrawSessionModel(session, { anchorNodeId: ANCHOR });

    const replEdge = edges.find((e) => e.id === "repl:run-b");
    expect(replEdge).toBeDefined();
    expect(replEdge!.source).toBe("drawcard:run-b");
    expect(replEdge!.target).toBe("drawcard:run-a");
    expect(replEdge!.data?.relation).toBe("replacement_of");
  });

  // ── DS-3b. replacement_of edge: target NOT in session → no edge ─────────────
  it("does NOT emit replacement_of edge when the target run_id is not in this session", () => {
    const cardB = makeDrawCard({ card_id: "c-b", run_id: "run-b", index: 0, replacement_of_run_id: "run-outside" });
    const session = makeDrawSession({ draw_id: DRAW_ID, cards: [cardB] });

    const { edges } = buildDrawSessionModel(session, { anchorNodeId: ANCHOR });

    expect(edges.find((e) => e.id === "repl:run-b")).toBeUndefined();
  });

  // ── DS-4. winner/favorite/eliminated flags surface on card node data ─────────
  it("is_winner, favorite, eliminated flags are correctly set on card node data", () => {
    const WINNER_RUN = "run-winner";
    const cardWinner = makeDrawCard({ card_id: "c-w", run_id: WINNER_RUN, index: 0, favorite: true });
    const cardElim   = makeDrawCard({ card_id: "c-e", run_id: "run-elim", index: 1, eliminated: true });
    const cardPlain  = makeDrawCard({ card_id: "c-p", run_id: "run-plain", index: 2 });
    const session = makeDrawSession({ draw_id: DRAW_ID, winner_run_id: WINNER_RUN, cards: [cardWinner, cardElim, cardPlain] });

    const { nodes } = buildDrawSessionModel(session, { anchorNodeId: ANCHOR });

    const winNode   = nodes.find((n) => n.id === `drawcard:${WINNER_RUN}`);
    const elimNode  = nodes.find((n) => n.id === "drawcard:run-elim");
    const plainNode = nodes.find((n) => n.id === "drawcard:run-plain");

    expect(winNode!.data.is_winner).toBe(true);
    expect(winNode!.data.favorite).toBe(true);
    expect(winNode!.data.eliminated).toBe(false);

    expect(elimNode!.data.eliminated).toBe(true);
    expect(elimNode!.data.is_winner).toBe(false);

    expect(plainNode!.data.is_winner).toBe(false);
    expect(plainNode!.data.favorite).toBe(false);
  });

  // ── DS-5. empty cards → just draw_session node + draw_from edge ────────────
  it("empty cards list → 1 draw_session node + 1 draw_from edge", () => {
    const session = makeDrawSession({ draw_id: DRAW_ID, cards: [] });

    const { nodes, edges } = buildDrawSessionModel(session, { anchorNodeId: ANCHOR });

    expect(nodes).toHaveLength(1);
    expect(nodes[0].id).toBe(`draw:${DRAW_ID}`);
    expect(edges).toHaveLength(1);
    expect(edges[0].id).toBe(`drawfrom:${DRAW_ID}`);
  });

  // ── DS-6. draw_session node data fields ─────────────────────────────────────
  it("draw_session node data carries expected summary fields", () => {
    const session = makeDrawSession({
      draw_id: DRAW_ID,
      status: "running",
      requested_count: 4,
      completed_count: 2,
      running_count: 1,
      failed_count: 1,
      winner_run_id: "run-w",
      cards: [
        makeDrawCard({ card_id: "c1", run_id: "run-c1", index: 0 }),
        makeDrawCard({ card_id: "c2", run_id: "run-c2", index: 1 }),
      ],
    });

    const { nodes } = buildDrawSessionModel(session, { anchorNodeId: ANCHOR });
    const dsNode = nodes.find((n) => n.id === `draw:${DRAW_ID}`)!;

    expect(dsNode.data.draw_id).toBe(DRAW_ID);
    expect(dsNode.data.status).toBe("running");
    expect(dsNode.data.requested_count).toBe(4);
    expect(dsNode.data.completed_count).toBe(2);
    expect(dsNode.data.running_count).toBe(1);
    expect(dsNode.data.failed_count).toBe(1);
    expect(dsNode.data.winner_run_id).toBe("run-w");
    expect(dsNode.data.card_count).toBe(2);
    expect(dsNode.position).toEqual({ x: 0, y: 0 });
  });

  // ── DS-7. draw_card node data fields ─────────────────────────────────────────
  it("draw_card node data carries expected fields", () => {
    const card = makeDrawCard({
      card_id: "c-detail",
      run_id: "run-detail",
      index: 3,
      group_id: "grp-abc",
      status: "completed",
      label: "Card detail",
      strategy_label: "high",
      final_score: 0.85,
      can_use_for_fusion: true,
    });
    const session = makeDrawSession({ draw_id: DRAW_ID, cards: [card] });

    const { nodes } = buildDrawSessionModel(session, { anchorNodeId: ANCHOR });
    const dcNode = nodes.find((n) => n.id === "drawcard:run-detail")!;

    expect(dcNode.data.type).toBe("draw_card");
    expect(dcNode.data.draw_id).toBe(DRAW_ID);
    expect(dcNode.data.run_id).toBe("run-detail");
    expect(dcNode.data.card_id).toBe("c-detail");
    expect(dcNode.data.group_id).toBe("grp-abc");
    expect(dcNode.data.index).toBe(3);
    expect(dcNode.data.status).toBe("completed");
    expect(dcNode.data.label).toBe("Card detail");
    expect(dcNode.data.strategy_label).toBe("high");
    expect(dcNode.data.final_score).toBe(0.85);
    expect(dcNode.data.can_use_for_fusion).toBe(true);
    expect(dcNode.position).toEqual({ x: 0, y: 0 });
  });
});

// ─── buildRegionConstraintModel tests ────────────────────────────────────────

function makeRegion(overrides: Partial<RegionConstraint> & { id: string }): RegionConstraint {
  return {
    label: `Region ${overrides.id}`,
    mode: "modify",
    instruction: `Instruction for ${overrides.id}`,
    geometry_type: "rect",
    geometry: { x: 0.1, y: 0.1, w: 0.3, h: 0.3 },
    strength: 0.7,
    ...overrides,
  };
}

describe("buildRegionConstraintModel", () => {
  const ANCHOR = "run:run-root-0001";

  // ── RC-1. empty regions → empty output ──────────────────────────────────────
  it("empty regions array → empty nodes and edges", () => {
    const { nodes, edges } = buildRegionConstraintModel([], { anchorNodeId: ANCHOR });
    expect(nodes).toEqual([]);
    expect(edges).toEqual([]);
  });

  // ── RC-2. N regions → N nodes + N constraint_applies edges ─────────────────
  it("3 regions → 3 region_constraint nodes + 3 constraint_applies edges", () => {
    const regions = [
      makeRegion({ id: "r1", label: "Sky", mode: "protect" }),
      makeRegion({ id: "r2", label: "Water", mode: "modify" }),
      makeRegion({ id: "r3", label: "Ground", mode: "protect" }),
    ];

    const { nodes, edges } = buildRegionConstraintModel(regions, { anchorNodeId: ANCHOR });

    expect(nodes).toHaveLength(3);
    expect(edges).toHaveLength(3);

    // Correct node ids
    expect(nodes.map((n) => n.id)).toEqual(["region:r1", "region:r2", "region:r3"]);
    // Correct edge ids
    expect(edges.map((e) => e.id)).toEqual(["applies:r1", "applies:r2", "applies:r3"]);
  });

  // ── RC-3. edges have correct source/target ───────────────────────────────────
  it("each constraint_applies edge has source=anchorNodeId and target=region:{id}", () => {
    const CUSTOM_ANCHOR = "cp:run-abc:candidate:selected";
    const regions = [
      makeRegion({ id: "rx" }),
      makeRegion({ id: "ry" }),
    ];

    const { edges } = buildRegionConstraintModel(regions, { anchorNodeId: CUSTOM_ANCHOR });

    for (const edge of edges) {
      expect(edge.source).toBe(CUSTOM_ANCHOR);
      expect(edge.data?.relation).toBe("constraint_applies");
    }
    expect(edges[0].target).toBe("region:rx");
    expect(edges[1].target).toBe("region:ry");
  });

  // ── RC-4. node data carries all region fields ────────────────────────────────
  it("node data carries region_id, label, mode, instruction, strength, geometry", () => {
    const region = makeRegion({
      id: "detail-r",
      label: "Detailed Region",
      mode: "protect",
      instruction: "Keep this area unchanged",
      strength: 0.9,
      geometry: { x: 0.2, y: 0.3, w: 0.4, h: 0.5 },
    });

    const { nodes } = buildRegionConstraintModel([region], { anchorNodeId: ANCHOR });
    const node = nodes[0];

    expect(node.id).toBe("region:detail-r");
    expect(node.type).toBe("region_constraint");
    expect(node.data.type).toBe("region_constraint");
    expect(node.data.region_id).toBe("detail-r");
    expect(node.data.label).toBe("Detailed Region");
    expect(node.data.mode).toBe("protect");
    expect(node.data.instruction).toBe("Keep this area unchanged");
    expect(node.data.strength).toBe(0.9);
    expect(node.data.geometry).toEqual({ x: 0.2, y: 0.3, w: 0.4, h: 0.5 });
    expect(node.position).toEqual({ x: 0, y: 0 });
  });

  // ── RC-5. mode badge distinction (modify vs protect) ─────────────────────────
  it("correctly carries mode=modify and mode=protect on separate nodes", () => {
    const regions = [
      makeRegion({ id: "mod-1", mode: "modify", label: "Modify zone", strength: 0.5 }),
      makeRegion({ id: "prot-1", mode: "protect", label: "Protect zone", strength: 1.0 }),
    ];

    const { nodes } = buildRegionConstraintModel(regions, { anchorNodeId: ANCHOR });

    const modNode = nodes.find((n) => n.id === "region:mod-1");
    const protNode = nodes.find((n) => n.id === "region:prot-1");

    expect(modNode).toBeDefined();
    expect(modNode!.data.mode).toBe("modify");
    expect(modNode!.data.strength).toBe(0.5);

    expect(protNode).toBeDefined();
    expect(protNode!.data.mode).toBe("protect");
    expect(protNode!.data.strength).toBe(1.0);
  });

  // ── RC-6. deterministic: same input → same output ───────────────────────────
  it("is deterministic: two calls with identical input produce deeply-equal output", () => {
    const regions = [
      makeRegion({ id: "det-1" }),
      makeRegion({ id: "det-2" }),
    ];

    const a = buildRegionConstraintModel(regions, { anchorNodeId: ANCHOR });
    const b = buildRegionConstraintModel(regions, { anchorNodeId: ANCHOR });
    expect(a).toEqual(b);
  });

  // ── RC-7. ordering is preserved (array order) ────────────────────────────────
  it("preserves array order in emitted nodes/edges", () => {
    const ids = ["z-last", "a-first", "m-mid"];
    const regions = ids.map((id) => makeRegion({ id }));

    const { nodes, edges } = buildRegionConstraintModel(regions, { anchorNodeId: ANCHOR });

    expect(nodes.map((n) => n.id)).toEqual(ids.map((id) => `region:${id}`));
    expect(edges.map((e) => e.id)).toEqual(ids.map((id) => `applies:${id}`));
  });
});

// ─── buildFusionModel tests ───────────────────────────────────────────────────

function makeFusionRegion(overrides: Partial<FusionRegion> & { id: string }): FusionRegion {
  return {
    label: `FusionRegion ${overrides.id}`,
    source_run_id: "run-src-default",
    instruction: `Instruction for ${overrides.id}`,
    geometry_type: "rect",
    geometry: { x: 0.1, y: 0.1, w: 0.3, h: 0.3 },
    strength: 0.5,
    blend_mode: "soft",
    feather: 0.08,
    ...overrides,
  };
}

function makeFusion(overrides: Partial<FusionStatus> & { fusion_id: string }): FusionStatus {
  return {
    status: "draft",
    base_run_id: "run-base-0001",
    source_run_ids: [],
    output_run_id: null,
    composite_target_url: null,
    regions: [],
    error: null,
    ...overrides,
  };
}

describe("buildFusionModel", () => {
  const FUSION_ID = "fus-test-0001";
  const BASE_ANCHOR = "run:run-base-0001";
  const SRC_A = "run-src-aaaa";
  const SRC_B = "run-src-bbbb";
  const OUT_RUN = "run-out-0001";
  const OUT_ANCHOR = `run:${OUT_RUN}`;

  // ── FM-1. Full fusion: base + 2 sources + output → 1 node + 4 edges ────────
  it("fusion with base + 2 sources + output: 1 node + fusion_base + 2 fusion_source + fusion_output edges", () => {
    const fusion = makeFusion({
      fusion_id: FUSION_ID,
      status: "completed",
      base_run_id: "run-base-0001",
      source_run_ids: [SRC_A, SRC_B],
      output_run_id: OUT_RUN,
      regions: [makeFusionRegion({ id: "r1" }), makeFusionRegion({ id: "r2" })],
    });

    const { nodes, edges } = buildFusionModel(fusion, {
      baseAnchorNodeId: BASE_ANCHOR,
      sourceAnchorNodeIds: { [SRC_A]: `run:${SRC_A}`, [SRC_B]: `run:${SRC_B}` },
      outputAnchorNodeId: OUT_ANCHOR,
    });

    // 1 fusion_plan node
    expect(nodes).toHaveLength(1);
    const fpNode = nodes[0];
    expect(fpNode.id).toBe(`fusion:${FUSION_ID}`);
    expect(fpNode.type).toBe("fusion_plan");
    expect(fpNode.data.type).toBe("fusion_plan");
    expect(fpNode.data.fusion_id).toBe(FUSION_ID);
    expect(fpNode.data.status).toBe("completed");
    expect(fpNode.data.base_run_id).toBe("run-base-0001");
    expect(fpNode.data.source_run_ids).toEqual([SRC_A, SRC_B]);
    expect(fpNode.data.output_run_id).toBe(OUT_RUN);
    expect(fpNode.data.region_count).toBe(2);
    expect(fpNode.data.label).toBe(`Fusion ${FUSION_ID.slice(-4)}`);
    expect(fpNode.position).toEqual({ x: 0, y: 0 });

    // 4 edges total
    expect(edges).toHaveLength(4);

    // fusion_base edge
    const baseEdge = edges.find((e) => e.id === `fbase:${FUSION_ID}`);
    expect(baseEdge).toBeDefined();
    expect(baseEdge!.source).toBe(BASE_ANCHOR);
    expect(baseEdge!.target).toBe(`fusion:${FUSION_ID}`);
    expect(baseEdge!.data?.relation).toBe("fusion_base");

    // fusion_source edges (in source_run_ids order)
    const srcEdgeA = edges.find((e) => e.id === `fsrc:${FUSION_ID}:${SRC_A}`);
    expect(srcEdgeA).toBeDefined();
    expect(srcEdgeA!.source).toBe(`run:${SRC_A}`);
    expect(srcEdgeA!.target).toBe(`fusion:${FUSION_ID}`);
    expect(srcEdgeA!.data?.relation).toBe("fusion_source");

    const srcEdgeB = edges.find((e) => e.id === `fsrc:${FUSION_ID}:${SRC_B}`);
    expect(srcEdgeB).toBeDefined();
    expect(srcEdgeB!.source).toBe(`run:${SRC_B}`);
    expect(srcEdgeB!.target).toBe(`fusion:${FUSION_ID}`);
    expect(srcEdgeB!.data?.relation).toBe("fusion_source");

    // fusion_output edge
    const outEdge = edges.find((e) => e.id === `fout:${FUSION_ID}`);
    expect(outEdge).toBeDefined();
    expect(outEdge!.source).toBe(`fusion:${FUSION_ID}`);
    expect(outEdge!.target).toBe(OUT_ANCHOR);
    expect(outEdge!.data?.relation).toBe("fusion_output");
  });

  // ── FM-2. No anchors → just the fusion_plan node, no edges ─────────────────
  it("no anchors provided → 1 fusion_plan node and 0 edges", () => {
    const fusion = makeFusion({
      fusion_id: FUSION_ID,
      source_run_ids: [SRC_A, SRC_B],
      output_run_id: OUT_RUN,
      regions: [makeFusionRegion({ id: "r1" })],
    });

    const { nodes, edges } = buildFusionModel(fusion, {});

    expect(nodes).toHaveLength(1);
    expect(nodes[0].id).toBe(`fusion:${FUSION_ID}`);
    expect(edges).toHaveLength(0);
  });

  // ── FM-3. output edge absent when output_run_id is null ──────────────────────
  it("output edge is absent when output_run_id is null even if outputAnchorNodeId is provided", () => {
    const fusion = makeFusion({
      fusion_id: FUSION_ID,
      source_run_ids: [SRC_A],
      output_run_id: null,
    });

    const { edges } = buildFusionModel(fusion, {
      baseAnchorNodeId: BASE_ANCHOR,
      sourceAnchorNodeIds: { [SRC_A]: `run:${SRC_A}` },
      outputAnchorNodeId: OUT_ANCHOR,
    });

    // Should have base + 1 source, but NO output edge
    expect(edges.find((e) => e.id === `fout:${FUSION_ID}`)).toBeUndefined();
    expect(edges.map((e) => e.data?.relation)).toContain("fusion_base");
    expect(edges.map((e) => e.data?.relation)).toContain("fusion_source");
    expect(edges).toHaveLength(2);
  });

  // ── FM-4. data carries correct fields for status/region_count/output_run_id ─
  it("node data carries status, region_count, and output_run_id correctly", () => {
    const fusion = makeFusion({
      fusion_id: FUSION_ID,
      status: "running",
      output_run_id: OUT_RUN,
      regions: [makeFusionRegion({ id: "r1" }), makeFusionRegion({ id: "r2" }), makeFusionRegion({ id: "r3" })],
    });

    const { nodes } = buildFusionModel(fusion, {});
    const fpNode = nodes[0];

    expect(fpNode.data.status).toBe("running");
    expect(fpNode.data.region_count).toBe(3);
    expect(fpNode.data.output_run_id).toBe(OUT_RUN);
  });

  // ── FM-5. source_run_ids order is preserved in edges ─────────────────────────
  it("fusion_source edges preserve source_run_ids array order", () => {
    const SRC_1 = "run-src-1111";
    const SRC_2 = "run-src-2222";
    const SRC_3 = "run-src-3333";
    const fusion = makeFusion({
      fusion_id: FUSION_ID,
      source_run_ids: [SRC_1, SRC_2, SRC_3],
    });

    const { edges } = buildFusionModel(fusion, {
      sourceAnchorNodeIds: {
        [SRC_1]: `run:${SRC_1}`,
        [SRC_2]: `run:${SRC_2}`,
        [SRC_3]: `run:${SRC_3}`,
      },
    });

    const sourceEdges = edges.filter((e) => e.data?.relation === "fusion_source");
    expect(sourceEdges).toHaveLength(3);
    expect(sourceEdges[0].id).toBe(`fsrc:${FUSION_ID}:${SRC_1}`);
    expect(sourceEdges[1].id).toBe(`fsrc:${FUSION_ID}:${SRC_2}`);
    expect(sourceEdges[2].id).toBe(`fsrc:${FUSION_ID}:${SRC_3}`);
  });

  // ── FM-6. source anchor missing for one run_id → that edge omitted ───────────
  it("omits fusion_source edge for a run whose anchor is not in sourceAnchorNodeIds", () => {
    const SRC_KNOWN = "run-src-known";
    const SRC_MISSING = "run-src-missing";
    const fusion = makeFusion({
      fusion_id: FUSION_ID,
      source_run_ids: [SRC_KNOWN, SRC_MISSING],
    });

    const { edges } = buildFusionModel(fusion, {
      sourceAnchorNodeIds: { [SRC_KNOWN]: `run:${SRC_KNOWN}` },
      // SRC_MISSING is not in the map
    });

    const sourceEdges = edges.filter((e) => e.data?.relation === "fusion_source");
    expect(sourceEdges).toHaveLength(1);
    expect(sourceEdges[0].id).toBe(`fsrc:${FUSION_ID}:${SRC_KNOWN}`);
  });

  // ── FM-7. deterministic: same input → deeply-equal output ────────────────────
  it("is deterministic: two calls with identical input produce deeply-equal output", () => {
    const fusion = makeFusion({
      fusion_id: FUSION_ID,
      source_run_ids: [SRC_A, SRC_B],
      output_run_id: OUT_RUN,
      regions: [makeFusionRegion({ id: "r1" })],
    });
    const opts = {
      baseAnchorNodeId: BASE_ANCHOR,
      sourceAnchorNodeIds: { [SRC_A]: `run:${SRC_A}`, [SRC_B]: `run:${SRC_B}` },
      outputAnchorNodeId: OUT_ANCHOR,
    };

    const a = buildFusionModel(fusion, opts);
    const b = buildFusionModel(fusion, opts);
    expect(a).toEqual(b);
  });
});
