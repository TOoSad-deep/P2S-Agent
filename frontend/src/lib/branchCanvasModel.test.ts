import { describe, it, expect } from "vitest";
import {
  buildBranchCanvasModel,
  MAX_EXPANDED_RUNS,
  MAX_VISIBLE_CHECKPOINTS_PER_RUN,
  type BuildBranchCanvasInput,
} from "./branchCanvasModel";
import type { BranchTreeNode, CheckpointTimelineEntry } from "../hooks/usePngShader";

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
});
