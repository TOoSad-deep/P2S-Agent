import { describe, it, expect } from "vitest";
import {
  TERMINAL_STATUSES,
  NON_TERMINAL_STATUSES,
  isTerminalStatus,
  shouldKeepPolling,
  mergeStrategyFromServer,
} from "./runStatus";
import { FALLBACK_DEFAULT_STRATEGY, type StrategyConfig } from "./strategy-presets";

// ─── Bug 1: lifecycle status vocabulary ──────────────────────────────────────

describe("status vocabulary sets", () => {
  it("treats completed/failed/cancelled as terminal", () => {
    expect(TERMINAL_STATUSES.has("completed")).toBe(true);
    expect(TERMINAL_STATUSES.has("failed")).toBe(true);
    expect(TERMINAL_STATUSES.has("cancelled")).toBe(true);
  });

  it("treats running/queued/acquired/pending as non-terminal", () => {
    expect(NON_TERMINAL_STATUSES.has("running")).toBe(true);
    expect(NON_TERMINAL_STATUSES.has("queued")).toBe(true);
    expect(NON_TERMINAL_STATUSES.has("acquired")).toBe(true);
    expect(NON_TERMINAL_STATUSES.has("pending")).toBe(true);
  });

  it("keeps the terminal/non-terminal sets disjoint", () => {
    for (const s of TERMINAL_STATUSES) {
      expect(NON_TERMINAL_STATUSES.has(s)).toBe(false);
    }
  });
});

describe("isTerminalStatus", () => {
  it("is true for terminal statuses", () => {
    expect(isTerminalStatus("completed")).toBe(true);
    expect(isTerminalStatus("failed")).toBe(true);
    expect(isTerminalStatus("cancelled")).toBe(true);
  });

  it("is false for non-terminal statuses", () => {
    expect(isTerminalStatus("running")).toBe(false);
    expect(isTerminalStatus("queued")).toBe(false);
    expect(isTerminalStatus("acquired")).toBe(false);
    expect(isTerminalStatus("pending")).toBe(false);
  });

  it("is case-insensitive and trims whitespace", () => {
    expect(isTerminalStatus("COMPLETED")).toBe(true);
    expect(isTerminalStatus(" Completed ")).toBe(true);
    expect(isTerminalStatus("Running")).toBe(false);
  });
});

describe("shouldKeepPolling", () => {
  // The core Bug-1 regression: queued/acquired children MUST keep polling.
  it("keeps polling for queued and acquired (the bug)", () => {
    expect(shouldKeepPolling("queued")).toBe(true);
    expect(shouldKeepPolling("acquired")).toBe(true);
  });

  it("keeps polling for running and pending", () => {
    expect(shouldKeepPolling("running")).toBe(true);
    expect(shouldKeepPolling("pending")).toBe(true);
  });

  it("stops polling for terminal statuses", () => {
    expect(shouldKeepPolling("completed")).toBe(false);
    expect(shouldKeepPolling("failed")).toBe(false);
    expect(shouldKeepPolling("cancelled")).toBe(false);
  });

  it("is case-insensitive", () => {
    expect(shouldKeepPolling("QUEUED")).toBe(true);
    expect(shouldKeepPolling("Completed")).toBe(false);
  });

  // An unknown status should not freeze the loop forever: default to terminal
  // (stop) so we never poll an undefined lifecycle indefinitely.
  it("treats unknown/empty statuses as terminal (stop)", () => {
    expect(shouldKeepPolling("")).toBe(false);
    expect(shouldKeepPolling(undefined)).toBe(false);
    expect(shouldKeepPolling("weird_unknown_phase")).toBe(false);
    expect(isTerminalStatus("weird_unknown_phase")).toBe(true);
  });
});

// ─── Bug 3: strategy back-sync must not clobber pending local edits ───────────

describe("mergeStrategyFromServer", () => {
  const local: StrategyConfig = {
    ...FALLBACK_DEFAULT_STRATEGY,
    mode: "custom",
    refinement_threshold: 0.9,
    refinement_patience: 3,
    max_iterations: 4,
  };

  it("applies server values for keys with no pending local edit", () => {
    const server = { refinement_threshold: 0.6, refinement_patience: 5 };
    const merged = mergeStrategyFromServer(local, server, new Set());
    expect(merged.refinement_threshold).toBe(0.6);
    expect(merged.refinement_patience).toBe(5);
  });

  it("skips server back-sync for keys with a pending local edit (the bug)", () => {
    const server = { refinement_threshold: 0.6, refinement_patience: 5 };
    const pending = new Set<string>(["refinement_threshold"]);
    const merged = mergeStrategyFromServer(local, server, pending);
    // local edit preserved
    expect(merged.refinement_threshold).toBe(0.9);
    // non-pending key still back-synced
    expect(merged.refinement_patience).toBe(5);
  });

  it("preserves all pending keys at once", () => {
    const server = { refinement_threshold: 0.6, refinement_patience: 5, max_iterations: 8 };
    const pending = new Set<string>(["refinement_threshold", "max_iterations"]);
    const merged = mergeStrategyFromServer(local, server, pending);
    expect(merged.refinement_threshold).toBe(0.9);
    expect(merged.max_iterations).toBe(4);
    expect(merged.refinement_patience).toBe(5);
  });

  it("returns a new object and does not mutate local", () => {
    const server = { refinement_threshold: 0.6 };
    const merged = mergeStrategyFromServer(local, server, new Set());
    expect(merged).not.toBe(local);
    expect(local.refinement_threshold).toBe(0.9);
  });

  it("ignores undefined/missing server (returns local copy)", () => {
    const merged = mergeStrategyFromServer(local, undefined, new Set());
    expect(merged.refinement_threshold).toBe(0.9);
    expect(merged).not.toBe(local);
  });

  it("applies non-pending server keys while skipping pending ones", () => {
    // server carries multiple keys; merge applies them like the original code
    // did, but a pending key present in server is still skipped.
    const server = { refinement_threshold: 0.6, max_refinement_iterations: 12 };
    const merged = mergeStrategyFromServer(local, server, new Set(["refinement_threshold"]));
    expect(merged.refinement_threshold).toBe(0.9);
    expect(merged.max_refinement_iterations).toBe(12);
  });
});
