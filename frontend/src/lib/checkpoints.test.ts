import { describe, it, expect } from "vitest";
import { deriveCheckpoints } from "./checkpoints";
import type { PngShaderResult, RefinementEntry } from "../hooks/usePngShader";

function makeResult(overrides: Partial<PngShaderResult>): PngShaderResult {
  return overrides as PngShaderResult;
}

function makeEntry(overrides: Partial<RefinementEntry> & { iteration: number }): RefinementEntry {
  return {
    compile_glsl: "void mainImage(){}",
    score_after: 0.5,
    improved: false,
    ...overrides,
  } as RefinementEntry;
}

describe("deriveCheckpoints", () => {
  it("returns [] for a null result", () => {
    expect(deriveCheckpoints(null)).toEqual([]);
  });

  it("marks a directed-accepted iteration (accepted=true, improved=false) as accepted", () => {
    const result = makeResult({
      refinement_history: [
        makeEntry({ iteration: 1, accepted: true, improved: false, human_goal_override: "accepted_score_drop" }),
      ],
    });
    const [cp] = deriveCheckpoints(result);
    expect(cp.id).toBe("refinement:iter:1");
    expect(cp.accepted).toBe(true);
  });

  it("marks a rejected iteration (accepted=false, improved=false) as not accepted", () => {
    const result = makeResult({
      refinement_history: [makeEntry({ iteration: 2, accepted: false, improved: false })],
    });
    expect(deriveCheckpoints(result)[0].accepted).toBe(false);
  });

  it("falls back to improved when accepted is absent (legacy entries)", () => {
    const result = makeResult({
      refinement_history: [makeEntry({ iteration: 3, improved: true })],
    });
    expect(deriveCheckpoints(result)[0].accepted).toBe(true);
  });

  it("skips iterations without compiled GLSL", () => {
    const result = makeResult({
      refinement_history: [makeEntry({ iteration: 4, compile_glsl: "  ", accepted: true })],
    });
    expect(deriveCheckpoints(result)).toEqual([]);
  });

  it("emits a final:selected checkpoint when selected_glsl is present", () => {
    const result = makeResult({
      selected_glsl: "void mainImage(){}",
      quality_router: { final_score: 0.83 } as PngShaderResult["quality_router"],
    });
    const final = deriveCheckpoints(result).find((c) => c.id === "final:selected");
    expect(final?.accepted).toBe(true);
    expect(final?.score).toBe(0.83);
  });
});
