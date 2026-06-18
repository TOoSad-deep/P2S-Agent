import { describe, it, expect } from "vitest";
import { nextRegionId } from "./RegionMaskEditor";
import type { RegionConstraint } from "../hooks/usePngShader";

function region(id: string): RegionConstraint {
  return {
    id,
    label: id,
    mode: "modify",
    instruction: "",
    geometry_type: "rect",
    geometry: { x: 0, y: 0, w: 0.1, h: 0.1 },
    strength: 0.5,
  };
}

describe("nextRegionId", () => {
  it("starts at region_1 when empty", () => {
    expect(nextRegionId([])).toBe("region_1");
  });

  it("increments past the highest existing suffix", () => {
    expect(nextRegionId([region("region_1"), region("region_2")])).toBe("region_3");
  });

  it("does NOT collide after a middle region is deleted (regression)", () => {
    // [1,2,3] with region_2 deleted → must not reuse region_3.
    const remaining = [region("region_1"), region("region_3")];
    const next = nextRegionId(remaining);
    expect(next).toBe("region_4");
    expect(remaining.some((r) => r.id === next)).toBe(false);
  });

  it("ignores ids that aren't region_<n>", () => {
    expect(nextRegionId([region("region_5_ab12"), region("custom")])).toBe("region_1");
  });
});
