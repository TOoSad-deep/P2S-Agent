import { describe, it, expect } from "vitest";
import { nextShaderTime } from "./shader-renderer";

// Bug 1: the rAF animate() loop must respect a frozen time so that
// setTime()/__setShaderTime deterministic control survives across frames.
// nextShaderTime is the pure per-frame decision: frozen value wins over clock.
describe("nextShaderTime", () => {
  it("returns the frozen value, ignoring the running clock, when frozen", () => {
    expect(nextShaderTime(2.5, 100.0)).toBe(2.5);
  });

  it("returns the frozen value even when it is 0 (not falsy-skipped)", () => {
    expect(nextShaderTime(0, 42.0)).toBe(0);
  });

  it("returns a negative frozen value verbatim", () => {
    expect(nextShaderTime(-1.25, 9.0)).toBe(-1.25);
  });

  it("returns the clock value when not frozen (null)", () => {
    expect(nextShaderTime(null, 7.0)).toBe(7.0);
  });

  it("tracks the clock across frames when not frozen", () => {
    expect(nextShaderTime(null, 0.016)).toBe(0.016);
    expect(nextShaderTime(null, 0.032)).toBe(0.032);
  });
});
