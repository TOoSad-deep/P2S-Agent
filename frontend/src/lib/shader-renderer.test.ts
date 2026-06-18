import { describe, it, expect } from "vitest";
import { nextShaderTime, restoreActions } from "./shader-renderer";

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

// Bug 2: WebGL context-loss recovery. On `webglcontextrestored` the renderer
// must decide (a) whether to recompile a shader — only if one was compiled
// before the loss — and (b) whether to resume the rAF loop — only if it was
// actively rendering when the context was lost. restoreActions is that pure
// decision so it can be unit-tested without a real GL context (jsdom has none).
describe("restoreActions", () => {
  it("recompiles and resumes when a shader was compiled and rendering was active", () => {
    expect(restoreActions(true, "void mainImage(){}")).toEqual({
      shouldRecompile: true,
      shouldResume: true,
    });
  });

  it("recompiles but does NOT resume when a shader exists but rendering was paused", () => {
    expect(restoreActions(false, "void mainImage(){}")).toEqual({
      shouldRecompile: true,
      shouldResume: false,
    });
  });

  it("does not recompile when there is no saved fragment source", () => {
    expect(restoreActions(true, null)).toEqual({
      shouldRecompile: false,
      shouldResume: true,
    });
  });

  it("treats an empty-string source as nothing to recompile", () => {
    expect(restoreActions(true, "")).toEqual({
      shouldRecompile: false,
      shouldResume: true,
    });
  });

  it("does nothing when there was no shader and rendering was inactive", () => {
    expect(restoreActions(false, null)).toEqual({
      shouldRecompile: false,
      shouldResume: false,
    });
  });
});
