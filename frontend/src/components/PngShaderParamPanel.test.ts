import { describe, it, expect } from "vitest";
import { nextResetBaseline } from "./PngShaderParamPanel";

// The "Reset all parameters" button must restore the values present when a NEW
// shader was loaded — NOT the just-tuned values. Both consumers feed the
// panel's own param edits back into the same `glsl` prop, so the panel must
// distinguish an *external* new shader from an *echo* of its own emission.
//
// nextResetBaseline(incoming, prevBaseline, lastEmitted) decides the next reset
// baseline whenever a new `glsl` prop arrives.
describe("nextResetBaseline", () => {
  it("captures the incoming shader as the baseline on first load (no prior emission)", () => {
    const loaded = "#define A 1.0";
    expect(nextResetBaseline(loaded, "", null)).toBe(loaded);
  });

  it("keeps the pre-tuning baseline when the prop merely echoes the panel's own edit", () => {
    const loaded = "#define A 1.0";
    const edited = "#define A 2.0";
    // 1) shader loads -> baseline = loaded
    let baseline = nextResetBaseline(loaded, "", null);
    expect(baseline).toBe(loaded);
    // 2) panel emits `edited`; parent feeds it straight back as the `glsl` prop.
    //    lastEmitted === incoming, so this is an echo -> baseline must NOT move.
    baseline = nextResetBaseline(edited, baseline, edited);
    expect(baseline).toBe(loaded); // pre-fix bug: this returned `edited`
  });

  it("re-baselines when a genuinely different shader is loaded externally", () => {
    const a = "#define A 1.0";
    const b = "#define B 5.0";
    // Even if a prior edit of `a` is still the last emission, switching to a
    // different external shader must capture a fresh baseline.
    expect(nextResetBaseline(b, a, "#define A 9.0")).toBe(b);
  });

  it("re-baselines to an external shader that coincidentally has the same #define names but different values", () => {
    // Identity-by-#define-names would wrongly treat these as the same shader;
    // echo-tracking correctly re-baselines because it is not our own emission.
    const presetA = "#define HUE 0.10";
    const presetB = "#define HUE 0.90";
    expect(nextResetBaseline(presetB, presetA, presetA)).toBe(presetB);
  });
});
