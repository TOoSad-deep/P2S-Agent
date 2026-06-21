import { describe, it, expect } from "vitest";
import { updateShaderParam, type ShaderParameter } from "./glsl-parser";

function floatDefine(name: string, value: number): ShaderParameter {
  return {
    name,
    type: "float",
    value,
    min: 0,
    max: 1,
    step: 0.01,
    category: "define",
  };
}

// Bug 3: updateShaderParam regex must escape the param name and anchor the
// value match so it only rewrites the intended #define line.
describe("updateShaderParam", () => {
  it("updating param `scale` does NOT corrupt a line containing `scale_factor`", () => {
    const code = [
      "#define scale_factor 3.0",
      "#define scale 1.0",
    ].join("\n");
    const param = floatDefine("scale", 1.0);
    const out = updateShaderParam(code, param, 2.0);
    expect(out).toContain("#define scale_factor 3.0");
    expect(out).toContain("#define scale 2");
    // scale_factor must be untouched.
    expect(out).not.toContain("#define scale_factor 2");
  });

  it("replaces a param value exactly once and correctly", () => {
    const code = "#define INTENSITY 0.75";
    const param = floatDefine("INTENSITY", 0.75);
    const out = updateShaderParam(code, param, 1.5);
    expect(out).toBe("#define INTENSITY 1.5");
    // Only one occurrence of the define after replacement.
    const matches = out.match(/#define INTENSITY/g) ?? [];
    expect(matches.length).toBe(1);
  });

  it("does not match when the param name appears as a substring prefix", () => {
    // `radius` should not match `radiusOuter`.
    const code = [
      "#define radiusOuter 5.0",
      "#define radius 2.0",
    ].join("\n");
    const param = floatDefine("radius", 2.0);
    const out = updateShaderParam(code, param, 9.0);
    expect(out).toContain("#define radiusOuter 5.0");
    expect(out).toContain("#define radius 9");
  });

  it("treats param names with regex-special chars literally (no crash, no partial rewrite)", () => {
    // A pathological name containing regex metacharacters must be escaped.
    const code = "#define A.B 1.0\n#define AXB 2.0";
    const param = floatDefine("A.B", 1.0);
    const out = updateShaderParam(code, param, 7.0);
    // Only the literal "A.B" line should change; "AXB" (which `A.B` would match
    // unescaped) must remain intact.
    expect(out).toContain("#define A.B 7");
    expect(out).toContain("#define AXB 2.0");
  });

  it("emits a GLSL float literal for whole-number scalar values (BUG-003)", () => {
    // GLSL ES is strictly typed: `0` is an int literal and breaks float math
    // (`'*' : wrong operand types`). A scalar #define edited to a whole number
    // must stay a float (`0.0`), matching how vec components are written.
    const code = "#define CENTER_X 0.5";
    const param = floatDefine("CENTER_X", 0.5);
    expect(updateShaderParam(code, param, 0)).toBe("#define CENTER_X 0.0");
    expect(updateShaderParam(code, param, 1)).toBe("#define CENTER_X 1.0");
    expect(updateShaderParam(code, param, 2)).toBe("#define CENTER_X 2.0");
  });

  it("preserves fractional scalar values precisely", () => {
    const code = "#define K 0.5";
    const param = floatDefine("K", 0.5);
    expect(updateShaderParam(code, param, 0.25)).toBe("#define K 0.25");
  });

  it("updates a vec define and leaves a name-prefixed sibling intact", () => {
    const code = [
      "#define CENTER_OFFSET vec2(1.0, 2.0)",
      "#define CENTER vec2(0.0, 0.0)",
    ].join("\n");
    const param: ShaderParameter = {
      name: "CENTER",
      type: "vec2",
      value: [0, 0],
      min: 0,
      max: 1,
      step: 0.001,
      category: "define",
    };
    const out = updateShaderParam(code, param, [0.5, 0.25]);
    expect(out).toContain("#define CENTER_OFFSET vec2(1.0, 2.0)");
    expect(out).toContain("#define CENTER vec2(0.5, 0.25)");
  });
});
