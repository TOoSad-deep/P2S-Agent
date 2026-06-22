import { describe, it, expect } from "vitest";
import { EXAMPLE_SHADER } from "./playground-example";
import { parseShader } from "./glsl-parser";
import { toShaderToyFragment } from "./shader-format";

describe("EXAMPLE_SHADER", () => {
  it("已是 mainImage 形态（toShaderToyFragment 原样返回）", () => {
    expect(EXAMPLE_SHADER).toContain("void mainImage(");
    expect(toShaderToyFragment(EXAMPLE_SHADER)).toBe(EXAMPLE_SHADER);
  });

  it("暴露可调 #define 参数供参数面板解析", () => {
    const names = parseShader(EXAMPLE_SHADER).parameters.map((p) => p.name);
    expect(names).toContain("SPEED");
    expect(names).toContain("COLOR_A");
    expect(names).toContain("COLOR_B");
    expect(names).toContain("RADIUS");
  });
});
