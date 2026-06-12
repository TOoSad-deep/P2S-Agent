export function toShaderToyFragment(glsl: string): string {
  if (!glsl.trimStart().startsWith("#version 300 es")) {
    return glsl;
  }

  return glsl
    .replace(/^\s*#version\s+300\s+es\s*\n/, "")
    .replace(/^\s*precision\s+\w+\s+float\s*;\s*\n/m, "")
    .replace(/^\s*out\s+vec4\s+fragColor\s*;\s*\n/m, "")
    .replace(/^\s*uniform\s+vec2\s+iResolution\s*;\s*\n/m, "")
    .replace(/^\s*uniform\s+float\s+iTime\s*;\s*\n/m, "")
    .replace(/\bvoid\s+main\s*\(\s*\)\s*\{/, "void mainImage(out vec4 fragColor, in vec2 fragCoord) {")
    .replace(/\bgl_FragCoord\.xy\b/g, "fragCoord");
}
