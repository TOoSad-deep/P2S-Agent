/** Playground "Example" 按钮用的自包含示例 shader。
 *  已是 Shadertoy mainImage 形态（故 toShaderToyFragment 原样透传），并暴露
 *  若干 #define 参数，让参数面板能渲染出滑块/颜色控件。 */
export const EXAMPLE_SHADER = `#define SPEED 1.0
#define COLOR_A vec3(0.10, 0.30, 0.80)
#define COLOR_B vec3(0.90, 0.40, 0.20)
#define RADIUS 0.40

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    float d = length(uv) - RADIUS;
    float pulse = 0.5 + 0.5 * sin(iTime * SPEED);
    vec3 col = mix(COLOR_A, COLOR_B, smoothstep(-0.05, 0.05, d));
    col *= 0.6 + 0.4 * pulse;
    fragColor = vec4(col, 1.0);
}
`;
