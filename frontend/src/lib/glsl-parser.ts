// lib/glsl-parser.ts
// Simple GLSL parser for extracting parameters from shader code

export interface ShaderParameter {
  name: string;
  type: 'float' | 'vec2' | 'vec3' | 'vec4' | 'int' | 'bool';
  value: number | number[];
  min?: number;
  max?: number;
  step?: number;
  /** Per-component ranges for vec2/vec3/vec4 (length matches component count). */
  componentRanges?: { min: number; max: number; step: number }[];
  category: 'define' | 'uniform';
}

export interface ParsedShader {
  parameters: ShaderParameter[];
  uniforms: string[];
  defines: Map<string, string>;
}

interface NumericRange { min: number; max: number; step: number }

/** Decide a sensible slider range from the parameter name + its current value.
 *  Centralized so scalar #defines, vec components, and uniforms all behave the
 *  same way. */
function inferRange(name: string, value: number): NumericRange {
  const lower = name.toLowerCase();
  const abs = Math.abs(value);

  // Color components — uniformly [0, 1].
  if (
    /(^|_)color($|_)/.test(lower) ||
    /(^|_)col($|_)/.test(lower) ||
    /(^|_)tint($|_)/.test(lower) ||
    /(^|_)rgb($|_)/.test(lower) ||
    /(^|_)hue($|_)/.test(lower) ||
    /_r$|_g$|_b$/.test(lower) ||
    lower.startsWith('bg_') ||
    lower.startsWith('color_') ||
    lower.endsWith('_color')
  ) {
    return { min: 0, max: 1, step: 0.001 };
  }

  if (lower.includes('opacity') || lower.includes('alpha')) {
    return { min: 0, max: 1, step: 0.01 };
  }
  if (lower.includes('speed') || lower.includes('time')) {
    return { min: 0, max: 5, step: 0.01 };
  }
  if (lower.includes('intensity') || lower.includes('strength') || lower.includes('brightness') || lower.includes('bloom') || lower.includes('glow')) {
    return { min: 0, max: Math.max(2, abs * 2), step: 0.01 };
  }
  if (lower.includes('count') || lower.includes('iter') || lower.includes('samples')) {
    return { min: 1, max: Math.max(100, abs * 2), step: 1 };
  }
  if (lower.includes('falloff') || lower.includes('power') || lower.includes('exponent')) {
    return { min: 0.1, max: Math.max(8, abs * 2), step: 0.01 };
  }
  if (lower.includes('radius') || lower.includes('size') || lower.includes('thickness') || lower.includes('corner') || lower.includes('width') || lower.includes('height')) {
    return { min: 0.001, max: Math.max(1, abs * 2), step: 0.001 };
  }
  if (lower.includes('center') || lower.includes('position') || lower.includes('offset') || lower.includes('_pos') || lower.endsWith('_x') || lower.endsWith('_y')) {
    return { min: 0, max: 1, step: 0.001 };
  }
  if (lower.includes('amount')) {
    return { min: 0, max: 0.5, step: 0.001 };
  }
  if (lower.includes('dir_') || lower.includes('grad_c')) {
    return { min: -1, max: 1, step: 0.01 };
  }
  if (lower.includes('ab_')) {
    return { min: 0.001, max: Math.max(0.5, abs * 2), step: 0.001 };
  }
  if (lower.includes('translate') || lower.includes('scale') || lower.includes('zoom')) {
    return { min: -1, max: Math.max(2, abs * 2), step: 0.01 };
  }
  if (lower.includes('angle') || lower.includes('rotate')) {
    return { min: -Math.PI, max: Math.PI, step: 0.01 };
  }
  if (lower.includes('threshold')) {
    return { min: 0, max: 1, step: 0.001 };
  }

  // Fallback: pivot around the current value. Guard against value=0 producing
  // a zero-width slider by enforcing a minimum span.
  const span = Math.max(abs * 2, 1);
  if (value < 0) {
    return { min: -span, max: span, step: Math.max(span / 200, 0.001) };
  }
  return { min: 0, max: span, step: Math.max(span / 200, 0.001) };
}

function parseScalarLiteral(s: string): number | null {
  const num = parseFloat(s);
  return Number.isFinite(num) ? num : null;
}

/** Parse the inner args of a vec2/vec3/vec4 literal. Returns null on shape
 *  mismatch (e.g. a single arg like vec3(0.5) that should broadcast — we skip
 *  broadcast forms because writing them back is ambiguous). */
function parseVecLiteral(s: string, expectedLen: 2 | 3 | 4): number[] | null {
  const m = s.match(/vec[234]\s*\(\s*([^)]*)\s*\)/);
  if (!m) return null;
  const parts = m[1]
    .split(',')
    .map(p => p.trim())
    .filter(p => p.length > 0)
    .map(parseScalarLiteral);
  if (parts.length !== expectedLen) return null;
  if (parts.some(v => v === null)) return null;
  return parts as number[];
}

function detectVecType(rhs: string): 'vec2' | 'vec3' | 'vec4' | null {
  const m = rhs.match(/^\s*vec([234])\s*\(/);
  if (!m) return null;
  return `vec${m[1]}` as 'vec2' | 'vec3' | 'vec4';
}

// Parse #define constants — both scalar floats and vec2/vec3/vec4 literals.
function parseDefines(code: string): ShaderParameter[] {
  const parameters: ShaderParameter[] = [];
  const defineRegex = /^\s*#define\s+(\w+)\s+(.+)$/gm;
  let match;

  while ((match = defineRegex.exec(code)) !== null) {
    const name = match[1];
    const valueStr = match[2].trim();

    const vecType = detectVecType(valueStr);
    if (vecType) {
      const expected = vecType === 'vec2' ? 2 : vecType === 'vec3' ? 3 : 4;
      const components = parseVecLiteral(valueStr, expected);
      if (!components) continue;
      // Each component gets its own range, named with a virtual suffix so the
      // heuristic still kicks in for things like "CORE_COLOR" (treated as
      // color) or "CENTER" (treated as position).
      const componentRanges = components.map((v) => inferRange(name, v));
      parameters.push({
        name,
        type: vecType,
        value: components,
        componentRanges,
        // Top-level min/max/step are best-effort for code that doesn't read
        // componentRanges — fall back to the first component's range.
        min: componentRanges[0].min,
        max: componentRanges[0].max,
        step: componentRanges[0].step,
        category: 'define',
      });
      continue;
    }

    const numValue = parseScalarLiteral(valueStr);
    if (numValue === null) continue;
    const { min, max, step } = inferRange(name, numValue);
    parameters.push({
      name,
      type: 'float',
      value: numValue,
      min,
      max,
      step,
      category: 'define',
    });
  }

  return parameters;
}

// Parse uniform declarations
function parseUniforms(code: string): ShaderParameter[] {
  const parameters: ShaderParameter[] = [];
  const uniformRegex = /^\s*uniform\s+(\w+)\s+(\w+)\s*;?\s*$/gm;
  let match;

  while ((match = uniformRegex.exec(code)) !== null) {
    const type = match[1] as ShaderParameter['type'];
    const name = match[2];

    let value: number | number[];
    let min = 0;
    let max = 1;
    let step = 0.01;

    switch (type) {
      case 'float':
        value = 0.5;
        break;
      case 'int':
        value = 1;
        min = 0;
        max = 100;
        step = 1;
        break;
      case 'bool':
        value = 1;
        min = 0;
        max = 1;
        step = 1;
        break;
      case 'vec2':
        value = [0.5, 0.5];
        break;
      case 'vec3':
        value = [0.5, 0.5, 0.5];
        break;
      case 'vec4':
        value = [0.5, 0.5, 0.5, 1.0];
        break;
      default:
        continue;
    }

    if (type === 'float' || type === 'int' || type === 'bool') {
      const r = inferRange(name, value as number);
      min = r.min; max = r.max; step = r.step;
    }

    parameters.push({
      name,
      type,
      value,
      min,
      max,
      step,
      category: 'uniform',
    });
  }

  return parameters;
}

// Main parse function
export function parseShader(code: string): ParsedShader {
  const defines = parseDefines(code);
  const uniforms = parseUniforms(code);

  // Extract all uniform names for the renderer
  const uniformNames: string[] = [];
  const uniformRegex = /^\s*uniform\s+\w+\s+(\w+)\s*;?\s*$/gm;
  let match;
  while ((match = uniformRegex.exec(code)) !== null) {
    uniformNames.push(match[1]);
  }

  // Extract all defines
  const definesMap = new Map<string, string>();
  const defineRegex = /^\s*#define\s+(\w+)\s+(.+)$/gm;
  while ((match = defineRegex.exec(code)) !== null) {
    definesMap.set(match[1], match[2].trim());
  }

  return {
    parameters: [...defines, ...uniforms],
    uniforms: uniformNames,
    defines: definesMap,
  };
}

function formatComponent(v: number): string {
  // Keep the number readable but accurate enough for shader output.
  if (Number.isInteger(v)) return v.toFixed(1);
  return parseFloat(v.toFixed(6)).toString();
}

/** Escape a string for literal use inside a RegExp. Without this, a param name
 *  containing regex metacharacters (e.g. ".", "+") would be interpreted as a
 *  pattern and could match — and corrupt — unrelated #define lines. */
function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/** Build the anchored #define-rewrite regex for a given param name. The name is
 *  escaped and bounded by whitespace so only the exact `#define <name> ...`
 *  line is matched (never a name-prefixed sibling like `<name>_factor`). */
function defineLineRegex(name: string): RegExp {
  return new RegExp(`^(\\s*#define\\s+${escapeRegExp(name)}\\s+).+$`, 'm');
}

// Update shader code with a new parameter value.
export function updateShaderParam(
  code: string,
  param: ShaderParameter,
  newValue: number | number[],
): string {
  if (param.category !== 'define') return code;

  if (param.type === 'vec2' || param.type === 'vec3' || param.type === 'vec4') {
    const arr = Array.isArray(newValue) ? newValue : [newValue];
    const literal = `${param.type}(${arr.map(formatComponent).join(', ')})`;
    return code.replace(defineLineRegex(param.name), `$1${literal}`);
  }

  // Scalar `float` defines: emit a GLSL float literal (e.g. 0 → "0.0"), never a
  // bare int. GLSL ES is strictly typed — an int literal in float math throws
  // `'*' : wrong operand types` and breaks shader compilation (BUG-003).
  const valueStr = Array.isArray(newValue)
    ? newValue.map(formatComponent).join(', ')
    : formatComponent(newValue);
  return code.replace(defineLineRegex(param.name), `$1${valueStr}`);
}

// Generate uniform declarations for custom parameters
export function generateUniformDeclarations(params: ShaderParameter[]): string {
  return params
    .filter(p => p.category === 'uniform')
    .map(p => `uniform ${p.type} ${p.name};`)
    .join('\n');
}
