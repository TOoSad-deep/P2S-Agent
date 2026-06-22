import { useState, useEffect, useMemo, useCallback, useRef } from "react";
import { SlidersHorizontal, RotateCcw, ChevronDown, ChevronRight } from "lucide-react";
import { parseShader, updateShaderParam, type ShaderParameter } from "../lib/glsl-parser";

interface PngShaderParamPanelProps {
  glsl: string | null;
  onGlslChange: (glsl: string) => void;
}

/** Decide the reset baseline when a new `glsl` prop arrives.
 *
 *  Both consumers feed the panel's own param edits back through the same `glsl`
 *  prop, so an incoming value that equals our last emission is just an *echo* of
 *  a live edit — the reset baseline must stay put. Any other value is a
 *  genuinely external (new/different) shader and becomes the fresh baseline. */
export function nextResetBaseline(
  incomingGlsl: string,
  prevBaseline: string,
  lastEmitted: string | null,
): string {
  return incomingGlsl === lastEmitted ? prevBaseline : incomingGlsl;
}

interface ParamGroup {
  label: string;
  /** Pure scalar / scalar-vec sliders. */
  scalarParams: ShaderParameter[];
  /** vec3 defines treated as colors (#define X_COLOR vec3(...)). */
  vec3Colors: ShaderParameter[];
  /** vec2 defines treated as paired XY positions / sizes. */
  vec2Pairs: ShaderParameter[];
  /** Generic vec4 defines (rare — falls back to a row of sliders). */
  vec4Generic: ShaderParameter[];
  /** Triplets recovered from L*_*_r/_g/_b naming convention (deterministic
   *  compiler output). */
  colorTriplets: ColorTriplet[];
}

interface ColorTriplet {
  label: string;
  r: ShaderParameter;
  g: ShaderParameter;
  b: ShaderParameter;
}

const COLOR_NAME_RE = /(^|_)(color|col|tint|rgb|hue)($|_)|_color$|^color_|^bg_color$/i;

function isColorVec3(p: ShaderParameter): boolean {
  return p.type === "vec3" && COLOR_NAME_RE.test(p.name);
}

function isPositionVec2(p: ShaderParameter): boolean {
  return p.type === "vec2";
}

/** Bucket params by the leading identifier (e.g. CORE_FALLOFF / CORE_RADIUS →
 *  "CORE"). Falls back to "Misc" when the name has no underscore. */
function bucketKey(name: string): string {
  // Deterministic compiler naming has its own shape.
  const layerMatch = name.match(/^L(\d+)_/);
  if (layerMatch) return `Layer ${layerMatch[1]}`;
  if (name.startsWith("bg_") || name === "bg_color") return "Background";

  // LLM-style: uppercase prefixed names like CORE_FALLOFF, COLOR_BACKGROUND.
  // Use the first underscored segment as the bucket; the "COLOR_*" shape is
  // common enough to deserve its own bucket.
  const segs = name.split("_");
  if (segs.length === 1) return "Misc";
  // COLOR_X / COLOR_Y / COLOR_Z → "Colors" bucket
  if (segs[0].toLowerCase() === "color") return "Colors";
  return segs[0];
}

function extractGroups(params: ShaderParameter[]): ParamGroup[] {
  const defines = params.filter(p => p.category === "define");

  // Pre-pass: recover *_r/*_g/*_b triplets so they don't pollute the bucket
  // count. This covers deterministic compiler output (lowercase) and LLM
  // Shadertoy output such as COLOR_CENTER_R/G/B (uppercase).
  const tripletRecovered = new Set<string>();
  const tripletByBucket = new Map<string, ColorTriplet[]>();

  const colorChannelCandidates = defines.filter(p => p.type === "float" && /_r$/i.test(p.name));
  for (const rp of colorChannelCandidates) {
    const base = rp.name.slice(0, -2);
    const gp = defines.find(p => p.type === "float" && p.name.toLowerCase() === `${base}_g`.toLowerCase());
    const bp = defines.find(p => p.type === "float" && p.name.toLowerCase() === `${base}_b`.toLowerCase());
    if (!gp || !bp) continue;
    const bucket = bucketKey(rp.name);
    const labelSegs = base.split("_").filter(s => !/^L\d+$/.test(s));
    const label = labelSegs.length > 0 ? labelSegs.join(" ") : "color";
    const arr = tripletByBucket.get(bucket) ?? [];
    arr.push({ label, r: rp, g: gp, b: bp });
    tripletByBucket.set(bucket, arr);
    tripletRecovered.add(rp.name);
    tripletRecovered.add(gp.name);
    tripletRecovered.add(bp.name);
  }

  const buckets = new Map<string, ParamGroup>();
  function ensureBucket(label: string): ParamGroup {
    let g = buckets.get(label);
    if (!g) {
      g = {
        label,
        scalarParams: [],
        vec3Colors: [],
        vec2Pairs: [],
        vec4Generic: [],
        colorTriplets: [],
      };
      buckets.set(label, g);
    }
    return g;
  }

  // Seed with recovered triplets.
  for (const [bucket, triplets] of tripletByBucket) {
    ensureBucket(bucket).colorTriplets.push(...triplets);
  }

  for (const p of defines) {
    if (tripletRecovered.has(p.name)) continue;
    const g = ensureBucket(bucketKey(p.name));
    if (p.type === "vec3" && isColorVec3(p)) {
      g.vec3Colors.push(p);
    } else if (p.type === "vec2" && isPositionVec2(p)) {
      g.vec2Pairs.push(p);
    } else if (p.type === "vec4") {
      g.vec4Generic.push(p);
    } else {
      g.scalarParams.push(p);
    }
  }

  // Sort: Background → Layer N (numeric) → known LLM groups (Colors first) → others
  // alphabetical → Misc last.
  const groups = [...buckets.values()];
  groups.sort((a, b) => {
    const rank = (label: string): [number, number, string] => {
      if (label === "Background") return [0, 0, label];
      const layer = label.match(/^Layer (\d+)$/);
      if (layer) return [1, parseInt(layer[1], 10), label];
      if (label === "Colors") return [2, 0, label];
      if (label === "Misc") return [9, 0, label];
      return [3, 0, label.toLowerCase()];
    };
    const ra = rank(a.label);
    const rb = rank(b.label);
    if (ra[0] !== rb[0]) return ra[0] - rb[0];
    if (ra[1] !== rb[1]) return ra[1] - rb[1];
    return ra[2] < rb[2] ? -1 : ra[2] > rb[2] ? 1 : 0;
  });

  return groups;
}

function formatParamName(name: string): string {
  return name
    .replace(/^L\d+_/, "")
    .replace(/^bg_/, "")
    .replace(/_/g, " ");
}

function rgbToHex(r: number, g: number, b: number): string {
  const toHex = (v: number) => Math.round(Math.max(0, Math.min(1, v)) * 255).toString(16).padStart(2, "0");
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [
    parseInt(h.slice(0, 2), 16) / 255,
    parseInt(h.slice(2, 4), 16) / 255,
    parseInt(h.slice(4, 6), 16) / 255,
  ];
}

function readCurrentScalar(code: string, name: string, fallback: number): number {
  const match = code.match(new RegExp(`#define\\s+${name}\\s+([\\d.eE+-]+)`));
  if (!match) return fallback;
  const num = parseFloat(match[1]);
  return Number.isFinite(num) ? num : fallback;
}

function readCurrentVec(code: string, name: string, fallback: number[]): number[] {
  const match = code.match(new RegExp(`#define\\s+${name}\\s+vec[234]\\s*\\(([^)]*)\\)`));
  if (!match) return fallback;
  const parts = match[1].split(",").map(s => parseFloat(s.trim()));
  if (parts.length !== fallback.length || parts.some(v => !Number.isFinite(v))) return fallback;
  return parts;
}

function TripletColorControl({ triplet, currentGlsl, onGlslChange }: {
  triplet: ColorTriplet;
  currentGlsl: string;
  onGlslChange: (glsl: string) => void;
}) {
  const rVal = readCurrentScalar(currentGlsl, triplet.r.name, triplet.r.value as number);
  const gVal = readCurrentScalar(currentGlsl, triplet.g.name, triplet.g.value as number);
  const bVal = readCurrentScalar(currentGlsl, triplet.b.name, triplet.b.value as number);
  const hex = rgbToHex(rVal, gVal, bVal);

  const handleColorChange = useCallback((newHex: string) => {
    const [r, g, b] = hexToRgb(newHex);
    let code = updateShaderParam(currentGlsl, triplet.r, r);
    code = updateShaderParam(code, triplet.g, g);
    code = updateShaderParam(code, triplet.b, b);
    onGlslChange(code);
  }, [currentGlsl, triplet, onGlslChange]);

  return (
    <div className="flex items-center gap-2 py-1">
      <span className="text-xs text-[var(--text-muted)] flex-1 capitalize truncate" title={triplet.label}>
        {formatParamName(triplet.label)}
      </span>
      <div className="flex items-center gap-1">
        <input
          type="color"
          value={hex}
          onChange={(e) => handleColorChange(e.target.value)}
          className="w-6 h-6 rounded border border-[var(--border-color)] cursor-pointer p-0"
        />
        <span className="text-xs font-mono text-[var(--text-secondary)]">{hex}</span>
      </div>
    </div>
  );
}

function Vec3ColorControl({ param, currentGlsl, onGlslChange }: {
  param: ShaderParameter;
  currentGlsl: string;
  onGlslChange: (glsl: string) => void;
}) {
  const fallback = (param.value as number[]) ?? [0.5, 0.5, 0.5];
  const [r, g, b] = readCurrentVec(currentGlsl, param.name, fallback);
  const hex = rgbToHex(r, g, b);

  const handleColorChange = useCallback((newHex: string) => {
    const next = hexToRgb(newHex);
    onGlslChange(updateShaderParam(currentGlsl, param, next));
  }, [currentGlsl, param, onGlslChange]);

  return (
    <div className="flex items-center gap-2 py-1">
      <span className="text-xs text-[var(--text-muted)] flex-1 capitalize truncate" title={param.name}>
        {formatParamName(param.name)}
      </span>
      <div className="flex items-center gap-1">
        <input
          type="color"
          value={hex}
          onChange={(e) => handleColorChange(e.target.value)}
          className="w-6 h-6 rounded border border-[var(--border-color)] cursor-pointer p-0"
        />
        <span className="text-xs font-mono text-[var(--text-secondary)]">{hex}</span>
      </div>
    </div>
  );
}

function Vec2PairControl({ param, currentGlsl, onGlslChange }: {
  param: ShaderParameter;
  currentGlsl: string;
  onGlslChange: (glsl: string) => void;
}) {
  const fallback = (param.value as number[]) ?? [0.5, 0.5];
  const current = readCurrentVec(currentGlsl, param.name, fallback);
  const ranges = param.componentRanges ?? [
    { min: 0, max: 1, step: 0.001 },
    { min: 0, max: 1, step: 0.001 },
  ];

  const handleChange = useCallback((idx: number, v: number) => {
    const next = [...current];
    next[idx] = v;
    onGlslChange(updateShaderParam(currentGlsl, param, next));
  }, [current, currentGlsl, param, onGlslChange]);

  return (
    <div className="py-1">
      <div className="text-xs text-[var(--text-muted)] mb-0.5 capitalize truncate" title={param.name}>
        {formatParamName(param.name)}
      </div>
      {(["x", "y"] as const).map((axis, idx) => (
        <div key={axis} className="flex items-center gap-2 ml-2">
          <span className="text-[10px] text-[var(--text-muted)] w-3">{axis}</span>
          <input
            type="range"
            min={ranges[idx].min}
            max={ranges[idx].max}
            step={ranges[idx].step}
            value={current[idx]}
            onChange={(e) => handleChange(idx, parseFloat(e.target.value))}
            className="flex-1 h-1 bg-[var(--border-color)] rounded-lg appearance-none cursor-pointer"
          />
          <input
            type="number"
            value={current[idx].toFixed(3)}
            onChange={(e) => { const v = parseFloat(e.target.value); if (Number.isFinite(v)) handleChange(idx, v); }}
            className="w-16 px-1 py-0.5 bg-[var(--bg-tertiary)] border border-[var(--border-color)] rounded text-xs font-mono text-[var(--text-primary)] text-right focus:border-[var(--accent-primary)] focus:outline-none"
            step={ranges[idx].step}
          />
        </div>
      ))}
    </div>
  );
}

function Vec4GenericControl({ param, currentGlsl, onGlslChange }: {
  param: ShaderParameter;
  currentGlsl: string;
  onGlslChange: (glsl: string) => void;
}) {
  const fallback = (param.value as number[]) ?? [0, 0, 0, 1];
  const current = readCurrentVec(currentGlsl, param.name, fallback);
  const ranges = param.componentRanges ?? current.map(() => ({ min: 0, max: 1, step: 0.001 }));

  const handleChange = useCallback((idx: number, v: number) => {
    const next = [...current];
    next[idx] = v;
    onGlslChange(updateShaderParam(currentGlsl, param, next));
  }, [current, currentGlsl, param, onGlslChange]);

  const axes = ["x", "y", "z", "w"];
  return (
    <div className="py-1">
      <div className="text-xs text-[var(--text-muted)] mb-0.5 capitalize truncate" title={param.name}>
        {formatParamName(param.name)}
      </div>
      {current.map((v, idx) => (
        <div key={idx} className="flex items-center gap-2 ml-2">
          <span className="text-[10px] text-[var(--text-muted)] w-3">{axes[idx]}</span>
          <input
            type="range"
            min={ranges[idx].min}
            max={ranges[idx].max}
            step={ranges[idx].step}
            value={v}
            onChange={(e) => handleChange(idx, parseFloat(e.target.value))}
            className="flex-1 h-1 bg-[var(--border-color)] rounded-lg appearance-none cursor-pointer"
          />
          <input
            type="number"
            value={v.toFixed(3)}
            onChange={(e) => { const v = parseFloat(e.target.value); if (Number.isFinite(v)) handleChange(idx, v); }}
            className="w-16 px-1 py-0.5 bg-[var(--bg-tertiary)] border border-[var(--border-color)] rounded text-xs font-mono text-[var(--text-primary)] text-right focus:border-[var(--accent-primary)] focus:outline-none"
            step={ranges[idx].step}
          />
        </div>
      ))}
    </div>
  );
}

function SliderControl({ param, currentGlsl, onGlslChange }: {
  param: ShaderParameter;
  currentGlsl: string;
  onGlslChange: (glsl: string) => void;
}) {
  const currentVal = readCurrentScalar(currentGlsl, param.name, typeof param.value === "number" ? param.value : 0);

  const handleChange = useCallback((newVal: number) => {
    onGlslChange(updateShaderParam(currentGlsl, param, newVal));
  }, [currentGlsl, param, onGlslChange]);

  const min = param.min ?? 0;
  const max = param.max ?? 1;
  const step = param.step ?? 0.001;

  return (
    <div className="py-1">
      <div className="flex items-center gap-2">
        <span className="text-xs text-[var(--text-muted)] w-24 truncate capitalize" title={param.name}>
          {formatParamName(param.name)}
        </span>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={currentVal}
          onChange={(e) => handleChange(parseFloat(e.target.value))}
          className="flex-1 h-1 bg-[var(--border-color)] rounded-lg appearance-none cursor-pointer"
        />
        <input
          type="number"
          value={currentVal.toFixed(3)}
          onChange={(e) => { const v = parseFloat(e.target.value); if (Number.isFinite(v)) handleChange(v); }}
          className="w-16 px-1 py-0.5 bg-[var(--bg-tertiary)] border border-[var(--border-color)]
                     rounded text-xs font-mono text-[var(--text-primary)] text-right
                     focus:border-[var(--accent-primary)] focus:outline-none"
          step={step}
        />
      </div>
    </div>
  );
}

function groupCount(group: ParamGroup): number {
  return group.scalarParams.length + group.vec3Colors.length + group.vec2Pairs.length + group.vec4Generic.length + group.colorTriplets.length;
}

function GroupSection({ group, currentGlsl, onGlslChange }: {
  group: ParamGroup;
  currentGlsl: string;
  onGlslChange: (glsl: string) => void;
}) {
  const [expanded, setExpanded] = useState(true);
  const count = groupCount(group);
  if (count === 0) return null;

  return (
    <div className="border-b border-[var(--border-color)] last:border-b-0">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1 w-full px-2 py-1.5 text-xs font-medium text-[var(--text-secondary)]
                   hover:text-[var(--text-primary)] transition-colors"
      >
        {expanded ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        {group.label}
        <span className="text-[var(--text-muted)] ml-auto">{count}</span>
      </button>
      {expanded && (
        <div className="px-2 pb-2">
          {group.colorTriplets.map(t => (
            <TripletColorControl key={t.r.name} triplet={t} currentGlsl={currentGlsl} onGlslChange={onGlslChange} />
          ))}
          {group.vec3Colors.map(p => (
            <Vec3ColorControl key={p.name} param={p} currentGlsl={currentGlsl} onGlslChange={onGlslChange} />
          ))}
          {group.vec2Pairs.map(p => (
            <Vec2PairControl key={p.name} param={p} currentGlsl={currentGlsl} onGlslChange={onGlslChange} />
          ))}
          {group.vec4Generic.map(p => (
            <Vec4GenericControl key={p.name} param={p} currentGlsl={currentGlsl} onGlslChange={onGlslChange} />
          ))}
          {group.scalarParams.map(p => (
            <SliderControl key={p.name} param={p} currentGlsl={currentGlsl} onGlslChange={onGlslChange} />
          ))}
        </div>
      )}
    </div>
  );
}

export default function PngShaderParamPanel({ glsl, onGlslChange }: PngShaderParamPanelProps) {
  const [currentGlsl, setCurrentGlsl] = useState(glsl || "");
  const [originalGlsl, setOriginalGlsl] = useState(glsl || "");
  // The last GLSL this panel pushed out via onGlslChange. Parents echo param
  // edits straight back into the `glsl` prop, so we use this to tell our own
  // edit-echo (keep the reset baseline) from an externally-loaded new shader
  // (re-capture the baseline).
  const lastEmittedRef = useRef<string | null>(null);

  useEffect(() => {
    if (glsl) {
      // Snapshot the ref now: the setOriginalGlsl updater runs lazily during the
      // next render, by which point lastEmittedRef has already been advanced to
      // `glsl` — reading it inside the updater would make every change look like
      // an echo and freeze the baseline.
      const lastEmitted = lastEmittedRef.current;
      setCurrentGlsl(glsl);
      setOriginalGlsl(prev => nextResetBaseline(glsl, prev, lastEmitted));
      lastEmittedRef.current = glsl;
    }
  }, [glsl]);

  const parsed = useMemo(() => (currentGlsl ? parseShader(currentGlsl) : null), [currentGlsl]);
  const groups = useMemo(() => (parsed ? extractGroups(parsed.parameters) : []), [parsed]);

  const handleGlslChange = useCallback((newGlsl: string) => {
    lastEmittedRef.current = newGlsl;
    setCurrentGlsl(newGlsl);
    onGlslChange(newGlsl);
  }, [onGlslChange]);

  const handleReset = useCallback(() => {
    lastEmittedRef.current = originalGlsl;
    setCurrentGlsl(originalGlsl);
    onGlslChange(originalGlsl);
  }, [originalGlsl, onGlslChange]);

  if (!glsl) {
    return (
      <div className="flex items-center justify-center h-full text-sm text-[var(--text-muted)]">
        No shader selected
      </div>
    );
  }

  const totalParams = groups.reduce((sum, g) => sum + groupCount(g), 0);

  return (
    <div className="flex flex-col h-full bg-[var(--bg-secondary)] rounded-lg border border-[var(--border-color)] overflow-hidden">
      <div className="flex items-center gap-2 px-3 py-2 border-b border-[var(--border-color)]">
        <SlidersHorizontal size={14} className="text-[var(--accent-primary)]" />
        <span className="text-sm font-medium text-[var(--text-primary)]">Parameters</span>
        <span className="text-xs text-[var(--text-muted)] ml-1">({totalParams})</span>
        <button
          onClick={handleReset}
          className="ml-auto p-1 rounded hover:bg-[var(--bg-tertiary)] text-[var(--text-muted)]
                     hover:text-[var(--text-primary)] transition-colors"
          title="Reset all parameters"
        >
          <RotateCcw size={12} />
        </button>
      </div>
      <div className="flex-1 overflow-y-auto">
        {groups.length === 0 ? (
          <div className="flex items-center justify-center h-full text-xs text-[var(--text-muted)]">
            No tunable parameters found
          </div>
        ) : (
          groups.map((g) => (
            <GroupSection key={g.label} group={g} currentGlsl={currentGlsl} onGlslChange={handleGlslChange} />
          ))
        )}
      </div>
    </div>
  );
}
