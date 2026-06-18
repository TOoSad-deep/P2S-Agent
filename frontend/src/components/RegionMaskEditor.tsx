// RegionMaskEditor.tsx — Rectangle region editor for V4.2 Region/Mask Constraints.
// Presentational + local state only; no fetch, no app state. All mutations are immutable.
import { useRef, useState, useCallback } from "react";
import type { RegionConstraint } from "../hooks/usePngShader";

// ─── Props ─────────────────────────────────────────────────────────────────────

export interface RegionMaskEditorProps {
  imageUrl?: string | null;
  regions: RegionConstraint[];
  onChange: (next: RegionConstraint[]) => void;
  disabled?: boolean;
}

// ─── Drag state ────────────────────────────────────────────────────────────────

interface DragState {
  startX: number;
  startY: number;
  currentX: number;
  currentY: number;
}

// ─── Helpers ───────────────────────────────────────────────────────────────────

/** Clamp v to [lo, hi]. */
function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

/** Normalize a drag from two corner points into a {x,y,w,h} with positive w,h,
 *  clamped to [0,1], ensuring x+w ≤ 1 and y+h ≤ 1. */
function normalizeRect(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
): { x: number; y: number; w: number; h: number } {
  const x = clamp(Math.min(x1, x2), 0, 1);
  const y = clamp(Math.min(y1, y2), 0, 1);
  const w = clamp(Math.max(x1, x2), 0, 1) - x;
  const h = clamp(Math.max(y1, y2), 0, 1) - y;
  return { x, y, w: clamp(w, 0, 1 - x), h: clamp(h, 0, 1 - y) };
}

/** Next collision-free region id. Uses (highest existing `region_<n>` suffix)+1
 *  rather than `regions.length + 1`, which collides after a middle region is
 *  deleted (e.g. delete region_2 of [1,2,3] → length+1 = 3 = existing region_3,
 *  causing duplicate React keys and id-matched updates/deletes hitting two
 *  regions). Ids in other formats are ignored — they can't collide with
 *  `region_<n>` anyway. */
export function nextRegionId(regions: RegionConstraint[]): string {
  let max = 0;
  for (const r of regions) {
    const m = /^region_(\d+)$/.exec(r.id);
    if (m) max = Math.max(max, parseInt(m[1], 10));
  }
  return `region_${max + 1}`;
}

/** Get pixel coords relative to an SVG element from a mouse event. */
function svgRelative(
  e: React.MouseEvent<SVGSVGElement>,
  svg: SVGSVGElement,
): { px: number; py: number } {
  const rect = svg.getBoundingClientRect();
  return {
    px: e.clientX - rect.left,
    py: e.clientY - rect.top,
  };
}

/** Convert pixel coords to normalized [0,1] space given box dimensions. */
function toNorm(px: number, py: number, w: number, h: number): { nx: number; ny: number } {
  return { nx: clamp(px / w, 0, 1), ny: clamp(py / h, 0, 1) };
}

// ─── Region colors ─────────────────────────────────────────────────────────────

const MODE_COLORS: Record<RegionConstraint["mode"], { border: string; bg: string; label: string }> = {
  modify: { border: "#10b981", bg: "rgba(16,185,129,0.12)", label: "#10b981" },
  protect: { border: "#f59e0b", bg: "rgba(245,158,11,0.12)", label: "#f59e0b" },
};

// ─── Component ─────────────────────────────────────────────────────────────────

export default function RegionMaskEditor({
  imageUrl,
  regions,
  onChange,
  disabled,
}: RegionMaskEditorProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [drag, setDrag] = useState<DragState | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  // ── Drag handlers ──────────────────────────────────────────────────────────

  const handleMouseDown = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      if (disabled || !svgRef.current) return;
      // Only start drag on direct SVG background, not on region overlays
      if ((e.target as Element).closest("[data-region]")) return;
      const { px, py } = svgRelative(e, svgRef.current);
      const svg = svgRef.current;
      const w = svg.getBoundingClientRect().width;
      const h = svg.getBoundingClientRect().height;
      const { nx, ny } = toNorm(px, py, w, h);
      setDrag({ startX: nx, startY: ny, currentX: nx, currentY: ny });
      e.preventDefault();
    },
    [disabled],
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      if (!drag || !svgRef.current) return;
      const { px, py } = svgRelative(e, svgRef.current);
      const svg = svgRef.current;
      const bRect = svg.getBoundingClientRect();
      const { nx, ny } = toNorm(px, py, bRect.width, bRect.height);
      setDrag((prev) => (prev ? { ...prev, currentX: nx, currentY: ny } : null));
    },
    [drag],
  );

  const handleMouseUp = useCallback(() => {
    if (!drag) return;
    const rect = normalizeRect(drag.startX, drag.startY, drag.currentX, drag.currentY);
    // Ignore degenerate drags (area < 0.001)
    if (rect.w * rect.h >= 0.001) {
      const id = nextRegionId(regions);
      const n = id.slice("region_".length);
      const newRegion: RegionConstraint = {
        id,
        label: `region ${n}`,
        mode: "modify",
        instruction: "",
        geometry_type: "rect",
        geometry: rect,
        strength: 0.5,
      };
      onChange([...regions, newRegion]);
      setSelectedId(id);
    }
    setDrag(null);
  }, [drag, regions, onChange]);

  const handleMouseLeave = useCallback(() => {
    // Cancel drag if mouse leaves the canvas
    if (drag) setDrag(null);
  }, [drag]);

  // ── Region mutations ───────────────────────────────────────────────────────

  const updateRegion = useCallback(
    (id: string, patch: Partial<RegionConstraint>) => {
      onChange(regions.map((r) => (r.id === id ? { ...r, ...patch } : r)));
    },
    [regions, onChange],
  );

  const deleteRegion = useCallback(
    (id: string) => {
      onChange(regions.filter((r) => r.id !== id));
      if (selectedId === id) setSelectedId(null);
    },
    [regions, onChange, selectedId],
  );

  // ── Drag preview rect in normalized coords ─────────────────────────────────

  const dragRect = drag
    ? normalizeRect(drag.startX, drag.startY, drag.currentX, drag.currentY)
    : null;

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col gap-2">
      {/* Canvas */}
      <div
        className="relative w-full rounded-md overflow-hidden border border-[var(--border-color)]"
        style={{ aspectRatio: "16/9", background: "var(--bg-tertiary)" }}
      >
        {!imageUrl && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <span className="text-[11px] text-[var(--text-muted)]">无预览 no preview</span>
          </div>
        )}
        <svg
          ref={svgRef}
          width="100%"
          height="100%"
          viewBox="0 0 1 1"
          preserveAspectRatio="none"
          className={`absolute inset-0 ${disabled ? "cursor-not-allowed" : "cursor-crosshair"}`}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeave}
        >
          {/* Backdrop image */}
          {imageUrl && (
            <image
              href={imageUrl}
              x="0"
              y="0"
              width="1"
              height="1"
              preserveAspectRatio="xMidYMid slice"
            />
          )}

          {/* Existing regions */}
          {regions.map((r) => {
            const { border, bg, label: labelColor } = MODE_COLORS[r.mode];
            const isSelected = r.id === selectedId;
            return (
              <g
                key={r.id}
                data-region={r.id}
                onClick={(e) => {
                  e.stopPropagation();
                  setSelectedId((prev) => (prev === r.id ? null : r.id));
                }}
                style={{ cursor: disabled ? "not-allowed" : "pointer" }}
              >
                <rect
                  x={r.geometry.x}
                  y={r.geometry.y}
                  width={r.geometry.w}
                  height={r.geometry.h}
                  fill={bg}
                  stroke={border}
                  strokeWidth={isSelected ? 0.006 : 0.003}
                  strokeDasharray={isSelected ? undefined : "0.02 0.01"}
                />
                {/* Label */}
                <text
                  x={r.geometry.x + 0.01}
                  y={r.geometry.y + 0.04}
                  fontSize="0.04"
                  fill={labelColor}
                  style={{ pointerEvents: "none", userSelect: "none" }}
                >
                  {r.label}
                </text>
              </g>
            );
          })}

          {/* Live drag preview */}
          {dragRect && (
            <rect
              x={dragRect.x}
              y={dragRect.y}
              width={dragRect.w}
              height={dragRect.h}
              fill="rgba(99,102,241,0.15)"
              stroke="#6366f1"
              strokeWidth={0.004}
              strokeDasharray="0.02 0.01"
              style={{ pointerEvents: "none" }}
            />
          )}
        </svg>
      </div>

      {/* Region list */}
      {regions.length === 0 ? (
        <p className="text-[10px] text-[var(--text-muted)] text-center">
          {disabled ? "已禁用" : "拖拽以绘制矩形区域 / Drag to draw a region"}
        </p>
      ) : (
        <div className="flex flex-col gap-1.5">
          {regions.map((r) => {
            const { border } = MODE_COLORS[r.mode];
            const isSelected = r.id === selectedId;
            return (
              <div
                key={r.id}
                className="flex flex-col gap-1 px-2 py-1.5 rounded-md border transition-all"
                style={{
                  borderColor: isSelected ? border : "var(--border-color)",
                  background: isSelected ? `${border}10` : "var(--bg-tertiary)",
                  cursor: "pointer",
                }}
                onClick={() => setSelectedId((prev) => (prev === r.id ? null : r.id))}
              >
                {/* Header row */}
                <div className="flex items-center gap-1.5">
                  <span
                    className="text-[10px] font-mono px-1 py-0.5 rounded shrink-0"
                    style={{ background: `${border}22`, color: border }}
                  >
                    {r.mode === "modify" ? "modify" : "protect"}
                  </span>
                  <span className="text-[11px] text-[var(--text-primary)] flex-1 truncate">
                    {r.label}
                  </span>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      deleteRegion(r.id);
                    }}
                    disabled={disabled}
                    title="删除区域 Delete region"
                    className="text-[10px] text-[var(--text-muted)] hover:text-red-400 transition-colors disabled:opacity-40 disabled:cursor-not-allowed shrink-0"
                  >
                    ✕
                  </button>
                </div>

                {/* Expanded editor */}
                {isSelected && (
                  <div
                    className="flex flex-col gap-1.5 pt-1 border-t border-[var(--border-color)]"
                    onClick={(e) => e.stopPropagation()}
                  >
                    {/* Label */}
                    <div className="flex items-center gap-1.5">
                      <span className="text-[10px] text-[var(--text-muted)] w-14 shrink-0">标签 Label</span>
                      <input
                        type="text"
                        value={r.label}
                        onChange={(e) => updateRegion(r.id, { label: e.target.value })}
                        disabled={disabled}
                        className="flex-1 text-[11px] px-1.5 py-0.5 rounded bg-[var(--bg-secondary)] border border-[var(--border-color)] text-[var(--text-primary)] disabled:opacity-40"
                      />
                    </div>

                    {/* Mode toggle */}
                    <div className="flex items-center gap-1.5">
                      <span className="text-[10px] text-[var(--text-muted)] w-14 shrink-0">模式 Mode</span>
                      <div className="flex items-center gap-0.5 bg-[var(--bg-secondary)] rounded-md p-0.5">
                        {(["modify", "protect"] as const).map((m) => (
                          <button
                            key={m}
                            onClick={() => updateRegion(r.id, { mode: m })}
                            disabled={disabled}
                            className={`px-1.5 py-0.5 text-[10px] rounded-md transition-all disabled:opacity-40 ${
                              r.mode === m
                                ? m === "modify"
                                  ? "bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-medium"
                                  : "bg-gradient-to-r from-amber-500 to-amber-600 text-white font-medium"
                                : "text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)]"
                            }`}
                          >
                            {m === "modify" ? "修改 Modify" : "保护 Protect"}
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Instruction */}
                    <div className="flex flex-col gap-0.5">
                      <span className="text-[10px] text-[var(--text-muted)]">指令 Instruction</span>
                      <input
                        type="text"
                        value={r.instruction}
                        onChange={(e) => updateRegion(r.id, { instruction: e.target.value })}
                        disabled={disabled}
                        placeholder="例：增强光晕 / enhance glow"
                        className="w-full text-[11px] px-1.5 py-0.5 rounded bg-[var(--bg-secondary)] border border-[var(--border-color)] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] disabled:opacity-40"
                      />
                    </div>

                    {/* Strength */}
                    <div className="flex flex-col gap-0.5">
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] text-[var(--text-muted)]">强度 Strength</span>
                        <span className="text-[10px] font-mono text-[var(--text-primary)]">
                          {r.strength.toFixed(2)}
                        </span>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={1}
                        step={0.05}
                        value={r.strength}
                        onChange={(e) =>
                          updateRegion(r.id, { strength: parseFloat(e.target.value) })
                        }
                        disabled={disabled}
                        className="w-full accent-emerald-500 disabled:opacity-40"
                      />
                    </div>

                    {/* Geometry readout */}
                    <p className="text-[10px] text-[var(--text-muted)] font-mono">
                      x={r.geometry.x.toFixed(3)} y={r.geometry.y.toFixed(3)}{" "}
                      w={r.geometry.w.toFixed(3)} h={r.geometry.h.toFixed(3)}
                    </p>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
