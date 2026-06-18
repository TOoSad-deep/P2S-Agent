// FusionBuilderPanel.tsx — V4.5 Local Fusion builder panel.
// Presentational only; all state owned by parent (BranchCanvasWorkspace).
import { useCallback, useEffect, useState } from "react";
import { Plus, Trash2, Layers, Play, Image } from "lucide-react";
import type { FusionRegion, FusionStatus } from "../hooks/usePngShader";

// ─── FallbackImg ────────────────────────────────────────────────────────────────
// <img> that swaps to a fallback on load error and — crucially — retries when
// `src` changes. The previous inline `onError → style.display="none"` latched the
// element hidden: React doesn't reset an imperatively-set inline style when only
// the src changes, so picking a different (valid) image after a failed one left
// it blank. Resetting the error state on src change fixes that recovery.
function FallbackImg({
  src,
  alt,
  className,
  fallback = null,
}: {
  src: string;
  alt: string;
  className?: string;
  fallback?: React.ReactNode;
}) {
  const [errored, setErrored] = useState(false);
  useEffect(() => {
    setErrored(false);
  }, [src]);
  if (errored) return <>{fallback}</>;
  return <img src={src} alt={alt} className={className} onError={() => setErrored(true)} />;
}

// ─── Public types ──────────────────────────────────────────────────────────────

export interface FusionDraft {
  base_run_id: string | null;
  draw_session_id?: string | null;
  feedback: string;
  regions: FusionRegion[];
}

export interface FusionCandidate {
  run_id: string;
  label: string;
  thumbnail_url?: string | null;
}

interface FusionBuilderPanelProps {
  draft: FusionDraft;
  candidates: FusionCandidate[];
  fusion: FusionStatus | null;
  baseImageUrl?: string | null;
  onChange: (next: FusionDraft) => void;
  onCreate: () => void;
  onComposite: () => void;
  onRun: () => void;
  onPreviewRun: (runId: string) => void;
  disabled?: boolean;
}

// ─── Helpers ───────────────────────────────────────────────────────────────────

function makeRegionId(n: number): string {
  return `region_${n}_${Date.now().toString(36)}`;
}

function defaultRegion(candidates: FusionCandidate[], n: number): FusionRegion {
  return {
    id: makeRegionId(n),
    label: `Region ${n + 1}`,
    source_run_id: candidates[0]?.run_id ?? "",
    instruction: "",
    geometry_type: "rect",
    geometry: { x: 0.25, y: 0.25, w: 0.5, h: 0.5 },
    strength: 0.5,
    blend_mode: "soft",
    feather: 0.08,
  };
}

const BLEND_MODES: { value: FusionRegion["blend_mode"]; label: string }[] = [
  { value: "soft", label: "软融合 Soft" },
  { value: "replace_target", label: "替换 Replace" },
  { value: "protect_base", label: "保护底图 Protect base" },
];

// ─── RegionRow sub-component ──────────────────────────────────────────────────

interface RegionRowProps {
  region: FusionRegion;
  candidates: FusionCandidate[];
  onChange: (next: FusionRegion) => void;
  onDelete: () => void;
  disabled?: boolean;
}

function RegionRow({ region, candidates, onChange, onDelete, disabled }: RegionRowProps) {
  const upd = useCallback(
    <K extends keyof FusionRegion>(key: K, value: FusionRegion[K]) =>
      onChange({ ...region, [key]: value }),
    [region, onChange],
  );

  return (
    <div className="flex flex-col gap-1.5 p-2 rounded-md border border-[var(--border-color)] bg-[var(--bg-tertiary)]">
      {/* Source dropdown */}
      <div className="flex items-center gap-1.5">
        <span className="text-[10px] text-[var(--text-muted)] shrink-0 w-12">源 Source</span>
        <select
          value={region.source_run_id}
          onChange={(e) => upd("source_run_id", e.target.value)}
          disabled={disabled || candidates.length === 0}
          className="flex-1 text-[11px] px-1.5 py-0.5 rounded bg-[var(--bg-secondary)] border border-[var(--border-color)] text-[var(--text-primary)] disabled:opacity-40"
        >
          {candidates.length === 0 && (
            <option value="">无候选 / No candidates</option>
          )}
          {candidates.map((c) => (
            <option key={c.run_id} value={c.run_id}>
              {c.label} ({c.run_id.slice(-6)})
            </option>
          ))}
        </select>
        <button
          onClick={onDelete}
          disabled={disabled}
          title="删除区域 Delete region"
          className="p-1 rounded text-[var(--text-muted)] hover:text-red-400 transition-all disabled:opacity-40 shrink-0"
        >
          <Trash2 className="w-3 h-3" />
        </button>
      </div>

      {/* Label */}
      <div className="flex items-center gap-1.5">
        <span className="text-[10px] text-[var(--text-muted)] shrink-0 w-12">标签 Label</span>
        <input
          type="text"
          value={region.label}
          onChange={(e) => upd("label", e.target.value)}
          disabled={disabled}
          placeholder="e.g. 天空 Sky"
          className="flex-1 text-[11px] px-1.5 py-0.5 rounded bg-[var(--bg-secondary)] border border-[var(--border-color)] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] disabled:opacity-40"
        />
      </div>

      {/* Instruction */}
      <div className="flex items-center gap-1.5">
        <span className="text-[10px] text-[var(--text-muted)] shrink-0 w-12">指令 Instr.</span>
        <input
          type="text"
          value={region.instruction}
          onChange={(e) => upd("instruction", e.target.value)}
          disabled={disabled}
          placeholder="e.g. 保持云彩纹理"
          className="flex-1 text-[11px] px-1.5 py-0.5 rounded bg-[var(--bg-secondary)] border border-[var(--border-color)] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] disabled:opacity-40"
        />
      </div>

      {/* Blend mode */}
      <div className="flex items-center gap-1.5">
        <span className="text-[10px] text-[var(--text-muted)] shrink-0 w-12">混合 Blend</span>
        <select
          value={region.blend_mode}
          onChange={(e) => upd("blend_mode", e.target.value as FusionRegion["blend_mode"])}
          disabled={disabled}
          className="flex-1 text-[11px] px-1.5 py-0.5 rounded bg-[var(--bg-secondary)] border border-[var(--border-color)] text-[var(--text-primary)] disabled:opacity-40"
        >
          {BLEND_MODES.map(({ value, label }) => (
            <option key={value} value={value}>{label}</option>
          ))}
        </select>
      </div>

      {/* Strength slider */}
      <div className="flex items-center gap-1.5">
        <span className="text-[10px] text-[var(--text-muted)] shrink-0 w-12">强度 Str.</span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={region.strength}
          onChange={(e) => upd("strength", parseFloat(e.target.value))}
          disabled={disabled}
          className="flex-1 accent-emerald-500 disabled:opacity-40"
        />
        <span className="text-[10px] font-mono text-[var(--text-muted)] shrink-0 w-8 text-right">
          {region.strength.toFixed(2)}
        </span>
      </div>

      {/* Geometry display (normalized rect — read-only summary) */}
      <p className="text-[10px] text-[var(--text-muted)] font-mono">
        rect ({region.geometry.x.toFixed(2)},{region.geometry.y.toFixed(2)})&nbsp;
        {region.geometry.w.toFixed(2)}×{region.geometry.h.toFixed(2)}
      </p>
    </div>
  );
}

// ─── Main component ────────────────────────────────────────────────────────────

export default function FusionBuilderPanel({
  draft,
  candidates,
  fusion,
  baseImageUrl,
  onChange,
  onCreate,
  onComposite,
  onRun,
  onPreviewRun,
  disabled = false,
}: FusionBuilderPanelProps) {
  const canCreate =
    !disabled &&
    !!draft.base_run_id &&
    draft.regions.length > 0 &&
    draft.regions.some((r) => r.source_run_id !== "");

  const handleAddRegion = useCallback(() => {
    onChange({
      ...draft,
      regions: [...draft.regions, defaultRegion(candidates, draft.regions.length)],
    });
  }, [draft, candidates, onChange]);

  const handleRegionChange = useCallback(
    (idx: number, next: FusionRegion) => {
      const regions = draft.regions.map((r, i) => (i === idx ? next : r));
      onChange({ ...draft, regions });
    },
    [draft, onChange],
  );

  const handleDeleteRegion = useCallback(
    (idx: number) => {
      onChange({ ...draft, regions: draft.regions.filter((_, i) => i !== idx) });
    },
    [draft, onChange],
  );

  return (
    <div className="flex flex-col gap-2.5">
      {/* Header */}
      <div className="flex items-center gap-2">
        <Layers className="w-4 h-4 text-[var(--accent-primary)] shrink-0" />
        <p className="text-xs font-medium text-[var(--text-primary)]">
          融合构建器
          <span className="ml-2 text-[var(--text-muted)] font-normal">Fusion Builder</span>
        </p>
      </div>

      {/* Base image picker */}
      <div className="flex flex-col gap-1.5">
        <p className="text-[11px] font-medium text-[var(--text-secondary)]">
          底图 Base image
        </p>
        {baseImageUrl && (
          <FallbackImg
            src={baseImageUrl}
            alt="Base"
            className="w-full h-20 object-cover rounded border border-[var(--border-color)]"
          />
        )}
        <div className="flex flex-wrap gap-1.5">
          {candidates.map((c) => {
            const isBase = c.run_id === draft.base_run_id;
            return (
              <button
                key={c.run_id}
                onClick={() => onChange({ ...draft, base_run_id: c.run_id })}
                disabled={disabled}
                title={c.run_id}
                className={`flex flex-col items-center gap-0.5 px-1.5 py-1 rounded border transition-all disabled:opacity-40 text-[10px] ${
                  isBase
                    ? "border-emerald-500 bg-emerald-500/10 text-emerald-400"
                    : "border-[var(--border-color)] bg-[var(--bg-tertiary)] text-[var(--text-muted)] hover:border-[var(--border-hover)] hover:text-[var(--text-primary)]"
                }`}
              >
                {c.thumbnail_url ? (
                  <FallbackImg
                    src={c.thumbnail_url}
                    alt={c.label}
                    className="w-10 h-7 object-cover rounded"
                    fallback={<Image className="w-4 h-4" />}
                  />
                ) : (
                  <Image className="w-4 h-4" />
                )}
                <span className="truncate max-w-[48px]">{c.label}</span>
              </button>
            );
          })}
          {candidates.length === 0 && (
            <p className="text-[11px] text-[var(--text-muted)]">
              暂无候选 — 请先启动抽卡 / Start a draw session first
            </p>
          )}
        </div>
      </div>

      {/* Feedback */}
      <div className="flex flex-col gap-1">
        <label className="text-[11px] font-medium text-[var(--text-secondary)]">
          反馈 Feedback
        </label>
        <textarea
          value={draft.feedback}
          onChange={(e) => onChange({ ...draft, feedback: e.target.value })}
          disabled={disabled}
          rows={2}
          placeholder="融合目标说明，例如：保留A的天空，使用B的水面。"
          className="w-full text-xs p-2 rounded bg-[var(--bg-tertiary)] border border-[var(--border-color)] text-[var(--text-primary)] resize-y placeholder:text-[var(--text-muted)] disabled:opacity-40"
        />
      </div>

      {/* Regions */}
      <div className="flex flex-col gap-1.5">
        <div className="flex items-center justify-between">
          <p className="text-[11px] font-medium text-[var(--text-secondary)]">
            区域 Regions ({draft.regions.length})
          </p>
          <button
            onClick={handleAddRegion}
            disabled={disabled || candidates.length === 0}
            className="flex items-center gap-1 px-2 py-0.5 text-[11px] rounded transition-all disabled:opacity-40 disabled:cursor-not-allowed bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]"
          >
            <Plus className="w-3 h-3" />
            添加区域 / Add region
          </button>
        </div>
        {draft.regions.map((region, idx) => (
          <RegionRow
            key={region.id}
            region={region}
            candidates={candidates}
            onChange={(next) => handleRegionChange(idx, next)}
            onDelete={() => handleDeleteRegion(idx)}
            disabled={disabled}
          />
        ))}
        {draft.regions.length === 0 && (
          <p className="text-[11px] text-[var(--text-muted)] py-1">
            暂无区域 — 点击上方添加 / Click "Add region" above
          </p>
        )}
      </div>

      {/* Create fusion button */}
      {!fusion && (
        <button
          onClick={onCreate}
          disabled={!canCreate}
          className="flex items-center justify-center gap-1.5 w-full py-1.5 text-xs font-medium rounded-md transition-all disabled:opacity-40 disabled:cursor-not-allowed bg-gradient-to-r from-emerald-500 to-emerald-600 text-white hover:from-emerald-600 hover:to-emerald-700"
        >
          <Layers className="w-3.5 h-3.5" />
          创建融合 / Create fusion
        </button>
      )}

      {/* Active fusion status */}
      {fusion && (
        <div className="flex flex-col gap-1.5 p-2 rounded-md border border-[var(--border-color)] bg-[var(--bg-tertiary)]">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-[11px] font-medium text-[var(--text-secondary)]">
              融合 Fusion:
            </span>
            <span className={`text-[11px] px-1.5 py-0.5 rounded font-mono ${
              fusion.status === "completed" ? "bg-emerald-500/10 text-emerald-400"
              : fusion.status === "failed" ? "bg-red-500/10 text-red-400"
              : fusion.status === "running" ? "bg-yellow-400/10 text-yellow-400"
              : "bg-[var(--bg-secondary)] text-[var(--text-muted)]"
            }`}>
              {fusion.status}
            </span>
            <span className="text-[10px] text-[var(--text-muted)] font-mono truncate">
              {fusion.fusion_id.slice(-8)}
            </span>
          </div>

          {fusion.error && (
            <p className="text-[11px] text-red-400 leading-snug">{fusion.error}</p>
          )}

          {/* Composite target */}
          <button
            onClick={onComposite}
            disabled={disabled || fusion.status === "running" || fusion.status === "completed"}
            className="flex items-center justify-center gap-1.5 w-full py-1 text-[11px] rounded-md transition-all disabled:opacity-40 disabled:cursor-not-allowed bg-[var(--bg-secondary)] text-[var(--text-secondary)] hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]"
          >
            <Image className="w-3 h-3" />
            生成合成目标 / Composite target
          </button>

          {fusion.composite_target_url && (
            <FallbackImg
              src={fusion.composite_target_url}
              alt="Composite target"
              className="w-full rounded border border-[var(--border-color)]"
            />
          )}

          {/* Run fusion */}
          <button
            onClick={onRun}
            disabled={disabled || fusion.status !== "target_ready"}
            className="flex items-center justify-center gap-1.5 w-full py-1 text-[11px] rounded-md transition-all disabled:opacity-40 disabled:cursor-not-allowed bg-gradient-to-r from-emerald-500 to-emerald-600 text-white hover:from-emerald-600 hover:to-emerald-700"
          >
            <Play className="w-3 h-3" />
            开始融合优化 / Start fusion refine
          </button>

          {/* Output run */}
          {fusion.output_run_id && (
            <div className="flex items-center gap-1.5">
              <span className="text-[11px] text-[var(--text-muted)]">输出 Output:</span>
              <span className="text-[11px] font-mono text-[var(--text-primary)] truncate flex-1">
                {fusion.output_run_id.slice(-12)}
              </span>
              <button
                onClick={() => onPreviewRun(fusion.output_run_id!)}
                disabled={disabled}
                className="px-2 py-0.5 text-[11px] rounded transition-all disabled:opacity-40 bg-[var(--bg-secondary)] text-[var(--text-secondary)] hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]"
              >
                预览 Preview
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
