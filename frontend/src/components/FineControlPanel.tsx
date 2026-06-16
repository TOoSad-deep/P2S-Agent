// FineControlPanel.tsx — Structured constraints UI for V4.1/V4.2.
// Presentational + props-driven; no data fetching. All changes are immutable updates via onChange.
import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import type { HumanConstraintSpec } from "../hooks/usePngShader";
import RegionMaskEditor from "./RegionMaskEditor";

export const DEFAULT_CONSTRAINT_SPEC: HumanConstraintSpec = {
  locks: {},
  targets: {},
  edit_strength: 0.5,
  regions: [],
  use_preferences: true,
};

/** Returns true when the spec carries non-default values that should be sent to the backend. */
export function isMeaningfulConstraint(spec: HumanConstraintSpec): boolean {
  // Any lock enabled
  if (Object.values(spec.locks).some(Boolean)) return true;
  // Any target set to non-keep
  if (Object.values(spec.targets).some((t) => t !== "keep")) return true;
  // Edit strength differs from default
  if (spec.edit_strength !== 0.5) return true;
  // Preferences opt-out
  if (!spec.use_preferences) return true;
  // Any region constraints defined (V4.2)
  if (spec.regions.length > 0) return true;
  return false;
}

// ─── Constants ─────────────────────────────────────────────────────────────────

const LOCK_KEYS: { key: string; label: string }[] = [
  { key: "preserve_layout", label: "保持构图 Layout" },
  { key: "preserve_palette", label: "保持调色 Palette" },
  { key: "preserve_background", label: "保护背景 Background" },
  { key: "small_edits_only", label: "仅小幅改动 Small edits" },
];

const TARGET_KEYS: { key: string; label: string }[] = [
  { key: "brightness", label: "亮度 Brightness" },
  { key: "contrast", label: "对比 Contrast" },
  { key: "detail", label: "细节 Detail" },
  { key: "reflection", label: "反射 Reflection" },
];

type TargetValue = "keep" | "increase" | "decrease";

const TARGET_OPTIONS: { value: TargetValue; label: string }[] = [
  { value: "keep", label: "保持 Keep" },
  { value: "increase", label: "增强 +" },
  { value: "decrease", label: "减弱 −" },
];

// ─── Props ─────────────────────────────────────────────────────────────────────

interface FineControlPanelProps {
  value: HumanConstraintSpec;
  onChange: (next: HumanConstraintSpec) => void;
  disabled?: boolean;
  imageUrl?: string | null;
}

// ─── Component ─────────────────────────────────────────────────────────────────

export default function FineControlPanel({ value, onChange, disabled, imageUrl }: FineControlPanelProps) {
  const [regionsOpen, setRegionsOpen] = useState(false);
  const setLock = (key: string, checked: boolean) => {
    onChange({ ...value, locks: { ...value.locks, [key]: checked } });
  };

  const setTarget = (key: string, target: TargetValue) => {
    onChange({ ...value, targets: { ...value.targets, [key]: target } });
  };

  const setEditStrength = (strength: number) => {
    onChange({ ...value, edit_strength: strength });
  };

  const setUsePreferences = (use: boolean) => {
    onChange({ ...value, use_preferences: use });
  };

  return (
    <div className="flex flex-col gap-2">
      {/* ── Global locks ── */}
      <div className="flex flex-col gap-1">
        <span className="text-[10px] font-medium text-[var(--text-muted)] uppercase tracking-wide">
          锁定 Locks
        </span>
        <div className="grid grid-cols-2 gap-1">
          {LOCK_KEYS.map(({ key, label }) => (
            <label
              key={key}
              className={`flex items-center gap-1.5 text-[11px] text-[var(--text-secondary)] ${disabled ? "opacity-40 cursor-not-allowed" : "cursor-pointer"}`}
            >
              <input
                type="checkbox"
                checked={!!value.locks[key]}
                onChange={(e) => setLock(key, e.target.checked)}
                disabled={disabled}
                className="accent-emerald-500"
              />
              {label}
            </label>
          ))}
        </div>
      </div>

      {/* ── Targets ── */}
      <div className="flex flex-col gap-1">
        <span className="text-[10px] font-medium text-[var(--text-muted)] uppercase tracking-wide">
          目标调整 Targets
        </span>
        <div className="flex flex-col gap-1">
          {TARGET_KEYS.map(({ key, label }) => {
            const current: TargetValue = (value.targets[key] as TargetValue | undefined) ?? "keep";
            return (
              <div key={key} className="flex items-center gap-1.5">
                <span className="text-[11px] text-[var(--text-secondary)] w-24 shrink-0">{label}</span>
                <div className="flex items-center gap-0.5 bg-[var(--bg-tertiary)] rounded-md p-0.5">
                  {TARGET_OPTIONS.map(({ value: opt, label: optLabel }) => (
                    <button
                      key={opt}
                      onClick={() => setTarget(key, opt)}
                      disabled={disabled}
                      className={`px-1.5 py-0.5 text-[10px] rounded-md transition-all disabled:opacity-40 disabled:cursor-not-allowed ${
                        current === opt
                          ? "bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-medium"
                          : "text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)]"
                      }`}
                    >
                      {optLabel}
                    </button>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* ── Edit strength ── */}
      <div className="flex flex-col gap-1">
        <div className="flex items-center justify-between">
          <span className="text-[10px] font-medium text-[var(--text-muted)] uppercase tracking-wide">
            编辑强度 Edit Strength
          </span>
          <span className="text-[11px] font-mono text-[var(--text-primary)]">
            {value.edit_strength.toFixed(2)}
          </span>
        </div>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={value.edit_strength}
          onChange={(e) => setEditStrength(parseFloat(e.target.value))}
          disabled={disabled}
          className="w-full accent-emerald-500 disabled:opacity-40"
        />
        <div className="flex justify-between text-[10px] text-[var(--text-muted)]">
          <span>0 保守 Subtle</span>
          <span>1 激进 Bold</span>
        </div>
      </div>

      {/* ── Use preferences ── */}
      <label className={`flex items-center gap-1.5 text-[11px] text-[var(--text-secondary)] ${disabled ? "opacity-40 cursor-not-allowed" : "cursor-pointer"}`}>
        <input
          type="checkbox"
          checked={value.use_preferences}
          onChange={(e) => setUsePreferences(e.target.checked)}
          disabled={disabled}
          className="accent-emerald-500"
        />
        使用我的偏好 Use my preferences
      </label>

      {/* ── Region constraints (V4.2) ── */}
      <div className="border border-[var(--border-color)] rounded-md overflow-hidden">
        <button
          onClick={() => setRegionsOpen((prev) => !prev)}
          disabled={disabled}
          className="flex items-center gap-1.5 w-full px-2 py-1.5 text-[11px] text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)] transition-all disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {regionsOpen ? (
            <ChevronDown className="w-3 h-3 shrink-0" />
          ) : (
            <ChevronRight className="w-3 h-3 shrink-0" />
          )}
          区域约束 / Regions
          {value.regions.length > 0 && (
            <span className="ml-auto text-[10px] font-mono px-1 py-0.5 rounded bg-emerald-500/20 text-emerald-400">
              {value.regions.length}
            </span>
          )}
        </button>
        {regionsOpen && (
          <div className="px-2 pb-2 pt-1 border-t border-[var(--border-color)] bg-[var(--bg-tertiary)]">
            <RegionMaskEditor
              imageUrl={imageUrl}
              regions={value.regions}
              onChange={(regions) => onChange({ ...value, regions })}
              disabled={disabled}
            />
          </div>
        )}
      </div>
    </div>
  );
}
