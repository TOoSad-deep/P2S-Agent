import { useState } from "react";
import { ChevronDown, ChevronRight, Square } from "lucide-react";
import {
  ALL_PROTECTED_ASPECTS,
  type FailureType,
  type ProtectedAspect,
  type StrategyConfig,
  type StrategyMode,
  type RefinementMode,
} from "../lib/strategy-presets";
import type { ParamMetaJSON } from "../hooks/useStrategyConfig";

const PRESET_BUTTONS: { mode: Exclude<StrategyMode, "custom">; label: string }[] = [
  { mode: "fast",       label: "Fast" },
  { mode: "balanced",   label: "Balanced" },
  { mode: "quality",    label: "Quality" },
  { mode: "aggressive", label: "Aggressive" },
];

const REFINEMENT_SEGMENTS: { mode: RefinementMode; label: string }[] = [
  { mode: "off",  label: "关闭" },
  { mode: "auto", label: "自动" },
  { mode: "on",   label: "开启" },
];

const FAILURE_OPTIONS: { value: FailureType; label: string }[] = [
  { value: null,          label: "自动" },
  { value: "color",       label: "色彩" },
  { value: "structure",   label: "结构" },
  { value: "parameter",   label: "参数" },
  { value: "layer_order", label: "层序" },
];

interface SliderRowProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  disabled?: boolean;
  onChange: (v: number) => void;
  format?: (v: number) => string;
}

function SliderRow({ label, value, min, max, step, disabled, onChange, format }: SliderRowProps) {
  return (
    <div className="flex items-center gap-3 text-xs">
      <span className="w-28 text-[var(--text-muted)]">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className="flex-1 accent-[var(--accent-primary)] disabled:opacity-40"
      />
      <span className="w-16 text-right font-mono text-[var(--text-primary)]">
        {format ? format(value) : value}
      </span>
    </div>
  );
}

interface Props {
  strategy: StrategyConfig;
  loading: boolean;
  onApplyPreset: (mode: Exclude<StrategyMode, "custom">) => void;
  onChange: (partial: Partial<StrategyConfig>) => void;
  onStop?: () => void;
  stopPending?: boolean;
  paramMeta?: Record<string, ParamMetaJSON>;
}

export default function StrategyControlPanel({
  strategy,
  loading,
  onApplyPreset,
  onChange,
  onStop,
  stopPending = false,
  paramMeta,
}: Props) {
  const [advancedOpen, setAdvancedOpen] = useState(false);

  // Mid-run lock: only fields that can be patched live remain editable.
  // refinement_mode / force_failure_type / protected_aspects / presets
  // all change pipeline branching decisions and can't take effect live.
  const liveLocked = loading;

  const meta = (key: string, fallbackMin: number, fallbackMax: number, fallbackStep: number) => ({
    min: paramMeta?.[key]?.min ?? fallbackMin,
    max: paramMeta?.[key]?.max ?? fallbackMax,
    step: paramMeta?.[key]?.step ?? fallbackStep,
  });

  const toggleAspect = (aspect: ProtectedAspect) => {
    const next = strategy.protected_aspects.includes(aspect)
      ? strategy.protected_aspects.filter((a) => a !== aspect)
      : [...strategy.protected_aspects, aspect];
    onChange({ protected_aspects: next });
  };

  return (
    <div className="bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-lg">
      {/* Row 1: refinement mode pill */}
      <div className="px-3 py-2 flex items-center gap-3 border-b border-[var(--border-color)]">
        <span className="text-xs font-medium text-[var(--text-primary)]">闭环优化</span>
        <span className="text-[11px] text-[var(--text-muted)]">Self Optimize</span>
        <div className="flex items-center gap-0.5 bg-[var(--bg-tertiary)] rounded-md p-0.5 ml-auto">
          {REFINEMENT_SEGMENTS.map(({ mode, label }) => (
            <button
              key={mode}
              onClick={() => onChange({ refinement_mode: mode })}
              disabled={loading || liveLocked}
              title={loading ? "运行中不可改，停止后重新运行生效" : undefined}
              className={`px-2.5 py-1 text-xs rounded-md transition-all disabled:opacity-40 ${
                strategy.refinement_mode === mode
                  ? "bg-gradient-to-r from-blue-500 to-blue-600 text-white font-medium shadow-sm shadow-blue-500/25"
                  : "text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)]"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {/* Row 2: presets */}
      <div className="px-3 py-2 flex items-center gap-2 border-b border-[var(--border-color)]">
        <span className="text-xs font-medium text-[var(--text-primary)]">策略预设</span>
        <div className="flex items-center gap-1 flex-1">
          {PRESET_BUTTONS.map(({ mode, label }) => (
            <button
              key={mode}
              onClick={() => onApplyPreset(mode)}
              disabled={loading}
              title={loading ? "运行中不可改，停止后重新运行生效" : undefined}
              className={`px-3 py-1.5 text-xs rounded-md transition-all font-medium disabled:opacity-40 ${
                strategy.mode === mode
                  ? "bg-gradient-to-r from-emerald-500 to-emerald-600 text-white shadow-sm shadow-emerald-500/25"
                  : "bg-[var(--bg-tertiary)] text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)]"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
        <span className="text-[11px] text-[var(--text-muted)] ml-auto">
          当前: <span className="font-mono text-[var(--text-primary)]">{strategy.mode}</span>
        </span>
      </div>

      {/* Row 3: advanced toggle + stop button */}
      <div className="px-3 py-2 flex items-center gap-2">
        <button
          onClick={() => setAdvancedOpen((v) => !v)}
          className="flex items-center gap-1 text-xs text-[var(--text-primary)] hover:text-[var(--accent-primary)]"
        >
          {advancedOpen ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
          高级策略
        </button>
        {loading && onStop && (
          <button
            onClick={onStop}
            disabled={stopPending}
            className="ml-auto flex items-center gap-1 px-2.5 py-1 text-xs bg-red-500/20 hover:bg-red-500/30 disabled:opacity-60 text-red-400 rounded transition-all"
            title="Stop immediately and accept current best"
          >
            <Square className="w-3.5 h-3.5" />
            {stopPending ? "停止中..." : "立即停止"}
          </button>
        )}
      </div>

      {/* Advanced sliders */}
      {advancedOpen && (
        <div className="px-3 pb-3 flex flex-col gap-2 border-t border-[var(--border-color)] pt-3">
          <SliderRow
            label="闭环迭代上限"
            value={strategy.max_refinement_iterations}
            {...meta("max_refinement_iterations", 0, 20, 1)}
            onChange={(v) => onChange({ max_refinement_iterations: v })}
          />
          <SliderRow
            label="优化触发阈值"
            value={strategy.refinement_threshold}
            {...meta("refinement_threshold", 0.5, 1.0, 0.01)}
            format={(v) => v.toFixed(2)}
            onChange={(v) => onChange({ refinement_threshold: v })}
          />
          <SliderRow
            label="高分早停"
            value={strategy.refinement_high_score_stop}
            {...meta("refinement_high_score_stop", 0.7, 1.0, 0.01)}
            format={(v) => v.toFixed(2)}
            onChange={(v) => onChange({ refinement_high_score_stop: v })}
          />
          <SliderRow
            label="最小改进"
            value={strategy.refinement_min_improvement}
            {...meta("refinement_min_improvement", 0.001, 0.05, 0.001)}
            format={(v) => v.toFixed(3)}
            onChange={(v) => onChange({ refinement_min_improvement: v })}
          />
          <SliderRow
            label="无改进忍耐"
            value={strategy.refinement_patience}
            {...meta("refinement_patience", 1, 5, 1)}
            onChange={(v) => onChange({ refinement_patience: v })}
          />
          <SliderRow
            label="优化器预算"
            value={strategy.max_iterations}
            {...meta("max_iterations", 0, 8, 1)}
            onChange={(v) => onChange({ max_iterations: v })}
          />
          <SliderRow
            label="残差增层"
            value={strategy.max_added_layers}
            {...meta("max_added_layers", 0, 8, 1)}
            disabled={loading}
            onChange={(v) => onChange({ max_added_layers: v })}
          />
          <div className="flex items-center gap-3 text-xs">
            <span className="w-28 text-[var(--text-muted)]">VLM 评审</span>
            <label className="flex items-center gap-2 text-[var(--text-primary)]">
              <input
                type="checkbox"
                checked={strategy.vlm_judge_enabled === 1}
                disabled={loading}
                onChange={(e) => onChange({ vlm_judge_enabled: e.target.checked ? 1 : 0 })}
                className="accent-[var(--accent-primary)] disabled:opacity-40"
              />
              <span className="text-[11px] text-[var(--text-muted)]">
                {strategy.vlm_judge_enabled === 1 ? "开启" : "关闭"}
              </span>
            </label>
          </div>
          <SliderRow
            label="近平局阈值"
            value={strategy.vlm_tie_epsilon}
            {...meta("vlm_tie_epsilon", 0, 0.2, 0.01)}
            disabled={loading || strategy.vlm_judge_enabled !== 1}
            format={(v) => v.toFixed(2)}
            onChange={(v) => onChange({ vlm_tie_epsilon: v })}
          />

          <div className="flex items-start gap-3 text-xs mt-1">
            <span className="w-28 text-[var(--text-muted)]">修复取向</span>
            <div className="flex flex-wrap gap-1 flex-1">
              {FAILURE_OPTIONS.map(({ value, label }) => (
                <button
                  key={String(value)}
                  onClick={() => onChange({ force_failure_type: value })}
                  disabled={loading}
                  title={loading ? "运行中不可改，停止后重新运行生效" : undefined}
                  className={`px-2 py-0.5 rounded text-[11px] transition-all disabled:opacity-40 ${
                    strategy.force_failure_type === value
                      ? "bg-[var(--accent-primary)] text-white"
                      : "bg-[var(--bg-tertiary)] text-[var(--text-muted)] hover:text-[var(--text-primary)]"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          <div className="flex items-start gap-3 text-xs">
            <span className="w-28 text-[var(--text-muted)]">保护层级</span>
            <div className="flex flex-wrap gap-1 flex-1">
              {ALL_PROTECTED_ASPECTS.map((aspect) => {
                const active = strategy.protected_aspects.includes(aspect);
                return (
                  <button
                    key={aspect}
                    onClick={() => toggleAspect(aspect)}
                    disabled={loading}
                    title={loading ? "运行中不可改，停止后重新运行生效" : undefined}
                    className={`px-2 py-0.5 rounded text-[11px] transition-all disabled:opacity-40 ${
                      active
                        ? "bg-[var(--accent-primary)] text-white"
                        : "bg-[var(--bg-tertiary)] text-[var(--text-muted)] hover:text-[var(--text-primary)]"
                    }`}
                  >
                    {active ? "✓ " : ""}
                    {aspect}
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
