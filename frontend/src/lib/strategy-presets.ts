// Strategy presets and types shared by the PNG-Shader closed-loop UI.

import type { StrategyConfigJSON } from "../hooks/useStrategyConfig";

export type StrategyMode =
  | "fast"
  | "balanced"
  | "quality"
  | "aggressive"
  | "custom";

export type RefinementMode = "off" | "auto" | "on";

export type FailureType =
  | null
  | "color"
  | "structure"
  | "parameter"
  | "layer_order";

export const ALL_PROTECTED_ASPECTS = [
  "layer_count",
  "primitive_types",
  "background",
  "visual_causality",
  "technique_plan",
  "tunable_parameters",
] as const;
export type ProtectedAspect = (typeof ALL_PROTECTED_ASPECTS)[number];

export interface StrategyConfig {
  mode: StrategyMode;
  max_iterations: number;              // optimizer budget, 0-8
  max_refinement_iterations: number;   // LLM refinement loop, 0-20
  refinement_mode: RefinementMode;
  refinement_threshold: number;        // 0.5-1.0
  refinement_high_score_stop: number;  // 0.7-1.0
  refinement_min_improvement: number;  // 0.001-0.05
  refinement_patience: number;         // 1-5
  max_added_layers: number;            // 0-8
  vlm_judge_enabled: number;           // 0/1
  vlm_tie_epsilon: number;             // 0.0-0.2
  force_failure_type: FailureType;
  protected_aspects: ProtectedAspect[];
}

const DEFAULT_PROTECTED: ProtectedAspect[] = [
  "layer_count",
  "primitive_types",
  "background",
];

export const FALLBACK_PRESETS: Record<Exclude<StrategyMode, "custom">, StrategyConfig> = {
  fast: {
    mode: "fast",
    max_iterations: 2,
    max_refinement_iterations: 1,
    refinement_mode: "auto",
    refinement_threshold: 0.75,
    refinement_high_score_stop: 0.88,
    refinement_min_improvement: 0.02,
    refinement_patience: 1,
    max_added_layers: 0,
    vlm_judge_enabled: 0,
    vlm_tie_epsilon: 0.05,
    force_failure_type: null,
    protected_aspects: [...DEFAULT_PROTECTED],
  },
  balanced: {
    mode: "balanced",
    max_iterations: 5,
    max_refinement_iterations: 3,
    refinement_mode: "auto",
    refinement_threshold: 0.80,
    refinement_high_score_stop: 0.92,
    refinement_min_improvement: 0.01,
    refinement_patience: 2,
    max_added_layers: 4,
    vlm_judge_enabled: 0,
    vlm_tie_epsilon: 0.05,
    force_failure_type: null,
    protected_aspects: [...DEFAULT_PROTECTED],
  },
  quality: {
    mode: "quality",
    max_iterations: 8,
    max_refinement_iterations: 5,
    refinement_mode: "auto",
    refinement_threshold: 0.85,
    refinement_high_score_stop: 0.95,
    refinement_min_improvement: 0.005,
    refinement_patience: 3,
    max_added_layers: 6,
    vlm_judge_enabled: 1,
    vlm_tie_epsilon: 0.05,
    force_failure_type: null,
    protected_aspects: [...DEFAULT_PROTECTED],
  },
  aggressive: {
    mode: "aggressive",
    max_iterations: 6,
    max_refinement_iterations: 5,
    refinement_mode: "on",
    refinement_threshold: 0.90,
    refinement_high_score_stop: 0.97,
    refinement_min_improvement: 0.003,
    refinement_patience: 4,
    max_added_layers: 6,
    vlm_judge_enabled: 1,
    vlm_tie_epsilon: 0.05,
    force_failure_type: null,
    protected_aspects: [...DEFAULT_PROTECTED],
  },
};

// Spread to avoid aliasing FALLBACK_PRESETS.balanced — preset objects must remain
// frozen identities for detectMode() to work correctly.
export const FALLBACK_DEFAULT_STRATEGY: StrategyConfig = {
  ...FALLBACK_PRESETS.balanced,
  protected_aspects: [...FALLBACK_PRESETS.balanced.protected_aspects],
};

// 'mode' is intentionally excluded — it is the derived output, not an input.
const COMPARED_FIELDS: (keyof StrategyConfig)[] = [
  "max_iterations",
  "max_refinement_iterations",
  "refinement_mode",
  "refinement_threshold",
  "refinement_high_score_stop",
  "refinement_min_improvement",
  "refinement_patience",
  "max_added_layers",
  "vlm_judge_enabled",
  "vlm_tie_epsilon",
  "force_failure_type",
  "protected_aspects",
];

function sameAspects(a: readonly string[], b: readonly string[]): boolean {
  if (a.length !== b.length) return false;
  const sa = [...a].sort();
  const sb = [...b].sort();
  return sa.every((v, i) => v === sb[i]);
}

export function detectMode(cfg: StrategyConfig): StrategyMode {
  for (const [name, preset] of Object.entries(FALLBACK_PRESETS) as [
    Exclude<StrategyMode, "custom">,
    StrategyConfig,
  ][]) {
    const matches = COMPARED_FIELDS.every((key) => {
      if (key === "protected_aspects") {
        return sameAspects(cfg.protected_aspects, preset.protected_aspects);
      }
      return cfg[key] === preset[key];
    });
    if (matches) return name;
  }
  return "custom";
}

/** Project StrategyConfig into the shape POSTed to backend as input_spec.quality. */
export function toQualityOverrides(cfg: StrategyConfig): Record<string, unknown> {
  // The backend mode validator only accepts {fast, balanced, quality, aggressive};
  // "custom" is a frontend-only label, so we send "balanced" as the substrate.
  return {
    mode: cfg.mode === "custom" ? "balanced" : cfg.mode,
    max_iterations: cfg.max_iterations,
    max_refinement_iterations: cfg.max_refinement_iterations,
    refinement_mode: cfg.refinement_mode,
    refinement_threshold: cfg.refinement_threshold,
    refinement_high_score_stop: cfg.refinement_high_score_stop,
    refinement_min_improvement: cfg.refinement_min_improvement,
    refinement_patience: cfg.refinement_patience,
    max_added_layers: cfg.max_added_layers,
    vlm_judge_enabled: cfg.vlm_judge_enabled,
    vlm_tie_epsilon: cfg.vlm_tie_epsilon,
    force_failure_type: cfg.force_failure_type,
    protected_aspects: cfg.protected_aspects,
  };
}

export function buildPresetsFromAPI(
  apiConfig: StrategyConfigJSON,
): Record<Exclude<StrategyMode, "custom">, StrategyConfig> {
  const result = {} as Record<Exclude<StrategyMode, "custom">, StrategyConfig>;
  for (const [mode, preset] of Object.entries(apiConfig.presets)) {
    result[mode as Exclude<StrategyMode, "custom">] = {
      mode: mode as Exclude<StrategyMode, "custom">,
      max_iterations: preset.max_iterations,
      max_refinement_iterations: preset.max_refinement_iterations,
      refinement_mode: mode === "aggressive" ? "on" : "auto",
      refinement_threshold: preset.refinement_threshold,
      refinement_high_score_stop: preset.refinement_high_score_stop,
      refinement_min_improvement: preset.refinement_min_improvement,
      refinement_patience: preset.refinement_patience,
      max_added_layers: preset.max_added_layers,
      vlm_judge_enabled: preset.vlm_judge_enabled,
      vlm_tie_epsilon: preset.vlm_tie_epsilon,
      force_failure_type: null,
      protected_aspects: [...DEFAULT_PROTECTED],
    };
  }
  return result;
}

export function buildDefaultStrategyFromAPI(
  apiConfig: StrategyConfigJSON,
): StrategyConfig {
  const balanced = apiConfig.presets["balanced"];
  return {
    mode: "balanced",
    max_iterations: balanced.max_iterations,
    max_refinement_iterations: balanced.max_refinement_iterations,
    refinement_mode: "auto",
    refinement_threshold: balanced.refinement_threshold,
    refinement_high_score_stop: balanced.refinement_high_score_stop,
    refinement_min_improvement: balanced.refinement_min_improvement,
    refinement_patience: balanced.refinement_patience,
    max_added_layers: balanced.max_added_layers,
    vlm_judge_enabled: balanced.vlm_judge_enabled,
    vlm_tie_epsilon: balanced.vlm_tie_epsilon,
    force_failure_type: null,
    protected_aspects: [...DEFAULT_PROTECTED],
  };
}
