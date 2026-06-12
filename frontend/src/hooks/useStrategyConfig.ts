import { useState, useEffect } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "";

export interface ParamMetaJSON {
  default: number;
  min: number;
  max: number;
  step: number;
  integer: boolean;
  label: string;
  description: string;
}

export interface PresetJSON {
  description: string;
  max_iterations: number;
  max_refinement_iterations: number;
  refinement_threshold: number;
  refinement_high_score_stop: number;
  refinement_min_improvement: number;
  refinement_patience: number;
  max_added_layers: number;
  vlm_judge_enabled: number;
  vlm_tie_epsilon: number;
}

export interface StrategyConfigJSON {
  params: Record<string, ParamMetaJSON>;
  presets: Record<string, PresetJSON>;
}

let cachedConfig: StrategyConfigJSON | null = null;

export function useStrategyConfig(): {
  config: StrategyConfigJSON | null;
  loading: boolean;
  error: string | null;
} {
  const [config, setConfig] = useState<StrategyConfigJSON | null>(cachedConfig);
  const [loading, setLoading] = useState(!cachedConfig);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (cachedConfig) return;

    let cancelled = false;

    fetch(`${API_BASE}/api/strategy-config`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data: StrategyConfigJSON) => {
        if (!cancelled) {
          cachedConfig = data;
          setConfig(data);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err.message);
          setLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  return { config, loading, error };
}
