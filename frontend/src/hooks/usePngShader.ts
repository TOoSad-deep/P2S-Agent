// usePngShader.ts
import { useState, useCallback, useEffect, useRef } from "react";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Debounced<T extends (...args: any[]) => void> = T & { cancel: () => void };
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function debounce<T extends (...args: any[]) => void>(fn: T, ms: number): Debounced<T> {
  let timer: ReturnType<typeof setTimeout> | null = null;
  const wrapped = ((...args: Parameters<T>) => {
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => {
      timer = null;
      fn(...args);
    }, ms);
  }) as Debounced<T>;
  wrapped.cancel = () => {
    if (timer) {
      clearTimeout(timer);
      timer = null;
    }
  };
  return wrapped;
}
import {
  FALLBACK_DEFAULT_STRATEGY,
  FALLBACK_PRESETS,
  detectMode,
  toQualityOverrides,
  type StrategyConfig,
  type StrategyMode,
} from "../lib/strategy-presets";

export interface LlmIO {
  system_prompt: string;
  user_prompt: string;
  raw_response: string;
  mode: string;
  /** Compiled GLSL produced by the initial LLM call, snapshotted before any
   *  later refinement mutates the candidate's compile_glsl in place. */
  compile_glsl?: string | null;
  image_paths?: string[];
}

export interface CandidateEntry {
  id: string;
  source: string; // "baseline" | "rule" | "cv" | "fallback" | "llm"
  output_kind: string; // "dsl" | "glsl"
  enabled: boolean;
  priority: number;
  validation_valid: boolean;
  validation_errors?: string[];
  compile_success: boolean;
  compile_errors?: string[];
  compile_glsl: string;
  previewable?: boolean;
  score_status?: "pending" | "scored" | "preview_only" | "disabled" | "validation_failed" | "compile_failed" | string;
  final_score: number;
  objective_metrics?: Record<string, unknown>;
  quality_router?: QualityRouterOutput | null;
  quality_band: string | null;
  selected: boolean;
  reason: string[];
  llm_io?: LlmIO | null;
  glsl_metadata?: Record<string, unknown>;
}

export interface Scoreboard {
  total: number;
  enabled: number;
  compiled: number;
  selected_id: string | null;
  candidates: CandidateEntry[];
}

export interface Preprocess {
  width: number;
  height: number;
  has_alpha: boolean;
  alpha_coverage: number;
  palette: string[];
  color_count_estimate: number;
  edge_sharpness: number;
  component_count_estimate: number;
  texture_score: number;
  photo_like_score: number;
  gradient_score: number;
}

export interface QualityRouterOutput {
  status: string; // "pass" | "preview" | "failed" | "unsupported"
  quality_band: string; // "excellent" | "good" | "acceptable" | "poor"
  next_action: string;
  final_score: number;
  failure_type: string;
  reason: string[];
  protected_aspects: string[];
}

export interface RefinementEntry {
  iteration: number;
  score_before: number;
  score_after: number | null;
  delta?: number | null;
  improved: boolean;
  meaningful_improvement?: boolean;
  llm_io: LlmIO | null;
  llm_duration_ms?: number | null;
  error: string | null;
  error_type?: string | null;
  /** GLSL compiled from this iteration's revised DSL, if compile succeeded. */
  compile_glsl?: string | null;
}

export interface PngShaderResult {
  run_id: string;
  run_dir?: string;
  status: string;
  error?: string;
  filename?: string;
  content_type?: string;
  submitted_at?: number;
  preprocess?: Preprocess | null;
  input_spec?: Record<string, unknown> | null;
  scoreboard?: Scoreboard | null;
  selected_candidate_id?: string | null;
  selected_dsl?: Record<string, unknown> | null;
  selected_glsl?: string | null;
  objective_metrics?: Record<string, unknown>;
  quality_router?: QualityRouterOutput | null;
  optimization?: Record<string, unknown> | null;
  revision?: Record<string, unknown> | null;
  refinement_summary?: Record<string, unknown> | null;
  refinement_history?: RefinementEntry[];
  candidate_details?: Record<string, unknown>[];
  strategy?: Partial<StrategyConfig> | null;
  stop_requested?: boolean;
  strategy_revision?: number;
}

export type LlmMode = "off" | "auto" | "on";

const API_BASE = import.meta.env.VITE_API_BASE || "";

const LLM_MODE_SPEC: Record<LlmMode, object> = {
  off:  { candidates: { llm_enabled: false } },
  auto: { candidates: { llm_enabled: true, llm_implementation: "auto", glsl_render_enabled: true } },
  on:   { candidates: { llm_enabled: true, llm_implementation: "shadertoy_glsl", glsl_render_enabled: true } },
};

// Subset of fields that can be live-updated mid-run.
// refinement_mode / force_failure_type / protected_aspects affect
// pipeline branching decisions made at run-start and can't change live.
const LIVE_KEYS = new Set<keyof StrategyConfig>([
  "max_iterations",
  "max_refinement_iterations",
  "refinement_threshold",
  "refinement_high_score_stop",
  "refinement_min_improvement",
  "refinement_patience",
]);

function mergeInputSpecs(...specs: object[]): object {
  return specs.reduce<Record<string, unknown>>((merged, spec) => {
    for (const [key, value] of Object.entries(spec)) {
      if (
        value &&
        typeof value === "object" &&
        !Array.isArray(value) &&
        merged[key] &&
        typeof merged[key] === "object" &&
        !Array.isArray(merged[key])
      ) {
        merged[key] = { ...(merged[key] as Record<string, unknown>), ...(value as Record<string, unknown>) };
      } else {
        merged[key] = value;
      }
    }
    return merged;
  }, {});
}

export function usePngShader() {
  const [runId, setRunId] = useState<string | null>(null);
  const [result, setResult] = useState<PngShaderResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [llmMode, setLlmMode] = useState<LlmMode>("off");
  const [strategy, setStrategy] = useState<StrategyConfig>(FALLBACK_DEFAULT_STRATEGY);
  const [stopPending, setStopPending] = useState(false);
  const pollingRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const activeRunRef = useRef<string | null>(null);

  const stopPolling = useCallback(() => {
    if (pollingRef.current) {
      clearTimeout(pollingRef.current);
      pollingRef.current = null;
    }
  }, []);

  const pollStatus = useCallback(async (id: string) => {
    if (activeRunRef.current !== id) return;

    try {
      const response = await fetch(`${API_BASE}/png-shader/status/${id}`);
      if (!response.ok) {
        const text = await response.text();
        throw new Error(`Status failed (${response.status}): ${text}`);
      }

      const data: PngShaderResult = await response.json();
      if (activeRunRef.current !== id) return;

      setResult(data);
      if (data.status === "running") {
        // Spec §5.2: sync server strategy back so multi-client edits stay coherent
        if (data.strategy && typeof data.strategy === "object") {
          setStrategy((prev) => {
            const merged = { ...prev, ...data.strategy };
            merged.mode = detectMode(merged as StrategyConfig);
            return merged as StrategyConfig;
          });
        }
        pollingRef.current = setTimeout(() => pollStatus(id), 1000);
        return;
      }

      setStopPending(false);
      stopPolling();
      activeRunRef.current = null;
      setLoading(false);
      if (data.status === "failed") {
        setError(data.error || "PNG shader pipeline failed");
      }
    } catch (err) {
      if (activeRunRef.current !== id) return;
      setStopPending(false);
      stopPolling();
      activeRunRef.current = null;
      setLoading(false);
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [stopPolling]);

  const runPngShader = useCallback(async (file: File): Promise<void> => {
    stopPolling();
    setStopPending(false);
    setLoading(true);
    setError(null);
    setResult(null);
    activeRunRef.current = null;

    try {
      const formData = new FormData();
      formData.append("image", file);
      formData.append(
        "input_spec_json",
        JSON.stringify(
          mergeInputSpecs(LLM_MODE_SPEC[llmMode], { quality: toQualityOverrides(strategy) })
        )
      );

      const response = await fetch(`${API_BASE}/png-shader/run`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(`Request failed (${response.status}): ${text}`);
      }

      const data: PngShaderResult = await response.json();
      setRunId(data.run_id);
      setResult(data);
      activeRunRef.current = data.run_id;

      if (data.status === "running") {
        pollingRef.current = setTimeout(() => pollStatus(data.run_id), 1000);
        return;
      }

      activeRunRef.current = null;
      setLoading(false);
      if (data.status === "failed") {
        setError(data.error || "PNG shader pipeline failed");
      }
    } catch (err) {
      activeRunRef.current = null;
      setError(err instanceof Error ? err.message : String(err));
      setLoading(false);
    }
  }, [llmMode, strategy, pollStatus, stopPolling]);

  const pendingPatchRef = useRef<Partial<StrategyConfig>>({});

  // runIdRef keeps the latest runId accessible inside the stable debounced
  // callback without re-creating the debounce timer on every render.
  const runIdRef = useRef<string | null>(null);
  useEffect(() => {
    runIdRef.current = runId;
  }, [runId]);

  // Stable debounced flush — null-initialised, assigned once after mount.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const debouncedFlushRef = useRef<Debounced<((...args: any[]) => void)> | null>(null);
  useEffect(() => {
    const inst = debounce(async () => {
      const id = activeRunRef.current ?? runIdRef.current;
      if (!id) return;
      const patch = pendingPatchRef.current;
      pendingPatchRef.current = {};
      if (!Object.keys(patch).length) return;
      try {
        await fetch(`${API_BASE}/png-shader/runs/${id}/strategy`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ quality: patch }),
        });
      } catch {
        // best-effort; next poll will re-sync
      }
    }, 250);
    debouncedFlushRef.current = inst;
    return () => {
      inst.cancel();
      debouncedFlushRef.current = null;
    };
  }, []); // intentionally empty — this debounce is created once

  const updateStrategyLive = useCallback((partial: Partial<StrategyConfig>) => {
    pendingPatchRef.current = { ...pendingPatchRef.current, ...partial };
    debouncedFlushRef.current?.();
  }, []);

  const stopRun = useCallback(async () => {
    const id = activeRunRef.current ?? runId;
    if (!id) return;
    setStopPending(true);
    try {
      await fetch(`${API_BASE}/png-shader/runs/${id}/stop`, { method: "POST" });
    } catch {
      setStopPending(false);
    }
  }, [runId]);

  const setStrategyPartial = useCallback((partial: Partial<StrategyConfig>) => {
    setStrategy((prev) => {
      const merged = { ...prev, ...partial };
      merged.mode = detectMode(merged);
      return merged;
    });
    if (activeRunRef.current) {
      const liveable: Partial<StrategyConfig> = {};
      for (const [k, v] of Object.entries(partial)) {
        if (LIVE_KEYS.has(k as keyof StrategyConfig)) {
          (liveable as Record<string, unknown>)[k] = v;
        }
      }
      if (Object.keys(liveable).length) updateStrategyLive(liveable);
    }
  }, [updateStrategyLive]);

  const applyPreset = useCallback((preset: Exclude<StrategyMode, "custom">) => {
    setStrategy({
      ...FALLBACK_PRESETS[preset],
      protected_aspects: [...FALLBACK_PRESETS[preset].protected_aspects],
    });
  }, []);

  const clearResult = useCallback(() => {
    stopPolling();
    activeRunRef.current = null;
    setRunId(null);
    setResult(null);
    setError(null);
    setLoading(false);
    setStopPending(false);
  }, [stopPolling]);

  useEffect(() => stopPolling, [stopPolling]);

  return {
    runId,
    result,
    loading,
    error,
    runPngShader,
    clearResult,
    llmMode,
    setLlmMode,
    strategy,
    setStrategyPartial,
    applyPreset,
    updateStrategyLive,
    stopRun,
    stopPending,
  };
}

export type { StrategyMode, StrategyConfig, RefinementMode } from "../lib/strategy-presets";
