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
import { logFrontendEvent, makeRequestId } from "../lib/logger";
import type { ModelSelection } from "../lib/models";

export interface LlmIO {
  system_prompt: string;
  user_prompt: string;
  raw_response: string;
  mode: string;
  /** Compiled GLSL produced by the initial LLM call, snapshotted before any
   *  later refinement mutates the candidate's compile_glsl in place. */
  compile_glsl?: string | null;
  image_paths?: string[];
  attempts?: Array<{
    mode?: string;
    raw_response?: string;
    parse_success?: boolean;
    raw_response_len?: number;
    json_candidate_count?: number;
    candidate_shapes?: string[];
    parsed_layer_count?: number | null;
  }>;
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
  /** V1.2 directed acceptance: whether this iteration was accepted as best, and
   *  whether the human-goal VLM judge overrode a small score drop. */
  accepted?: boolean | null;
  human_goal_override?: string | null;
  best_score_after?: number | null;
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
  // Human-in-loop branch refinement (V1).
  lineage?: BranchLineage | null;
  parent_run_id?: string | null;
  source_checkpoint_id?: string | null;
}

export type BranchMode = "continue" | "refine" | "polish";

export interface BranchLineage {
  parent_run_id?: string;
  root_run_id?: string;
  source_checkpoint_id?: string;
  source_checkpoint_label?: string;
  mode?: string;
  feedback?: string;
}

export interface PipelineCheckpointMeta {
  id: string;
  kind: "candidate" | "refinement_iter" | "final";
  label: string;
  score?: number | null;
  iteration?: number | null;
  accepted?: boolean | null;
  has_glsl: boolean;
}

export interface CheckpointTimelineEntry {
  id: string;
  run_id?: string | null;
  kind: "candidate" | "refinement_iter" | "final";
  label: string;
  iteration?: number | null;
  score?: number | null;
  score_before?: number | null;
  delta?: number | null;
  accepted?: boolean | null;
  human_goal_override?: string | null;
  changes_summary?: string | null;
  has_glsl: boolean;
  artifact_ids?: { render?: string; shader?: string; llm_io?: string };
}

export interface BranchTreeNode {
  run_id: string;
  root_run_id: string;
  parent_run_id?: string | null;
  source_checkpoint_id?: string | null;
  source_checkpoint_label?: string | null;
  title?: string | null;
  mode?: string | null;
  feedback?: string | null;
  status: string;
  final_score?: number | null;
  created_at?: number | null;
  completed_at?: number | null;
  favorite?: boolean;
  children: BranchTreeNode[];
}

export interface BranchTreeResponse {
  root_run_id: string;
  active_run_id: string;
  tree: BranchTreeNode;
}

export interface RunMetadataPatch {
  title?: string;
  favorite?: boolean;
  tags?: string[];
}

export interface RunMetadataRecord {
  run_id: string;
  title?: string | null;
  favorite?: boolean;
  tags?: string[];
  // backend returns the full RunLineageRecord as a dict; keep it permissive
  [key: string]: unknown;
}

export interface BranchRefineRequest {
  checkpoint_id: string;
  feedback: string;
  mode: BranchMode;
  locks?: Record<string, boolean>;
  stop_parent?: boolean;
  quality?: Partial<StrategyConfig>;
}

export type LlmMode = "off" | "auto" | "on";

export interface ParameterizeResult {
  glsl: string;
  tunable_parameters: { name: string; value: unknown; raw: string; role: string }[];
  param_count_before: number;
  param_count_after: number;
  warnings: string[];
}

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
  // True only while a branch-refine request is in flight. Kept separate from
  // `loading` (the parent run's lifecycle) so the human-loop panel can branch
  // from an existing checkpoint while the parent is still running.
  const [branchPending, setBranchPending] = useState(false);
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
      const requestId = makeRequestId("status");
      const response = await fetch(`${API_BASE}/png-shader/status/${id}`, {
        headers: { "x-request-id": requestId, "x-run-id": id },
      });
      if (!response.ok) {
        const text = await response.text();
        logFrontendEvent("api_status_failed", { run_id: id, request_id: requestId, status: response.status }, "warn");
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
      logFrontendEvent("api_status_terminal", {
        run_id: id,
        request_id: requestId,
        status: data.status,
        final_score: data.quality_router?.final_score,
      });
      if (data.status === "failed") {
        setError(data.error || "PNG shader pipeline failed");
      }
    } catch (err) {
      if (activeRunRef.current !== id) return;
      setStopPending(false);
      stopPolling();
      activeRunRef.current = null;
      setLoading(false);
      logFrontendEvent("api_status_error", { run_id: id, error: err instanceof Error ? err.message : String(err) }, "error");
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [stopPolling]);

  const runPngShader = useCallback(async (
    file: File,
    seedGlsl?: string,
    modelSelection?: ModelSelection | null,
  ): Promise<void> => {
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
          mergeInputSpecs(
            LLM_MODE_SPEC[llmMode],
            { quality: toQualityOverrides(strategy) },
            modelSelection ? { model: modelSelection } : {},
          )
        )
      );
      if (seedGlsl && seedGlsl.trim()) {
        formData.append("seed_glsl", seedGlsl);
      }

      const requestId = makeRequestId("run");
      logFrontendEvent("api_run_submit", {
        request_id: requestId,
        filename: file.name,
        size: file.size,
        llm_mode: llmMode,
        refinement_mode: strategy.refinement_mode,
      });
      const response = await fetch(`${API_BASE}/png-shader/run`, {
        method: "POST",
        headers: { "x-request-id": requestId },
        body: formData,
      });

      if (!response.ok) {
        const text = await response.text();
        logFrontendEvent("api_run_failed", { request_id: requestId, status: response.status }, "warn");
        throw new Error(`Request failed (${response.status}): ${text}`);
      }

      const data: PngShaderResult = await response.json();
      logFrontendEvent("api_run_accepted", { request_id: requestId, run_id: data.run_id, status: data.status });
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
      logFrontendEvent("api_run_error", { error: err instanceof Error ? err.message : String(err) }, "error");
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
        const requestId = makeRequestId("strategy");
        const response = await fetch(`${API_BASE}/png-shader/runs/${id}/strategy`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json", "x-request-id": requestId, "x-run-id": id },
          body: JSON.stringify({ quality: patch }),
        });
        logFrontendEvent("api_strategy_patch", {
          request_id: requestId,
          run_id: id,
          status: response.status,
          fields: Object.keys(patch),
        }, response.ok ? "debug" : "warn");
      } catch {
        logFrontendEvent("api_strategy_patch_error", { run_id: id, fields: Object.keys(patch) }, "warn");
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
      const requestId = makeRequestId("stop");
      const response = await fetch(`${API_BASE}/png-shader/runs/${id}/stop`, {
        method: "POST",
        headers: { "x-request-id": requestId, "x-run-id": id },
      });
      logFrontendEvent("api_stop_run", { request_id: requestId, run_id: id, status: response.status }, response.ok ? "info" : "warn");
    } catch {
      logFrontendEvent("api_stop_run_error", { run_id: id }, "warn");
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

  const parameterizeGlsl = useCallback(async (glsl: string): Promise<ParameterizeResult> => {
    const id = activeRunRef.current ?? runId ?? "none";
    const requestId = makeRequestId("parameterize");
    const formData = new FormData();
    formData.append("glsl", glsl);
    logFrontendEvent("api_parameterize_submit", { request_id: requestId, run_id: id, glsl_len: glsl.length });
    const response = await fetch(`${API_BASE}/png-shader/parameterize/${id}`, {
      method: "POST",
      headers: { "x-request-id": requestId, "x-run-id": id },
      body: formData,
    });
    if (!response.ok) {
      const text = await response.text();
      logFrontendEvent("api_parameterize_failed", { request_id: requestId, run_id: id, status: response.status }, "warn");
      throw new Error(`Parameterize failed (${response.status}): ${text}`);
    }
    const data: ParameterizeResult = await response.json();
    logFrontendEvent("api_parameterize_done", {
      request_id: requestId,
      run_id: id,
      param_count_before: data.param_count_before,
      param_count_after: data.param_count_after,
    });
    return data;
  }, [runId]);

  const fetchCheckpoints = useCallback(async (id: string): Promise<PipelineCheckpointMeta[]> => {
    const requestId = makeRequestId("checkpoints");
    const response = await fetch(`${API_BASE}/png-shader/runs/${id}/checkpoints`, {
      headers: { "x-request-id": requestId, "x-run-id": id },
    });
    if (!response.ok) {
      const text = await response.text();
      logFrontendEvent("api_checkpoints_failed", { request_id: requestId, run_id: id, status: response.status }, "warn");
      throw new Error(`Checkpoints failed (${response.status}): ${text}`);
    }
    const data = await response.json();
    return (data.checkpoints ?? []) as PipelineCheckpointMeta[];
  }, []);

  const fetchTimeline = useCallback(async (id: string): Promise<CheckpointTimelineEntry[]> => {
    const requestId = makeRequestId("timeline");
    const response = await fetch(`${API_BASE}/png-shader/runs/${id}/timeline`, {
      headers: { "x-request-id": requestId, "x-run-id": id },
    });
    if (!response.ok) {
      const text = await response.text();
      logFrontendEvent("api_timeline_failed", { request_id: requestId, run_id: id, status: response.status }, "warn");
      throw new Error(`Timeline failed (${response.status}): ${text}`);
    }
    const data = await response.json();
    return (data.timeline ?? []) as CheckpointTimelineEntry[];
  }, []);

  const fetchBranches = useCallback(async (id: string): Promise<BranchTreeResponse> => {
    const requestId = makeRequestId("branches");
    const response = await fetch(`${API_BASE}/png-shader/runs/${id}/branches`, {
      headers: { "x-request-id": requestId, "x-run-id": id },
    });
    if (!response.ok) {
      const text = await response.text();
      logFrontendEvent("api_branches_failed", { request_id: requestId, run_id: id, status: response.status }, "warn");
      throw new Error(`Branches failed (${response.status}): ${text}`);
    }
    const data = await response.json();
    return data as BranchTreeResponse;
  }, []);

  const updateRunMetadata = useCallback(async (id: string, patch: RunMetadataPatch): Promise<RunMetadataRecord> => {
    const requestId = makeRequestId("metadata");
    const response = await fetch(`${API_BASE}/png-shader/runs/${id}/metadata`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json", "x-request-id": requestId, "x-run-id": id },
      body: JSON.stringify(patch),
    });
    if (!response.ok) {
      const text = await response.text();
      logFrontendEvent("api_metadata_failed", { request_id: requestId, run_id: id, status: response.status }, "warn");
      throw new Error(`Metadata failed (${response.status}): ${text}`);
    }
    const data = await response.json();
    return data as RunMetadataRecord;
  }, []);

  const switchRun = useCallback(async (id: string): Promise<void> => {
    // Intentionally keep the previous `result` visible while the target run's
    // status loads, so consumers can show it as a placeholder. `runId` flips
    // once the fetch resolves; on error we leave the prior result as a fallback.
    stopPolling();
    setStopPending(false);
    setError(null);
    setLoading(true);
    activeRunRef.current = null;
    try {
      const requestId = makeRequestId("switch-run");
      const response = await fetch(`${API_BASE}/png-shader/status/${id}`, {
        headers: { "x-request-id": requestId, "x-run-id": id },
      });
      if (!response.ok) {
        const text = await response.text();
        logFrontendEvent("api_switch_run_failed", { request_id: requestId, run_id: id, status: response.status }, "warn");
        throw new Error(`Switch run failed (${response.status}): ${text}`);
      }
      const data: PngShaderResult = await response.json();
      setRunId(data.run_id ?? id);
      setResult(data);
      activeRunRef.current = data.run_id ?? id;
      if (data.status === "running") {
        pollingRef.current = setTimeout(() => pollStatus(data.run_id ?? id), 1000);
      } else {
        activeRunRef.current = null;
        setLoading(false);
        if (data.status === "failed") setError(data.error || "Run failed");
      }
    } catch (err) {
      activeRunRef.current = null;
      setLoading(false);
      logFrontendEvent("api_switch_run_error", { run_id: id, error: err instanceof Error ? err.message : String(err) }, "error");
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [pollStatus, stopPolling]);

  // Create a directed-refinement child run from a parent checkpoint and switch
  // the active run to it, reusing the existing polling loop.
  const branchRefine = useCallback(async (
    parentRunId: string,
    request: BranchRefineRequest,
  ): Promise<string | null> => {
    stopPolling();
    setStopPending(false);
    setBranchPending(true);
    setLoading(true);
    setError(null);
    activeRunRef.current = null;
    try {
      const requestId = makeRequestId("branch-refine");
      logFrontendEvent("api_branch_refine_submit", {
        request_id: requestId,
        parent_run_id: parentRunId,
        checkpoint_id: request.checkpoint_id,
        mode: request.mode,
        feedback_len: request.feedback.length,
      });
      const response = await fetch(`${API_BASE}/png-shader/runs/${parentRunId}/branch-refine`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "x-request-id": requestId, "x-run-id": parentRunId },
        body: JSON.stringify(request),
      });
      if (!response.ok) {
        const text = await response.text();
        logFrontendEvent("api_branch_refine_failed", { request_id: requestId, parent_run_id: parentRunId, status: response.status }, "warn");
        throw new Error(`Branch refine failed (${response.status}): ${text}`);
      }
      const data: PngShaderResult = await response.json();
      logFrontendEvent("api_branch_refine_accepted", { request_id: requestId, run_id: data.run_id, parent_run_id: parentRunId });
      setRunId(data.run_id);
      setResult(data);
      activeRunRef.current = data.run_id;
      if (data.status === "running") {
        pollingRef.current = setTimeout(() => pollStatus(data.run_id), 1000);
      } else {
        activeRunRef.current = null;
        setLoading(false);
        if (data.status === "failed") setError(data.error || "Branch refine failed");
      }
      return data.run_id;
    } catch (err) {
      activeRunRef.current = null;
      setLoading(false);
      logFrontendEvent("api_branch_refine_error", { error: err instanceof Error ? err.message : String(err) }, "error");
      setError(err instanceof Error ? err.message : String(err));
      return null;
    } finally {
      setBranchPending(false);
    }
  }, [pollStatus, stopPolling]);

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
    parameterizeGlsl,
    clearResult,
    llmMode,
    setLlmMode,
    strategy,
    setStrategyPartial,
    applyPreset,
    updateStrategyLive,
    stopRun,
    stopPending,
    branchPending,
    fetchCheckpoints,
    fetchTimeline,
    fetchBranches,
    updateRunMetadata,
    switchRun,
    branchRefine,
  };
}

export type { StrategyMode, StrategyConfig, RefinementMode } from "../lib/strategy-presets";
