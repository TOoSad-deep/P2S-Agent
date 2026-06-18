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
import { shouldKeepPolling, mergeStrategyFromServer } from "../lib/runStatus";
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
  variant_group_id?: string | null;
  variant_index?: number | null;
  variant_label?: string | null;
  draw_session_id?: string | null;
  draw_card_index?: number | null;
  replacement_of_run_id?: string | null;
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

export interface RegionConstraint {
  id: string;
  label: string;
  mode: "modify" | "protect";
  instruction: string;
  geometry_type: "rect";
  geometry: { x: number; y: number; w: number; h: number };
  strength: number;
}

export interface HumanConstraintSpec {
  locks: Record<string, boolean>;
  targets: Record<string, "keep" | "increase" | "decrease">;
  edit_strength: number;           // 0..1
  regions: RegionConstraint[];     // V4.2 region/mask constraints
  use_preferences: boolean;
}

export interface BranchRefineRequest {
  checkpoint_id: string;
  feedback: string;
  mode: BranchMode;
  locks?: Record<string, boolean>;
  stop_parent?: boolean;
  quality?: Partial<StrategyConfig>;
  constraints?: HumanConstraintSpec;
}

export interface ExploreVariantsRequest {
  checkpoint_id: string;
  feedback: string;
  variant_count?: number;       // 2..6, default 4
  diversity?: "low" | "medium" | "high";
  mode?: string;                // "explore"
  quality?: Partial<StrategyConfig>;
  stop_parent?: boolean;
  constraints?: HumanConstraintSpec;
}

export interface ExploreVariantsResponse {
  group_id: string;
  status: string;
  parent_run_id: string;
  source_checkpoint_id: string;
  child_run_ids: string[];
}

export interface VariantStatusEntry {
  run_id: string;
  variant_index: number;
  label: string;
  status: string;
  final_score?: number | null;
  current_score?: number | null;
  selected_glsl?: string | null;
  thumbnail_url?: string | null;
  changes_summary?: string | null;
  error?: string | null;
  favorite?: boolean;
  preference_score?: number;
  recommended?: boolean;
}

export interface VariantGroupStatus {
  group_id: string;
  parent_run_id: string;
  source_checkpoint_id: string;
  feedback: string;
  status: "queued" | "running" | "completed" | "partial_failed" | "failed" | "cancelled" | string;
  winner_run_id?: string | null;
  variants: VariantStatusEntry[];
  preference_enabled?: boolean;
}

export interface DrawCardStatus {
  card_id: string;
  run_id: string;
  group_id: string | null;
  index: number;
  status: string;
  label: string;
  strategy_label?: string | null;
  final_score?: number | null;
  current_score?: number | null;
  thumbnail_url?: string | null;
  feedback?: string | null;
  favorite?: boolean;
  eliminated?: boolean;
  tags?: string[];
  replacement_of_run_id?: string | null;
  can_use_for_fusion?: boolean;
  error?: string | null;
}

export interface DrawSessionStatus {
  draw_id: string;
  parent_run_id: string;
  source_checkpoint_id: string;
  feedback: string;
  status: "queued" | "running" | "completed" | "partial_failed" | "failed" | "cancelled";
  requested_count: number;
  completed_count: number;
  running_count: number;
  failed_count: number;
  winner_run_id?: string | null;
  group_ids: string[];
  cards: DrawCardStatus[];
}

export interface CreateDrawSessionRequest {
  checkpoint_id?: string;
  feedback: string;
  card_count?: number;       // 2..12, default 8
  diversity?: string;        // "low" | "medium" | "high"
  quality?: Record<string, unknown>;
  constraints?: HumanConstraintSpec;
  mode?: string;
  stop_parent?: boolean;
}
export interface CreateDrawSessionResponse {
  draw_id: string;
  status: string;
  parent_run_id: string;
  source_checkpoint_id: string;
  group_ids: string[];
  card_run_ids: string[];
}
export interface DrawMoreRequest {
  card_count?: number;       // default 4
  diversity?: string;
  quality?: Record<string, unknown>;
  constraints?: HumanConstraintSpec;
}
export interface DrawMoreResponse {
  draw_id: string;
  status: string;
  group_ids: string[];
  card_run_ids: string[];
}
export interface RedrawCardResponse {
  draw_id: string;
  group_id: string;
  replaced_run_id: string;
  replacement_run_id: string;
}
export type DrawCardEventType =
  | "favorite" | "eliminate" | "tag" | "note"
  | "use_as_fusion_base" | "use_as_region_source";

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
  // Bug 4: side-panel operations (draw sessions, fusion) get a SEPARATE error
  // channel so their failures don't overwrite / conflate with the main run
  // error shown for the active pipeline run.
  const [sidePanelError, setSidePanelError] = useState<string | null>(null);
  const [llmMode, setLlmMode] = useState<LlmMode>("off");
  const [strategy, setStrategy] = useState<StrategyConfig>(FALLBACK_DEFAULT_STRATEGY);
  const [stopPending, setStopPending] = useState(false);
  const pollingRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const activeRunRef = useRef<string | null>(null);
  // Field keys with an un-acked local strategy PATCH in flight. The poll
  // back-sync skips these so a slider edited mid-run is not clobbered by the
  // server's stale value (Bug 3 last-writer race).
  const pendingPatchRef = useRef<Partial<StrategyConfig>>({});

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
      // Bug 1: keep polling for ALL non-terminal phases (running/queued/
      // acquired/pending). Backend emits queued/acquired for variant/draw
      // children that are not yet terminal; treating them as terminal froze
      // the result view at the queued snapshot.
      if (shouldKeepPolling(data.status)) {
        // Spec §5.2: sync server strategy back so multi-client edits stay
        // coherent — but Bug 3: skip keys with an un-acked local PATCH so a
        // slider edited mid-run is not clobbered by the server's stale value.
        if (data.strategy && typeof data.strategy === "object") {
          const pendingKeys = new Set(Object.keys(pendingPatchRef.current));
          setStrategy((prev) => {
            const merged = mergeStrategyFromServer(prev, data.strategy, pendingKeys);
            merged.mode = detectMode(merged);
            return merged;
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

      if (shouldKeepPolling(data.status)) {
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
      // Bug 2: fetch() does not throw on 4xx/5xx, so a rejected stop would skip
      // the catch and leave stopPending stuck true forever (Stop button stays
      // permanently disabled). Re-enable the control on any non-ok response.
      if (!response.ok) {
        const text = await response.text().catch(() => "");
        setStopPending(false);
        setError(`Stop run failed (${response.status})${text ? `: ${text}` : ""}`);
      }
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
      // Bug 1: queued/acquired children keep polling — double-clicking a
      // queued node must re-poll until terminal, not freeze at the snapshot.
      if (shouldKeepPolling(data.status)) {
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
      if (shouldKeepPolling(data.status)) {
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
    }
  }, [pollStatus, stopPolling]);

  // V3: Variant exploration — submit, poll, stop, select winner, rate.
  const exploreVariants = useCallback(async (
    parentRunId: string,
    request: ExploreVariantsRequest,
  ): Promise<ExploreVariantsResponse | null> => {
    const requestId = makeRequestId("explore-variants");
    logFrontendEvent("api_explore_variants_submit", {
      request_id: requestId,
      parent_run_id: parentRunId,
      checkpoint_id: request.checkpoint_id,
      feedback_len: request.feedback.length,
      variant_count: request.variant_count,
    });
    const response = await fetch(`${API_BASE}/png-shader/runs/${parentRunId}/explore-variants`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-request-id": requestId, "x-run-id": parentRunId },
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      const text = await response.text();
      logFrontendEvent("api_explore_variants_failed", { request_id: requestId, parent_run_id: parentRunId, status: response.status }, "warn");
      throw new Error(`Explore variants failed (${response.status}): ${text}`);
    }
    const data: ExploreVariantsResponse = await response.json();
    return data;
  }, []);

  const fetchVariantGroup = useCallback(async (groupId: string): Promise<VariantGroupStatus> => {
    const requestId = makeRequestId("variant-group");
    const response = await fetch(`${API_BASE}/png-shader/variant-groups/${groupId}`, {
      headers: { "x-request-id": requestId },
    });
    if (!response.ok) {
      const text = await response.text();
      logFrontendEvent("api_variant_group_failed", { request_id: requestId, group_id: groupId, status: response.status }, "warn");
      throw new Error(`Variant group failed (${response.status}): ${text}`);
    }
    const data: VariantGroupStatus = await response.json();
    return data;
  }, []);

  const stopVariantGroup = useCallback(async (groupId: string): Promise<void> => {
    const requestId = makeRequestId("variant-group-stop");
    try {
      const response = await fetch(`${API_BASE}/png-shader/variant-groups/${groupId}/stop`, {
        method: "POST",
        headers: { "x-request-id": requestId },
      });
      logFrontendEvent("api_variant_group_stop", { request_id: requestId, group_id: groupId, status: response.status }, response.ok ? "info" : "warn");
    } catch {
      logFrontendEvent("api_variant_group_stop", { group_id: groupId, error: "network error" }, "warn");
    }
  }, []);

  const selectVariantWinner = useCallback(async (groupId: string, runId: string, reason?: string): Promise<void> => {
    const requestId = makeRequestId("variant-winner");
    const response = await fetch(`${API_BASE}/png-shader/variant-groups/${groupId}/winner`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-request-id": requestId },
      body: JSON.stringify({ winner_run_id: runId, reason }),
    });
    if (!response.ok) {
      const text = await response.text();
      logFrontendEvent("api_variant_winner", { request_id: requestId, group_id: groupId, run_id: runId, status: response.status }, "warn");
      throw new Error(`Select winner failed (${response.status}): ${text}`);
    }
    logFrontendEvent("api_variant_winner", { request_id: requestId, group_id: groupId, run_id: runId, status: response.status });
  }, []);

  const rateVariant = useCallback(async (
    groupId: string,
    runId: string,
    rating: number,
    reason?: string,
    tags?: string[],
  ): Promise<void> => {
    const requestId = makeRequestId("variant-rate");
    try {
      const response = await fetch(`${API_BASE}/png-shader/variant-groups/${groupId}/ratings`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "x-request-id": requestId },
        body: JSON.stringify({ run_id: runId, rating, reason, tags }),
      });
      logFrontendEvent("api_variant_rate", { request_id: requestId, group_id: groupId, run_id: runId, rating, status: response.status }, response.ok ? "info" : "warn");
    } catch {
      logFrontendEvent("api_variant_rate", { group_id: groupId, run_id: runId, error: "network error" }, "warn");
    }
  }, []);

  // V3.5: Draw-session API layer
  const createDrawSession = useCallback(async (
    parentRunId: string,
    request: CreateDrawSessionRequest,
  ): Promise<CreateDrawSessionResponse | null> => {
    const requestId = makeRequestId("draw-session-create");
    logFrontendEvent("api_draw_session_create", {
      request_id: requestId,
      parent_run_id: parentRunId,
      checkpoint_id: request.checkpoint_id,
      feedback_len: request.feedback.length,
      card_count: request.card_count,
    });
    try {
      const response = await fetch(`${API_BASE}/png-shader/runs/${parentRunId}/draw-session`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "x-request-id": requestId, "x-run-id": parentRunId },
        body: JSON.stringify(request),
      });
      if (!response.ok) {
        const text = await response.text();
        logFrontendEvent("api_draw_session_create_failed", { request_id: requestId, parent_run_id: parentRunId, status: response.status }, "warn");
        setSidePanelError(`Create draw session failed (${response.status}): ${text}`);
        return null;
      }
      const data: CreateDrawSessionResponse = await response.json();
      return data;
    } catch (err) {
      logFrontendEvent("api_draw_session_create_failed", { parent_run_id: parentRunId, error: err instanceof Error ? err.message : String(err) }, "warn");
      setSidePanelError(err instanceof Error ? err.message : String(err));
      return null;
    }
  }, []);

  const fetchDrawSession = useCallback(async (drawId: string): Promise<DrawSessionStatus> => {
    const requestId = makeRequestId("draw-session");
    const response = await fetch(`${API_BASE}/png-shader/draw-sessions/${drawId}`, {
      headers: { "x-request-id": requestId },
    });
    if (!response.ok) {
      const text = await response.text();
      logFrontendEvent("api_draw_session_failed", { request_id: requestId, draw_id: drawId, status: response.status }, "warn");
      throw new Error(`Draw session fetch failed (${response.status}): ${text}`);
    }
    const data: DrawSessionStatus = await response.json();
    return data;
  }, []);

  const drawMore = useCallback(async (
    drawId: string,
    request: DrawMoreRequest,
  ): Promise<DrawMoreResponse | null> => {
    const requestId = makeRequestId("draw-more");
    logFrontendEvent("api_draw_more", { request_id: requestId, draw_id: drawId, card_count: request.card_count });
    try {
      const response = await fetch(`${API_BASE}/png-shader/draw-sessions/${drawId}/draw-more`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "x-request-id": requestId },
        body: JSON.stringify(request),
      });
      if (!response.ok) {
        const text = await response.text();
        logFrontendEvent("api_draw_more_failed", { request_id: requestId, draw_id: drawId, status: response.status }, "warn");
        setSidePanelError(`Draw more failed (${response.status}): ${text}`);
        return null;
      }
      const data: DrawMoreResponse = await response.json();
      return data;
    } catch (err) {
      logFrontendEvent("api_draw_more_failed", { draw_id: drawId, error: err instanceof Error ? err.message : String(err) }, "warn");
      setSidePanelError(err instanceof Error ? err.message : String(err));
      return null;
    }
  }, []);

  const redrawCard = useCallback(async (
    drawId: string,
    runId: string,
    opts?: { reason?: string; diversity?: string },
  ): Promise<RedrawCardResponse | null> => {
    const requestId = makeRequestId("redraw-card");
    logFrontendEvent("api_redraw_card", { request_id: requestId, draw_id: drawId, run_id: runId });
    try {
      const response = await fetch(`${API_BASE}/png-shader/draw-sessions/${drawId}/redraw`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "x-request-id": requestId },
        body: JSON.stringify({ run_id: runId, ...opts }),
      });
      if (!response.ok) {
        const text = await response.text();
        logFrontendEvent("api_redraw_card_failed", { request_id: requestId, draw_id: drawId, run_id: runId, status: response.status }, "warn");
        setSidePanelError(`Redraw card failed (${response.status}): ${text}`);
        return null;
      }
      const data: RedrawCardResponse = await response.json();
      return data;
    } catch (err) {
      logFrontendEvent("api_redraw_card_failed", { draw_id: drawId, run_id: runId, error: err instanceof Error ? err.message : String(err) }, "warn");
      setSidePanelError(err instanceof Error ? err.message : String(err));
      return null;
    }
  }, []);

  const cardEvent = useCallback(async (
    drawId: string,
    runId: string,
    eventType: DrawCardEventType,
    opts?: { value?: unknown; reason?: string; tags?: string[] },
  ): Promise<void> => {
    const requestId = makeRequestId("card-event");
    try {
      const response = await fetch(`${API_BASE}/png-shader/draw-sessions/${drawId}/cards/${runId}/event`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "x-request-id": requestId },
        body: JSON.stringify({ event_type: eventType, ...opts }),
      });
      logFrontendEvent("api_card_event", { request_id: requestId, draw_id: drawId, run_id: runId, event_type: eventType, status: response.status }, response.ok ? "info" : "warn");
    } catch {
      logFrontendEvent("api_card_event_failed", { draw_id: drawId, run_id: runId, event_type: eventType, error: "network error" }, "warn");
    }
  }, []);

  // ─── V4.3 Preference API ─────────────────────────────────────────────────────

  const fetchPreferenceProfile = useCallback(async (): Promise<PreferenceProfile> => {
    const requestId = makeRequestId("pref-profile");
    const response = await fetch(`${API_BASE}/png-shader/preferences/profile`, {
      headers: { "x-request-id": requestId },
    });
    if (!response.ok) {
      const text = await response.text();
      logFrontendEvent("api_pref_profile_failed", { request_id: requestId, status: response.status }, "warn");
      throw new Error(`Preference profile failed (${response.status}): ${text}`);
    }
    const data: PreferenceProfile = await response.json();
    return data;
  }, []);

  const patchPreferenceProfile = useCallback(async (
    patch: Partial<Pick<PreferenceProfile, "enabled" | "default_locks" | "positive_preferences" | "negative_preferences" | "score_drop_tolerance_hint">>,
  ): Promise<PreferenceProfile | null> => {
    const requestId = makeRequestId("pref-patch");
    try {
      const response = await fetch(`${API_BASE}/png-shader/preferences/profile`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json", "x-request-id": requestId },
        body: JSON.stringify(patch),
      });
      if (!response.ok) {
        logFrontendEvent("api_pref_patch_failed", { request_id: requestId, status: response.status }, "warn");
        return null;
      }
      const data: PreferenceProfile = await response.json();
      return data;
    } catch (err) {
      logFrontendEvent("api_pref_patch_error", { error: err instanceof Error ? err.message : String(err) }, "warn");
      return null;
    }
  }, []);

  const rebuildPreferences = useCallback(async (): Promise<PreferenceProfile | null> => {
    const requestId = makeRequestId("pref-rebuild");
    try {
      const response = await fetch(`${API_BASE}/png-shader/preferences/rebuild`, {
        method: "POST",
        headers: { "x-request-id": requestId },
      });
      if (!response.ok) {
        logFrontendEvent("api_pref_rebuild_failed", { request_id: requestId, status: response.status }, "warn");
        return null;
      }
      const data: PreferenceProfile = await response.json();
      return data;
    } catch (err) {
      logFrontendEvent("api_pref_rebuild_error", { error: err instanceof Error ? err.message : String(err) }, "warn");
      return null;
    }
  }, []);

  const clearPreferences = useCallback(async (): Promise<void> => {
    const requestId = makeRequestId("pref-clear");
    try {
      const response = await fetch(`${API_BASE}/png-shader/preferences/clear`, {
        method: "POST",
        headers: { "x-request-id": requestId },
      });
      logFrontendEvent("api_pref_clear", { request_id: requestId, status: response.status }, response.ok ? "info" : "warn");
    } catch {
      logFrontendEvent("api_pref_clear_error", {}, "warn");
    }
  }, []);

  const postPreferenceEvent = useCallback(async (event: PreferenceEventInput): Promise<void> => {
    const requestId = makeRequestId("pref-event");
    try {
      const response = await fetch(`${API_BASE}/png-shader/preferences/events`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "x-request-id": requestId },
        body: JSON.stringify(event),
      });
      logFrontendEvent("api_pref_event", { request_id: requestId, event_type: event.event_type, status: response.status }, response.ok ? "info" : "warn");
    } catch {
      logFrontendEvent("api_pref_event_error", { event_type: event.event_type }, "warn");
    }
  }, []);

  // ─── V4.5 Fusion API ─────────────────────────────────────────────────────────

  const createFusion = useCallback(async (
    request: CreateFusionRequest,
  ): Promise<{ fusion_id: string; status: string } | null> => {
    const requestId = makeRequestId("fusion-create");
    logFrontendEvent("api_fusion_create", { request_id: requestId, base_run_id: request.base_run_id, region_count: request.regions.length });
    try {
      const response = await fetch(`${API_BASE}/png-shader/fusions`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "x-request-id": requestId },
        body: JSON.stringify(request),
      });
      if (!response.ok) {
        const text = await response.text();
        logFrontendEvent("api_fusion_create_failed", { request_id: requestId, status: response.status }, "warn");
        setSidePanelError(`Create fusion failed (${response.status}): ${text}`);
        return null;
      }
      const data = await response.json();
      return data as { fusion_id: string; status: string };
    } catch (err) {
      logFrontendEvent("api_fusion_create_error", { error: err instanceof Error ? err.message : String(err) }, "warn");
      setSidePanelError(err instanceof Error ? err.message : String(err));
      return null;
    }
  }, []);

  const fetchFusion = useCallback(async (fusionId: string): Promise<FusionStatus> => {
    const requestId = makeRequestId("fusion-fetch");
    const response = await fetch(`${API_BASE}/png-shader/fusions/${fusionId}`, {
      headers: { "x-request-id": requestId },
    });
    if (!response.ok) {
      const text = await response.text();
      logFrontendEvent("api_fusion_fetch_failed", { request_id: requestId, fusion_id: fusionId, status: response.status }, "warn");
      throw new Error(`Fusion fetch failed (${response.status}): ${text}`);
    }
    const data: FusionStatus = await response.json();
    return data;
  }, []);

  const generateCompositeTarget = useCallback(async (fusionId: string): Promise<FusionStatus | null> => {
    const requestId = makeRequestId("fusion-composite");
    logFrontendEvent("api_fusion_composite", { request_id: requestId, fusion_id: fusionId });
    try {
      const response = await fetch(`${API_BASE}/png-shader/fusions/${fusionId}/composite-target`, {
        method: "POST",
        headers: { "x-request-id": requestId },
      });
      if (!response.ok) {
        const text = await response.text();
        logFrontendEvent("api_fusion_composite_failed", { request_id: requestId, fusion_id: fusionId, status: response.status }, "warn");
        setSidePanelError(`Generate composite target failed (${response.status}): ${text}`);
        return null;
      }
      const data: FusionStatus = await response.json();
      return data;
    } catch (err) {
      logFrontendEvent("api_fusion_composite_error", { fusion_id: fusionId, error: err instanceof Error ? err.message : String(err) }, "warn");
      setSidePanelError(err instanceof Error ? err.message : String(err));
      return null;
    }
  }, []);

  const runFusion = useCallback(async (
    fusionId: string,
    body?: { quality?: Record<string, unknown>; directed_acceptance?: Record<string, unknown> },
  ): Promise<{ fusion_id: string; status: string; output_run_id: string } | null> => {
    const requestId = makeRequestId("fusion-run");
    logFrontendEvent("api_fusion_run", { request_id: requestId, fusion_id: fusionId });
    try {
      const response = await fetch(`${API_BASE}/png-shader/fusions/${fusionId}/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "x-request-id": requestId },
        body: JSON.stringify(body ?? {}),
      });
      if (!response.ok) {
        const text = await response.text();
        logFrontendEvent("api_fusion_run_failed", { request_id: requestId, fusion_id: fusionId, status: response.status }, "warn");
        setSidePanelError(`Run fusion failed (${response.status}): ${text}`);
        return null;
      }
      const data = await response.json();
      return data as { fusion_id: string; status: string; output_run_id: string };
    } catch (err) {
      logFrontendEvent("api_fusion_run_error", { fusion_id: fusionId, error: err instanceof Error ? err.message : String(err) }, "warn");
      setSidePanelError(err instanceof Error ? err.message : String(err));
      return null;
    }
  }, []);

  const clearSidePanelError = useCallback(() => setSidePanelError(null), []);

  const clearResult = useCallback(() => {
    stopPolling();
    activeRunRef.current = null;
    setRunId(null);
    setResult(null);
    setError(null);
    setSidePanelError(null);
    setLoading(false);
    setStopPending(false);
  }, [stopPolling]);

  useEffect(() => stopPolling, [stopPolling]);

  return {
    runId,
    result,
    loading,
    error,
    sidePanelError,
    clearSidePanelError,
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
    fetchCheckpoints,
    fetchTimeline,
    fetchBranches,
    updateRunMetadata,
    switchRun,
    branchRefine,
    exploreVariants,
    fetchVariantGroup,
    stopVariantGroup,
    selectVariantWinner,
    rateVariant,
    createDrawSession,
    fetchDrawSession,
    drawMore,
    redrawCard,
    cardEvent,
    fetchPreferenceProfile,
    patchPreferenceProfile,
    rebuildPreferences,
    clearPreferences,
    postPreferenceEvent,
    createFusion,
    fetchFusion,
    generateCompositeTarget,
    runFusion,
  };
}

export type { StrategyMode, StrategyConfig, RefinementMode } from "../lib/strategy-presets";

// ─── V4.3 Preference types ────────────────────────────────────────────────────

export interface PreferenceProfile {
  schema_version: number;
  updated_at: number;
  enabled: boolean;
  default_locks: Record<string, boolean>;
  positive_preferences: string[];
  negative_preferences: string[];
  preferred_variant_labels: string[];
  score_drop_tolerance_hint: number;
  summary_source_event_count: number;
}

export interface PreferenceEventInput {
  event_type?: string;
  run_id?: string | null;
  group_id?: string | null;
  feedback?: string | null;
  winner_run_id?: string | null;
  rating?: number | null;
  reason?: string | null;
  tags?: string[];
  context?: Record<string, unknown>;
}

// ─── V4.5 Fusion types ────────────────────────────────────────────────────────

export interface FusionRegion {
  id: string;
  label: string;
  source_run_id: string;
  instruction: string;
  geometry_type: "rect";
  geometry: { x: number; y: number; w: number; h: number };
  strength: number;
  blend_mode: "soft" | "replace_target" | "protect_base";
  feather: number;
}

export interface FusionStatus {
  fusion_id: string;
  status: "draft" | "target_ready" | "running" | "completed" | "failed";
  base_run_id: string;
  source_run_ids: string[];
  output_run_id?: string | null;
  composite_target_url?: string | null;
  regions: FusionRegion[];
  error?: string | null;
}

export interface CreateFusionRequest {
  base_run_id: string;
  draw_session_id?: string | null;
  feedback?: string;
  source_run_ids?: string[];
  regions: FusionRegion[];
}
