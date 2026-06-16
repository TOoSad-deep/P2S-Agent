// PngShaderView.tsx
import { useRef, useState, useCallback, useEffect } from "react";
import { Upload, Play, Loader, X, Cpu } from "lucide-react";
import type { LlmMode, CandidateEntry } from "../hooks/usePngShader";
import SceneGraphPanel from "./SceneGraphPanel";
import CandidateScoreboard from "./CandidateScoreboard";
import QualityRouterPanel from "./QualityRouterPanel";
import ImageDiffPanel from "./ImageDiffPanel";
import DslLayerPanel from "./DslLayerPanel";
import LlmIOPanel, { type LlmPreviewSelection } from "./LlmIOPanel";
import PngShaderParamPanel from "./PngShaderParamPanel";
import HumanLoopPanel from "./HumanLoopPanel";
import BranchWorkspacePanel from "./BranchWorkspacePanel";
import BranchCanvasWorkspace from "./BranchCanvasWorkspace";
import type {
  PngShaderResult,
  BranchRefineRequest,
  CheckpointTimelineEntry,
  BranchTreeResponse,
  RunMetadataPatch,
  RunMetadataRecord,
} from "../hooks/usePngShader";
import { deriveCheckpoints } from "../lib/checkpoints";
import StrategyControlPanel from "./StrategyControlPanel";
import ModelSelectorPanel from "./ModelSelectorPanel";
import type { ModelControls } from "../hooks/useModels";
import type { StrategyConfig, StrategyMode } from "../lib/strategy-presets";
import { useStrategyConfig } from "../hooks/useStrategyConfig";

const LLM_SEGMENTS: { mode: LlmMode; label: string; sublabel: string; desc: string }[] = [
  { mode: "off",  label: "关",   sublabel: "Off",   desc: "仅使用确定性算法 (快速)" },
  { mode: "auto", label: "自动", sublabel: "Auto",  desc: "按图像复杂度自动选择 (均衡)" },
  { mode: "on",   label: "开",   sublabel: "On",    desc: "强制调用 LLM 生成着色器 (慢)" },
];

interface Props {
  result: PngShaderResult | null;
  loading: boolean;
  error: string | null;
  onRun: (file: File, seedGlsl?: string) => void;
  inputImageUrl: string | null;
  llmMode: LlmMode;
  onLlmModeChange: (mode: LlmMode) => void;
  modelControls: ModelControls;
  strategy: StrategyConfig;
  onStrategyPartial: (partial: Partial<StrategyConfig>) => void;
  onApplyPreset: (mode: Exclude<StrategyMode, "custom">) => void;
  onStop: () => void;
  stopPending?: boolean;
  /** True only while a branch-refine request is in flight (independent of the
   *  parent run's loading state), so branching is allowed mid-run. */
  branchPending?: boolean;
  parameterizeGlsl: (glsl: string) => Promise<{ glsl: string; param_count_before: number; param_count_after: number }>;
  onBranchRefine: (request: BranchRefineRequest) => void;
  runId: string | null;
  fetchTimeline: (id: string) => Promise<CheckpointTimelineEntry[]>;
  fetchBranches: (id: string) => Promise<BranchTreeResponse>;
  updateRunMetadata: (id: string, patch: RunMetadataPatch) => Promise<RunMetadataRecord>;
  switchRun: (id: string) => void;
  branchRefine: (parentRunId: string, request: BranchRefineRequest) => Promise<string | null>;
}

function candidatePreviewGlsl(candidate: CandidateEntry | null, result: PngShaderResult | null): string | null {
  const candidateGlsl = candidate?.compile_glsl?.trim() ? candidate.compile_glsl : null;
  if (candidateGlsl) return candidateGlsl;

  if (
    candidate &&
    candidate.id === result?.selected_candidate_id &&
    result?.selected_glsl?.trim()
  ) {
    return result.selected_glsl;
  }

  return null;
}

export default function PngShaderView({
  result,
  loading,
  error,
  onRun,
  inputImageUrl,
  llmMode,
  onLlmModeChange,
  modelControls,
  strategy,
  onStrategyPartial,
  onApplyPreset,
  onStop,
  stopPending,
  branchPending,
  parameterizeGlsl,
  onBranchRefine,
  runId,
  fetchTimeline,
  fetchBranches,
  updateRunMetadata,
  switchRun,
  branchRefine,
}: Props) {
  const { config: strategyConfig } = useStrategyConfig();
  const [parameterizing, setParameterizing] = useState(false);
  const [branchCheckpointId, setBranchCheckpointId] = useState<string | null>(null);
  const [workspaceView, setWorkspaceView] = useState<"list" | "canvas">("list");

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [seedEnabled, setSeedEnabled] = useState(false);
  const [seedGlsl, setSeedGlsl] = useState("");

  const handleSeedFile = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    file
      .text()
      .then((text) => setSeedGlsl(text))
      .catch((err) => console.error("seed file read failed", err));
  }, []);

  const [dragging, setDragging] = useState(false);
  const [previewCandidateId, setPreviewCandidateId] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback((file: File) => {
    if (file.type === "image/png" || file.name.endsWith(".png")) {
      setSelectedFile(file);
    }
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault();
      setDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handleRun = useCallback(() => {
    if (selectedFile) onRun(selectedFile, seedEnabled ? seedGlsl : undefined);
  }, [selectedFile, onRun, seedEnabled, seedGlsl]);

  const handleClear = useCallback(() => {
    setSelectedFile(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, []);

  // Per-result working GLSL: param-slider edits and "补全可调参数" results are
  // stored keyed by the active result (candidate id / llm iteration / selected
  // output) so switching between results and back preserves each one's edits.
  const [workingGlslByKey, setWorkingGlslByKey] = useState<Record<string, string>>({});
  const [llmPreview, setLlmPreview] = useState<LlmPreviewSelection>({ glsl: null, label: null, key: null });
  const activeKeyRef = useRef<string | null>(null);

  useEffect(() => {
    setWorkingGlslByKey({});
    setPreviewCandidateId(null);
    setLlmPreview({ glsl: null, label: null, key: null });
    setBranchCheckpointId(null);
  }, [result?.run_id]);

  // Clicking a candidate row clears any LLM-panel preview override so the
  // candidate selection actually wins the preview chain (otherwise llmPreview
  // would keep overriding because it has higher priority than candidatePreview).
  const handleCandidateClick = useCallback((id: string) => {
    setPreviewCandidateId(prev => prev === id ? null : id);
    setLlmPreview({ glsl: null, label: null, key: null });
    // Clicking a candidate also makes it the directed-branch start point.
    setBranchCheckpointId(`candidate:${id}`);
  }, []);

  const handleParamGlslChange = useCallback((glsl: string) => {
    const key = activeKeyRef.current;
    if (!key) return;
    setWorkingGlslByKey(prev => ({ ...prev, [key]: glsl }));
  }, []);

  // Clicking the Initial Call tab or an iteration card overrides the candidate
  // row selection so the two preview controls don't fight each other.
  const handleLlmPreviewSelect = useCallback((selection: LlmPreviewSelection) => {
    setLlmPreview(selection);
    if (selection.glsl) setPreviewCandidateId(null);
  }, []);

  const candidates: CandidateEntry[] = result?.scoreboard?.candidates ?? [];
  const previewCandidate = previewCandidateId
    ? candidates.find(c => c.id === previewCandidateId) ?? null
    : null;
  const candidatePreview = candidatePreviewGlsl(previewCandidate, result);
  const llmPreviewGlsl = llmPreview.glsl?.trim() ? llmPreview.glsl : null;

  const previewGlsl = llmPreviewGlsl ?? candidatePreview;
  const previewLabel = llmPreviewGlsl ? llmPreview.label : previewCandidate?.id ?? null;
  const displayCandidateId = llmPreviewGlsl
    ? null
    : previewCandidate?.id ?? result?.selected_candidate_id ?? null;

  // Identity of the currently shown result; keys the per-result working GLSL.
  const activeKey = llmPreviewGlsl
    ? `llm:${llmPreview.key ?? "initial"}`
    : previewCandidate?.id ?? result?.selected_candidate_id ?? (result ? "__selected__" : null);
  activeKeyRef.current = activeKey;
  const workingGlsl = activeKey ? workingGlslByKey[activeKey] ?? null : null;

  // Preview priority (highest first):
  // 1. workingGlsl — this result's param-slider edits / parameterization (keyed)
  // 2. llmPreview  — Initial Call tab or a refinement iteration card
  // 3. candidatePreview — candidate row in the scoreboard
  // 4. result.selected_glsl — pipeline's chosen output
  const activeGlsl = workingGlsl ?? previewGlsl ?? result?.selected_glsl ?? null;
  const activeQualityRouter = previewCandidate?.quality_router ?? result?.quality_router ?? null;

  const llmCandidate = candidates.find(c => c.source === "llm");
  const llmIO = llmCandidate?.llm_io ?? null;

  const [paramError, setParamError] = useState<string | null>(null);
  const handleParameterize = useCallback(async () => {
    if (!activeGlsl || parameterizing) return;
    setParameterizing(true);
    setParamError(null);
    try {
      const res = await parameterizeGlsl(activeGlsl);
      if (res?.glsl) {
        const key = activeKeyRef.current;
        if (key) setWorkingGlslByKey(prev => ({ ...prev, [key]: res.glsl }));
      }
    } catch (err) {
      setParamError(err instanceof Error ? err.message : String(err));
    } finally {
      setParameterizing(false);
    }
  }, [activeGlsl, parameterizing, parameterizeGlsl]);

  return (
    <div className="flex flex-col gap-4">
      {/* Upload zone */}
      <div className="flex-shrink-0">
        <div
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => !selectedFile && fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-xl p-5 transition-all cursor-pointer flex items-center gap-4 ${
            dragging
              ? "border-emerald-400 bg-emerald-500/10 shadow-lg shadow-emerald-500/20"
              : selectedFile
              ? "border-[var(--border-hover)] bg-[var(--bg-card)] cursor-default"
              : "border-[var(--border-color)] bg-[var(--bg-secondary)] hover:border-emerald-400/50 hover:bg-emerald-500/5"
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".png,image/png"
            className="hidden"
            onChange={handleInputChange}
          />

          {selectedFile ? (
            <>
              <Upload className="w-5 h-5 text-[var(--accent-primary)] flex-shrink-0" />
              <span className="text-sm text-[var(--text-primary)] flex-1 truncate font-mono">
                {selectedFile.name}
              </span>
              <button
                onClick={(e) => { e.stopPropagation(); handleClear(); }}
                className="p-1 rounded hover:bg-[var(--bg-tertiary)] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-all"
              >
                <X className="w-4 h-4" />
              </button>
              <button
                onClick={(e) => { e.stopPropagation(); handleRun(); }}
                disabled={loading}
                className="flex items-center gap-2 px-5 py-2 bg-gradient-to-r from-emerald-500 to-emerald-600 hover:from-emerald-400 hover:to-emerald-500 disabled:opacity-50 disabled:cursor-not-allowed text-white text-sm rounded-lg transition-all font-semibold shadow-lg shadow-emerald-500/25 hover:shadow-emerald-500/40"
                title="Run"
              >
                {loading ? (
                  <><Loader className="w-4 h-4 animate-spin" /> 运行中...</>
                ) : (
                  <><Play className="w-4 h-4" /> 运行</>
                )}
              </button>
            </>
          ) : (
            <>
              <Upload className="w-5 h-5 text-[var(--text-muted)] flex-shrink-0" />
              <div>
                <p className="text-sm text-[var(--text-muted)]">拖入 PNG 或点击上传</p>
                <p className="text-xs text-[var(--text-muted)]/60 mt-0.5">Drop a PNG here, or click to upload</p>
              </div>
            </>
          )}
        </div>

        <div className="mt-2 flex flex-col gap-2">
          {/* Seed GLSL: start the closed loop from an existing shader */}
          <div className="flex flex-col gap-2 px-3 py-2 bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-lg">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={seedEnabled}
                onChange={(e) => setSeedEnabled(e.target.checked)}
                disabled={loading}
                className="accent-emerald-500"
              />
              <span className="text-xs font-medium text-[var(--text-primary)]">
                从已有着色器开始
                <span className="ml-2 text-[var(--text-muted)] font-normal">Seed GLSL</span>
              </span>
            </label>
            {seedEnabled && (
              <>
                <textarea
                  value={seedGlsl}
                  onChange={(e) => setSeedGlsl(e.target.value)}
                  disabled={loading}
                  placeholder="粘贴已有 GLSL（Shadertoy mainImage 或普通 main 片元着色器）"
                  rows={6}
                  className="w-full text-xs font-mono p-2 rounded bg-[var(--bg-tertiary)] border border-[var(--border-color)] text-[var(--text-primary)] resize-y"
                />
                <input
                  type="file"
                  accept=".glsl,.frag,.txt,text/plain"
                  onChange={handleSeedFile}
                  disabled={loading}
                  className="text-xs text-[var(--text-muted)]"
                />
              </>
            )}
          </div>
          {/* LLM mode selector */}
          <div className="flex items-center gap-3 px-3 py-2 bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-lg min-w-0">
            <Cpu className={`w-4 h-4 flex-shrink-0 ${llmMode !== "off" ? "text-[var(--accent-primary)]" : "text-[var(--text-muted)]"}`} />
            <div className="flex-1 min-w-0">
              <p className="text-xs font-medium text-[var(--text-primary)] leading-tight">
                AI 模型候选
                <span className="ml-2 text-[var(--text-muted)] font-normal">AI Candidate</span>
              </p>
              <p className="text-[11px] text-[var(--text-muted)] leading-tight truncate">
                {LLM_SEGMENTS.find(s => s.mode === llmMode)?.desc}
              </p>
            </div>
            <div className="flex items-center gap-0.5 bg-[var(--bg-tertiary)] rounded-md p-0.5 flex-shrink-0">
              {LLM_SEGMENTS.map(({ mode, label, sublabel }) => (
                <button
                  key={mode}
                  onClick={() => onLlmModeChange(mode)}
                  disabled={loading}
                  title={sublabel}
                  className={`px-2.5 py-1 text-xs rounded-md transition-all disabled:opacity-40 ${
                    llmMode === mode
                      ? "bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-medium shadow-sm shadow-emerald-500/25"
                      : "text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)]"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          {/* Model selector */}
          <ModelSelectorPanel controls={modelControls} loading={loading} />

          <StrategyControlPanel
            strategy={strategy}
            loading={loading}
            onApplyPreset={onApplyPreset}
            onChange={onStrategyPartial}
            onStop={onStop}
            stopPending={stopPending}
            paramMeta={strategyConfig?.params}
          />

          {/* Human-in-loop directed branch refinement */}
          {result && (
            <HumanLoopPanel
              checkpoints={deriveCheckpoints(result)}
              selectedCheckpointId={branchCheckpointId}
              onSelectCheckpoint={setBranchCheckpointId}
              onSubmit={onBranchRefine}
              lineage={result.lineage ?? null}
              // Gate on the branch request, not the parent run: a user can branch
              // from an existing checkpoint while the parent is still running.
              disabled={branchPending}
              busy={branchPending}
            />
          )}

          {/* Branch workspace: Canvas / List toggle + panels */}
          {result && runId && (
            <div className="flex flex-col gap-2">
              {/* Segmented toggle */}
              <div className="flex items-center gap-0.5 self-start bg-[var(--bg-tertiary)] rounded-md p-0.5">
                {(["list", "canvas"] as const).map((view) => (
                  <button
                    key={view}
                    onClick={() => setWorkspaceView(view)}
                    className={`px-2.5 py-1 text-xs rounded-md transition-all ${
                      workspaceView === view
                        ? "bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-medium shadow-sm shadow-emerald-500/25"
                        : "text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)]"
                    }`}
                  >
                    {view === "canvas" ? "画布 Canvas" : "列表 List"}
                  </button>
                ))}
              </div>

              {workspaceView === "canvas" ? (
                <BranchCanvasWorkspace
                  runId={runId}
                  result={result}
                  fetchBranches={fetchBranches}
                  fetchTimeline={fetchTimeline}
                  switchRun={switchRun}
                  updateRunMetadata={updateRunMetadata}
                  branchRefine={branchRefine}
                  disabled={loading}
                />
              ) : (
                <BranchWorkspacePanel
                  runId={runId}
                  result={result}
                  activeCheckpointId={branchCheckpointId}
                  onCheckpointSelect={setBranchCheckpointId}
                  onSwitchRun={switchRun}
                  fetchTimeline={fetchTimeline}
                  fetchBranches={fetchBranches}
                  updateRunMetadata={updateRunMetadata}
                  disabled={loading}
                />
              )}
            </div>
          )}
        </div>

        {error && (
          <div className="mt-2 px-3 py-2 bg-red-500/10 border border-red-500/30 rounded-lg">
            <p className="text-xs text-red-400">{error}</p>
          </div>
        )}
        {paramError && (
          <div className="mt-2 px-3 py-2 bg-red-500/10 border border-red-500/30 rounded-lg">
            <p className="text-xs text-red-400">补全可调参数失败：{paramError}</p>
          </div>
        )}
      </div>

      {/* Top row: SceneGraph | Scoreboard | QualityRouter */}
      <div className="grid grid-cols-3 gap-3 min-h-[420px]">
        <div className="min-w-0 overflow-hidden">
          <SceneGraphPanel preprocess={result?.preprocess ?? null} />
        </div>
        <div className="min-w-0 overflow-hidden">
          <CandidateScoreboard
            scoreboard={result?.scoreboard ?? null}
            previewId={previewCandidateId}
            onCandidateClick={handleCandidateClick}
          />
        </div>
        <div className="min-w-0 overflow-hidden">
          <QualityRouterPanel qualityRouter={activeQualityRouter} />
        </div>
      </div>

      {/* Bottom row: ImageDiff | ParamPanel | DslLayer | LlmIO */}
      <div
        className="grid gap-3 min-h-[520px]"
        style={{ gridTemplateColumns: "30fr 20fr 18fr 32fr" }}
      >
        <div className="min-w-0 overflow-hidden">
          <ImageDiffPanel
            inputImageUrl={inputImageUrl}
            selectedGlsl={result?.selected_glsl ?? null}
            previewGlsl={workingGlsl ?? previewGlsl}
            previewLabel={workingGlsl ? "edited" : previewLabel}
          />
        </div>
        <div className="min-w-0 overflow-hidden">
          <PngShaderParamPanel
            glsl={activeGlsl}
            onGlslChange={handleParamGlslChange}
          />
        </div>
        <div className="min-w-0 overflow-hidden">
          <DslLayerPanel
            selectedGlsl={activeGlsl}
            selectedCandidateId={displayCandidateId}
            scoreboard={result?.scoreboard ?? null}
            onParameterize={handleParameterize}
            parameterizing={parameterizing}
          />
        </div>
        <div className="min-w-0 overflow-hidden">
          <LlmIOPanel
            key={result?.run_id ?? "no-run"}
            llmIO={llmIO}
            llmMode={llmMode}
            refinementSummary={result?.refinement_summary}
            refinementHistory={result?.refinement_history}
            onPreviewSelect={handleLlmPreviewSelect}
          />
        </div>
      </div>
    </div>
  );
}
