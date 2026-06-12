// LlmIOPanel.tsx
import { useState, useEffect, useCallback, useRef } from "react";
import { Cpu, ChevronDown, ChevronRight, TrendingUp, TrendingDown, Minus, AlertCircle, Maximize2, X } from "lucide-react";
import type { LlmIO, RefinementEntry } from "../hooks/usePngShader";

export interface LlmPreviewSelection {
  /** Compiled GLSL to preview. Null clears any LLM-panel preview override. */
  glsl: string | null;
  /** Short label shown in preview chips (e.g. "初始调用", "iter 2"). */
  label: string | null;
  /** Stable identifier for diff/dedupe purposes. */
  key: string | null;
}

interface Props {
  llmIO: LlmIO | null | undefined;
  llmMode: string;
  refinementSummary?: Record<string, unknown> | null;
  refinementHistory?: RefinementEntry[];
  /** Called when the user navigates between the initial call or an iteration
   *  card. Parent stores this and uses it to drive the shader preview. */
  onPreviewSelect?: (selection: LlmPreviewSelection) => void;
}

type TopTab = "initial" | "refinement";
type IOTab = "system" | "user" | "response";

function CollapsibleBlock({ text, maxLines = 12 }: { text: string; maxLines?: number }) {
  const [expanded, setExpanded] = useState(false);
  const lines = text.split("\n");
  const preview = expanded ? text : lines.slice(0, maxLines).join("\n");
  const hasMore = lines.length > maxLines;
  return (
    <div className="flex flex-col gap-1 min-h-0 flex-1 overflow-hidden">
      <div className="flex-1 overflow-auto bg-[var(--bg-primary)] rounded-lg min-h-0">
        <pre className="p-3 text-[10px] font-mono text-[var(--text-primary)] leading-relaxed whitespace-pre-wrap break-words">
          {preview}
        </pre>
      </div>
      {hasMore && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-1 text-[10px] text-[var(--accent-primary)] hover:text-[var(--text-primary)] transition-colors flex-shrink-0"
        >
          {expanded
            ? <><ChevronDown className="w-3 h-3" /> 收起</>
            : <><ChevronRight className="w-3 h-3" /> 展开全部（{lines.length} 行）</>}
        </button>
      )}
    </div>
  );
}

function IOView({ io }: { io: LlmIO }) {
  const [tab, setTab] = useState<IOTab>("response");
  const TABS: { id: IOTab; label: string; sub: string }[] = [
    { id: "system", label: "系统", sub: "System" },
    { id: "user", label: "用户", sub: "User" },
    { id: "response", label: "输出", sub: "Response" },
  ];
  return (
    <div className="flex flex-col gap-2 flex-1 min-h-0 overflow-hidden">
      <div className="flex gap-0.5 bg-[var(--bg-tertiary)] rounded-lg p-0.5 flex-shrink-0">
        {TABS.map(({ id, label, sub }) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className={`flex-1 py-1 px-1 text-[10px] rounded transition-all min-w-0 ${
              tab === id
                ? "bg-[var(--accent-primary)] text-white font-medium"
                : "text-[var(--text-muted)] hover:text-[var(--text-primary)]"
            }`}
          >
            <span className="block truncate">{label}</span>
            <span className="block opacity-60 text-[9px] truncate">{sub}</span>
          </button>
        ))}
      </div>
      <div className="flex-1 min-h-0 overflow-hidden flex flex-col">
        {tab === "system" && <CollapsibleBlock text={io.system_prompt} />}
        {tab === "user" && <CollapsibleBlock text={io.user_prompt} />}
        {tab === "response" && (
          <>
            <div className="flex items-center gap-2 mb-1 flex-shrink-0 flex-wrap">
              <span className="text-[10px] text-[var(--text-muted)]">
                mode: <span className="text-[var(--text-primary)]">{io.mode}</span>
              </span>
              <span className="text-[10px] text-[var(--text-muted)]">
                {io.raw_response.length} chars
              </span>
            </div>
            <CollapsibleBlock text={io.raw_response} maxLines={20} />
          </>
        )}
      </div>
    </div>
  );
}

function FullScreenIO({ entry, onClose }: { entry: RefinementEntry; onClose: () => void }) {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-6">
      <div className="bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-xl w-full max-w-5xl h-[85vh] flex flex-col overflow-hidden">
        <div className="flex items-center justify-between p-4 border-b border-[var(--border-color)] flex-shrink-0">
          <h3 className="text-sm font-semibold text-[var(--text-primary)]">
            迭代 {entry.iteration} · LLM I/O
            <span className="ml-2 text-xs font-normal text-[var(--text-muted)]">
              {entry.score_before.toFixed(3)}
              {entry.score_after !== null && (
                <> → {entry.score_after.toFixed(3)} ({(entry.delta ?? 0) >= 0 ? "+" : ""}{(entry.delta ?? 0).toFixed(3)})</>
              )}
            </span>
          </h3>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-[var(--bg-tertiary)] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
            title="关闭 (Esc)"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
        <div className="flex-1 min-h-0 overflow-hidden p-4 flex flex-col">
          {entry.error ? (
            <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
              <p className="text-xs text-red-400">{entry.error}</p>
            </div>
          ) : entry.llm_io ? (
            <IOView io={entry.llm_io} />
          ) : (
            <p className="text-xs text-[var(--text-muted)]">无 I/O 记录</p>
          )}
        </div>
      </div>
    </div>
  );
}

function RefinementView({
  history,
  summary,
  activeIter,
  onActiveIterChange,
}: {
  history: RefinementEntry[];
  summary?: Record<string, unknown> | null;
  activeIter: number | null;
  onActiveIterChange: (iter: number | null) => void;
}) {
  const [fullScreenIter, setFullScreenIter] = useState<number | null>(null);
  const mode = typeof summary?.mode === "string" ? summary.mode : null;
  const decision = typeof summary?.decision === "string" ? summary.decision : null;
  const stopReason = typeof summary?.stop_reason === "string" ? summary.stop_reason : null;
  const enabled = typeof summary?.enabled === "boolean" ? summary.enabled : null;

  if (history.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center gap-2 p-3">
        <p className="text-xs text-[var(--text-muted)] text-center">
          未触发闭环优化
          <span className="block text-[10px] text-[var(--text-muted)]/70">No refinement triggered</span>
        </p>
        {summary && (
          <div className="flex items-center gap-1 flex-wrap justify-center text-[10px] text-[var(--text-muted)]">
            {mode && <span className="px-1.5 py-0.5 rounded bg-[var(--bg-tertiary)]">mode: {mode}</span>}
            {decision && <span className="px-1.5 py-0.5 rounded bg-[var(--bg-tertiary)]">{decision}</span>}
          </div>
        )}
      </div>
    );
  }

  const initialScore = history[0].score_before;
  const finalEntry = history[history.length - 1];
  const finalScore = finalEntry.score_after ?? finalEntry.score_before;
  const delta = finalScore - initialScore;

  const activeEntry = activeIter !== null ? history.find(e => e.iteration === activeIter) : null;
  const fullScreenEntry = fullScreenIter !== null ? history.find(e => e.iteration === fullScreenIter) : null;

  return (
    <div className="flex flex-col gap-2 flex-1 min-h-0 overflow-hidden">
      {/* Score summary — two rows so narrow columns don't overflow */}
      <div className="flex flex-col gap-1.5 px-2 py-1.5 bg-[var(--bg-tertiary)] rounded-lg flex-shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-[var(--text-muted)]">初始</span>
          <span className="text-xs font-mono text-[var(--text-primary)]">{initialScore.toFixed(3)}</span>
          <div className="flex-1 h-1 bg-[var(--bg-secondary)] rounded-full overflow-hidden min-w-[20px]">
            <div className="h-full bg-[var(--accent-primary)] rounded-full" style={{ width: `${Math.max(0, Math.min(100, finalScore * 100))}%` }} />
          </div>
          <span className="text-xs font-mono text-[var(--text-primary)]">{finalScore.toFixed(3)}</span>
          <span className={`text-[10px] font-medium ${delta > 0 ? "text-green-400" : delta < 0 ? "text-red-400" : "text-[var(--text-muted)]"}`}>
            {delta > 0 ? "+" : ""}{delta.toFixed(3)}
          </span>
        </div>
        <div className="flex items-center gap-1 flex-wrap">
          {mode && (
            <span className={`text-[9px] px-1.5 py-0.5 rounded ${
              enabled ? "bg-emerald-500/20 text-emerald-400" : "bg-[var(--bg-secondary)] text-[var(--text-muted)]"
            }`}>
              {mode}
            </span>
          )}
          <span className="text-[9px] px-1.5 py-0.5 rounded bg-[var(--bg-secondary)] text-[var(--text-muted)]">
            {history.length} iter
          </span>
          {stopReason && (
            <span className="text-[9px] px-1.5 py-0.5 rounded bg-[var(--bg-secondary)] text-[var(--text-muted)] truncate">
              {stopReason}
            </span>
          )}
        </div>
      </div>

      {/* Iteration cards */}
      <div className="flex gap-1 flex-shrink-0 overflow-x-auto pb-1">
        {history.map(entry => {
          const isActive = activeIter === entry.iteration;
          return (
            <button
              key={entry.iteration}
              onClick={() => onActiveIterChange(activeIter === entry.iteration ? null : entry.iteration)}
              className={`flex-shrink-0 flex flex-col items-center gap-0.5 px-2 py-1 rounded-lg border text-[10px] transition-all ${
                isActive
                  ? "border-[var(--accent-primary)] bg-[var(--accent-primary)]/15"
                  : "border-[var(--border-color)] hover:border-[var(--accent-primary)]/50"
              }`}
              title={entry.error ?? `Iter ${entry.iteration}: ${entry.score_before.toFixed(3)} → ${entry.score_after?.toFixed(3) ?? "—"}`}
            >
              <span className="text-[var(--text-muted)] text-[9px]">#{entry.iteration}</span>
              <div className="flex items-center gap-0.5">
                {entry.error ? (
                  <AlertCircle className="w-3 h-3 text-red-400" />
                ) : entry.improved ? (
                  <TrendingUp className="w-3 h-3 text-green-400" />
                ) : (
                  <TrendingDown className="w-3 h-3 text-orange-400" />
                )}
                {entry.score_after !== null ? (
                  <span className={entry.improved ? "text-green-400" : "text-orange-400"}>
                    {entry.score_after.toFixed(2)}
                  </span>
                ) : (
                  <span className="text-red-400">✗</span>
                )}
              </div>
              {typeof entry.delta === "number" && entry.score_after !== null && (
                <span className={`text-[9px] ${entry.delta > 0 ? "text-green-400/80" : entry.delta < 0 ? "text-red-400/80" : "text-[var(--text-muted)]"}`}>
                  {entry.delta > 0 ? "+" : ""}{entry.delta.toFixed(2)}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Detail for selected iteration */}
      <div className="flex-1 min-h-0 overflow-hidden flex flex-col">
        {activeEntry ? (
          <>
            <div className="flex items-center justify-between gap-2 mb-1.5 px-1 flex-shrink-0">
              <div className="flex items-center gap-2 min-w-0 flex-wrap">
                <span className="text-[10px] font-medium text-[var(--text-primary)]">
                  迭代 {activeEntry.iteration}
                </span>
                <span className="text-[10px] text-[var(--text-muted)] font-mono">
                  {activeEntry.score_before.toFixed(3)}
                  {activeEntry.score_after !== null && (
                    <> → {activeEntry.score_after.toFixed(3)}</>
                  )}
                </span>
                {typeof activeEntry.llm_duration_ms === "number" && (
                  <span className="text-[9px] text-[var(--text-muted)]">
                    {(activeEntry.llm_duration_ms / 1000).toFixed(1)}s
                  </span>
                )}
              </div>
              {(activeEntry.llm_io || activeEntry.error) && (
                <button
                  onClick={() => setFullScreenIter(activeEntry.iteration)}
                  className="p-0.5 rounded hover:bg-[var(--bg-tertiary)] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors flex-shrink-0"
                  title="放大查看 / Fullscreen"
                >
                  <Maximize2 className="w-3 h-3" />
                </button>
              )}
            </div>
            {activeEntry.error ? (
              <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg flex-shrink-0 overflow-auto">
                <p className="text-[10px] text-red-400 whitespace-pre-wrap break-words">{activeEntry.error}</p>
              </div>
            ) : activeEntry.llm_io ? (
              <IOView io={activeEntry.llm_io} />
            ) : (
              <p className="text-[10px] text-[var(--text-muted)] px-1">无 I/O 记录</p>
            )}
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center px-2">
              <Minus className="w-5 h-5 text-[var(--text-muted)]/30 mx-auto mb-1" />
              <p className="text-[10px] text-[var(--text-muted)]">点击迭代卡片查看 I/O</p>
              <p className="text-[9px] text-[var(--text-muted)]/60">Click an iteration card</p>
            </div>
          </div>
        )}
      </div>

      {fullScreenEntry && (
        <FullScreenIO entry={fullScreenEntry} onClose={() => setFullScreenIter(null)} />
      )}
    </div>
  );
}

export default function LlmIOPanel({ llmIO, llmMode, refinementSummary, refinementHistory = [], onPreviewSelect }: Props) {
  const hasRefinement = refinementHistory.length > 0;
  const refinementImproved = refinementHistory.some(e => e.improved);
  const [topTab, setTopTab] = useState<TopTab>(
    hasRefinement && refinementImproved ? "refinement" : "initial"
  );
  const [activeIter, setActiveIter] = useState<number | null>(null);
  const modeLabel = llmMode === "off" ? "关闭" : llmMode === "auto" ? "自动" : "强制开启";
  const noData = !llmIO && !hasRefinement;
  const userSelectedRef = useRef(false);

  // Emit-on-click: every user click on a tab or iter card fires a fresh
  // selection. This is intentional — using only useEffect with [topTab,
  // activeIter] deps means clicking an already-active tab is a no-op, so we
  // can't restore the LLM preview after a candidate-row click stole it.
  const emitInitial = useCallback(() => {
    if (!onPreviewSelect) return;
    const glsl = llmIO?.compile_glsl?.trim() ? llmIO.compile_glsl : null;
    onPreviewSelect({
      glsl: glsl ?? null,
      label: glsl ? "初始调用" : null,
      key: glsl ? "initial" : null,
    });
  }, [llmIO, onPreviewSelect]);

  const emitIter = useCallback((iter: number | null) => {
    if (!onPreviewSelect) return;
    if (iter === null) {
      onPreviewSelect({ glsl: null, label: null, key: null });
      return;
    }
    const entry = refinementHistory.find(e => e.iteration === iter);
    const glsl = entry?.compile_glsl?.trim() ? entry.compile_glsl : null;
    onPreviewSelect({
      glsl: glsl ?? null,
      label: glsl ? `iter ${iter}` : null,
      key: glsl ? `iter-${iter}` : null,
    });
  }, [refinementHistory, onPreviewSelect]);

  const handleSelectInitial = useCallback(() => {
    userSelectedRef.current = true;
    setTopTab("initial");
    emitInitial();
  }, [emitInitial]);

  const handleSelectRefinement = useCallback(() => {
    userSelectedRef.current = true;
    setTopTab("refinement");
    emitIter(activeIter);
  }, [activeIter, emitIter]);

  const handleIterChange = useCallback((iter: number | null) => {
    userSelectedRef.current = true;
    setActiveIter(iter);
    emitIter(iter);
  }, [emitIter]);

  // Polling can refresh refinement_history / llmIO mid-run. If the user has
  // an active selection, refresh the parent's preview with the new GLSL so
  // they see the latest compiled output, not a stale snapshot.
  // Guarded by userSelectedRef: only fire after explicit user interaction,
  // otherwise the initial llmIO arrival would auto-inject LLM GLSL into the
  // preview and override the scoreboard's CV-selected candidate.
  useEffect(() => {
    if (!onPreviewSelect || !userSelectedRef.current) return;
    if (topTab === "initial" && llmIO) emitInitial();
    else if (topTab === "refinement" && activeIter !== null) emitIter(activeIter);
    // intentionally NOT depending on topTab/activeIter — those are handled by
    // explicit click handlers above. This effect only re-runs when the data
    // backing the current selection arrives via polling.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [llmIO, refinementHistory]);

  return (
    <div className="bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-xl p-3 h-full flex flex-col overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between gap-2 mb-2 flex-shrink-0 flex-wrap">
        <h3 className="text-sm font-semibold text-[var(--text-primary)] leading-tight">
          模型 I/O
          <span className="block text-[10px] font-normal text-[var(--text-muted)] mt-0.5">LLM Input / Output</span>
        </h3>
        <div className="flex items-center gap-1.5 flex-wrap">
          <Cpu className={`w-3.5 h-3.5 ${llmMode !== "off" ? "text-[var(--accent-primary)]" : "text-[var(--text-muted)]"}`} />
          <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${
            llmMode === "off"
              ? "bg-[var(--bg-tertiary)] text-[var(--text-muted)]"
              : "bg-[var(--accent-primary)]/20 text-[var(--accent-primary)]"
          }`}>
            {modeLabel}
          </span>
          {hasRefinement && (
            <span className="text-[10px] px-1.5 py-0.5 rounded font-medium bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
              {refinementHistory.length} 轮
            </span>
          )}
        </div>
      </div>

      {noData ? (
        <div className="flex-1 flex flex-col items-center justify-center gap-2 p-3">
          <Cpu className="w-8 h-8 text-[var(--text-muted)]/30" />
          <p className="text-xs text-[var(--text-muted)] text-center whitespace-pre-line">
            {llmMode === "off"
              ? "AI 模式已关闭\nAI mode is off"
              : "本次运行未产生 LLM 调用\nNo LLM call in this run"}
          </p>
        </div>
      ) : (
        <>
          {/* Top-level tabs: Initial call | Refinement loop */}
          <div className="flex gap-0.5 mb-2 bg-[var(--bg-tertiary)] rounded-lg p-0.5 flex-shrink-0">
            <button
              onClick={handleSelectInitial}
              disabled={!llmIO}
              className={`flex-1 py-1 px-1 text-[10px] rounded transition-all disabled:opacity-30 min-w-0 ${
                topTab === "initial"
                  ? "bg-[var(--accent-primary)] text-white font-medium"
                  : "text-[var(--text-muted)] hover:text-[var(--text-primary)]"
              }`}
            >
              <span className="block truncate">初始调用</span>
              <span className="block opacity-60 text-[9px] truncate">Initial Call</span>
            </button>
            <button
              onClick={handleSelectRefinement}
              className={`flex-1 py-1 px-1 text-[10px] rounded transition-all relative min-w-0 ${
                topTab === "refinement"
                  ? "bg-[var(--accent-primary)] text-white font-medium"
                  : "text-[var(--text-muted)] hover:text-[var(--text-primary)]"
              }`}
            >
              <span className="block truncate">闭环优化</span>
              <span className="block opacity-60 text-[9px] truncate">Refinement Loop</span>
              {hasRefinement && (
                <span className="absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full bg-emerald-400" />
              )}
            </button>
          </div>

          <div className="flex-1 min-h-0 overflow-hidden flex flex-col">
            {topTab === "initial" && llmIO && <IOView io={llmIO} />}
            {topTab === "refinement" && (
              <RefinementView
                history={refinementHistory}
                summary={refinementSummary}
                activeIter={activeIter}
                onActiveIterChange={handleIterChange}
              />
            )}
          </div>
        </>
      )}
    </div>
  );
}
