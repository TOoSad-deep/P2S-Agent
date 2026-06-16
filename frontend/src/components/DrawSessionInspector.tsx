// DrawSessionInspector.tsx — draw-session panel (V3.5 Batch Draw).
// Pure presentational; no data fetching. Shows start form or active session card grid.
import { useState } from "react";
import { Layers2, StopCircle, RotateCcw } from "lucide-react";
import type {
  DrawSessionStatus,
  DrawCardStatus,
  DrawCardEventType,
  CreateDrawSessionRequest,
  DrawMoreRequest,
} from "../hooks/usePngShader";
import { fmtScore, truncate } from "../lib/format";
import DrawCard from "./DrawCard";

// ─── statusDot (local copy) ────────────────────────────────────────────────────

function statusDot(status: string): React.ReactElement {
  const color =
    status === "completed"
      ? "bg-emerald-500"
      : status === "running" || status === "queued"
      ? "bg-yellow-400 animate-pulse"
      : status === "failed"
      ? "bg-red-500"
      : status === "partial_failed"
      ? "bg-orange-400"
      : status === "cancelled"
      ? "bg-[var(--text-muted)]"
      : "bg-[var(--text-muted)]";
  return <span className={`inline-block w-2 h-2 rounded-full shrink-0 ${color}`} />;
}

// ─── Constants ─────────────────────────────────────────────────────────────────

const TERMINAL_STATUSES = new Set(["completed", "failed", "cancelled", "partial_failed"]);

type Diversity = "low" | "medium" | "high";
type FilterKey = "all" | "favorite" | "completed" | "running" | "failed" | "eliminated";
type SortKey = "score" | "created" | "strategy";

const CARD_COUNTS_START = [4, 8, 12] as const;
const CARD_COUNTS_MORE = [4, 8] as const;

// ─── Props ─────────────────────────────────────────────────────────────────────

export interface DrawSessionInspectorProps {
  parentRunId: string;
  checkpointId?: string;
  session: DrawSessionStatus | null;
  fusionEnabled?: boolean;
  onStartDraw: (parentRunId: string, request: CreateDrawSessionRequest) => void;
  onDrawMore: (drawId: string, request: DrawMoreRequest) => void;
  onRedrawCard: (drawId: string, runId: string) => void;
  onCardEvent: (
    drawId: string,
    runId: string,
    eventType: DrawCardEventType,
    opts?: { value?: unknown; reason?: string; tags?: string[] }
  ) => void;
  onPreviewCard: (runId: string) => void;
  onContinueCard: (runId: string) => void;
  onSelectWinner?: (drawId: string, runId: string) => void;
  onStopDraw?: (drawId: string) => void;
  disabled?: boolean;
  error?: string | null;
}

// ─── Filter + sort helpers ─────────────────────────────────────────────────────

function filterCards(cards: DrawCardStatus[], filter: FilterKey): DrawCardStatus[] {
  switch (filter) {
    case "favorite":
      return cards.filter((c) => c.favorite);
    case "completed":
      return cards.filter((c) => c.status === "completed");
    case "running":
      // treat "queued" as running
      return cards.filter((c) => c.status === "running" || c.status === "queued");
    case "failed":
      return cards.filter((c) => c.status === "failed");
    case "eliminated":
      return cards.filter((c) => c.eliminated);
    default:
      return cards;
  }
}

function sortCards(cards: DrawCardStatus[], sort: SortKey): DrawCardStatus[] {
  const copy = [...cards];
  switch (sort) {
    case "score":
      return copy.sort((a, b) => {
        const sa = a.final_score ?? a.current_score ?? -Infinity;
        const sb = b.final_score ?? b.current_score ?? -Infinity;
        return sb - sa; // desc
      });
    case "strategy":
      return copy.sort((a, b) => {
        const la = (a.strategy_label ?? a.label ?? "").toLowerCase();
        const lb = (b.strategy_label ?? b.label ?? "").toLowerCase();
        if (la < lb) return -1;
        if (la > lb) return 1;
        return a.index - b.index; // deterministic secondary
      });
    case "created":
    default:
      return copy.sort((a, b) => a.index - b.index); // asc
  }
}

// ─── StartForm sub-component ───────────────────────────────────────────────────

interface StartFormProps {
  parentRunId: string;
  checkpointId?: string;
  onStartDraw: (parentRunId: string, request: CreateDrawSessionRequest) => void;
  disabled?: boolean;
}

function StartForm({ parentRunId, checkpointId, onStartDraw, disabled }: StartFormProps) {
  const [feedback, setFeedback] = useState("");
  const [cardCount, setCardCount] = useState<number>(8);
  const [diversity, setDiversity] = useState<Diversity>("medium");

  const canSubmit = !disabled && feedback.trim().length > 0;

  const handleStart = () => {
    if (!canSubmit) return;
    onStartDraw(parentRunId, {
      checkpoint_id: checkpointId ?? "final:selected",
      feedback: feedback.trim(),
      card_count: cardCount,
      diversity,
    });
  };

  return (
    <div className="flex flex-col gap-2">
      {/* Feedback */}
      <textarea
        value={feedback}
        onChange={(e) => setFeedback(e.target.value)}
        disabled={disabled}
        rows={3}
        placeholder="例如：尝试不同风格的云彩处理，保持整体蓝色调。"
        className="w-full text-xs p-2 rounded bg-[var(--bg-tertiary)] border border-[var(--border-color)] text-[var(--text-primary)] resize-y placeholder:text-[var(--text-muted)] disabled:opacity-40"
      />

      {/* Card count */}
      <div className="flex items-center gap-1.5">
        <span className="text-[11px] text-[var(--text-muted)] shrink-0">数量 Count</span>
        <div className="flex items-center gap-0.5 bg-[var(--bg-tertiary)] rounded-md p-0.5">
          {CARD_COUNTS_START.map((n) => (
            <button
              key={n}
              onClick={() => setCardCount(n)}
              disabled={disabled}
              className={`px-2 py-0.5 text-[11px] rounded-md transition-all disabled:opacity-40 ${
                cardCount === n
                  ? "bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-medium"
                  : "text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)]"
              }`}
            >
              {n}
            </button>
          ))}
        </div>
      </div>

      {/* Diversity */}
      <div className="flex items-center gap-1.5">
        <span className="text-[11px] text-[var(--text-muted)] shrink-0">多样性 Diversity</span>
        <div className="flex items-center gap-0.5 bg-[var(--bg-tertiary)] rounded-md p-0.5">
          {(["low", "medium", "high"] as const).map((d) => (
            <button
              key={d}
              onClick={() => setDiversity(d)}
              disabled={disabled}
              className={`px-2 py-0.5 text-[11px] rounded-md transition-all disabled:opacity-40 ${
                diversity === d
                  ? "bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-medium"
                  : "text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)]"
              }`}
            >
              {d}
            </button>
          ))}
        </div>
      </div>

      {/* Submit */}
      <button
        onClick={handleStart}
        disabled={!canSubmit}
        className="flex items-center justify-center gap-1.5 w-full py-1.5 text-xs font-medium rounded-md transition-all disabled:opacity-40 disabled:cursor-not-allowed bg-gradient-to-r from-emerald-500 to-emerald-600 text-white hover:from-emerald-600 hover:to-emerald-700"
      >
        <Layers2 className="w-3.5 h-3.5" />
        开始抽卡 / Start draw
      </button>
    </div>
  );
}

// ─── DrawMore sub-component ────────────────────────────────────────────────────

interface DrawMoreFormProps {
  drawId: string;
  onDrawMore: (drawId: string, request: DrawMoreRequest) => void;
  disabled?: boolean;
}

function DrawMoreForm({ drawId, onDrawMore, disabled }: DrawMoreFormProps) {
  const [cardCount, setCardCount] = useState<number>(4);
  const [diversity, setDiversity] = useState<Diversity>("medium");

  return (
    <div className="flex flex-wrap items-center gap-1.5 pt-2 border-t border-[var(--border-color)]">
      <span className="text-[11px] text-[var(--text-muted)] shrink-0">再抽 Draw more:</span>

      {/* Count */}
      <div className="flex items-center gap-0.5 bg-[var(--bg-tertiary)] rounded-md p-0.5">
        {CARD_COUNTS_MORE.map((n) => (
          <button
            key={n}
            onClick={() => setCardCount(n)}
            disabled={disabled}
            className={`px-2 py-0.5 text-[11px] rounded-md transition-all disabled:opacity-40 ${
              cardCount === n
                ? "bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-medium"
                : "text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)]"
            }`}
          >
            {n}
          </button>
        ))}
      </div>

      {/* Diversity */}
      <div className="flex items-center gap-0.5 bg-[var(--bg-tertiary)] rounded-md p-0.5">
        {(["low", "medium", "high"] as const).map((d) => (
          <button
            key={d}
            onClick={() => setDiversity(d)}
            disabled={disabled}
            className={`px-2 py-0.5 text-[11px] rounded-md transition-all disabled:opacity-40 ${
              diversity === d
                ? "bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-medium"
                : "text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)]"
            }`}
          >
            {d}
          </button>
        ))}
      </div>

      {/* Trigger */}
      <button
        onClick={() => onDrawMore(drawId, { card_count: cardCount, diversity })}
        disabled={disabled}
        className="flex items-center gap-1 px-2 py-0.5 text-[11px] rounded-md transition-all disabled:opacity-40 disabled:cursor-not-allowed bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-medium hover:from-emerald-600 hover:to-emerald-700"
      >
        <RotateCcw className="w-3 h-3" />
        再抽 / Draw more
      </button>
    </div>
  );
}

// ─── Main component ────────────────────────────────────────────────────────────

export default function DrawSessionInspector({
  parentRunId,
  checkpointId,
  session,
  fusionEnabled = false,
  onStartDraw,
  onDrawMore,
  onRedrawCard,
  onCardEvent,
  onPreviewCard,
  onContinueCard,
  onSelectWinner: _onSelectWinner,
  onStopDraw,
  disabled = false,
  error,
}: DrawSessionInspectorProps) {
  // Local UI state for filter + sort (only used when session != null)
  const [filter, setFilter] = useState<FilterKey>("all");
  const [sort, setSort] = useState<SortKey>("score");

  // ── Start form ────────────────────────────────────────────────────────────
  if (session === null) {
    return (
      <div className="flex flex-col gap-2">
        <p className="text-[11px] font-medium text-[var(--text-secondary)]">
          批量抽卡 <span className="text-[var(--text-muted)] font-normal">Batch Draw</span>
        </p>
        <StartForm
          parentRunId={parentRunId}
          checkpointId={checkpointId}
          onStartDraw={onStartDraw}
          disabled={disabled}
        />
        {error && (
          <p className="text-[11px] text-red-400 leading-snug">{error}</p>
        )}
      </div>
    );
  }

  // ── Active session ────────────────────────────────────────────────────────
  const isTerminal = TERMINAL_STATUSES.has(session.status);
  const canStop = !!onStopDraw && !isTerminal;

  const visibleCards = sortCards(filterCards(session.cards, filter), sort);

  const FILTER_OPTIONS: { key: FilterKey; label: string }[] = [
    { key: "all", label: "全部 All" },
    { key: "favorite", label: "收藏" },
    { key: "completed", label: "完成" },
    { key: "running", label: "运行中" },
    { key: "failed", label: "失败" },
    { key: "eliminated", label: "已淘汰" },
  ];

  const SORT_OPTIONS: { key: SortKey; label: string }[] = [
    { key: "score", label: "分数 Score" },
    { key: "created", label: "序号 Index" },
    { key: "strategy", label: "策略 Strategy" },
  ];

  return (
    <div className="flex flex-col gap-2">
      {/* Session header */}
      <div className="flex items-start gap-2">
        <div className="flex flex-col gap-0.5 flex-1 min-w-0">
          <div className="flex items-center gap-1.5 flex-wrap">
            {statusDot(session.status)}
            <span className="text-[11px] font-medium text-[var(--text-primary)] truncate">
              {truncate(session.feedback, 60)}
            </span>
            {session.winner_run_id && (
              <span className="text-[10px] px-1 py-0.5 rounded bg-emerald-500/10 text-emerald-400 font-mono shrink-0">
                winner set
              </span>
            )}
          </div>
          <p className="text-[10px] text-[var(--text-muted)]">
            {session.completed_count}/{session.requested_count} done
            {" · "}
            {session.running_count} running
            {" · "}
            {session.failed_count} failed
            {" · "}
            <span className="font-mono">{fmtScore(undefined)}</span>
          </p>
        </div>

        {/* Stop button */}
        {canStop && (
          <button
            onClick={() => onStopDraw?.(session.draw_id)}
            disabled={disabled}
            title="停止抽卡 Stop draw"
            className="flex items-center gap-1 px-2 py-0.5 text-[11px] rounded transition-all shrink-0 disabled:opacity-40 disabled:cursor-not-allowed bg-[var(--bg-tertiary)] text-[var(--text-secondary)] hover:bg-red-500/20 hover:text-red-400"
          >
            <StopCircle className="w-3 h-3" />
            停止 Stop
          </button>
        )}
      </div>

      {/* Filter row */}
      <div className="flex items-center gap-0.5 flex-wrap">
        <span className="text-[10px] text-[var(--text-muted)] shrink-0 mr-0.5">筛选:</span>
        {FILTER_OPTIONS.map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setFilter(key)}
            className={`px-1.5 py-0.5 text-[10px] rounded transition-all ${
              filter === key
                ? "bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-medium"
                : "text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)] bg-[var(--bg-tertiary)]"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Sort row */}
      <div className="flex items-center gap-0.5 flex-wrap">
        <span className="text-[10px] text-[var(--text-muted)] shrink-0 mr-0.5">排序:</span>
        {SORT_OPTIONS.map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setSort(key)}
            className={`px-1.5 py-0.5 text-[10px] rounded transition-all ${
              sort === key
                ? "bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-medium"
                : "text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)] bg-[var(--bg-tertiary)]"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Card grid */}
      {visibleCards.length > 0 ? (
        <div className="grid grid-cols-2 gap-2">
          {visibleCards.map((card) => (
            <DrawCard
              key={card.run_id}
              card={card}
              isWinner={card.run_id === session.winner_run_id}
              fusionEnabled={fusionEnabled}
              onPreview={onPreviewCard}
              onFavorite={(rid, next) =>
                onCardEvent(session.draw_id, rid, "favorite", { value: next })
              }
              onEliminate={(rid, next) =>
                onCardEvent(session.draw_id, rid, "eliminate", { value: next })
              }
              onRedraw={(rid) => onRedrawCard(session.draw_id, rid)}
              onContinue={onContinueCard}
              onUseAsBase={
                fusionEnabled
                  ? (rid) => onCardEvent(session.draw_id, rid, "use_as_fusion_base")
                  : undefined
              }
              onUseRegion={
                fusionEnabled
                  ? (rid) => onCardEvent(session.draw_id, rid, "use_as_region_source")
                  : undefined
              }
              disabled={disabled}
            />
          ))}
        </div>
      ) : (
        <p className="text-[11px] text-[var(--text-muted)] py-3 text-center">
          暂无卡片 No cards match filter
        </p>
      )}

      {/* Draw more */}
      <DrawMoreForm
        drawId={session.draw_id}
        onDrawMore={onDrawMore}
        disabled={disabled}
      />

      {/* Error */}
      {error && (
        <p className="text-[11px] text-red-400 leading-snug">{error}</p>
      )}
    </div>
  );
}
