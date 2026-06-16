// DrawCard.tsx — single draw-card cell (V3.5 Batch Draw).
// Pure presentational; no data fetching. Mirror VariantCard styling from BranchCanvasInspector.
import { memo, useState } from "react";
import { Star, Eye, RefreshCw, GitBranch, X, Layers, Crosshair } from "lucide-react";
import type { DrawCardStatus } from "../hooks/usePngShader";
import { fmtScore } from "../lib/format";

// ─── statusDot (local copy — not exported from BranchCanvasInspector) ─────────

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

// ─── Props ────────────────────────────────────────────────────────────────────

export interface DrawCardProps {
  card: DrawCardStatus;
  isWinner?: boolean;
  fusionEnabled?: boolean;
  onPreview: (runId: string) => void;
  onFavorite: (runId: string, next: boolean) => void;
  onEliminate: (runId: string, next: boolean) => void;
  onRedraw: (runId: string) => void;
  onContinue: (runId: string) => void;
  onUseAsBase?: (runId: string) => void;
  onUseRegion?: (runId: string) => void;
  disabled?: boolean;
}

// ─── Component ────────────────────────────────────────────────────────────────

const DrawCard = memo(function DrawCard({
  card,
  isWinner = false,
  fusionEnabled = false,
  onPreview,
  onFavorite,
  onEliminate,
  onRedraw,
  onContinue,
  onUseAsBase,
  onUseRegion,
  disabled = false,
}: DrawCardProps) {
  const [imgError, setImgError] = useState(false);
  const isFavorite = isWinner || !!card.favorite;
  const isEliminated = !!card.eliminated;

  return (
    <div
      className={`flex flex-col gap-1.5 px-2 py-1.5 rounded-md border transition-all ${
        isWinner
          ? "border-emerald-500/40 bg-emerald-500/5"
          : isEliminated
          ? "border-[var(--border-color)] opacity-50"
          : "border-[var(--border-color)] bg-[var(--bg-tertiary)]"
      }`}
    >
      {/* Thumbnail */}
      <div className="relative w-full aspect-video rounded overflow-hidden bg-[var(--bg-tertiary)] shrink-0">
        {card.thumbnail_url && !imgError ? (
          <img
            src={card.thumbnail_url}
            alt={card.label}
            onError={() => setImgError(true)}
            className="w-full h-full object-cover"
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <span className="text-[10px] text-[var(--text-muted)] opacity-50">No preview</span>
          </div>
        )}
      </div>

      {/* Header row */}
      <div className="flex items-center gap-1.5 flex-wrap">
        <span className="text-[10px] font-mono px-1 py-0.5 rounded bg-[var(--bg-secondary)] text-[var(--text-muted)] shrink-0">
          #{card.index}
        </span>
        <span
          className={`text-[11px] font-medium text-[var(--text-primary)] truncate flex-1 min-w-0 ${
            isEliminated ? "line-through" : ""
          }`}
        >
          {card.label}
        </span>
        {isFavorite && (
          <span title="收藏 Favorite / Winner" className="shrink-0">
            <Star className="w-3.5 h-3.5 text-emerald-400 fill-current" />
          </span>
        )}
      </div>

      {/* Status row */}
      <div className="flex items-center gap-1.5 flex-wrap">
        {statusDot(card.status)}
        <span className="text-[11px] font-mono text-[var(--text-muted)] shrink-0">
          {fmtScore(card.final_score ?? card.current_score)}
        </span>
        {card.strategy_label && (
          <span className="text-[10px] text-[var(--text-muted)] truncate">{card.strategy_label}</span>
        )}
      </div>

      {/* Replacement indicator */}
      {card.replacement_of_run_id && (
        <p className="text-[10px] text-[var(--text-muted)]">↻ replacement</p>
      )}

      {/* Error */}
      {card.error && (
        <p className="text-[11px] text-red-400 leading-snug">{card.error}</p>
      )}

      {/* Action buttons */}
      <div className="flex flex-wrap items-center gap-1 pt-0.5">
        {/* Preview */}
        <button
          onClick={() => onPreview(card.run_id)}
          disabled={disabled}
          title="预览 Preview"
          className="px-2 py-0.5 text-[11px] rounded transition-all disabled:opacity-40 disabled:cursor-not-allowed bg-[var(--bg-secondary)] text-[var(--text-secondary)] hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)] flex items-center gap-1"
        >
          <Eye className="w-3 h-3" />
          预览 Preview
        </button>

        {/* Favorite */}
        <button
          onClick={() => onFavorite(card.run_id, !card.favorite)}
          disabled={disabled}
          title={card.favorite ? "取消收藏 Unfavorite" : "收藏 Favorite"}
          className={`p-1 rounded transition-all disabled:opacity-40 disabled:cursor-not-allowed ${
            card.favorite
              ? "text-emerald-400"
              : "text-[var(--text-muted)] hover:text-emerald-400"
          }`}
        >
          <Star className={`w-3.5 h-3.5 ${card.favorite ? "fill-current" : ""}`} />
        </button>

        {/* Eliminate */}
        <button
          onClick={() => onEliminate(card.run_id, !card.eliminated)}
          disabled={disabled}
          title={isEliminated ? "恢复 Restore" : "淘汰 Eliminate"}
          className={`p-1 rounded transition-all disabled:opacity-40 disabled:cursor-not-allowed ${
            isEliminated
              ? "text-red-400"
              : "text-[var(--text-muted)] hover:text-red-400"
          }`}
        >
          <X className="w-3.5 h-3.5" />
        </button>

        {/* Redraw */}
        <button
          onClick={() => onRedraw(card.run_id)}
          disabled={disabled}
          title="重新抽卡 Redraw"
          className="px-2 py-0.5 text-[11px] rounded transition-all disabled:opacity-40 disabled:cursor-not-allowed bg-[var(--bg-secondary)] text-[var(--text-secondary)] hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)] flex items-center gap-1"
        >
          <RefreshCw className="w-3 h-3" />
          重抽 Redraw
        </button>

        {/* Continue — only for completed */}
        {card.status === "completed" && (
          <button
            onClick={() => onContinue(card.run_id)}
            disabled={disabled}
            title="从此卡继续优化 Continue refining"
            className="px-2 py-0.5 text-[11px] rounded transition-all disabled:opacity-40 disabled:cursor-not-allowed bg-[var(--bg-secondary)] text-[var(--text-secondary)] hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)] flex items-center gap-1"
          >
            <GitBranch className="w-3 h-3" />
            继续 Continue
          </button>
        )}

        {/* Fusion: Use as base — always rendered, disabled unless fusionEnabled */}
        <button
          onClick={() => onUseAsBase?.(card.run_id)}
          disabled={disabled || !fusionEnabled || !onUseAsBase}
          title={fusionEnabled ? "用作融合基底 Use as fusion base" : "Available in V4.5"}
          className="p-1 rounded transition-all disabled:opacity-40 disabled:cursor-not-allowed text-[var(--text-muted)] hover:text-[var(--text-primary)]"
        >
          <Layers className="w-3.5 h-3.5" />
        </button>

        {/* Fusion: Use region — always rendered, disabled unless fusionEnabled */}
        <button
          onClick={() => onUseRegion?.(card.run_id)}
          disabled={disabled || !fusionEnabled || !onUseRegion}
          title={fusionEnabled ? "用作区域源 Use as region source" : "Available in V4.5"}
          className="p-1 rounded transition-all disabled:opacity-40 disabled:cursor-not-allowed text-[var(--text-muted)] hover:text-[var(--text-primary)]"
        >
          <Crosshair className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  );
});

export default DrawCard;
