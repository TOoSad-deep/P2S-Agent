// BranchCompareStrip.tsx — side-by-side render thumbnails for the active run and its parent (V2).
import { useState } from "react";
import type { BranchTreeNode } from "../hooks/usePngShader";
import { findNode, findParent } from "../lib/branchTree";

const API_BASE = import.meta.env.VITE_API_BASE || "";

interface Props {
  activeRunId: string | null;
  tree: BranchTreeNode | null;
}

interface TileProps {
  runId: string;
  label: string;
}

function Tile({ runId, label }: TileProps) {
  const [errored, setErrored] = useState(false);
  const src = `${API_BASE}/png-shader/runs/${runId}/artifacts/selected_render`;

  return (
    <div className="flex flex-col items-center gap-1">
      {errored ? (
        <div
          className="w-32 h-32 flex items-center justify-center rounded bg-[var(--bg-tertiary)] border border-[var(--border-color)]"
        >
          <span className="text-[10px] text-[var(--text-muted)] text-center px-1">
            无预览 / No preview
          </span>
        </div>
      ) : (
        <img
          src={src}
          alt={label}
          className="w-32 h-32 object-contain rounded border border-[var(--border-color)] bg-[var(--bg-tertiary)]"
          onError={() => setErrored(true)}
        />
      )}
      <span className="text-[10px] text-[var(--text-muted)] text-center">{label}</span>
    </div>
  );
}

// Cap sibling tiles so a wide variant fan-out can't blow out the strip.
const MAX_SIBLING_TILES = 4;

function siblingLabel(node: BranchTreeNode): string {
  const tag = node.title?.trim() || node.variant_label?.trim();
  const short = node.run_id.slice(-4);
  return tag ? `兄弟 / ${tag}` : `兄弟 / Sibling ${short}`;
}

export default function BranchCompareStrip({ activeRunId, tree }: Props) {
  if (!activeRunId || !tree) {
    return null;
  }

  const activeNode = findNode(tree, activeRunId);
  const parentNode = activeNode ? findParent(tree, activeRunId) : null;

  // If we can't find the active node in the tree at all, don't render.
  if (!activeNode) return null;

  // Siblings share the same parent; useful to compare alternatives at the same
  // branch point (sibling / two-node compare, not just current vs parent).
  const siblings = parentNode
    ? parentNode.children.filter((c) => c.run_id !== activeRunId)
    : [];
  const shownSiblings = siblings.slice(0, MAX_SIBLING_TILES);
  const hiddenSiblingCount = siblings.length - shownSiblings.length;

  return (
    // M-4: borderless layout — card chrome lives in BranchWorkspacePanel only.
    <div className="flex flex-col gap-2 px-1 py-1">
      <p className="text-xs font-medium text-[var(--text-primary)] leading-tight">
        对比
        <span className="ml-2 text-[var(--text-muted)] font-normal">Compare</span>
      </p>
      <div className="flex items-start gap-4 flex-wrap">
        {/* I-1: key resets errored state when the run changes */}
        <Tile key={activeRunId} runId={activeRunId} label="当前 / Current" />
        {parentNode && (
          <Tile key={parentNode.run_id} runId={parentNode.run_id} label="父级 / Parent" />
        )}
        {shownSiblings.map((s) => (
          <Tile key={s.run_id} runId={s.run_id} label={siblingLabel(s)} />
        ))}
        {hiddenSiblingCount > 0 && (
          <div className="flex flex-col items-center justify-center w-32 h-32 rounded bg-[var(--bg-tertiary)] border border-[var(--border-color)]">
            <span className="text-[11px] text-[var(--text-muted)]">+{hiddenSiblingCount}</span>
            <span className="text-[10px] text-[var(--text-muted)]">更多 / more</span>
          </div>
        )}
      </div>
    </div>
  );
}
