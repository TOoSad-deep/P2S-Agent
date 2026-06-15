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

export default function BranchCompareStrip({ activeRunId, tree }: Props) {
  if (!activeRunId || !tree) {
    return null;
  }

  const activeNode = findNode(tree, activeRunId);
  const parentNode = activeNode ? findParent(tree, activeRunId) : null;

  // If we can't find the active node in the tree at all, don't render.
  if (!activeNode) return null;

  return (
    // M-4: borderless layout — card chrome lives in BranchWorkspacePanel only.
    <div className="flex flex-col gap-2 px-1 py-1">
      <p className="text-xs font-medium text-[var(--text-primary)] leading-tight">
        对比
        <span className="ml-2 text-[var(--text-muted)] font-normal">Compare</span>
      </p>
      <div className="flex items-start gap-4">
        {/* I-1: key resets errored state when the run changes */}
        <Tile key={activeRunId} runId={activeRunId} label="当前 / Current" />
        {parentNode && (
          <Tile key={parentNode.run_id} runId={parentNode.run_id} label="父级 / Parent" />
        )}
      </div>
    </div>
  );
}
