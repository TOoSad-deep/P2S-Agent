// BranchCompareStrip.tsx — side-by-side render thumbnails for the active run and its parent (V2).
import { useState } from "react";
import type { BranchTreeNode } from "../hooks/usePngShader";

const API_BASE = import.meta.env.VITE_API_BASE || "";

interface Props {
  activeRunId: string | null;
  tree: BranchTreeNode | null;
}

/** Recursively search the tree for a node by run_id. */
function findNode(node: BranchTreeNode, runId: string): BranchTreeNode | null {
  if (node.run_id === runId) return node;
  for (const child of node.children) {
    const found = findNode(child, runId);
    if (found) return found;
  }
  return null;
}

/** Recursively find the parent of a node with the given run_id. */
function findParent(node: BranchTreeNode, runId: string): BranchTreeNode | null {
  for (const child of node.children) {
    if (child.run_id === runId) return node;
    const found = findParent(child, runId);
    if (found) return found;
  }
  return null;
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
    <div className="flex flex-col gap-2 px-3 py-2.5 bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-lg">
      <p className="text-xs font-medium text-[var(--text-primary)] leading-tight">
        对比
        <span className="ml-2 text-[var(--text-muted)] font-normal">Compare</span>
      </p>
      <div className="flex items-start gap-4">
        <Tile runId={activeRunId} label="当前 / Current" />
        {parentNode && (
          <Tile runId={parentNode.run_id} label="父级 / Parent" />
        )}
      </div>
    </div>
  );
}
