// branchTree.ts — shared recursive tree helpers for branch workspace components (V2).
import type { BranchTreeNode } from "../hooks/usePngShader";

/** Recursively search the tree for a node by run_id. */
export function findNode(node: BranchTreeNode, runId: string): BranchTreeNode | null {
  if (node.run_id === runId) return node;
  for (const child of node.children) {
    const found = findNode(child, runId);
    if (found) return found;
  }
  return null;
}

/** Recursively find the parent of a node with the given run_id. */
export function findParent(node: BranchTreeNode, runId: string): BranchTreeNode | null {
  for (const child of node.children) {
    if (child.run_id === runId) return node;
    const found = findParent(child, runId);
    if (found) return found;
  }
  return null;
}
