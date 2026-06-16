import type { Node, Edge } from "@xyflow/react";

export type BranchCanvasNodeType =
  | "input"
  | "run"
  | "checkpoint"
  | "branch_action"
  | "variant_group"
  | "variant_run"
  | "region_constraint"
  | "preference";

// React Flow v12 requires node data to extend Record<string, unknown>.
export interface BranchCanvasNodeData extends Record<string, unknown> {
  type: BranchCanvasNodeType;
  run_id?: string;
  checkpoint_id?: string;
  parent_run_id?: string | null;
  source_checkpoint_id?: string | null;
  title?: string | null;
  label: string;
  status?: string;
  score?: number | null;
  delta?: number | null;
  accepted?: boolean | null;
  favorite?: boolean;
  feedback?: string | null;
  thumbnail_artifact_id?: string | null;
  shader_artifact_id?: string | null;
  group_id?: string | null;
  collapsed?: boolean;
}

export type BranchCanvasEdgeRelation =
  | "timeline_next"
  | "branch_from"
  | "active_run"
  | "variant_child"
  | "constraint_applies"
  | "preference_influences";

export interface BranchCanvasEdgeData extends Record<string, unknown> {
  relation: BranchCanvasEdgeRelation;
  label?: string;
}

export type BranchCanvasNode = Node<BranchCanvasNodeData>;
export type BranchCanvasEdge = Edge<BranchCanvasEdgeData>;

// buildBranchCanvasModel: added in V2.1-2
