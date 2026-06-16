import "@xyflow/react/dist/style.css";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  type NodeTypes,
} from "@xyflow/react";
import type { BranchCanvasNode, BranchCanvasEdge } from "../lib/branchCanvasModel";

interface Props {
  nodes: BranchCanvasNode[];
  edges: BranchCanvasEdge[];
  nodeTypes?: NodeTypes;
  selectedNodeId?: string | null;
  onNodeClick?: (id: string) => void;
  onNodeDoubleClick?: (id: string) => void;
  onSelectionChange?: (ids: string[]) => void;
  onNodeDragStop?: (id: string, pos: { x: number; y: number }) => void;
}

export default function BranchCanvas({
  nodes,
  edges,
  nodeTypes,
  onNodeClick,
  onNodeDoubleClick,
  onSelectionChange,
  onNodeDragStop,
}: Props) {
  return (
    <div
      className="w-full h-[520px] rounded-md border overflow-hidden"
      style={{
        borderColor: "var(--color-border, #334155)",
        background: "var(--color-surface, #0f172a)",
      }}
    >
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        onNodeClick={(_, node) => onNodeClick?.(node.id)}
        onNodeDoubleClick={(_, node) => onNodeDoubleClick?.(node.id)}
        onNodeDragStop={(_, node) => onNodeDragStop?.(node.id, node.position)}
        onSelectionChange={({ nodes: selectedNodes }) =>
          onSelectionChange?.(selectedNodes.map((n) => n.id))
        }
      >
        <Background />
        <Controls />
        <MiniMap />
      </ReactFlow>
    </div>
  );
}
