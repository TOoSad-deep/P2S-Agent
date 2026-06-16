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
  /**
   * Must be a stable reference across renders (module constant or useMemo).
   * Passing a new object each render triggers React Flow warning 002 and
   * remounts all nodes.
   */
  nodeTypes?: NodeTypes;
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
        borderColor: "var(--border-color)",
        background: "var(--bg-secondary)",
      }}
    >
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        colorMode="dark"
        fitView
        onNodeClick={(_, node) => onNodeClick?.(node.id)}
        onNodeDoubleClick={(_, node) => onNodeDoubleClick?.(node.id)}
        onNodeDragStop={(_, node) => onNodeDragStop?.(node.id, node.position)} // NOTE: RF owns node position in uncontrolled mode; V2.1-6 must round-trip this into the layout model (or set nodesDraggable=false).
        onSelectionChange={({ nodes: selectedNodes }) =>
          onSelectionChange?.(selectedNodes.map((n) => n.id)) // node IDs only; edge selection is out of scope for the current design
        }
      >
        <Background />
        <Controls />
        <MiniMap />
      </ReactFlow>
    </div>
  );
}
