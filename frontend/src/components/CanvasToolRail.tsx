// CanvasToolRail.tsx — left floating tool rail rendered as a React Flow <Panel>.
// Lives INSIDE <ReactFlow> (as a child of <BranchCanvas>), so useReactFlow() is
// available for fit-view. Holds the fit-view + reset-layout actions plus the
// relocated run badge / status text from the old workspace toolbar.
import { Panel, useReactFlow } from "@xyflow/react";
import { Maximize2, RotateCcw } from "lucide-react";

interface Props {
  onResetLayout: () => void;
  activeRunShort?: string;
  statusLabel?: string;
}

export default function CanvasToolRail({
  onResetLayout,
  activeRunShort,
  statusLabel,
}: Props) {
  const { fitView } = useReactFlow();

  return (
    <Panel position="top-left" style={{ margin: 8 }}>
      <div
        className="canvas-panel flex flex-col gap-1.5 p-1.5"
      >
        {/* Icon action group */}
        <div className="flex items-center gap-1">
          <button
            onClick={() => fitView({ padding: 0.2 })}
            title="适应视图 / Fit view"
            className="flex items-center justify-center w-7 h-7 rounded transition-all hover:bg-[var(--bg-hover)]"
            style={{ color: "var(--text-secondary)" }}
          >
            <Maximize2 className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={onResetLayout}
            title="重置布局 / Reset layout"
            className="flex items-center justify-center w-7 h-7 rounded transition-all hover:bg-[var(--bg-hover)]"
            style={{ color: "var(--text-secondary)" }}
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>

          {/* Run badge + status */}
          <span
            className="text-[11px] font-mono px-1.5 py-0.5 rounded ml-0.5"
            style={{ background: "var(--bg-tertiary)", color: "var(--text-secondary)" }}
            title={activeRunShort ?? ""}
          >
            {activeRunShort ?? "—"}
          </span>
          <span className="text-[11px]" style={{ color: "var(--text-muted)" }}>
            {statusLabel ?? "—"}
          </span>
        </div>

        {/* Interaction legend */}
        <p
          className="text-[10px] leading-tight px-0.5"
          style={{ color: "var(--text-muted)" }}
        >
          单击预览 · 双击切换 / click=preview · dbl=switch
        </p>
      </div>
    </Panel>
  );
}
