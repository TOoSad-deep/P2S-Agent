// PreviewDock.tsx — bottom-right floating preview dock (P5).
// Single-click on a canvas node with a run_id shows its selected_render
// side-by-side with the reference image. Double-click still switches the active run.
import { useState } from "react";
import { Panel } from "@xyflow/react";
import type { BranchCanvasNode } from "../lib/branchCanvasModel";
import { fmtScore } from "../lib/format";

interface PreviewDockProps {
  referenceUrl?: string | null;   // inputImageUrl from context
  node: BranchCanvasNode | null;  // selectedNode (set by single-click)
}

const PREVIEW_OPEN_KEY = "p2s.canvas.previewOpen";

function loadPreviewOpen(): boolean {
  try {
    const raw = localStorage.getItem(PREVIEW_OPEN_KEY);
    if (raw === "0") return false;
    return true; // default true when absent or invalid
  } catch {
    return true;
  }
}

function savePreviewOpen(value: boolean): void {
  try {
    localStorage.setItem(PREVIEW_OPEN_KEY, value ? "1" : "0");
  } catch {
    // quota exceeded or SSR — ignore
  }
}

export default function PreviewDock({ referenceUrl, node }: PreviewDockProps) {
  const [open, setOpen] = useState(loadPreviewOpen);
  const [renderError, setRenderError] = useState(false);

  function toggleOpen(next: boolean): void {
    setOpen(next);
    savePreviewOpen(next);
  }

  const runId =
    typeof node?.data?.run_id === "string" ? node.data.run_id : null;

  // Reset error state when the node changes so the new img gets a fresh attempt.
  // Using a key on the img handles this — see renderImgKey below.
  const renderImgKey = runId ?? "none";

  const score =
    typeof node?.data?.score === "number"
      ? node.data.score
      : typeof node?.data?.final_score === "number"
        ? (node.data.final_score as number)
        : null;

  const nodeLabel =
    typeof node?.data?.label === "string" ? node.data.label : null;

  if (!open) {
    return (
      <Panel position="bottom-right" style={{ margin: 8 }}>
        <button
          onClick={() => toggleOpen(true)}
          title="展开预览 / Preview"
          className="canvas-panel flex items-center justify-center w-7 py-3 transition-all hover:bg-[var(--bg-hover)]"
          style={{
            color: "var(--text-secondary)",
            writingMode: "vertical-rl",
            fontSize: 11,
            letterSpacing: "0.05em",
          }}
        >
          预览
        </button>
      </Panel>
    );
  }

  return (
    <Panel position="bottom-right" style={{ margin: 8 }}>
      <div
        className="canvas-panel"
        style={{ width: 260 }}
      >
        {/* Header */}
        <div
          className="flex items-center justify-between px-3 py-1.5 border-b"
          style={{ borderColor: "var(--border-color)" }}
        >
          <span
            className="text-[11px] font-medium"
            style={{ color: "var(--text-secondary)" }}
          >
            预览 / Preview
          </span>
          <button
            onClick={() => toggleOpen(false)}
            title="折叠 / Collapse"
            className="flex items-center justify-center w-5 h-5 rounded transition-all hover:bg-[var(--bg-hover)]"
            style={{ color: "var(--text-muted)", fontSize: 13 }}
          >
            «
          </button>
        </div>

        {/* Body */}
        <div className="p-2">
          {runId ? (
            <>
              {/* Side-by-side image tiles */}
              <div className="flex gap-2">
                {/* Reference tile */}
                <div className="flex-1 flex flex-col gap-1">
                  <div
                    className="rounded overflow-hidden flex items-center justify-center"
                    style={{
                      height: 110,
                      background: "var(--bg-tertiary)",
                    }}
                  >
                    {referenceUrl ? (
                      <img
                        src={referenceUrl}
                        alt="Reference"
                        className="w-full h-full object-contain"
                      />
                    ) : (
                      <span
                        className="text-[10px] text-center leading-snug"
                        style={{ color: "var(--text-muted)" }}
                      >
                        无参考
                        <br />
                        No ref
                      </span>
                    )}
                  </div>
                  <span
                    className="text-[10px] text-center"
                    style={{ color: "var(--text-muted)" }}
                  >
                    参考 Reference
                  </span>
                </div>

                {/* Node render tile */}
                <div className="flex-1 flex flex-col gap-1">
                  <div
                    className="rounded overflow-hidden flex items-center justify-center"
                    style={{
                      height: 110,
                      background: "var(--bg-tertiary)",
                    }}
                  >
                    {renderError ? (
                      <span
                        className="text-[10px] text-center leading-snug"
                        style={{ color: "var(--text-muted)" }}
                      >
                        无渲染
                        <br />
                        No render
                      </span>
                    ) : (
                      <img
                        key={renderImgKey}
                        src={`/png-shader/runs/${runId}/artifacts/selected_render`}
                        alt="Node render"
                        className="w-full h-full object-contain"
                        onError={() => setRenderError(true)}
                        onLoad={() => setRenderError(false)}
                      />
                    )}
                  </div>
                  <span
                    className="text-[10px] text-center truncate"
                    style={{ color: "var(--text-muted)" }}
                    title={nodeLabel ?? undefined}
                  >
                    {nodeLabel
                      ? nodeLabel.length > 14
                        ? nodeLabel.slice(0, 13) + "…"
                        : nodeLabel
                      : "节点渲染"}
                  </span>
                </div>
              </div>

              {/* Score row */}
              {score !== null && (
                <div
                  className="mt-1.5 text-center text-[11px]"
                  style={{ color: "var(--text-secondary)" }}
                >
                  得分 Score:{" "}
                  <span className="font-medium">{fmtScore(score)}</span>
                </div>
              )}

              {/* Hint */}
              <p
                className="mt-2 text-[10px] text-center leading-snug"
                style={{ color: "var(--text-muted)" }}
              >
                双击节点切换为活动运行
                <br />
                double-click to switch
              </p>
            </>
          ) : (
            <p
              className="text-[11px] text-center py-3 leading-snug"
              style={{ color: "var(--text-muted)" }}
            >
              单击带渲染的节点预览
              <br />
              Click a node with a render to preview
            </p>
          )}
        </div>
      </div>
    </Panel>
  );
}
