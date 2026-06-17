// CanvasPage.tsx — Branch lineage graph view. Owns the list|canvas sub-toggle
// and renders BranchCanvasWorkspace / BranchWorkspacePanel. Reads everything
// from PngShaderContext (single usePngShader() lives at App root).
import { useState } from "react";
import BranchCanvasWorkspace from "../components/BranchCanvasWorkspace";
import BranchWorkspacePanel from "../components/BranchWorkspacePanel";
import { usePngShaderContext } from "../context/PngShaderContext";

export default function CanvasPage() {
  const {
    result,
    loading,
    runId,
    branchCheckpointId,
    setBranchCheckpointId,
    fetchBranches,
    fetchTimeline,
    switchRun,
    updateRunMetadata,
    branchRefine,
    exploreVariants,
    fetchVariantGroup,
    stopVariantGroup,
    selectVariantWinner,
    rateVariant,
    createDrawSession,
    fetchDrawSession,
    drawMore,
    redrawCard,
    cardEvent,
    createFusion,
    fetchFusion,
    generateCompositeTarget,
    runFusion,
  } = usePngShaderContext();

  // Default to "canvas" — this is the canvas page.
  const [workspaceView, setWorkspaceView] = useState<"list" | "canvas">("canvas");

  return (
    <div className="flex flex-col gap-2">
      {/* Branch workspace: Canvas / List toggle + panels */}
      {result && runId && (
        <div className="flex flex-col gap-2">
          {/* Segmented toggle */}
          <div className="flex items-center gap-0.5 self-start bg-[var(--bg-tertiary)] rounded-md p-0.5">
            {(["list", "canvas"] as const).map((view) => (
              <button
                key={view}
                onClick={() => setWorkspaceView(view)}
                className={`px-2.5 py-1 text-xs rounded-md transition-all ${
                  workspaceView === view
                    ? "bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-medium shadow-sm shadow-emerald-500/25"
                    : "text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)]"
                }`}
              >
                {view === "canvas" ? "画布 Canvas" : "列表 List"}
              </button>
            ))}
          </div>

          {workspaceView === "canvas" ? (
            <BranchCanvasWorkspace
              runId={runId}
              result={result}
              fetchBranches={fetchBranches}
              fetchTimeline={fetchTimeline}
              switchRun={switchRun}
              updateRunMetadata={updateRunMetadata}
              branchRefine={branchRefine}
              exploreVariants={exploreVariants}
              fetchVariantGroup={fetchVariantGroup}
              stopVariantGroup={stopVariantGroup}
              selectVariantWinner={selectVariantWinner}
              rateVariant={rateVariant}
              createDrawSession={createDrawSession}
              fetchDrawSession={fetchDrawSession}
              drawMore={drawMore}
              redrawCard={redrawCard}
              cardEvent={cardEvent}
              createFusion={createFusion}
              fetchFusion={fetchFusion}
              generateCompositeTarget={generateCompositeTarget}
              runFusion={runFusion}
              disabled={loading}
            />
          ) : (
            <BranchWorkspacePanel
              runId={runId}
              result={result}
              activeCheckpointId={branchCheckpointId}
              onCheckpointSelect={setBranchCheckpointId}
              onSwitchRun={switchRun}
              fetchTimeline={fetchTimeline}
              fetchBranches={fetchBranches}
              updateRunMetadata={updateRunMetadata}
              disabled={loading}
            />
          )}
        </div>
      )}
    </div>
  );
}
