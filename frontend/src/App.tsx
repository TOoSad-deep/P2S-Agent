import { usePngShader, type LlmMode, type BranchRefineRequest } from './hooks/usePngShader'
import { useModels } from './hooks/useModels'
import { useState, useCallback, useEffect, useRef } from 'react'
import { Sparkles, Zap } from 'lucide-react'
import StudioPage from './pages/StudioPage'
import CanvasPage from './pages/CanvasPage'
import type { StrategyConfig } from './lib/strategy-presets'
import { PngShaderProvider } from './context/PngShaderContext'

type Page = 'studio' | 'canvas'

export default function App() {
  const {
    runId,
    result,
    loading,
    error,
    runPngShader,
    parameterizeGlsl,
    llmMode,
    setLlmMode,
    strategy,
    setStrategyPartial,
    applyPreset,
    stopRun,
    stopPending,
    branchRefine,
    fetchTimeline,
    fetchBranches,
    updateRunMetadata,
    switchRun,
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
    fetchPreferenceProfile,
    patchPreferenceProfile,
    rebuildPreferences,
    clearPreferences,
    createFusion,
    fetchFusion,
    generateCompositeTarget,
    runFusion,
  } = usePngShader()

  const models = useModels()

  const [inputImageUrl, setInputImageUrl] = useState<string | null>(null)
  const inputImageUrlRef = useRef<string | null>(null)

  // branchCheckpointId is shared across the page boundary (HumanLoopPanel on
  // Studio + BranchWorkspacePanel on Canvas), so App owns it and passes it via
  // context.
  const [branchCheckpointId, setBranchCheckpointId] = useState<string | null>(null)

  // Page shell: Studio (build/analysis) ⇄ Canvas (branch lineage graph).
  const [page, setPage] = useState<Page>(() =>
    new URLSearchParams(window.location.search).get('view') === 'canvas' ? 'canvas' : 'studio'
  )
  // Canvas only becomes valid once a run exists; clamp to Studio otherwise.
  const showCanvas = !!(result && runId)
  const effectivePage: Page = showCanvas ? page : 'studio'

  const goToPage = useCallback((p: Page) => {
    setPage(p)
    history.pushState(
      {},
      '',
      p === 'canvas' ? '?view=canvas' : window.location.pathname
    )
  }, [])

  // Keep page in sync with browser back/forward. effectivePage clamps to Studio
  // when no run exists, so only the "canvas" intent needs to be honored here.
  useEffect(() => {
    const onPopState = () => {
      const view = new URLSearchParams(window.location.search).get('view')
      setPage(view === 'canvas' ? 'canvas' : 'studio')
    }
    window.addEventListener('popstate', onPopState)
    return () => window.removeEventListener('popstate', onPopState)
  }, [])

  // Normalize a stale `?view=canvas` deep link back to Studio when Canvas cannot
  // be shown (no completed run yet). Without this the address bar keeps
  // `?view=canvas` while the Studio empty state renders. (BUG-007)
  useEffect(() => {
    if (showCanvas) return
    if (new URLSearchParams(window.location.search).get('view') === 'canvas') {
      history.replaceState({}, '', window.location.pathname)
    }
    setPage('studio')
  }, [showCanvas])

  const ssimValue =
    typeof result?.objective_metrics?.simple_ssim === 'number'
      ? result.objective_metrics.simple_ssim
      : typeof result?.objective_metrics?.ssim === 'number'
        ? result.objective_metrics.ssim
        : null

  const handleRun = useCallback((file: File, seedGlsl?: string) => {
    const url = URL.createObjectURL(file)
    if (inputImageUrlRef.current) {
      URL.revokeObjectURL(inputImageUrlRef.current)
    }
    inputImageUrlRef.current = url
    setInputImageUrl(url)
    runPngShader(file, seedGlsl, models.selection)
  }, [runPngShader, models.selection])

  const handleLlmModeChange = useCallback((mode: LlmMode) => {
    setLlmMode(mode)
  }, [setLlmMode])

  const handleStrategyPartial = useCallback((partial: Partial<StrategyConfig>) => {
    setStrategyPartial(partial)
  }, [setStrategyPartial])

  const handleBranchRefine = useCallback((request: BranchRefineRequest) => {
    if (runId) branchRefine(runId, request)
  }, [runId, branchRefine])

  useEffect(() => {
    return () => {
      if (inputImageUrlRef.current) {
        URL.revokeObjectURL(inputImageUrlRef.current)
        inputImageUrlRef.current = null
      }
    }
  }, [])

  return (
    <div className="h-screen flex flex-col text-white" style={{ background: 'var(--bg-primary)' }}>
      {/* Header */}
      <header
        className="flex-none border-b px-6 py-4 backdrop-blur-xl z-50"
        style={{
          borderColor: 'var(--border-color)',
          background: 'rgba(10, 10, 12, 0.85)',
        }}
      >
        <div className="flex items-center justify-between max-w-[1600px] mx-auto">
          {/* Logo & Title */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-3">
              <div 
                className="w-10 h-10 rounded-xl flex items-center justify-center"
                style={{ 
                  background: 'linear-gradient(135deg, var(--accent-primary), var(--accent-secondary))',
                  boxShadow: '0 0 20px var(--glow-emerald)'
                }}
              >
                <Sparkles className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold flex items-center gap-2">
                  <span className="gradient-text">P2S</span>
                  <span style={{ color: 'var(--text-muted)' }}>|</span>
                  <span style={{ color: 'var(--text-primary)' }}>Shader Agent</span>
                </h1>
                <p className="text-xs mt-0.5" style={{ color: 'var(--text-muted)' }}>
                  AI-Powered PNG to GLSL Shader Pipeline
                </p>
              </div>
            </div>
          </div>

          {/* Status & Actions */}
          <div className="flex items-center gap-4">
            {/* Studio ⇄ Canvas page toggle (only once a run exists) */}
            {showCanvas && (
              <div className="flex items-center gap-0.5 bg-[var(--bg-tertiary)] rounded-md p-0.5">
                {([
                  { page: 'studio' as Page, label: '工作台 Studio' },
                  { page: 'canvas' as Page, label: '画布 Canvas' },
                ]).map(({ page: p, label }) => (
                  <button
                    key={p}
                    onClick={() => goToPage(p)}
                    className={`px-2.5 py-1 text-xs rounded-md transition-all ${
                      effectivePage === p
                        ? 'bg-gradient-to-r from-emerald-500 to-emerald-600 text-white font-medium shadow-sm shadow-emerald-500/25'
                        : 'text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-hover)]'
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
            )}

            {/* Pipeline Status */}
            {loading && (
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg" style={{ background: 'var(--glow-emerald)' }}>
                <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                <span className="text-xs font-medium text-emerald-400">Processing...</span>
              </div>
            )}

            {/* Quick Stats */}
            {result && !loading && (
              <div className="flex items-center gap-3">
                <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg" style={{ background: 'var(--bg-tertiary)' }}>
                  <Zap className="w-3.5 h-3.5" style={{ color: 'var(--accent-warning)' }} />
                  <span className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                    {result.candidate_details?.length || 0} candidates
                  </span>
                </div>
                {ssimValue != null && (
                  <div className={`score-badge ${
                    ssimValue >= 0.8 ? 'score-high' :
                    ssimValue >= 0.5 ? 'score-medium' : 'score-low'
                  }`}>
                    SSIM: {(ssimValue * 100).toFixed(0)}%
                  </div>
                )}
              </div>
            )}

            {/* Version Badge */}
            <span 
              className="text-xs px-2.5 py-1 rounded-full"
              style={{ 
                background: 'var(--bg-tertiary)', 
                color: 'var(--text-muted)',
                border: '1px solid var(--border-color)'
              }}
            >
              v1.0.0
            </span>
          </div>
        </div>
      </header>

      {/* Content area — fills remaining viewport below the header. */}
      <div className="flex-1 min-h-0">
        <PngShaderProvider value={{
          result,
          loading,
          error,
          onRun: handleRun,
          inputImageUrl,
          llmMode,
          onLlmModeChange: handleLlmModeChange,
          modelControls: models,
          strategy,
          onStrategyPartial: handleStrategyPartial,
          onApplyPreset: applyPreset,
          onStop: stopRun,
          stopPending,
          parameterizeGlsl,
          onBranchRefine: handleBranchRefine,
          runId,
          branchCheckpointId,
          setBranchCheckpointId,
          fetchTimeline,
          fetchBranches,
          updateRunMetadata,
          switchRun,
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
          fetchPreferenceProfile,
          patchPreferenceProfile,
          rebuildPreferences,
          clearPreferences,
          createFusion,
          fetchFusion,
          generateCompositeTarget,
          runFusion,
        }}>
          {/* Render BOTH pages and toggle via CSS visibility so the canvas
              stays mounted across switches → its polling effects keep running. */}
          {/* Studio: scrolls internally; owns the page footer (canvas is full-bleed/footer-less). */}
          <div
            className="h-full overflow-y-auto"
            style={{ display: effectivePage === 'studio' ? undefined : 'none' }}
          >
            <main className="max-w-[1600px] mx-auto px-6 py-6">
              <StudioPage />
            </main>

            {/* Footer */}
            <footer
              className="border-t py-4 mt-8"
              style={{ borderColor: 'var(--border-color)', background: 'var(--bg-secondary)' }}
            >
              <div className="max-w-[1600px] mx-auto px-6 flex items-center justify-between">
                <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                  P2S-Agent • Powered by LangGraph & FastAPI
                </p>
                <div className="flex items-center gap-4">
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    Built with React + Vite + Tailwind
                  </span>
                </div>
              </div>
            </footer>
          </div>

          {/* Canvas: full-bleed, fills the content area. */}
          {showCanvas && (
            <div
              className="h-full min-h-0"
              style={{ display: effectivePage === 'canvas' ? undefined : 'none' }}
            >
              <CanvasPage />
            </div>
          )}
        </PngShaderProvider>
      </div>
    </div>
  )
}
