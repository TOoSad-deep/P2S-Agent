import { usePngShader, type LlmMode, type BranchRefineRequest } from './hooks/usePngShader'
import { useModels } from './hooks/useModels'
import { useState, useCallback, useEffect, useRef } from 'react'
import { Sparkles, Zap } from 'lucide-react'
import PngShaderView from './components/PngShaderView'
import type { StrategyConfig } from './lib/strategy-presets'

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
    branchPending,
    branchRefine,
    fetchTimeline,
    fetchBranches,
    updateRunMetadata,
    switchRun,
  } = usePngShader()

  const models = useModels()

  const [inputImageUrl, setInputImageUrl] = useState<string | null>(null)
  const inputImageUrlRef = useRef<string | null>(null)

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
    <div className="min-h-screen text-white" style={{ background: 'var(--bg-primary)' }}>
      {/* Header */}
      <header 
        className="border-b px-6 py-4 backdrop-blur-xl sticky top-0 z-50"
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

      {/* Main Content */}
      <main className="max-w-[1600px] mx-auto px-6 py-6">
        <PngShaderView
          result={result}
          loading={loading}
          error={error}
          onRun={handleRun}
          inputImageUrl={inputImageUrl}
          llmMode={llmMode}
          onLlmModeChange={handleLlmModeChange}
          modelControls={models}
          strategy={strategy}
          onStrategyPartial={handleStrategyPartial}
          onApplyPreset={applyPreset}
          onStop={stopRun}
          stopPending={stopPending}
          branchPending={branchPending}
          parameterizeGlsl={parameterizeGlsl}
          onBranchRefine={handleBranchRefine}
          runId={runId}
          fetchTimeline={fetchTimeline}
          fetchBranches={fetchBranches}
          updateRunMetadata={updateRunMetadata}
          switchRun={switchRun}
          branchRefine={branchRefine}
        />
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
  )
}
