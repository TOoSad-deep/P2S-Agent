// SceneGraphPanel.tsx
import type { Preprocess } from "../hooks/usePngShader";

interface Props {
  preprocess: Preprocess | null;
}

interface ScoreBar {
  key: keyof Preprocess;
  label: string;
}

const SCORE_BARS: ScoreBar[] = [
  { key: "edge_sharpness", label: "边缘锐度" },
  { key: "texture_score", label: "纹理分" },
  { key: "photo_like_score", label: "照片相似度" },
  { key: "gradient_score", label: "渐变分" },
  { key: "alpha_coverage", label: "透明覆盖率" },
];

export default function SceneGraphPanel({ preprocess }: Props) {
  return (
    <div className="bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-xl p-4 h-full flex flex-col overflow-hidden">
      <h3 className="text-sm font-semibold text-[var(--text-primary)] mb-3 flex-shrink-0 leading-tight">
        预处理分析
        <span className="block text-[10px] font-normal text-[var(--text-muted)] mt-0.5">Preprocess Analysis</span>
      </h3>

      {!preprocess ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <p className="text-sm text-[var(--text-muted)]">运行流水线以查看分析</p>
            <p className="text-xs text-[var(--text-muted)]/60 mt-0.5">Run pipeline to see analysis</p>
          </div>
        </div>
      ) : (
        <div className="flex-1 overflow-auto space-y-3">
          {/* Size */}
          <div className="flex justify-between text-xs">
            <span className="text-[var(--text-muted)]">尺寸 <span className="text-[10px]">Size</span></span>
            <span className="text-[var(--text-primary)] font-mono">{preprocess.width}×{preprocess.height}</span>
          </div>

          {/* Alpha */}
          <div className="flex justify-between text-xs items-center">
            <span className="text-[var(--text-muted)]">透明通道 <span className="text-[10px]">Alpha</span></span>
            <div className="flex items-center gap-2">
              <span
                className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                  preprocess.has_alpha
                    ? "bg-blue-500/20 text-blue-400 border border-blue-500/30"
                    : "bg-gray-500/20 text-gray-400 border border-gray-500/30"
                }`}
              >
                {preprocess.has_alpha ? "含透明" : "不透明"}
              </span>
              <span className="text-[var(--text-primary)]">{(preprocess.alpha_coverage * 100).toFixed(1)}%</span>
            </div>
          </div>

          {/* Counts */}
          <div className="flex justify-between text-xs">
            <span className="text-[var(--text-muted)]">颜色数 <span className="text-[10px]">Colors</span></span>
            <span className="text-[var(--text-primary)]">{preprocess.color_count_estimate}</span>
          </div>
          <div className="flex justify-between text-xs">
            <span className="text-[var(--text-muted)]">连通域 <span className="text-[10px]">Components</span></span>
            <span className="text-[var(--text-primary)]">{preprocess.component_count_estimate}</span>
          </div>

          {/* Palette */}
          {preprocess.palette.length > 0 && (
            <div>
              <p className="text-xs text-[var(--text-muted)] mb-1.5">调色板 <span className="text-[10px]">Palette</span></p>
              <div className="flex gap-1 flex-wrap">
                {preprocess.palette.slice(0, 5).map((hex, i) => (
                  <div
                    key={i}
                    title={hex}
                    style={{ backgroundColor: hex }}
                    className="w-5 h-5 rounded border border-[var(--border-color)]"
                  />
                ))}
              </div>
            </div>
          )}

          {/* Score bars */}
          <div className="space-y-2">
            {SCORE_BARS.map(({ key, label }) => {
              const val = preprocess[key] as number;
              const pct = Math.min(100, Math.max(0, val * 100));
              return (
                <div key={key}>
                  <div className="flex justify-between mb-0.5">
                    <span className="text-xs text-[var(--text-muted)]">{label}</span>
                    <span className="text-xs text-[var(--text-primary)]">{val.toFixed(2)}</span>
                  </div>
                  <div className="h-2 bg-[var(--bg-tertiary)] rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all"
                      style={{ 
                        width: `${pct}%`,
                        background: pct > 70 
                          ? 'linear-gradient(90deg, #10b981, #34d399)' 
                          : pct > 40 
                            ? 'linear-gradient(90deg, #f59e0b, #fbbf24)'
                            : 'linear-gradient(90deg, #ef4444, #f87171)',
                        boxShadow: pct > 70 
                          ? '0 0 8px rgba(16, 185, 129, 0.4)'
                          : pct > 40 
                            ? '0 0 8px rgba(245, 158, 11, 0.4)'
                            : '0 0 8px rgba(239, 68, 68, 0.4)'
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
