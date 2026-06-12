// ArtifactBrowser.tsx
import { CheckCircle, Loader } from "lucide-react";
import type { PngShaderResult } from "../hooks/usePngShader";

interface Props {
  result: PngShaderResult | null;
}

export default function ArtifactBrowser({ result }: Props) {
  if (!result) {
    return (
      <div className="bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-xl p-4 h-full flex items-center justify-center">
        <div className="text-center">
          <p className="text-sm text-[var(--text-muted)]">尚未运行流水线</p>
          <p className="text-xs text-[var(--text-muted)]/60 mt-0.5">No pipeline run yet</p>
        </div>
      </div>
    );
  }

  const inputSpec = result.input_spec ?? {};
  const target = (inputSpec.target ?? {}) as Record<string, unknown>;
  const isComplete = result.status === "done" || result.status === "completed" || result.status === "pass";

  return (
    <div className="bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-xl p-4 h-full flex flex-col overflow-hidden">
      <h3 className="text-sm font-semibold text-[var(--text-primary)] mb-3 flex-shrink-0 leading-tight">
        产物浏览器
        <span className="block text-[10px] font-normal text-[var(--text-muted)] mt-0.5">Artifact Browser</span>
      </h3>

      <div className="flex-1 overflow-auto space-y-3">
        {/* Status */}
        <div className="flex items-center gap-2">
          {isComplete ? (
            <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0" />
          ) : (
            <Loader className="w-4 h-4 text-yellow-400 flex-shrink-0 animate-spin" />
          )}
          <span className="text-xs text-[var(--text-primary)] font-medium">
            {isComplete ? "流水线完成" : "运行中..."}
          </span>
        </div>

        {/* Run ID */}
        <div>
          <p className="text-xs text-[var(--text-muted)] mb-0.5">运行 ID <span className="text-[10px]">Run ID</span></p>
          <p className="text-xs font-mono text-[var(--text-primary)] bg-[var(--bg-tertiary)] px-2 py-1 rounded break-all">
            {result.run_id}
          </p>
        </div>

        {/* Status badge */}
        <div className="flex justify-between text-xs">
          <span className="text-[var(--text-muted)]">状态 <span className="text-[10px]">Status</span></span>
          <span className="text-[var(--text-primary)] font-mono">{result.status}</span>
        </div>

        {/* Input spec */}
        {Object.keys(inputSpec).length > 0 && (
          <div>
            <p className="text-xs text-[var(--text-muted)] mb-1.5">输入规格 <span className="text-[10px]">Input Spec</span></p>
            <div className="space-y-1">
              {["resolution", "backend", "max_shader_chars"].map((key) => {
                const val = target[key];
                if (val === undefined) return null;
                return (
                  <div key={key} className="flex justify-between text-xs">
                    <span className="text-[var(--text-muted)]">{key}</span>
                    <span className="text-[var(--text-primary)] font-mono">{String(val)}</span>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Scoreboard summary */}
        {result.scoreboard && (
        <div>
          <p className="text-xs text-[var(--text-muted)] mb-1.5">候选 <span className="text-[10px]">Candidates</span></p>
          <div className="grid grid-cols-3 gap-2">
            {[
              { label: "总数", value: result.scoreboard.total },
              { label: "启用", value: result.scoreboard.enabled },
              { label: "已编译", value: result.scoreboard.compiled },
            ].map(({ label, value }) => (
              <div key={label} className="bg-[var(--bg-tertiary)] rounded-lg p-2 text-center">
                <p className="text-base font-bold text-[var(--text-primary)]">{value}</p>
                <p className="text-[10px] text-[var(--text-muted)]">{label}</p>
              </div>
            ))}
          </div>
        </div>
        )}

        {/* Selected candidate */}
        {result.selected_candidate_id && (
          <div>
            <p className="text-xs text-[var(--text-muted)] mb-0.5">已选候选 <span className="text-[10px]">Selected Candidate</span></p>
            <p className="text-xs font-mono text-[var(--text-primary)] bg-[var(--bg-tertiary)] px-2 py-1 rounded break-all">
              {result.selected_candidate_id}
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
