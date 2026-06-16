import type { PngShaderResult, PipelineCheckpointMeta } from "../hooks/usePngShader";

/** Mirror of backend list_checkpoints: candidates with GLSL, iteration
 *  proposals with GLSL, then the final selected shader. */
export function deriveCheckpoints(result: PngShaderResult | null): PipelineCheckpointMeta[] {
  if (!result) return [];
  const out: PipelineCheckpointMeta[] = [];
  for (const c of result.scoreboard?.candidates ?? []) {
    const has = c.previewable ?? (c.compile_success && !!c.compile_glsl?.trim());
    if (!has) continue;
    out.push({
      id: `candidate:${c.id}`,
      kind: "candidate",
      label: c.selected ? "Selected baseline" : `Candidate ${c.id}`,
      score: c.final_score,
      iteration: null,
      accepted: c.selected,
      has_glsl: true,
    });
  }
  for (const e of result.refinement_history ?? []) {
    if (!e.compile_glsl?.trim()) continue;
    out.push({
      id: `refinement:iter:${e.iteration}`,
      kind: "refinement_iter",
      label: `Iteration ${e.iteration} proposal`,
      score: e.score_after,
      iteration: e.iteration,
      // Directed acceptance sets accepted=true while improved=false (score
      // intentionally dropped); fall back to improved for legacy entries.
      accepted: e.accepted ?? e.improved,
      has_glsl: true,
    });
  }
  if (result.selected_glsl?.trim()) {
    out.push({
      id: "final:selected",
      kind: "final",
      label: "Current best",
      score: result.quality_router?.final_score ?? null,
      iteration: null,
      accepted: true,
      has_glsl: true,
    });
  }
  return out;
}
