// branchRefineOptions.ts — single source of truth for branch-refine modes and locks.
import type { BranchMode } from "../hooks/usePngShader";

export const MODES: { mode: BranchMode; label: string; sub: string; desc: string }[] = [
  { mode: "refine", label: "定向", sub: "Refine", desc: "按反馈定向优化（强制至少一轮）" },
  { mode: "polish", label: "精修", sub: "Polish", desc: "结构尽量不变，仅小幅画质提升" },
  { mode: "continue", label: "继续", sub: "Continue", desc: "不注入目标，继续自动优化" },
];

export const LOCKS: { key: string; label: string }[] = [
  { key: "preserve_layout", label: "保持构图 Layout" },
  { key: "preserve_palette", label: "保持调色 Palette" },
  { key: "preserve_background", label: "保护背景 Background" },
  { key: "small_edits_only", label: "仅小幅改动 Small edits" },
];
