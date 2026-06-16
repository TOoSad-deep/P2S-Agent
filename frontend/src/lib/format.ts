export function fmtScore(score?: number | null): string {
  return typeof score === "number" ? score.toFixed(3) : "—";
}
/** Truncate to at most `max` characters, appending '…' when cut. */
export function truncate(text: string, max: number): string {
  return text.length > max ? text.slice(0, max - 1) + "…" : text;
}
