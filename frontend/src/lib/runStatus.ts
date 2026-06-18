// Shared run-lifecycle status vocabulary for the PNG-Shader closed-loop UI.
//
// Backend emits several non-terminal phases for variant/draw children
// (`queued`, an `acquired` acquisition phase, `pending`) in addition to
// `running`. The polling loop must keep polling for ALL non-terminal phases —
// treating `queued`/`acquired` as terminal freezes a selected node at its
// queued snapshot (see usePngShader Bug 1).

import type { StrategyConfig } from "./strategy-presets";

/** True-terminal lifecycle statuses: the poll loop stops here. */
export const TERMINAL_STATUSES: ReadonlySet<string> = new Set([
  "completed",
  "failed",
  "cancelled",
]);

/** Non-terminal lifecycle statuses: the poll loop must keep going. */
export const NON_TERMINAL_STATUSES: ReadonlySet<string> = new Set([
  "running",
  "queued",
  "acquired",
  "pending",
]);

function normalize(status: string | null | undefined): string {
  return (status ?? "").trim().toLowerCase();
}

/**
 * Whether a status is terminal (the poll loop should stop, clear loading, and
 * release activeRun). Unknown/empty statuses are treated as terminal so an
 * undefined lifecycle phase can never spin the loop forever.
 */
export function isTerminalStatus(status: string | null | undefined): boolean {
  const s = normalize(status);
  if (NON_TERMINAL_STATUSES.has(s)) return false;
  return true;
}

/**
 * Whether the poll loop should keep polling. Inverse of `isTerminalStatus` for
 * known statuses; `queued`/`acquired`/`pending`/`running` all keep polling.
 */
export function shouldKeepPolling(status: string | null | undefined): boolean {
  return NON_TERMINAL_STATUSES.has(normalize(status));
}

/**
 * Merge server-reported strategy fields back into the local strategy, skipping
 * any key that has an un-acked pending local PATCH (Bug 3 last-writer race).
 *
 * Pure so it can be TDD'd: given the current local strategy, the server's
 * strategy partial, and the set of field keys with pending local edits, return
 * a new merged object that trusts the server for non-pending keys and preserves
 * local values for pending keys.
 */
export function mergeStrategyFromServer(
  local: StrategyConfig,
  server: Partial<StrategyConfig> | null | undefined,
  pendingKeys: ReadonlySet<string>,
): StrategyConfig {
  const merged: Record<string, unknown> = { ...(local as unknown as Record<string, unknown>) };
  if (server && typeof server === "object") {
    for (const [key, value] of Object.entries(server)) {
      if (pendingKeys.has(key)) continue; // keep in-flight local edit
      merged[key] = value;
    }
  }
  return merged as unknown as StrategyConfig;
}
