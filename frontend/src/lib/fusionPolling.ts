// fusionPolling.ts — pure decision helper for the fusion status poller.
//
// A fusion is only ACTIVELY advancing while its status is "running". The other
// statuses are idle and wait for a user action before they can change:
//   - draft / target_ready: idle, advance only when the user composites/runs.
//   - completed / failed:    terminal, never change again.
// Polling any of those wastes a backend request every 2s indefinitely, so the
// poller should keep going ONLY while the status is "running".

import type { FusionStatus } from "../hooks/usePngShader";

type FusionStatusValue = FusionStatus["status"];

/**
 * Should the fusion poller keep polling (re-schedule another tick) given the
 * latest fetched status? Only `running` fusions are actively advancing; every
 * other status (draft / target_ready / completed / failed) is idle or terminal
 * and the poller must stop.
 */
export function shouldPollFusion(status: FusionStatusValue): boolean {
  return status === "running";
}
