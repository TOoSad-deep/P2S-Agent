type LogLevel = "debug" | "info" | "warn" | "error";

function isEnabled(): boolean {
  if (import.meta.env.DEV) return true;
  try {
    return window.localStorage.getItem("p2s_debug_logs") === "1";
  } catch {
    return false;
  }
}

export function makeRequestId(prefix = "web"): string {
  const random =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID().replace(/-/g, "").slice(0, 10)
      : Math.random().toString(36).slice(2, 12);
  return `${prefix}_${random}`;
}

export function logFrontendEvent(
  event: string,
  data: Record<string, unknown> = {},
  level: LogLevel = "info",
): void {
  if (!isEnabled()) return;
  const payload = {
    timestamp: new Date().toISOString(),
    event,
    ...data,
  };
  const fn = level === "debug" ? console.debug : level === "warn" ? console.warn : level === "error" ? console.error : console.info;
  fn(`[p2s] ${event}`, payload);
}
