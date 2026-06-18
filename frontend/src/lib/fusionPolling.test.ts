import { describe, it, expect } from "vitest";
import { shouldPollFusion } from "./fusionPolling";

describe("shouldPollFusion", () => {
  it("keeps polling while the fusion is actively running", () => {
    expect(shouldPollFusion("running")).toBe(true);
  });

  it("stops polling for an idle draft fusion (waits for user action)", () => {
    expect(shouldPollFusion("draft")).toBe(false);
  });

  it("stops polling for an idle target_ready fusion (waits for user action)", () => {
    expect(shouldPollFusion("target_ready")).toBe(false);
  });

  it("stops polling for a completed (terminal) fusion", () => {
    expect(shouldPollFusion("completed")).toBe(false);
  });

  it("stops polling for a failed (terminal) fusion", () => {
    expect(shouldPollFusion("failed")).toBe(false);
  });
});
