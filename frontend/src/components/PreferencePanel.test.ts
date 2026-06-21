import { describe, it, expect } from "vitest";
import { relativeTime } from "./PreferencePanel";

// BUG-008: the default/empty preference profile uses updated_at: 0.0, which must
// not render as the Unix epoch (1970). relativeTime must report "never updated".
describe("relativeTime", () => {
  it("returns a 'never updated' sentinel for a zero/default epoch", () => {
    expect(relativeTime(0)).toBe("未更新 Never updated");
  });

  it("returns a 'never updated' sentinel for negative epochs", () => {
    expect(relativeTime(-5)).toBe("未更新 Never updated");
  });
});
