import { afterEach, describe, expect, it } from "vitest";
import { clearHistory, loadHistory, saveHistory } from "./sessionHistory";

describe("session history", () => {
  afterEach(clearHistory);
  it("bounds, de-duplicates, and clears search history", () => {
    for (let index = 0; index < 10; index += 1) saveHistory({ text: `query-${index}`, mood: "joy" });
    expect(loadHistory()).toHaveLength(8);
    saveHistory({ text: "query-9", mood: "joy" });
    expect(loadHistory()).toHaveLength(8);
    clearHistory(); expect(loadHistory()).toEqual([]);
  });
});
