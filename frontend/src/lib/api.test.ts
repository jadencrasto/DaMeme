import { afterEach, describe, expect, it, vi } from "vitest";

import { getEmotionRecommendations, getTextRecommendations } from "./api";

describe("recommendation API client", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("constructs text recommendation requests with optional filters", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify([]), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    await getTextRecommendations({ text: "late meeting", emotion: "sadness", sentiment: "negative", usage: ["sarcasm", "parody"], limit: 5 });
    expect(fetchMock.mock.calls[0][0]).toContain("/api/v1/recommendations/text?");
    expect(fetchMock.mock.calls[0][0]).toContain("text=late+meeting");
    expect(fetchMock.mock.calls[0][0]).toContain("usage=sarcasm");
  });

  it("constructs emotion-only requests", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify([]), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    await getEmotionRecommendations({ emotion: "joy", usage: [], limit: 10 });
    expect(fetchMock.mock.calls[0][0]).toContain("/api/v1/recommendations/emotion?emotion=joy");
  });
});
