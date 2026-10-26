import { afterEach, describe, expect, it, vi } from "vitest";

import { getEmotionRecommendations, getTextRecommendations, submitFeedback } from "./api";

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

  it("constructs a mood-only A0 text request without an empty text parameter", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify([]), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    await getTextRecommendations({ mood: "anxiety", usage: [], limit: 10, excludeMemeIds: ["seen.jpg"] });
    expect(fetchMock.mock.calls[0][0]).toContain("mood=anxiety");
    expect(fetchMock.mock.calls[0][0]).toContain("exclude_meme_id=seen.jpg");
    expect(fetchMock.mock.calls[0][0]).not.toContain("text=");
  });

  it("posts isolated product feedback", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ accepted: true }), { status: 201 }));
    vi.stubGlobal("fetch", fetchMock);
    await submitFeedback({ request_id: "request-1", meme_id: "meme.jpg", selected_mood: "anxiety", retrieval_variant: "a0", value: "useful" });
    expect(fetchMock.mock.calls[0][0]).toContain("/api/v1/feedback");
    expect(fetchMock.mock.calls[0][1]).toMatchObject({ method: "POST" });
  });
});
