import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { Phase7RecommendationExperience } from "./Phase7RecommendationExperience";

const response = {
  predicted_emotions: {
    emotion_probabilities: { joy: 0.91, surprise: 0.62 },
    emotion_labels: ["joy", "surprise"],
  },
  recommendations: Array.from({ length: 6 }, (_, index) => ({
    meme_id: `meme-${index}.jpg`, image_available: false, image_url: null,
    semantic_similarity: 0.8, emotion_compatibility: 0.7, score: 0.78,
  })),
};

describe("Phase 7 text recommendations", () => {
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it("posts text, shows detected emotions, and limits cards to five", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify(response), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    render(<Phase7RecommendationExperience />);
    expect(screen.getByRole("button", { name: "Find Memes" })).toBeDisabled();
    fireEvent.change(screen.getByLabelText("How are you feeling?"), { target: { value: "I finally passed my exam" } });
    fireEvent.click(screen.getByRole("button", { name: "Find Memes" }));
    await screen.findByText("meme 0");
    expect(screen.getByText("joy (91%), surprise (62%)")).toBeInTheDocument();
    expect(screen.getAllByText(/meme [0-4]$/)).toHaveLength(5);
    expect(screen.queryByText("meme 5")).not.toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining("/api/v1/recommendations/text"), expect.objectContaining({ method: "POST", body: JSON.stringify({ text: "I finally passed my exam", top_k: 5 }) }));
  });

  it("shows a retryable backend error", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "Recommendation service is unavailable" }), { status: 503 }));
    vi.stubGlobal("fetch", fetchMock);
    render(<Phase7RecommendationExperience />);
    fireEvent.change(screen.getByLabelText("How are you feeling?"), { target: { value: "A rough day" } });
    fireEvent.click(screen.getByRole("button", { name: "Find Memes" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Recommendation service is unavailable");
    await waitFor(() => expect(screen.getByRole("button", { name: "Retry" })).toBeEnabled());
  });

  it("uses the verified direct-emotion endpoint when no text is entered", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(JSON.stringify({ emotion: "joy", recommendations: [] }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    render(<Phase7RecommendationExperience />);
    fireEvent.click(screen.getByRole("button", { name: "Joy" }));
    expect(screen.getByRole("button", { name: "Joy" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "Find Memes" })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "Find Memes" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(expect.stringContaining("/api/v1/recommendations/phase5/emotion?emotion=joy&limit=5"), expect.any(Object)));
  });
});
