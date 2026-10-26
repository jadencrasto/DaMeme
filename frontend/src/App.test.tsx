import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import App from "./App";

const recommendation = {
  meme_id: "exam_success.jpg", semantic_similarity: 0.82, emotion_relevance: 0.76,
  sentiment_relevance: null, final_score: 0.79, usage_categories: ["parody"],
  explanation: "text similarity 0.820; source joy probability 0.760",
  image_available: false, image_url: null,
};

function mockFetch(recommendations = [recommendation]) {
  vi.stubGlobal("fetch", vi.fn((url: string) => {
    if (url.includes("/status")) return Promise.resolve(new Response(JSON.stringify({ ready: true, detail: null, meme_count: 1 })));
    if (url.includes("/options")) return Promise.resolve(new Response(JSON.stringify({ emotions: ["joy", "sadness"], sentiments: ["positive"], usage_categories: ["parody"] })));
    if (url.includes("/emotions")) return Promise.resolve(new Response(JSON.stringify({ moods: [{ id: "anxiety", label: "Anxiety", group: "Stress and negative", description: "worry", synonyms: ["worried"], classifier_weights: { fear: .75 }, sentiment: "negative" }] })));
    return Promise.resolve(new Response(JSON.stringify(recommendations)));
  }));
}

describe("cinematic recommendation interface", () => {
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it("renders text recommendations and an unavailable-image placeholder", async () => {
    mockFetch(); render(<App />);
    await screen.findAllByRole("button", { name: "Anxiety" });
    fireEvent.change(screen.getByLabelText("What happened?"), { target: { value: "I passed my exam" } });
    fireEvent.click(screen.getByRole("button", { name: "Find my meme" }));
    expect(await screen.findByText("exam success")).toBeInTheDocument();
    expect(screen.getByLabelText("Meme image unavailable")).toBeInTheDocument();
  });

  it("supports mood-only mode and shows backend errors", async () => {
    mockFetch(); render(<App />);
    await screen.findByRole("tab", { name: "Just a mood" });
    fireEvent.click(screen.getByRole("tab", { name: "Just a mood" }));
    fireEvent.click(screen.getAllByRole("button", { name: "Anxiety" })[0]);
    fireEvent.click(screen.getByRole("button", { name: "Find my meme" }));
    await waitFor(() => expect(screen.getByText("exam success")).toBeInTheDocument());
  });

  it("fetches the mood taxonomy and sends a selected stable mood ID", async () => {
    mockFetch(); render(<App />);
    await screen.findAllByRole("button", { name: "Anxiety" });
    fireEvent.click(screen.getAllByRole("button", { name: "Anxiety" })[0]);
    fireEvent.change(screen.getByLabelText("What happened?"), { target: { value: "presentation" } });
    fireEvent.click(screen.getByRole("button", { name: "Find my meme" }));
    await waitFor(() => expect(fetch).toHaveBeenCalledWith(expect.stringContaining("mood=anxiety"), expect.any(Object)));
  });

  it("uses the selected bounded recommendation maximum and reports a smaller returned set", async () => {
    mockFetch(); render(<App />);
    await screen.findByLabelText("What happened?");
    expect(screen.getByLabelText("Show up to")).toHaveValue("10");
    fireEvent.change(screen.getByLabelText("Show up to"), { target: { value: "20" } });
    fireEvent.change(screen.getByLabelText("What happened?"), { target: { value: "an impossible Tuesday" } });
    fireEvent.click(screen.getByRole("button", { name: "Find my meme" }));
    await waitFor(() => expect(fetch).toHaveBeenCalledWith(expect.stringContaining("limit=20"), expect.any(Object)));
    expect(await screen.findByText("1 returned · up to 20")).toBeInTheDocument();
    expect(screen.getByText("We found fewer safe, varied matches than your selected maximum.")).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Show up to"), { target: { value: "50" } });
    fireEvent.click(screen.getByRole("button", { name: "Find my meme" }));
    await waitFor(() => expect(fetch).toHaveBeenCalledWith(expect.stringContaining("limit=50"), expect.any(Object)));
  });

  it("shows loading and backend-unavailable states", async () => {
    let resolveRecommendation: ((value: Response) => void) | undefined;
    vi.stubGlobal("fetch", vi.fn((url: string) => {
      if (url.includes("/status")) return Promise.resolve(new Response(JSON.stringify({ ready: true, detail: null, meme_count: 1 })));
      if (url.includes("/options")) return Promise.resolve(new Response(JSON.stringify({ emotions: ["joy"], sentiments: [], usage_categories: [] })));
      if (url.includes("/emotions")) return Promise.resolve(new Response(JSON.stringify({ moods: [] })));
      return new Promise<Response>((resolve) => { resolveRecommendation = resolve; });
    }));
    render(<App />);
    await screen.findByLabelText("What happened?");
    fireEvent.change(screen.getByLabelText("What happened?"), { target: { value: "too much work" } });
    fireEvent.click(screen.getByRole("button", { name: "Find my meme" }));
    expect(screen.getByText("Curating a fresh set of safe, varied picks…")).toBeInTheDocument();
    resolveRecommendation?.(new Response(JSON.stringify({ detail: "Recommendation assets are unavailable" }), { status: 503 }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Recommendation assets are unavailable");
  });
});
