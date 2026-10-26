import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { RecommendationCard } from "./RecommendationCard";

const recommendation = {
  meme_id: "tall-meme.jpg", semantic_similarity: 0.7, emotion_relevance: null,
  sentiment_relevance: null, final_score: 0.7, usage_categories: [], explanation: "semantic match",
  image_available: true, image_url: "/api/v1/memes/tall-meme.jpg/image",
};

describe("RecommendationCard", () => {
  afterEach(cleanup);
  it("uses contain sizing so source images are never cropped", () => {
    render(<RecommendationCard recommendation={recommendation} />);
    const image = screen.getByRole("img", { name: "Meme: tall meme" });
    expect(image).toHaveStyle({ width: "100%", height: "auto", maxWidth: "100%", objectFit: "contain" });
    expect(image.parentElement).toHaveClass("meme-media");
  });

  it("replaces a failed image with the accessible fallback", () => {
    render(<RecommendationCard recommendation={recommendation} />);
    fireEvent.error(screen.getByRole("img"));
    expect(screen.getByLabelText("Meme image unavailable")).toBeInTheDocument();
  });

  it("opens an accessible preview and closes it with Escape", () => {
    render(<RecommendationCard recommendation={recommendation} />);
    fireEvent.click(screen.getByRole("button", { name: "Inspect image" }));
    expect(screen.getByRole("dialog")).toBeInTheDocument();
    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("shows download and share actions without opening the preview", () => {
    render(<RecommendationCard recommendation={recommendation} />);
    expect(screen.getByRole("button", { name: "Download image" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Share" })).toBeInTheDocument();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });
});
