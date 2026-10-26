import { afterEach, describe, expect, it, vi } from "vitest";

import { downloadMeme, shareMeme } from "./memeActions";

describe("meme actions", () => {
  afterEach(() => vi.restoreAllMocks());

  it("downloads the original fetched image with its meme filename", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response(new Blob(["image"], { type: "image/png" }), { status: 200 })));
    const click = vi.spyOn(HTMLAnchorElement.prototype, "click").mockImplementation(() => undefined);
    vi.stubGlobal("URL", { createObjectURL: vi.fn(() => "blob:test"), revokeObjectURL: vi.fn() });
    await downloadMeme("/api/v1/memes/a.png/image", "a.png");
    expect(click).toHaveBeenCalledOnce();
  });

  it("copies a local URL when file sharing is unavailable", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });
    expect(await shareMeme("/api/v1/memes/a.jpg/image", "a.jpg")).toBe("copied-local-url");
    expect(writeText).toHaveBeenCalledWith("/api/v1/memes/a.jpg/image");
  });
});
