export type ShareOutcome = "shared" | "copied-local-url" | "unavailable";

const filenameFor = (memeId: string) => memeId.replace(/[^a-z0-9._-]+/gi, "-").replace(/^-+|-+$/g, "") || "dameme-meme.jpg";

async function imageFile(imageUrl: string, memeId: string): Promise<File> {
  const response = await fetch(imageUrl);
  if (!response.ok) throw new Error("The image could not be downloaded.");
  const blob = await response.blob();
  return new File([blob], filenameFor(memeId), { type: blob.type || "image/jpeg" });
}

export async function downloadMeme(imageUrl: string, memeId: string): Promise<void> {
  const file = await imageFile(imageUrl, memeId);
  const objectUrl = URL.createObjectURL(file);
  const link = document.createElement("a");
  link.href = objectUrl; link.download = file.name; document.body.append(link); link.click(); link.remove(); URL.revokeObjectURL(objectUrl);
}

export async function shareMeme(imageUrl: string, memeId: string): Promise<ShareOutcome> {
  if (navigator.share) {
    try { const file = await imageFile(imageUrl, memeId); if (!navigator.canShare || navigator.canShare({ files: [file] })) { await navigator.share({ title: "A DaMeme recommendation", text: "A meme recommendation from DaMeme", files: [file] }); return "shared"; } }
    catch (error) { if (error instanceof DOMException && error.name === "AbortError") return "unavailable"; }
  }
  if (navigator.clipboard?.writeText) { await navigator.clipboard.writeText(imageUrl); return "copied-local-url"; }
  return "unavailable";
}
