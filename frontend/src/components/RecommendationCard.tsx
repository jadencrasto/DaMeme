import { useState } from "react";

import { apiBaseUrl, submitFeedback, type FeedbackValue, type Recommendation } from "../lib/api";
import { downloadMeme, shareMeme } from "../lib/memeActions";
import { MemeInspectionDialog } from "./MemeInspectionDialog";

const titleFor = (identifier: string) => identifier.replace(/\.[^.]+$/, "").replaceAll(/[-_]+/g, " ");
// Follow the source dimensions; meme captions and panels must never be cropped.
const memeImageStyle = { width: "100%", height: "auto", maxWidth: "100%", objectFit: "contain" } as const;

export function RecommendationCard({ recommendation, onInspect, onFeedback, feedback }: { recommendation: Recommendation; onInspect?: () => void; onFeedback?: (value: FeedbackValue) => void; feedback?: FeedbackValue | "submitting" | "error" }) {
  const [imageFailed, setImageFailed] = useState(false);
  const [inspecting, setInspecting] = useState(false);
  const [localFeedback, setLocalFeedback] = useState<FeedbackValue | "submitting" | "error">();
  const [downloadState, setDownloadState] = useState<"idle" | "working" | "error">("idle");
  const [shareMessage, setShareMessage] = useState("");
  const imageUrl = recommendation.image_available && recommendation.image_url && !imageFailed ? `${apiBaseUrl}${recommendation.image_url}` : null;
  const displayedFeedback = feedback ?? localFeedback;
  async function submitLocalFeedback(value: FeedbackValue) { if (displayedFeedback) return; setLocalFeedback("submitting"); try { await submitFeedback({ request_id: sessionStorage.getItem("dameme-product-request") ?? globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`, meme_id: recommendation.meme_id, selected_mood: sessionStorage.getItem("dameme-product-mood") || undefined, retrieval_variant: "a0", value }); setLocalFeedback(value); } catch { setLocalFeedback("error"); } }
  async function download() { if (!imageUrl || downloadState === "working") return; setDownloadState("working"); try { await downloadMeme(imageUrl, recommendation.meme_id); setDownloadState("idle"); } catch { setDownloadState("error"); } }
  async function share() { if (!imageUrl) return; const outcome = await shareMeme(imageUrl, recommendation.meme_id); setShareMessage(outcome === "shared" ? "Shared." : outcome === "copied-local-url" ? "Local image URL copied. It is only reachable from this DaMeme server." : "Sharing is not supported in this browser."); }
  return <article className="meme-card">
      {imageUrl ? <button type="button" className="meme-media meme-preview" onClick={() => { setInspecting(true); onInspect?.(); }} aria-label={`Inspect meme: ${titleFor(recommendation.meme_id)}`}><img className="meme-image" style={memeImageStyle} src={imageUrl} alt={`Meme: ${titleFor(recommendation.meme_id)}`} loading="lazy" onError={() => setImageFailed(true)} /></button> : <div className="meme-placeholder" aria-label="Meme image unavailable"><span>DaMeme</span><small>Image available after local dataset setup</small></div>}
    <div className="meme-content">
      <p className="meme-id" title={recommendation.meme_id}>{titleFor(recommendation.meme_id)}</p>
      {imageUrl && <button type="button" className="inspect-button" onClick={() => { setInspecting(true); onInspect?.(); }}>Inspect image</button>}
      {imageUrl && <div className="card-actions" aria-label={`Actions for ${titleFor(recommendation.meme_id)}`}><button className="download-action" type="button" disabled={downloadState === "working"} onClick={() => void download()}>{downloadState === "working" ? "Downloading…" : "Download image"}</button><button className="share-action" type="button" onClick={() => void share()}>Share</button></div>}
      {downloadState === "error" && <p className="action-error" role="alert">Could not download this image. Try again.</p>}
      {shareMessage && <p className="action-status" role="status">{shareMessage}</p>}
      <div className="feedback" aria-label={`Feedback for ${titleFor(recommendation.meme_id)}`}><span>Was this useful?</span>{displayedFeedback === "useful" || displayedFeedback === "not_relevant" ? <strong role="status">Thanks for the feedback.</strong> : <><button className="feedback-action" type="button" disabled={displayedFeedback === "submitting"} onClick={() => onFeedback ? onFeedback("useful") : void submitLocalFeedback("useful")}>Useful</button><button className="feedback-action" type="button" disabled={displayedFeedback === "submitting"} onClick={() => onFeedback ? onFeedback("not_relevant") : void submitLocalFeedback("not_relevant")}>Not relevant</button>{displayedFeedback === "error" && <small role="alert">Could not save feedback. Try once more.</small>}</>}</div>
    </div>
      {inspecting && <MemeInspectionDialog recommendation={recommendation} onClose={() => setInspecting(false)} />}
    </article>;
}
