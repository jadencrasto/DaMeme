import { useEffect, useRef, useState } from "react";

import { apiBaseUrl, type Recommendation } from "../lib/api";

const titleFor = (identifier: string) => identifier.replace(/\.[^.]+$/, "").replaceAll(/[-_]+/g, " ");

export function MemeInspectionDialog({ recommendation, onClose }: { recommendation: Recommendation; onClose: () => void }) {
  const closeButton = useRef<HTMLButtonElement>(null);
  const previousFocus = useRef<HTMLElement | null>(null);
  const [imageFailed, setImageFailed] = useState(false);
  const imageUrl = recommendation.image_available && recommendation.image_url ? `${apiBaseUrl}${recommendation.image_url}` : null;
  useEffect(() => {
    previousFocus.current = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    closeButton.current?.focus();
    const handler = (event: KeyboardEvent) => { if (event.key === "Escape") onClose(); };
    window.addEventListener("keydown", handler);
    return () => { window.removeEventListener("keydown", handler); previousFocus.current?.focus(); };
  }, [onClose]);
  return <div className="inspection-backdrop" role="presentation" onMouseDown={onClose}>
    <section className="inspection-dialog" role="dialog" aria-modal="true" aria-labelledby="inspection-title" onMouseDown={(event) => event.stopPropagation()}>
      <div className="inspection-header"><h2 id="inspection-title">{titleFor(recommendation.meme_id)}</h2><button ref={closeButton} type="button" onClick={onClose}>Close preview</button></div>
      {imageUrl && !imageFailed ? <img className="inspection-image" src={imageUrl} alt={`Meme preview: ${titleFor(recommendation.meme_id)}`} onError={() => setImageFailed(true)} /> : <p role="status">This meme image is not available locally.</p>}
    </section>
  </div>;
}
