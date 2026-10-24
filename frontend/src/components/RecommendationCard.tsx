import { useState } from "react";

import { apiBaseUrl, type Recommendation } from "../lib/api";

const titleFor = (identifier: string) => identifier.replace(/\.[^.]+$/, "").replaceAll(/[-_]+/g, " ");
const score = (value: number | null) => value === null ? null : value.toFixed(3);
// Follow the source dimensions; meme captions and panels must never be cropped.
const memeImageStyle = { width: "100%", height: "auto", maxWidth: "100%", objectFit: "contain" } as const;

export function RecommendationCard({ recommendation }: { recommendation: Recommendation }) {
  const [imageFailed, setImageFailed] = useState(false);
  const imageUrl = recommendation.image_available && recommendation.image_url && !imageFailed ? `${apiBaseUrl}${recommendation.image_url}` : null;
  return <article className="meme-card">
    {imageUrl ? <div className="meme-media"><img className="meme-image" style={memeImageStyle} src={imageUrl} alt={`Meme: ${titleFor(recommendation.meme_id)}`} onError={() => setImageFailed(true)} /></div> : <div className="meme-placeholder" aria-label="Meme image unavailable"><span>DaMeme</span><small>Image available after local dataset setup</small></div>}
    <div className="meme-content">
      <p className="meme-id" title={recommendation.meme_id}>{titleFor(recommendation.meme_id)}</p>
      <p className="explanation">{recommendation.explanation}</p>
      <dl className="score-grid">
        {score(recommendation.semantic_similarity) && <div><dt>Semantic</dt><dd>{score(recommendation.semantic_similarity)}</dd></div>}
        {score(recommendation.emotion_relevance) && <div><dt>Emotion</dt><dd>{score(recommendation.emotion_relevance)}</dd></div>}
        <div><dt>Ranking score</dt><dd>{recommendation.final_score.toFixed(3)}</dd></div>
      </dl>
      {recommendation.usage_categories.length > 0 && <p className="tags">{recommendation.usage_categories.map((tag) => <span key={tag}>{tag}</span>)}</p>}
    </div>
  </article>;
}
