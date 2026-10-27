import { type FormEvent, useRef, useState } from "react";

import { RecommendationCard } from "./RecommendationCard";
import {
  type Phase6Recommendation,
  type Recommendation,
  type Emotion,
  getPhase5EmotionRecommendations,
  getPhase6TextRecommendations,
} from "../lib/api";

const emotionChoices: Emotion[] = ["anger", "fear", "joy", "sadness", "surprise", "disgust"];
const label = (emotion: Emotion) => emotion.charAt(0).toUpperCase() + emotion.slice(1);

const asRecommendation = (item: Phase6Recommendation): Recommendation => ({
  meme_id: item.meme_id,
  semantic_similarity: item.semantic_similarity,
  emotion_relevance: item.emotion_compatibility,
  sentiment_relevance: null,
  final_score: item.score,
  usage_categories: [],
  explanation: "Semantic and emotion-aware match.",
  image_available: item.image_available,
  image_url: item.image_url,
});

export function Phase7RecommendationExperience() {
  const [text, setText] = useState("");
  const [selectedEmotion, setSelectedEmotion] = useState<Emotion | "">("");
  const [results, setResults] = useState<Recommendation[]>([]);
  const [emotions, setEmotions] = useState<string[]>([]);
  const [probabilities, setProbabilities] = useState<Record<string, number>>({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requestVersion = useRef(0);
  const canSubmit = Boolean(text.trim() || selectedEmotion);

  async function search() {
    if (!canSubmit) return;
    const version = ++requestVersion.current;
    setLoading(true);
    setError(null);
    setResults([]);
    setEmotions([]);
    setProbabilities({});
    try {
      if (!text.trim() && selectedEmotion) {
        const response = await getPhase5EmotionRecommendations(selectedEmotion, 5);
        if (version !== requestVersion.current) return;
        setResults(response.recommendations.slice(0, 5).map(asRecommendation));
        return;
      }
      const response = await getPhase6TextRecommendations(text.trim(), 5);
      if (version !== requestVersion.current) return;
      setResults(response.recommendations.slice(0, 5).map(asRecommendation));
      setEmotions(response.predicted_emotions.emotion_labels);
      setProbabilities(response.predicted_emotions.emotion_probabilities);
    } catch (reason) {
      if (version === requestVersion.current) {
        setError(reason instanceof Error ? reason.message : "Could not reach the recommendation service.");
      }
    } finally {
      if (version === requestVersion.current) setLoading(false);
    }
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    void search();
  }

  return <section className="discovery" aria-label="Discover memes">
    <form className="discovery-panel" onSubmit={submit}>
      <div className="discovery-form">
        <label className="situation-label" htmlFor="situation">How are you feeling?
          <textarea id="situation" value={text} onChange={(event) => setText(event.target.value)} placeholder="When my code finally works after three hours of debugging" maxLength={1000} />
        </label>
        <fieldset className="emotion-picker phase7-emotion-picker">
          <legend>Or choose an emotion</legend>
          <div>{emotionChoices.map((emotion) => <button type="button" className={selectedEmotion === emotion ? "emotion active" : "emotion"} key={emotion} aria-pressed={selectedEmotion === emotion} onClick={() => setSelectedEmotion(selectedEmotion === emotion ? "" : emotion)}>{label(emotion)}</button>)}</div>
        </fieldset>
      </div>
      <div className="discovery-submit">
        <button type="submit" disabled={!canSubmit || loading}>{loading ? "Finding memes…" : "Find Memes"}</button>
        <span>{text.trim() ? "Your description will be used." : "Choose an emotion or tell us what is on your mind."}</span>
      </div>
    </form>
    {error && <section className="notice error" role="alert"><p>{error}</p><button type="button" onClick={() => void search()} disabled={!canSubmit || loading}>Retry</button></section>}
    {loading && <p className="searching" aria-live="polite">Finding memes that match your moment…</p>}
    {emotions.length > 0 && <section className="notice" aria-live="polite"><strong>Detected emotions: </strong>{emotions.map((emotion) => `${emotion} (${Math.round((probabilities[emotion] ?? 0) * 100)}%)`).join(", ")}</section>}
    {!loading && !error && results.length === 0 && <p className="empty-state">Describe how you are feeling to find a meme.</p>}
    {results.length > 0 && <section className="results" aria-live="polite" aria-label="Meme recommendations">
      <div className="results-heading"><div><p className="eyebrow">Your selection</p><h2>A meme for this moment.</h2><span>{results.length} recommendations</span></div></div>
      <div className="meme-grid">{results.map((item) => <RecommendationCard key={item.meme_id} recommendation={item} />)}</div>
    </section>}
  </section>;
}
