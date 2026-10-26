import { type FormEvent, useEffect, useRef, useState } from "react";

import { RecommendationCard } from "./RecommendationCard";
import { type FeedbackValue, type Mood, type Recommendation, getMoods, getRecommendationStatus, getTextRecommendations, submitFeedback } from "../lib/api";
import { clearHistory, loadHistory, saveHistory, type HistoryEntry } from "../lib/sessionHistory";

type FeedbackState = FeedbackValue | "submitting" | "error";
const identifier = () => globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;

export function RecommendationExperience() {
  const [text, setText] = useState(""); const [mood, setMood] = useState(""); const [moods, setMoods] = useState<Mood[]>([]);
  const [results, setResults] = useState<Recommendation[]>([]); const [seen, setSeen] = useState<Set<string>>(new Set()); const [batchIds, setBatchIds] = useState<Record<string, string>>({}); const [batchMoods, setBatchMoods] = useState<Record<string, string>>({}); const [activeSearch, setActiveSearch] = useState<HistoryEntry | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>(() => loadHistory()); const [loading, setLoading] = useState(false); const [error, setError] = useState<string | null>(null); const [exhausted, setExhausted] = useState(false); const [feedback, setFeedback] = useState<Record<string, FeedbackState>>({});
  const requestVersion = useRef(0);
  useEffect(() => { void Promise.all([getMoods(), getRecommendationStatus()]).then(([taxonomy, status]) => { setMoods(taxonomy.moods); if (!status.ready) setError(status.detail ?? "Recommendation assets are unavailable."); }).catch(() => setError("Could not reach the recommendation service.")); }, []);
  async function fetchBatch(entry: HistoryEntry, more: boolean) {
    const normalized = { text: entry.text.trim(), mood: entry.mood.trim() };
    if ((!normalized.text && !normalized.mood) || loading) { if (!normalized.text && !normalized.mood) setError("Describe a situation or select a mood."); return; }
    const version = ++requestVersion.current; setLoading(true); setError(null); if (!more) { setResults([]); setSeen(new Set()); setBatchIds({}); setBatchMoods({}); setFeedback({}); setExhausted(false); }
    const exclusions = more ? [...seen] : [];
    try {
      const rows = await getTextRecommendations({ text: normalized.text || undefined, mood: normalized.mood || undefined, usage: [], limit: 10, excludeMemeIds: exclusions });
      if (version !== requestVersion.current) return;
      const unique = rows.filter((row) => !exclusions.includes(row.meme_id)); const batchId = identifier();
      if (more && unique.length === 0) setExhausted(true);
      setResults((current) => more ? [...current, ...unique.filter((row) => !current.some((existing) => existing.meme_id === row.meme_id))] : unique);
      setSeen((current) => new Set([...current, ...unique.map((row) => row.meme_id)]));
      setBatchIds((current) => Object.fromEntries([...Object.entries(more ? current : {}), ...unique.map((row) => [row.meme_id, batchId])]));
      setBatchMoods((current) => Object.fromEntries([...Object.entries(more ? current : {}), ...unique.map((row) => [row.meme_id, normalized.mood])]));
      if (!more) { setActiveSearch(normalized); const next = saveHistory(normalized); setHistory(next); }
    } catch (reason) { if (version === requestVersion.current) setError(reason instanceof Error ? reason.message : "Could not load recommendations."); }
    finally { if (version === requestVersion.current) setLoading(false); }
  }
  function search(event?: FormEvent, entry = { text, mood }) { event?.preventDefault(); void fetchBatch(entry, false); }
  async function rate(item: Recommendation, value: FeedbackValue) { const requestId = batchIds[item.meme_id]; if (!requestId || feedback[item.meme_id]) return; setFeedback((current) => ({ ...current, [item.meme_id]: "submitting" })); try { await submitFeedback({ request_id: requestId, meme_id: item.meme_id, selected_mood: batchMoods[item.meme_id] || undefined, retrieval_variant: "a0", value }); setFeedback((current) => ({ ...current, [item.meme_id]: value })); } catch { setFeedback((current) => ({ ...current, [item.meme_id]: "error" })); } }
  return <section className="experience" aria-label="Meme recommendations"><form className="search-card" onSubmit={search}><label htmlFor="situation">Describe a situation<input id="situation" aria-describedby="situation-help" value={text} onChange={(event) => setText(event.target.value)} placeholder="e.g. surviving a Monday meeting" maxLength={1000} /></label><p id="situation-help">Optional when you select a mood.</p><fieldset className="mood-picker"><legend>Optional mood</legend><div>{moods.map((item) => <button type="button" className={mood === item.id ? "emotion active" : "emotion"} key={item.id} onClick={() => setMood(mood === item.id ? "" : item.id)}>{item.label}</button>)}</div></fieldset><button type="submit" disabled={(!text.trim() && !mood) || loading}>{loading ? "Finding matches…" : "Recommend memes"}</button></form><section className="history-panel" aria-label="Recent searches"><div><h2>Recent searches</h2><button type="button" onClick={() => { clearHistory(); setHistory([]); }}>Clear history</button></div>{history.length ? <ul>{history.map((entry) => <li key={`${entry.text}-${entry.mood}`}><button type="button" onClick={() => { setText(entry.text); setMood(entry.mood); void fetchBatch(entry, false); }}>{entry.text || "Mood-only search"}{entry.mood ? ` · ${entry.mood}` : ""}</button></li>)}</ul> : <p>No recent searches in this browser session.</p>}</section>{error && <p className="notice error" role="alert">{error}</p>}{results.length > 0 && <section className="results" aria-live="polite"><div className="results-heading"><h2>Recommendations</h2><span>{results.length} shown</span><button type="button" disabled={loading || exhausted} onClick={() => activeSearch && void fetchBatch(activeSearch, true)}>{exhausted ? "No more safe matches" : loading ? "Loading…" : "Show me more"}</button></div><div className="meme-grid">{results.map((item) => <RecommendationCard key={item.meme_id} recommendation={item} feedback={feedback[item.meme_id]} onFeedback={(value) => void rate(item, value)} />)}</div></section>}{!loading && !error && results.length === 0 && <p className="empty">Describe a situation or select a mood to start exploring.</p>}</section>;
}
