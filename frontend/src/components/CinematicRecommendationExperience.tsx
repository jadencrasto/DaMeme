import { type FormEvent, type KeyboardEvent, useEffect, useRef, useState } from "react";

import { MoodSelector } from "./MoodSelector";
import { RecommendationCard } from "./RecommendationCard";
import { type FeedbackValue, type Mood, type Recommendation, getMoods, getRecommendationStatus, getTextRecommendations, submitFeedback } from "../lib/api";
import { clearHistory, loadHistory, saveHistory, type HistoryEntry } from "../lib/sessionHistory";

type DiscoveryMode = "situation" | "mood";
type FeedbackState = FeedbackValue | "submitting" | "error";
const identifier = () => globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;

export function CinematicRecommendationExperience() {
  const [mode, setMode] = useState<DiscoveryMode>("situation");
  const [situation, setSituation] = useState(""); const [situationMood, setSituationMood] = useState(""); const [moodOnly, setMoodOnly] = useState("");
  const [moods, setMoods] = useState<Mood[]>([]); const [results, setResults] = useState<Recommendation[]>([]); const [seen, setSeen] = useState<Set<string>>(new Set());
  const [batchIds, setBatchIds] = useState<Record<string, string>>({}); const [batchMoods, setBatchMoods] = useState<Record<string, string>>({}); const [activeSearch, setActiveSearch] = useState<HistoryEntry | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>(() => loadHistory()); const [loading, setLoading] = useState(false); const [error, setError] = useState<string | null>(null); const [exhausted, setExhausted] = useState(false); const [feedback, setFeedback] = useState<Record<string, FeedbackState>>({});
  const requestVersion = useRef(0);

  useEffect(() => { void Promise.all([getMoods(), getRecommendationStatus()]).then(([taxonomy, status]) => { setMoods(taxonomy.moods); if (!status.ready) setError(status.detail ?? "Recommendation assets are unavailable."); }).catch(() => setError("Could not reach the recommendation service.")); }, []);
  const draft = (): HistoryEntry => mode === "situation" ? { text: situation.trim(), mood: situationMood } : { text: "", mood: moodOnly };
  const canSubmit = mode === "situation" ? Boolean(situation.trim()) : Boolean(moodOnly);

  async function fetchBatch(entry: HistoryEntry, more: boolean) {
    const normalized = { text: entry.text.trim(), mood: entry.mood.trim() };
    if ((!normalized.text && !normalized.mood) || loading) { if (!normalized.text && !normalized.mood) setError("Describe a situation or select a mood."); return; }
    const version = ++requestVersion.current; setLoading(true); setError(null);
    if (!more) { setResults([]); setSeen(new Set()); setBatchIds({}); setBatchMoods({}); setFeedback({}); setExhausted(false); }
    const exclusions = more ? [...seen] : [];
    try {
      const rows = await getTextRecommendations({ text: normalized.text || undefined, mood: normalized.mood || undefined, usage: [], limit: 10, excludeMemeIds: exclusions });
      if (version !== requestVersion.current) return;
      const unique = rows.filter((row) => !exclusions.includes(row.meme_id)); const batchId = identifier();
      if (more && unique.length === 0) setExhausted(true);
      setResults((current) => more ? [...current, ...unique.filter((row) => !current.some((item) => item.meme_id === row.meme_id))] : unique);
      setSeen((current) => new Set([...current, ...unique.map((row) => row.meme_id)]));
      setBatchIds((current) => Object.fromEntries([...Object.entries(more ? current : {}), ...unique.map((row) => [row.meme_id, batchId])]));
      setBatchMoods((current) => Object.fromEntries([...Object.entries(more ? current : {}), ...unique.map((row) => [row.meme_id, normalized.mood])]));
      if (!more) { setActiveSearch(normalized); setHistory(saveHistory(normalized)); }
    } catch (reason) { if (version === requestVersion.current) setError(reason instanceof Error ? reason.message : "Could not load recommendations."); }
    finally { if (version === requestVersion.current) setLoading(false); }
  }

  function submit(event: FormEvent) { event.preventDefault(); if (!canSubmit) { setError(mode === "situation" ? "Describe a situation to continue." : "Select a mood to continue."); return; } void fetchBatch(draft(), false); }
  function revisit(entry: HistoryEntry) { setMode(entry.text ? "situation" : "mood"); if (entry.text) { setSituation(entry.text); setSituationMood(entry.mood); } else setMoodOnly(entry.mood); void fetchBatch(entry, false); }
  async function rate(item: Recommendation, value: FeedbackValue) { const requestId = batchIds[item.meme_id]; if (!requestId || feedback[item.meme_id]) return; setFeedback((current) => ({ ...current, [item.meme_id]: "submitting" })); try { await submitFeedback({ request_id: requestId, meme_id: item.meme_id, selected_mood: batchMoods[item.meme_id] || undefined, retrieval_variant: "a0", value }); setFeedback((current) => ({ ...current, [item.meme_id]: value })); } catch { setFeedback((current) => ({ ...current, [item.meme_id]: "error" })); } }

  const tabKey = (event: KeyboardEvent<HTMLButtonElement>, next: DiscoveryMode) => { if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) { event.preventDefault(); setMode(next); } };
  return <section className="discovery" aria-label="Discover memes"><form className="discovery-panel" onSubmit={submit}><div className="discovery-tabs" role="tablist" aria-label="Discovery method"><button id="situation-tab" type="button" role="tab" aria-selected={mode === "situation"} aria-controls="situation-panel" tabIndex={mode === "situation" ? 0 : -1} onClick={() => setMode("situation")} onKeyDown={(event) => tabKey(event, "mood")}>Describe a situation</button><button id="mood-tab" type="button" role="tab" aria-selected={mode === "mood"} aria-controls="mood-panel" tabIndex={mode === "mood" ? 0 : -1} onClick={() => setMode("mood")} onKeyDown={(event) => tabKey(event, "situation")}>Just a mood</button></div>{mode === "situation" ? <div id="situation-panel" role="tabpanel" aria-labelledby="situation-tab" className="discovery-form"><label className="situation-label" htmlFor="situation">What happened?<textarea id="situation" value={situation} onChange={(event) => setSituation(event.target.value)} placeholder="When my code finally works after three hours of debugging" maxLength={1000} /></label><MoodSelector moods={moods} selected={situationMood} onChange={setSituationMood} label="Add a feeling" /></div> : <div id="mood-panel" role="tabpanel" aria-labelledby="mood-tab" className="discovery-form"><p className="mode-intro">No backstory needed. Pick the feeling you want a meme for.</p><MoodSelector moods={moods} selected={moodOnly} onChange={setMoodOnly} required label="Choose a mood" /></div>}<div className="discovery-submit"><button type="submit" disabled={loading}>{loading ? "Finding the right energy…" : "Find my meme"}</button><span>{mode === "situation" ? "A mood is optional here." : "A mood is required here."}</span></div></form>{error && <p className="notice error" role="alert">{error}</p>}{loading && <p className="searching" aria-live="polite">Curating a fresh set of safe, varied picks…</p>}{!loading && !error && results.length === 0 && <p className="empty-state">Start with a moment or a feeling. The rest is up to DaMeme.</p>}{results.length > 0 && <section className="results" aria-live="polite" aria-label="Meme recommendations"><div className="results-heading"><div><p className="eyebrow">Your selection</p><h2>A meme for this moment.</h2></div><button type="button" disabled={loading || exhausted || !activeSearch} onClick={() => activeSearch && void fetchBatch(activeSearch, true)}>{exhausted ? "That’s the set" : loading ? "Finding more…" : "Show me more"}</button></div>{exhausted && <p className="exhausted" role="status">You’ve seen every safe, varied match available for this search.</p>}<div className="meme-grid">{results.map((item) => <RecommendationCard key={item.meme_id} recommendation={item} feedback={feedback[item.meme_id]} onFeedback={(value) => void rate(item, value)} />)}</div></section>}<details className="history-drawer"><summary>Recent searches <span>{history.length}</span></summary><div>{history.length ? <ul>{history.map((entry) => <li key={`${entry.text}-${entry.mood}`}><button type="button" onClick={() => revisit(entry)}>{entry.text || "Just a mood"}<small>{entry.mood ? ` · ${entry.mood}` : ""}</small></button></li>)}</ul> : <p>No recent searches in this browser session.</p>}<button type="button" className="text-action" disabled={!history.length} onClick={() => { clearHistory(); setHistory([]); }}>Clear history</button></div></details></section>;
}
