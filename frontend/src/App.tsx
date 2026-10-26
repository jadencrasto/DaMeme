import { useEffect, useMemo, useState } from "react";

import { HealthStatus } from "./components/HealthStatus";
import { RecommendationCard } from "./components/RecommendationCard";
import { CinematicRecommendationExperienceC2 } from "./components/CinematicRecommendationExperienceC2";
import { type Emotion, type Mood, type Recommendation, type RecommendationOptions, getEmotionRecommendations, getMoods, getRecommendationOptions, getRecommendationStatus, getTextRecommendations } from "./lib/api";
import { saveHistory } from "./lib/sessionHistory";

type Mode = "text" | "emotion";
const label = (value: string) => value.charAt(0).toUpperCase() + value.slice(1);

// eslint-disable-next-line @typescript-eslint/no-unused-vars
function LegacyApp() {
  const [mode, setMode] = useState<Mode>("text"); const [options, setOptions] = useState<RecommendationOptions | null>(null);
  const [text, setText] = useState(""); const [emotion, setEmotion] = useState<Emotion | "">(""); const [moods, setMoods] = useState<Mood[]>([]); const [mood, setMood] = useState(""); const [moodSearch, setMoodSearch] = useState(""); const [sentiment, setSentiment] = useState(""); const [usage, setUsage] = useState<string[]>([]); const [limit, setLimit] = useState(10);
  const [results, setResults] = useState<Recommendation[]>([]); const [loading, setLoading] = useState(false); const [error, setError] = useState<string | null>(null); const [assetsReady, setAssetsReady] = useState<boolean | null>(null);
  const selectedUsage = useMemo(() => new Set(usage), [usage]); const canSubmit = mode === "text" ? text.trim().length > 0 : Boolean(emotion);
  useEffect(() => { let mounted = true; Promise.all([getRecommendationStatus(), getRecommendationOptions(), getMoods()]).then(([status, supported, taxonomy]) => { if (mounted) { setAssetsReady(status.ready); setOptions(supported); setMoods(taxonomy.moods); } }).catch((reason: unknown) => { if (mounted) setError(reason instanceof Error ? reason.message : "Could not reach the recommendation service."); }); return () => { mounted = false; }; }, []);
  async function runRecommendations() { if (!canSubmit) return; setLoading(true); setError(null); setResults([]); try { const result = mode === "text" ? await getTextRecommendations({ text: text.trim(), emotion: emotion || undefined, mood: mood || undefined, sentiment: sentiment as never || undefined, usage, limit }) : await getEmotionRecommendations({ emotion: emotion as Emotion, sentiment: sentiment as never || undefined, usage, limit }); const requestId = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`; sessionStorage.setItem("dameme-product-request", requestId); sessionStorage.setItem("dameme-product-mood", mode === "text" ? mood : ""); if (mode === "text") saveHistory({ text: text.trim(), mood }); setResults(result); } catch (reason) { setError(reason instanceof Error ? reason.message : "Could not load recommendations."); } finally { setLoading(false); } }
  function toggleUsage(category: string) { setUsage((current) => current.includes(category) ? current.filter((item) => item !== category) : [...current, category]); }
  const visibleMoods = moods.filter((item) => `${item.label} ${item.synonyms.join(" ")}`.toLowerCase().includes(moodSearch.toLowerCase()));
  return <main className="shell"><nav aria-label="Primary navigation" className="nav"><a className="brand" href="#top" aria-label="DaMeme home">Da<span>Meme</span></a><HealthStatus /></nav><section id="top" className="hero" aria-labelledby="hero-title"><p className="eyebrow">Emotion-aware meme discovery</p><h1 id="hero-title">Find the meme that <em>gets it.</em></h1><p className="intro">Describe a moment, choose a feeling, and explore source-backed meme recommendations.</p><form className="search-card" onSubmit={(event) => { event.preventDefault(); void runRecommendations(); }}><div className="mode-tabs" role="tablist" aria-label="Recommendation mode"><button type="button" role="tab" aria-selected={mode === "text"} className={mode === "text" ? "active" : ""} onClick={() => setMode("text")}>Describe a situation</button><button type="button" role="tab" aria-selected={mode === "emotion"} className={mode === "emotion" ? "active" : ""} onClick={() => setMode("emotion")}>Browse by emotion</button></div>{mode === "text" && <><label htmlFor="situation">Describe your mood or situation<input id="situation" value={text} onChange={(event) => setText(event.target.value)} placeholder="e.g. surviving a Monday meeting" maxLength={1000} /></label><fieldset className="mood-picker"><legend>Optional mood context</legend><input aria-label="Search moods" value={moodSearch} onChange={(event) => setMoodSearch(event.target.value)} placeholder="Search 32 moods" />{[...new Set(visibleMoods.map((item) => item.group))].map((group) => <div key={group}><small>{group}</small><div>{visibleMoods.filter((item) => item.group === group).map((item) => <button type="button" title={item.description} className={mood === item.id ? "emotion active" : "emotion"} key={item.id} onClick={() => setMood(mood === item.id ? "" : item.id)}>{item.label}</button>)}</div></div>)}{mood && <button type="button" className="clear-mood" onClick={() => setMood("")}>Clear mood</button>}</fieldset></>}<fieldset className="emotion-picker"><legend>{mode === "text" ? "Optional trained-emotion preference" : "Choose a trained emotion"}</legend><div>{options?.emotions.map((item) => <button type="button" className={emotion === item ? "emotion active" : "emotion"} key={item} onClick={() => setEmotion(emotion === item ? "" : item)}>{label(item)}</button>)}{!options && <small>Loading supported emotions…</small>}</div></fieldset><details className="advanced"><summary>Advanced filters</summary><div className="filters"><label>Sentiment<select value={sentiment} onChange={(event) => setSentiment(event.target.value)}><option value="">No preference</option>{options?.sentiments.map((item) => <option key={item} value={item}>{label(item)}</option>)}</select></label><label>Results<select value={limit} onChange={(event) => setLimit(Number(event.target.value))}>{[5, 10, 20].map((item) => <option key={item} value={item}>{item}</option>)}</select></label>{options?.usage_categories.length ? <fieldset className="usage"><legend>Usage</legend>{options.usage_categories.map((item) => <label key={item}><input type="checkbox" checked={selectedUsage.has(item)} onChange={() => toggleUsage(item)} /> {item.replaceAll("_", " ")}</label>)}</fieldset> : null}</div></details><div className="submit-row"><button type="submit" disabled={!canSubmit || loading || assetsReady === false}>{loading ? "Finding matches…" : "Recommend memes"}</button>{assetsReady === false && <p>Recommendation assets are not available locally.</p>}</div></form>{error && <section className="notice error" role="alert"><p>{error}</p><button type="button" onClick={() => void runRecommendations()}>Retry</button></section>}{loading && <p className="loading" aria-live="polite">Searching semantic and mood signals…</p>}{!loading && results.length > 0 && <section className="results" aria-live="polite"><div className="results-heading"><p className="eyebrow">Recommendations</p><span>{results.length} found</span></div><div className="meme-grid">{results.map((item) => <RecommendationCard key={item.meme_id} recommendation={item} />)}</div></section>}{!loading && !error && results.length === 0 && assetsReady && <p className="empty">Try describing a situation or choose an emotion to begin.</p>}</section><footer>Built as a research-first multimodal NLP project.</footer></main>;
}

/* Legacy Phase 1 landing screen retained in repository history:
  return (
    <main className="shell">
      <nav aria-label="Primary navigation" className="nav">
        <a className="brand" href="#top" aria-label="DaMeme home">Da<span>Meme</span></a>
        <HealthStatus />
      </nav>
      <section id="top" className="hero" aria-labelledby="hero-title">
        <p className="eyebrow">Emotion-aware meme discovery · Phase 1</p>
        <h1 id="hero-title">Find the meme that <em>gets it.</em></h1>
        <p className="intro">DaMeme is building a more human way to explore internet culture: semantic meaning, visual context, and emotion—together.</p>
        <form className="search-card" onSubmit={(event) => event.preventDefault()}>
          <label htmlFor="future-search">What’s the situation?</label>
          <div className="search-row">
            <input id="future-search" name="future-search" placeholder="e.g. surviving a Monday meeting" disabled />
            <button type="submit" disabled>Coming soon</button>
          </div>
          <p>Search and recommendations arrive in later phases. No models are running yet.</p>
        </form>
      </section>
      <footer>Built as a research-first multimodal NLP project.</footer>
    </main>
  );
}

*/

export default function App() {
  return <main className="shell"><nav aria-label="Primary navigation" className="nav"><a className="brand" href="#top">Da<span>Meme</span></a><div className="nav-links"><a href="#discover">Discover</a><HealthStatus /></div></nav><section id="top" className="hero"><p className="eyebrow">Emotion-aware meme discovery</p><h1>Every feeling deserves a <em>meme.</em></h1><p className="intro">Describe your moment or choose your mood. Find a meme that gets it.</p><div id="discover"><CinematicRecommendationExperienceC2 /></div></section><footer>Made for the moments that need a better reaction.</footer></main>;
}
