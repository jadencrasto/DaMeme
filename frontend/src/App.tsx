import { HealthStatus } from "./components/HealthStatus";

export default function App() {
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
