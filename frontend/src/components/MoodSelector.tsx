import { useMemo, useState } from "react";

import type { Mood } from "../lib/api";

const quickMoodIds = ["joy", "relief", "amusement", "sadness", "frustration", "anxiety", "anger", "calm"];

export function MoodSelector({ moods, selected, onChange, required, label }: { moods: Mood[]; selected: string; onChange: (mood: string) => void; required?: boolean; label: string }) {
  const [filter, setFilter] = useState("");
  const quickMoods = moods.filter((mood) => quickMoodIds.includes(mood.id));
  const filteredMoods = useMemo(() => moods.filter((mood) => `${mood.label} ${mood.synonyms.join(" ")} ${mood.description}`.toLowerCase().includes(filter.toLowerCase())), [filter, moods]);
  const select = (id: string) => onChange(selected === id && !required ? "" : id);
  return <fieldset className="mood-selector"><legend>{label}</legend><p className="field-note">{required ? "Choose one feeling to explore." : "Optional — add context without changing your situation."}</p><div className="quick-moods" aria-label="Popular moods">{quickMoods.map((mood) => <button type="button" key={mood.id} aria-pressed={selected === mood.id} className={selected === mood.id ? "selected" : ""} onClick={() => select(mood.id)}>{mood.label}</button>)}</div><label className="mood-search-label" htmlFor={`mood-search-${required ? "only" : "situation"}`}>Find another mood<input id={`mood-search-${required ? "only" : "situation"}`} type="search" value={filter} onChange={(event) => setFilter(event.target.value)} placeholder="Search all 32 moods" /></label><div className="mood-results" aria-label="All moods">{filteredMoods.map((mood) => <button type="button" key={mood.id} aria-pressed={selected === mood.id} className={selected === mood.id ? "selected" : ""} onClick={() => select(mood.id)}><span>{mood.label}</span></button>)}</div>{selected && !required && <button type="button" className="text-action" onClick={() => onChange("")}>Clear selected mood</button>}</fieldset>;
}
