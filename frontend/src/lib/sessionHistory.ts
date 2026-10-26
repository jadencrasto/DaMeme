export interface HistoryEntry { text: string; mood: string; }
const key = "dameme-search-history"; const maximumEntries = 8;
export function loadHistory(): HistoryEntry[] { try { const value = sessionStorage.getItem(key); const parsed: unknown = value ? JSON.parse(value) : []; return Array.isArray(parsed) ? parsed.filter((entry): entry is HistoryEntry => typeof entry?.text === "string" && typeof entry?.mood === "string").slice(0, maximumEntries) : []; } catch { return []; } }
export function saveHistory(entry: HistoryEntry): HistoryEntry[] { const next = [entry, ...loadHistory().filter((item) => item.text !== entry.text || item.mood !== entry.mood)].slice(0, maximumEntries); try { sessionStorage.setItem(key, JSON.stringify(next)); } catch { /* storage is optional */ } return next; }
export function clearHistory(): void { try { sessionStorage.removeItem(key); } catch { /* storage is optional */ } }
