import { useEffect, useState } from "react";
import { getFeedbackAnalytics, type FeedbackAnalytics } from "../lib/api";

export function FeedbackDashboard() {
  const [data, setData] = useState<FeedbackAnalytics | null>(null); const [error, setError] = useState(false);
  useEffect(() => { void getFeedbackAnalytics().then(setData).catch(() => setError(true)); }, []);
  if (error) return <section className="analytics-panel" aria-label="Local feedback dashboard"><p>Local feedback analytics are unavailable.</p></section>;
  if (!data) return <section className="analytics-panel" aria-busy="true">Loading local feedback…</section>;
  if (!Array.isArray(data.recent) || typeof data.total !== "number") return <section className="analytics-panel" aria-label="Local feedback dashboard"><p>Local feedback analytics are unavailable.</p></section>;
  return <section className="analytics-panel" aria-label="Local feedback dashboard"><h2>Local feedback</h2><p className="analytics-note">Product feedback is directional local input, not model accuracy.</p><div className="analytics-stats"><span><b>{data.total}</b> total</span><span><b>{data.useful}</b> useful</span><span><b>{data.not_relevant}</b> not relevant</span><span><b>{data.useful_percentage === null ? "—" : `${data.useful_percentage.toFixed(0)}%`}</b> useful</span></div>{data.recent.length ? <ul>{data.recent.map((item) => <li key={`${item.request_id}-${item.meme_id}`}>{item.meme_id} · {item.feedback_value}</li>)}</ul> : <p>No product feedback yet.</p>}</section>;
}
