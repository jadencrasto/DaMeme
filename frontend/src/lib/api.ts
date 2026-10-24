export type Emotion = "anger" | "fear" | "joy" | "sadness" | "surprise" | "disgust";
export type Sentiment = "positive" | "negative" | "neutral";

export interface Recommendation {
  meme_id: string;
  semantic_similarity: number | null;
  emotion_relevance: number | null;
  sentiment_relevance: number | null;
  final_score: number;
  usage_categories: string[];
  explanation: string;
  image_available: boolean;
  image_url: string | null;
}

export interface RecommendationOptions { emotions: Emotion[]; sentiments: Sentiment[]; usage_categories: string[]; }
export interface Mood { id: string; label: string; group: string; description: string; synonyms: string[]; classifier_weights: Record<string, number>; sentiment: Sentiment | null; }
export interface MoodTaxonomy { moods: Mood[]; }
export interface RecommendationStatus { ready: boolean; detail: string | null; meme_count: number; }

export class ApiError extends Error {
  constructor(message: string, public readonly status?: number) { super(message); }
}

export const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

async function request<T>(path: string): Promise<T> {
  const response = await fetch(`${apiBaseUrl}${path}`);
  if (!response.ok) {
    const body = await response.json().catch(() => null) as { detail?: string } | null;
    throw new ApiError(body?.detail ?? "The recommendation service is unavailable.", response.status);
  }
  return response.json() as Promise<T>;
}

function parameters(values: Record<string, string | number | undefined>, usage: string[]) {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(values)) if (value !== undefined && value !== "") query.set(key, String(value));
  usage.forEach((value) => query.append("usage", value));
  return query.toString();
}

export const getRecommendationStatus = () => request<RecommendationStatus>("/api/v1/recommendations/status");
export const getRecommendationOptions = () => request<RecommendationOptions>("/api/v1/recommendations/options");
export const getMoods = () => request<MoodTaxonomy>("/api/v1/emotions");

export function getTextRecommendations(input: { text: string; emotion?: Emotion; mood?: string; sentiment?: Sentiment; usage: string[]; limit: number; }) {
  const { usage, ...values } = input;
  return request<Recommendation[]>(`/api/v1/recommendations/text?${parameters(values, usage)}`);
}

export function getEmotionRecommendations(input: { emotion: Emotion; sentiment?: Sentiment; usage: string[]; limit: number; }) {
  const { usage, ...values } = input;
  return request<Recommendation[]>(`/api/v1/recommendations/emotion?${parameters(values, usage)}`);
}
