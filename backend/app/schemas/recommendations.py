from pydantic import BaseModel


class RecommendationResponse(BaseModel):
    meme_id: str
    semantic_similarity: float | None
    emotion_relevance: float | None
    sentiment_relevance: float | None
    final_score: float
    usage_categories: list[str]
    explanation: str
    image_available: bool
    image_url: str | None


class RecommendationStatusResponse(BaseModel):
    ready: bool
    detail: str | None
    meme_count: int


class RecommendationOptionsResponse(BaseModel):
    emotions: list[str]
    sentiments: list[str]
    usage_categories: list[str]
