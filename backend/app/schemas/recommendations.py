from typing import Literal

from pydantic import BaseModel, Field


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
    catalog_id: str = "memematch_281"
    image_count: int = 0
    database_ready: bool = False
    model_ready: bool = False


class RecommendationOptionsResponse(BaseModel):
    emotions: list[str]
    sentiments: list[str]
    usage_categories: list[str]


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=128)
    meme_id: str = Field(min_length=1, max_length=512)
    selected_mood: str | None = Field(default=None, max_length=64)
    retrieval_variant: Literal["a0"] = "a0"
    value: Literal["useful", "not_relevant"]


class FeedbackResponse(BaseModel):
    accepted: bool


class MoodResponse(BaseModel):
    id: str
    label: str
    group: str
    description: str
    synonyms: list[str]
    classifier_weights: dict[str, float]
    sentiment: str | None


class MoodTaxonomyResponse(BaseModel):
    moods: list[MoodResponse]
