from typing import Annotated
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from app.recommendation_engine import (
    RecommendationEngine,
    RecommendationUnavailable,
    get_recommendation_engine,
)
from app.recommendations import RankingWeights, Recommendation
from app.schemas.recommendations import (
    RecommendationOptionsResponse,
    RecommendationResponse,
    RecommendationStatusResponse,
)

router = APIRouter(tags=["recommendations"])
Engine = Annotated[RecommendationEngine, Depends(get_recommendation_engine)]
Emotion = Annotated[str | None, Query(pattern="^(anger|fear|joy|sadness|surprise|disgust)$")]
Sentiment = Annotated[str | None, Query(pattern="^(positive|negative|neutral)$")]


def _weights(semantic: float, emotion: float, sentiment: float) -> RankingWeights:
    try:
        return RankingWeights(semantic, emotion, sentiment)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def _response(item: Recommendation, engine: RecommendationEngine) -> RecommendationResponse:
    available = engine.image_path(item.meme_id) is not None
    return RecommendationResponse(
        meme_id=item.meme_id,
        semantic_similarity=item.semantic_similarity,
        emotion_relevance=item.emotion_relevance,
        sentiment_relevance=item.sentiment_relevance,
        final_score=item.final_score,
        usage_categories=list(item.usage_categories),
        explanation=item.explanation,
        image_available=available,
        image_url=f"/api/v1/memes/{quote(item.meme_id, safe='')}/image" if available else None,
    )


@router.get("/recommendations/status", response_model=RecommendationStatusResponse)
def status(engine: Engine) -> RecommendationStatusResponse:
    return RecommendationStatusResponse(**engine.status())


@router.get("/recommendations/options", response_model=RecommendationOptionsResponse)
def options(engine: Engine) -> RecommendationOptionsResponse:
    try:
        return RecommendationOptionsResponse(**engine.options())
    except RecommendationUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc


@router.get("/recommendations/text", response_model=list[RecommendationResponse])
def text_recommendations(
    engine: Engine,
    text: Annotated[str, Query(min_length=1, max_length=1000)],
    emotion: Emotion = None,
    sentiment: Sentiment = None,
    usage: Annotated[list[str] | None, Query(max_length=20)] = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
    candidate_limit: Annotated[int, Query(ge=1, le=281)] = 100,
    semantic_weight: Annotated[float, Query(ge=0)] = 0.8,
    emotion_weight: Annotated[float, Query(ge=0)] = 0.15,
    sentiment_weight: Annotated[float, Query(ge=0)] = 0.05,
    min_semantic_similarity: Annotated[float | None, Query(ge=-1, le=1)] = None,
    max_per_template: Annotated[int | None, Query(ge=1)] = None,
) -> list[RecommendationResponse]:
    if candidate_limit < limit:
        raise HTTPException(422, "candidate_limit must be at least limit")
    weights = _weights(semantic_weight, emotion_weight, sentiment_weight)
    try:
        rows = engine.recommend_text(text, emotion=emotion, sentiment=sentiment, usage=usage or [], weights=weights, candidate_limit=candidate_limit, min_semantic_similarity=min_semantic_similarity, diversity_max_per_template=max_per_template, limit=limit)
        return [_response(item, engine) for item in rows]
    except RecommendationUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc


@router.get("/recommendations/emotion", response_model=list[RecommendationResponse])
def emotion_recommendations(
    engine: Engine,
    emotion: Annotated[str, Query(pattern="^(anger|fear|joy|sadness|surprise|disgust)$")],
    sentiment: Sentiment = None,
    usage: Annotated[list[str] | None, Query(max_length=20)] = None,
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
    emotion_weight: Annotated[float, Query(ge=0)] = 1.0,
    sentiment_weight: Annotated[float, Query(ge=0)] = 1.0,
    max_per_template: Annotated[int | None, Query(ge=1)] = None,
) -> list[RecommendationResponse]:
    weights = _weights(0.0, emotion_weight, sentiment_weight)
    try:
        rows = engine.recommend_emotion(emotion, sentiment=sentiment, usage=usage or [], weights=weights, diversity_max_per_template=max_per_template, limit=limit)
        return [_response(item, engine) for item in rows]
    except RecommendationUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc


@router.get("/memes/{meme_id}/image")
def meme_image(meme_id: str, engine: Engine) -> FileResponse:
    try:
        image = engine.image_path(meme_id)
    except RecommendationUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    if image is None:
        raise HTTPException(404, "Meme image is not available locally")
    return FileResponse(image, filename=image.name)
