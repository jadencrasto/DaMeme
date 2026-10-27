from pathlib import Path

import pytest
from fastapi import HTTPException

from app.api.v1.recommendations import (
    _weights,
    emotion_recommendations,
    meme_image,
    options,
    status,
    text_recommendations,
)
from app.main import create_app
from app.moods import MOODS
from app.recommendation_engine import RecommendationUnavailable
from app.recommendations import Recommendation


class FakeEngine:
    def __init__(self, image: Path | None = None, unavailable: bool = False):
        self.image, self.unavailable, self.text_calls = image, unavailable, []

    def status(self) -> dict[str, object]:
        return {
            "ready": not self.unavailable,
            "detail": "missing assets" if self.unavailable else None,
            "meme_count": 1 if not self.unavailable else 0,
        }

    def options(self) -> dict[str, object]:
        if self.unavailable:
            raise RecommendationUnavailable("missing assets")
        return {"emotions": ["joy"], "sentiments": ["positive"], "usage_categories": ["sarcasm"]}

    @staticmethod
    def _row() -> Recommendation:
        return Recommendation(
            "meme.jpg",
            0.8,
            0.7,
            None,
            0.75,
            ("sarcasm",),
            "text similarity 0.800; source joy probability 0.700",
        )

    def recommend_text(self, text: str, **kwargs: object) -> list[Recommendation]:
        if self.unavailable:
            raise RecommendationUnavailable("missing assets")
        self.text_calls.append((text, kwargs))
        return [self._row()]

    def recommend_emotion(self, emotion: str, **_: object) -> list[Recommendation]:
        if self.unavailable:
            raise RecommendationUnavailable("missing assets")
        return [self._row()]

    def image_path(self, meme_id: str) -> Path | None:
        return self.image if meme_id == "meme.jpg" else None


def test_text_and_emotion_recommendations_are_serialized_without_model_loads() -> None:
    engine = FakeEngine()
    response = text_recommendations(engine, "study", "joy", None, None, 1, 10, 1, 1, 0)
    assert response[0].meme_id == "meme.jpg"
    assert response[0].image_url is None
    assert engine.text_calls[0][0] == "study"
    weights = engine.text_calls[0][1]["weights"]
    assert (weights.semantic, weights.emotion, weights.sentiment) == (1, 1, 0)
    assert emotion_recommendations(engine, "joy", None, None, 1, 1, 0)[0].meme_id == "meme.jpg"


def test_text_api_exposes_similarity_gate_and_template_cap() -> None:
    engine = FakeEngine()
    text_recommendations(engine, "study", "joy", None, None, 1, 10, 0.8, 0.15, 0.05, 0.2, 1)
    _, kwargs = engine.text_calls[0]
    assert kwargs["min_semantic_similarity"] == 0.2
    assert kwargs["diversity_max_per_template"] == 1


def test_text_api_defaults_to_a_larger_reranking_pool_and_safety_filter() -> None:
    engine = FakeEngine()
    text_recommendations(engine, "study")
    _, kwargs = engine.text_calls[0]
    assert kwargs["candidate_limit"] == 50
    assert kwargs["exclude_unsafe"] is True
    assert kwargs["diversity_max_per_template"] == 1
    assert kwargs["retrieval_mode"] == "semantic"
    assert kwargs["sentiment"] is None


def test_text_api_only_forwards_sentiment_when_the_caller_supplies_it() -> None:
    engine = FakeEngine()
    text_recommendations(engine, "I passed", sentiment="positive")
    assert engine.text_calls[0][1]["sentiment"] == "positive"


@pytest.mark.parametrize("mood", [item.id for item in MOODS])
def test_mood_only_requests_use_the_a0_text_route_for_every_supported_mood(mood: str) -> None:
    engine = FakeEngine()
    response = text_recommendations(engine, None, mood=mood, exclude_meme_id=["seen.jpg"])
    assert response[0].meme_id == "meme.jpg"
    assert engine.text_calls[0][0] == ""
    assert engine.text_calls[0][1]["mood"] == mood
    assert engine.text_calls[0][1]["exclude_meme_ids"] == {"seen.jpg"}


def test_text_route_rejects_a_blank_situation_without_a_mood() -> None:
    with pytest.raises(HTTPException) as invalid:
        text_recommendations(FakeEngine(), "   ")
    assert invalid.value.status_code == 422
    assert invalid.value.detail == "Provide a situation or select a mood"


def test_text_api_forwards_hybrid_controls_without_changing_response_schema() -> None:
    engine = FakeEngine()
    response = text_recommendations(
        engine,
        "I passed my exam",
        retrieval_mode="hybrid_cross_encoder",
        bm25_candidate_limit=75,
        cross_encoder_limit=25,
    )
    _, kwargs = engine.text_calls[0]
    assert response[0].model_dump().keys() == {
        "meme_id",
        "semantic_similarity",
        "emotion_relevance",
        "sentiment_relevance",
        "final_score",
        "usage_categories",
        "explanation",
        "image_available",
        "image_url",
    }
    assert kwargs["retrieval_mode"] == "hybrid_cross_encoder"
    assert kwargs["bm25_candidate_limit"] == 75
    assert kwargs["cross_encoder_limit"] == 25


def test_validation_unavailable_assets_and_safe_image_delivery(tmp_path: Path) -> None:
    engine = FakeEngine(unavailable=True)
    assert status(engine).ready is False
    with pytest.raises(HTTPException) as unavailable:
        options(engine)
    assert getattr(unavailable.value, "status_code", None) == 503
    valid = FakeEngine()
    with pytest.raises(HTTPException) as invalid:
        text_recommendations(valid, "x", None, None, None, 3, 2, 1, 1, 1)
    assert getattr(invalid.value, "status_code", None) == 422
    with pytest.raises(HTTPException):
        _weights(0, 0, 0)
    image = tmp_path / "images" / "meme.jpg"
    image.parent.mkdir()
    image.write_bytes(b"image")
    served = meme_image("meme.jpg", FakeEngine(image))
    assert served.path == image
    with pytest.raises(HTTPException) as missing:
        meme_image("../secret", FakeEngine(image))
    assert getattr(missing.value, "status_code", None) == 404


def test_existing_app_includes_recommendation_routes() -> None:
    paths = {route.path for route in create_app().routes}
    assert "/api/v1/recommendations/text" in paths
    assert "/api/v1/memes/{meme_id}/image" in paths
    assert "/api/v1/emotion/text" in paths
