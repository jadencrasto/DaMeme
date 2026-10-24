"""Lazy application boundary for local recommendation artifacts."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from app.core.config import get_settings
from app.recommendations import (
    MemeMatchMetadata,
    RankingWeights,
    Recommendation,
    RecommendationService,
)
from app.retrieval import RetrievalService, TextQueryEmbedder


class RecommendationUnavailable(RuntimeError):
    pass


class RecommendationEngine:
    def __init__(self, index_dir: Path, metadata_csv: Path, image_root: Path | None = None):
        self.index_dir, self.metadata_csv, self.image_root = index_dir, metadata_csv, image_root
        self._service: RecommendationService | None = None
        self._error: str | None = None

    def _load(self) -> RecommendationService:
        if self._service is not None:
            return self._service
        if self._error is not None:
            raise RecommendationUnavailable(self._error)
        try:
            metadata = MemeMatchMetadata.from_csv(self.metadata_csv, self.image_root)
            self._service = RecommendationService(RetrievalService.load(self.index_dir), metadata)
        except (FileNotFoundError, ValueError, OSError) as exc:
            self._error = str(exc)
            raise RecommendationUnavailable(self._error) from exc
        return self._service

    def status(self) -> dict[str, object]:
        try:
            service = self._load()
        except RecommendationUnavailable as exc:
            return {"ready": False, "detail": str(exc), "meme_count": 0}
        return {"ready": True, "detail": None, "meme_count": len(service.metadata.entries)}

    def options(self) -> dict[str, object]:
        service = self._load()
        usage = sorted({item for entry in service.metadata.entries.values() for item in entry.usage})
        return {"emotions": ["anger", "fear", "joy", "sadness", "surprise", "disgust"], "sentiments": ["positive", "negative", "neutral"], "usage_categories": usage}

    def recommend_text(
        self, text: str, *, emotion: str | None, sentiment: str | None, usage: list[str],
        weights: RankingWeights, candidate_limit: int, min_semantic_similarity: float | None,
        diversity_max_per_template: int | None, limit: int,
    ) -> list[Recommendation]:
        service = self._load()
        query = TextQueryEmbedder(service.retrieval).embed(text)
        from app.recommendations import DiversityConfig
        diversity = DiversityConfig(diversity_max_per_template) if diversity_max_per_template else None
        return service.recommend_text(query, emotion=emotion, sentiment=sentiment, usage_preferences=usage, weights=weights, candidate_limit=candidate_limit, min_semantic_similarity=min_semantic_similarity, diversity=diversity, limit=limit)

    def recommend_emotion(
        self, emotion: str, *, sentiment: str | None, usage: list[str], weights: RankingWeights,
        diversity_max_per_template: int | None, limit: int,
    ) -> list[Recommendation]:
        from app.recommendations import DiversityConfig
        diversity = DiversityConfig(diversity_max_per_template) if diversity_max_per_template else None
        return self._load().recommend_emotion(emotion, sentiment=sentiment, usage_preferences=usage, weights=weights, diversity=diversity, limit=limit)

    def image_path(self, meme_id: str) -> Path | None:
        entry = self._load().metadata.get(meme_id)
        if entry is None or entry.image_path is None:
            return None
        path = Path(entry.image_path)
        return path if path.is_file() else None


@lru_cache
def get_recommendation_engine() -> RecommendationEngine:
    settings = get_settings()
    return RecommendationEngine(
        settings.recommendation_index_dir,
        settings.recommendation_metadata_csv,
        settings.recommendation_image_root,
    )
