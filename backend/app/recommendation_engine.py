"""Lazy application boundary for local recommendation artifacts."""

from __future__ import annotations

import json
import warnings
from functools import lru_cache
from pathlib import Path

from app.core.config import get_settings
from app.hybrid_retrieval import (
    HybridRetriever,
    OfflineCrossEncoder,
    RetrievalMode,
    cross_encoder_with_fallback,
    reciprocal_rank_fusion,
)
from app.moods import mood_only_query, mood_query, primary_classifier_emotion, resolve_mood
from app.recommendations import (
    MemeMatchMetadata,
    RankingWeights,
    Recommendation,
    RecommendationService,
    SafetyConfig,
)
from app.retrieval import RetrievalService, TextQueryEmbedder


class RecommendationUnavailable(RuntimeError):
    pass


class RecommendationEngine:
    def __init__(
        self,
        index_dir: Path,
        metadata_csv: Path,
        image_root: Path | None = None,
        *,
        catalog_id: str = "custom",
        expected_count: int | None = None,
        expected_dimensions: tuple[int, int, int] | None = None,
        export_records: Path | None = None,
        export_metadata: Path | None = None,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        query_device: str = "cpu",
    ):
        self.index_dir, self.metadata_csv, self.image_root = index_dir, metadata_csv, image_root
        self.catalog_id, self.expected_count, self.expected_dimensions = catalog_id, expected_count, expected_dimensions
        self.export_records, self.export_metadata = export_records, export_metadata
        self.cross_encoder_model = cross_encoder_model
        self.query_device = query_device
        self.last_retrieval_latency_ms = 0.0
        self.last_cross_encoder_latency_ms = 0.0
        self.last_retrieval_warning: str | None = None
        self.last_retrieval_candidate_count = 0
        self.last_safe_candidate_count = 0
        self._service: RecommendationService | None = None
        self._hybrid: HybridRetriever | None = None
        self._cross_encoder: OfflineCrossEncoder | None = None
        self._error: str | None = None

    def _load(self) -> RecommendationService:
        if self._service is not None:
            return self._service
        if self._error is not None:
            raise RecommendationUnavailable(self._error)
        try:
            metadata = MemeMatchMetadata.from_csv(self.metadata_csv, self.image_root)
            retrieval = RetrievalService.load(self.index_dir)
            self._validate_catalog(retrieval, metadata)
            self._service = RecommendationService(retrieval, metadata)
            self._hybrid = HybridRetriever(retrieval, self._documents(retrieval, metadata))
        except (FileNotFoundError, ValueError, OSError) as exc:
            self._error = str(exc)
            raise RecommendationUnavailable(self._error) from exc
        return self._service

    @staticmethod
    def _documents(retrieval: RetrievalService, metadata: MemeMatchMetadata) -> dict[str, str]:
        documents = {}
        for record in retrieval.exported.records:
            meme_id = str(record["source_key"])
            context = record.get("context")
            if not isinstance(context, str) or not context.strip():
                entry = metadata.get(meme_id)
                context = "\n".join(value for value in (entry.local_context, entry.global_context) if value) if entry else ""
            documents[meme_id] = context
        return documents

    def _validate_catalog(self, retrieval: RetrievalService, metadata: MemeMatchMetadata) -> None:
        count = len(retrieval.exported.records)
        if self.expected_count is not None and count != self.expected_count:
            raise ValueError(f"{self.catalog_id} index must contain {self.expected_count} records, found {count}")
        if len(metadata.entries) != count:
            raise ValueError("metadata row count does not match retrieval record count")
        missing = [str(record["source_key"]) for record in retrieval.exported.records if metadata.get(str(record["source_key"])) is None]
        if missing:
            raise ValueError(f"metadata is missing retrieval source keys: {', '.join(missing[:3])}")
        dimensions = {name: index.d for name, index in retrieval.indexes.items()}
        if dimensions.get("fused") != dimensions.get("text", 0) + dimensions.get("image", 0):
            raise ValueError("fused index dimension is incompatible with text and image indexes")
        if self.expected_dimensions is not None and tuple(dimensions.get(name) for name in ("text", "image", "fused")) != self.expected_dimensions:
            raise ValueError(f"{self.catalog_id} index dimensions are incompatible")
        if self.export_records is not None:
            try:
                records = json.loads(self.export_records.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("invalid or missing exported records artifact") from exc
            if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
                raise ValueError("invalid or missing exported records artifact")
            completed = sorted((record for record in records if record.get("status") == "completed"), key=lambda record: record.get("embedding_index", -1))
            if [record.get("source_key") for record in completed] != [record["source_key"] for record in retrieval.exported.records]:
                raise ValueError("exported records do not align with retrieval index manifest")
        if self.export_metadata is not None:
            try:
                exported = json.loads(self.export_metadata.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("invalid or missing exported metadata artifact") from exc
            if not isinstance(exported, dict) or (exported.get("text_embedding_dimension"), exported.get("image_embedding_dimension")) != (dimensions["text"], dimensions["image"]):
                raise ValueError("exported metadata dimensions do not align with retrieval index")

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
        diversity_max_per_template: int | None, exclude_unsafe: bool, limit: int,
        retrieval_mode: RetrievalMode = "semantic", bm25_candidate_limit: int = 50,
        cross_encoder_limit: int = 20, mood: str | None = None, exclude_meme_ids: set[str] | None = None,
    ) -> list[Recommendation]:
        service = self._load()
        selected_mood = resolve_mood(mood)
        if text.strip():
            effective_query = mood_query(text, selected_mood)
        elif selected_mood is not None:
            effective_query = mood_only_query(selected_mood)
        else:
            raise ValueError("Provide a situation or select a mood")
        query = TextQueryEmbedder(service.retrieval, self.query_device).embed(effective_query)
        from app.recommendations import DiversityConfig
        diversity = DiversityConfig(diversity_max_per_template) if diversity_max_per_template else None
        if retrieval_mode not in {"semantic", "hybrid", "hybrid_cross_encoder"}:
            raise ValueError("unsupported retrieval mode")
        assert self._hybrid is not None
        retrieval = self._hybrid.search(retrieval_mode, effective_query, query, semantic_limit=candidate_limit, bm25_limit=bm25_candidate_limit)
        self.last_retrieval_latency_ms, self.last_cross_encoder_latency_ms = retrieval.retrieval_latency_ms, 0.0
        self.last_retrieval_warning = retrieval.warning
        self.last_retrieval_candidate_count = len(retrieval.candidates)
        safety = SafetyConfig(enabled=exclude_unsafe)
        ranking_emotion = emotion or primary_classifier_emotion(selected_mood)
        ranked = service.rank_candidates(retrieval.candidates, emotion=ranking_emotion, sentiment=sentiment, usage_preferences=usage, weights=weights, safety=safety, min_semantic_similarity=min_semantic_similarity, limit=max(candidate_limit, bm25_candidate_limit))
        if exclude_meme_ids:
            ranked = [item for item in ranked if item.meme_id not in exclude_meme_ids]
        self.last_safe_candidate_count = len(ranked)
        if retrieval_mode == "hybrid_cross_encoder":
            if self._cross_encoder is None:
                self._cross_encoder = OfflineCrossEncoder(self.cross_encoder_model)
            ordered_ids, self.last_cross_encoder_latency_ms, warning = cross_encoder_with_fallback(
                self._cross_encoder, effective_query, [item.meme_id for item in ranked], retrieval.documents, cross_encoder_limit
            )
            if warning is None:
                # Keep the existing semantic-first score order as one ranked
                # signal and fuse it with pairwise relevance ranks. This avoids
                # pretending logits and cosine scores are comparable while
                # retaining emotion/sentiment influence from Phase 6K.
                fused_ids = reciprocal_rank_fusion(
                    [item.meme_id for item in ranked],
                    ordered_ids,
                    limit=len(ranked),
                )
                positions = {meme_id: position for position, meme_id in enumerate(fused_ids)}
                ranked.sort(key=lambda item: positions[item.meme_id])
            else:
                # RRF ranking is retained as a safe offline fallback.
                warnings.warn(warning, RuntimeWarning, stacklevel=2)
                self.last_retrieval_warning = warning
        return service.apply_diversity(ranked, diversity, limit)

    def recommend_emotion(
        self, emotion: str, *, sentiment: str | None, usage: list[str], weights: RankingWeights,
        diversity_max_per_template: int | None, exclude_unsafe: bool, limit: int,
    ) -> list[Recommendation]:
        from app.recommendations import DiversityConfig
        diversity = DiversityConfig(diversity_max_per_template) if diversity_max_per_template else None
        return self._load().recommend_emotion(emotion, sentiment=sentiment, usage_preferences=usage, weights=weights, diversity=diversity, safety=SafetyConfig(enabled=exclude_unsafe), limit=limit)

    def image_path(self, meme_id: str) -> Path | None:
        entry = self._load().metadata.get(meme_id)
        if entry is None or entry.image_path is None:
            return None
        path = Path(entry.image_path)
        return path if path.is_file() else None

    def available_image_count(self) -> int:
        """Count image routes that can resolve without exposing filesystem paths."""
        service = self._load()
        return sum(self.image_path(meme_id) is not None for meme_id in service.metadata.entries)


@lru_cache
def get_recommendation_engine() -> RecommendationEngine:
    settings = get_settings()
    catalog = settings.active_catalog()
    return RecommendationEngine(
        catalog.index_dir,
        catalog.metadata_csv,
        catalog.image_root,
        catalog_id=catalog.catalog_id,
        expected_count=catalog.expected_count,
        expected_dimensions=catalog.expected_dimensions,
        export_records=catalog.export_records,
        export_metadata=catalog.export_metadata,
        cross_encoder_model=settings.recommendation_cross_encoder_model,
    )
