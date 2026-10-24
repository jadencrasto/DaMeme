"""Deterministic metadata-aware ranking over the local FAISS retrieval layer.

MemeMatch annotation probabilities are source metadata. They remain distinct from
DaMeme's Phase 4 classifier outputs and are not treated as calibrated labels.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.emotion import EMOTIONS, SENTIMENTS
from app.retrieval import RetrievalService, SearchResult

SOURCE_SCOPES = ("local", "global")


def _header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _probability(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if np.isfinite(result) and 0.0 <= result <= 1.0 else None


def _json_probabilities(row: dict[str, str], scope: str, label: str) -> list[float]:
    values = []
    for key, value in row.items():
        key = _header(key)
        if scope not in key or "emotion" not in key and "sentiment" not in key:
            continue
        try:
            decoded = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(decoded, dict):
            parsed = _probability(str(decoded.get(label, "")))
            if parsed is not None:
                values.append(parsed)
    return values


def _source_probabilities(row: dict[str, str], category: str, label: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for scope in SOURCE_SCOPES:
        candidates = []
        for key, value in row.items():
            normalized = _header(key)
            tokens = set(normalized.split("_"))
            if scope not in tokens or label not in tokens:
                continue
            # MemeMatch's unified export uses direct columns such as
            # ``local_sadness`` and ``global_positive``. Other supported
            # exports may include ``emotion``/``sentiment`` in the name.
            # The label sets do not overlap, so scope + requested label is
            # sufficient and avoids silently dropping the real annotations.
            parsed = _probability(value)
            if parsed is not None:
                candidates.append(parsed)
        candidates.extend(_json_probabilities(row, scope, label))
        if candidates:
            values[scope] = float(np.mean(candidates))
    return values


def _usage_categories(row: dict[str, str]) -> frozenset[str]:
    values = []
    for key, value in row.items():
        normalized = _header(key)
        if "usage" in normalized and value:
            values.extend(re.split(r"[|,;]", value))
    return frozenset(_header(value) for value in values if _header(value))


def _local_image_path(value: str | None, image_root: Path | None) -> str | None:
    """Resolve only portable image references below an explicitly supplied root."""
    if not value or image_root is None:
        return None
    root = image_root.resolve()
    supplied = Path(value)
    candidates = [root / supplied] if not supplied.is_absolute() else []
    candidates.extend((root / "images" / supplied.name, root / supplied.name))
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            resolved.relative_to(root)
        except ValueError:
            continue
        if resolved.is_file():
            return str(resolved)
    return None


@dataclass(frozen=True)
class MemeMetadata:
    meme_id: str
    emotion: dict[str, dict[str, float]]
    sentiment: dict[str, dict[str, float]]
    usage: frozenset[str]
    image_path: str | None = None

    def emotion_score(self, emotion: str) -> float | None:
        values = self.emotion.get(emotion, {})
        return float(np.mean(list(values.values()))) if values else None

    def sentiment_score(self, sentiment: str) -> float | None:
        values = self.sentiment.get(sentiment, {})
        return float(np.mean(list(values.values()))) if values else None


class MemeMatchMetadata:
    """Filename-keyed source annotations; never joins records by CSV position."""

    def __init__(self, entries: dict[str, MemeMetadata]):
        self.entries = entries

    @classmethod
    def from_csv(cls, path: Path, image_root: Path | None = None) -> MemeMatchMetadata:
        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except OSError as exc:
            raise FileNotFoundError(f"unable to read MemeMatch metadata: {path}") from exc
        if not rows:
            raise ValueError("MemeMatch metadata CSV has no rows")
        entries: dict[str, MemeMetadata] = {}
        for row in rows:
            # Retrieval records use source_key when it exists. Prefer it here
            # rather than relying on CSV column order when both are present.
            source_key = next(
                (value.strip() for key, value in row.items() if _header(key) == "source_key" and value),
                "",
            )
            filename = source_key or next(
                (value.strip() for key, value in row.items() if _header(key) == "filename" and value),
                "",
            )
            if not filename:
                raise ValueError("MemeMatch metadata row is missing source_key or filename")
            if filename in entries:
                raise ValueError(f"duplicate MemeMatch source key: {filename}")
            emotion = {label: _source_probabilities(row, "emotion", label) for label in EMOTIONS}
            sentiment = {label: _source_probabilities(row, "sentiment", label) for label in SENTIMENTS}
            image_path = _local_image_path(
                next((value for key, value in row.items() if _header(key) == "image_path" and value), None),
                image_root,
            )
            entries[filename] = MemeMetadata(filename, emotion, sentiment, _usage_categories(row), image_path)
        return cls(entries)

    def get(self, meme_id: str) -> MemeMetadata | None:
        return self.entries.get(meme_id)


@dataclass(frozen=True)
class RankingWeights:
    semantic: float = 1.0
    emotion: float = 1.0
    sentiment: float = 1.0

    def __post_init__(self) -> None:
        if any(value < 0 for value in (self.semantic, self.emotion, self.sentiment)):
            raise ValueError("ranking weights must be non-negative")
        if self.semantic + self.emotion + self.sentiment <= 0:
            raise ValueError("at least one ranking weight must be positive")

    def normalised(self, include_semantic: bool, include_emotion: bool, include_sentiment: bool) -> dict[str, float]:
        raw = {
            "semantic": self.semantic if include_semantic else 0.0,
            "emotion": self.emotion if include_emotion else 0.0,
            "sentiment": self.sentiment if include_sentiment else 0.0,
        }
        total = sum(raw.values())
        if total <= 0:
            return raw
        return {key: value / total for key, value in raw.items()}


def filename_template_key(meme_id: str) -> str:
    """Conservative optional filename heuristic, not a definitive template identity."""
    stem = Path(meme_id).stem.lower()
    return re.sub(r"[_-]\d+$", "", stem)


@dataclass(frozen=True)
class DiversityConfig:
    max_per_template: int = 1
    template_key: Callable[[str], str] = filename_template_key

    def __post_init__(self) -> None:
        if self.max_per_template < 1:
            raise ValueError("max_per_template must be positive")


@dataclass(frozen=True)
class Recommendation:
    meme_id: str
    semantic_similarity: float | None
    emotion_relevance: float | None
    sentiment_relevance: float | None
    final_score: float
    usage_categories: tuple[str, ...]
    explanation: str


class RecommendationService:
    def __init__(self, retrieval: RetrievalService, metadata: MemeMatchMetadata):
        self.retrieval, self.metadata = retrieval, metadata

    @staticmethod
    def _validate(emotion: str | None, sentiment: str | None) -> None:
        if emotion is not None and emotion not in EMOTIONS:
            raise ValueError(f"emotion must be one of: {', '.join(EMOTIONS)}")
        if sentiment is not None and sentiment not in SENTIMENTS:
            raise ValueError(f"sentiment must be one of: {', '.join(SENTIMENTS)}")

    @staticmethod
    def _explanation(
        semantic: float | None, emotion: str | None, emotion_score: float | None,
        sentiment: str | None, sentiment_score: float | None, usage: frozenset[str], preferences: frozenset[str],
    ) -> str:
        parts = []
        if semantic is not None:
            parts.append(f"text similarity {semantic:.3f}")
        if emotion is not None and emotion_score is not None:
            parts.append(f"source {emotion} probability {emotion_score:.3f}")
        if sentiment is not None and sentiment_score is not None:
            parts.append(f"source {sentiment} probability {sentiment_score:.3f}")
        if matched := sorted(usage & preferences):
            parts.append(f"matched usage: {', '.join(matched)}")
        return "; ".join(parts) or "included because source metadata is unavailable for optional ranking components"

    def _rank(
        self,
        candidates: Iterable[SearchResult | str],
        *,
        emotion: str | None,
        sentiment: str | None,
        usage_preferences: Iterable[str],
        weights: RankingWeights,
        diversity: DiversityConfig | None,
        min_semantic_similarity: float | None,
        limit: int,
    ) -> list[Recommendation]:
        if limit < 1:
            raise ValueError("recommendation limit must be positive")
        self._validate(emotion, sentiment)
        if min_semantic_similarity is not None and not -1.0 <= min_semantic_similarity <= 1.0:
            raise ValueError("min_semantic_similarity must be between -1 and 1")
        preferences = frozenset(_header(value) for value in usage_preferences if _header(value))
        ranked = []
        for candidate in candidates:
            meme_id = candidate.meme_id if isinstance(candidate, SearchResult) else candidate
            semantic = candidate.score if isinstance(candidate, SearchResult) else None
            if semantic is not None and min_semantic_similarity is not None and semantic < min_semantic_similarity:
                continue
            metadata = self.metadata.get(meme_id)
            if preferences and (metadata is None or not metadata.usage.intersection(preferences)):
                continue
            emotion_score = metadata.emotion_score(emotion) if metadata and emotion else None
            sentiment_score = metadata.sentiment_score(sentiment) if metadata and sentiment else None
            # Normalize once by request intent, not candidate metadata coverage.
            # Otherwise an item without emotion metadata is silently promoted by
            # receiving a semantic-only weight of 1.0 while its peers are blended.
            active = weights.normalised(semantic is not None, emotion is not None, sentiment is not None)
            semantic_relevance = (semantic + 1.0) / 2.0 if semantic is not None else 0.0
            score = active["semantic"] * semantic_relevance + active["emotion"] * (emotion_score or 0.0) + active["sentiment"] * (sentiment_score or 0.0)
            usage = metadata.usage if metadata else frozenset()
            ranked.append(
                Recommendation(
                    meme_id, semantic, emotion_score, sentiment_score, score, tuple(sorted(usage)),
                    self._explanation(semantic, emotion, emotion_score, sentiment, sentiment_score, usage, preferences),
                )
            )
        ranked.sort(key=lambda item: (-item.final_score, item.meme_id))
        if diversity is None:
            return ranked[:limit]
        chosen, template_counts = [], {}
        for item in ranked:
            key = diversity.template_key(item.meme_id)
            if template_counts.get(key, 0) >= diversity.max_per_template:
                continue
            template_counts[key] = template_counts.get(key, 0) + 1
            chosen.append(item)
            if len(chosen) == limit:
                break
        return chosen

    def recommend_text(
        self, query: np.ndarray, *, emotion: str | None = None, sentiment: str | None = None,
        usage_preferences: Iterable[str] = (), weights: RankingWeights | None = None,
        diversity: DiversityConfig | None = None, candidate_limit: int = 100,
        min_semantic_similarity: float | None = None, limit: int = 10,
    ) -> list[Recommendation]:
        candidates = self.retrieval.search("text", query, candidate_limit)
        return self._rank(candidates, emotion=emotion, sentiment=sentiment, usage_preferences=usage_preferences, weights=weights or RankingWeights(semantic=0.8, emotion=0.15, sentiment=0.05), diversity=diversity, min_semantic_similarity=min_semantic_similarity, limit=limit)

    def recommend_emotion(
        self, emotion: str, *, sentiment: str | None = None, usage_preferences: Iterable[str] = (),
        weights: RankingWeights | None = None, diversity: DiversityConfig | None = None, limit: int = 10,
    ) -> list[Recommendation]:
        candidates = sorted(self.metadata.entries)
        return self._rank(candidates, emotion=emotion, sentiment=sentiment, usage_preferences=usage_preferences, weights=weights or RankingWeights(semantic=0.0), diversity=diversity, min_semantic_similarity=None, limit=limit)
