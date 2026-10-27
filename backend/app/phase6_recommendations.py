"""Cached, portable serving adapter for the Phase 5 recommendation bundle."""

from __future__ import annotations

import pickle
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.emotion import EMOTIONS, PretrainedFeatureExtractor
from app.expanded_emotion import TEXT_DIMENSION
from app.phase5_retrieval import Phase5Bundle, RankedMeme


class Phase5Unavailable(RuntimeError):
    pass


class Phase6RecommendationService:
    def __init__(self, bundle_dir: Path, text_model: Path, media_root: Path | None) -> None:
        self.bundle_dir, self.text_model, self.media_root = bundle_dir, text_model, media_root
        self._bundle: Phase5Bundle | None = None
        self._payload: dict[object, object] | None = None
        self._extractor: PretrainedFeatureExtractor | None = None

    def _load_bundle(self) -> Phase5Bundle:
        if self._bundle is None:
            self._bundle = Phase5Bundle.load(self.bundle_dir)
        return self._bundle

    def _load_text_model(self) -> dict[object, object]:
        if self._payload is None:
            if not self.text_model.is_file():
                raise Phase5Unavailable("Phase 5 text model artifact is unavailable")
            try:
                payload = pickle.loads(self.text_model.read_bytes())
            except (OSError, pickle.UnpicklingError) as exc:
                raise Phase5Unavailable("Phase 5 text model artifact is unreadable") from exc
            if payload.get("model") != "text" or payload.get("label_order") != list(EMOTIONS):
                raise Phase5Unavailable("Phase 5 text model artifact is incompatible")
            self._payload = payload
        return self._payload

    def _embedder(self) -> PretrainedFeatureExtractor:
        if self._extractor is None:
            self._extractor = PretrainedFeatureExtractor("cpu")
        return self._extractor

    def readiness(self) -> dict[str, object]:
        try:
            bundle = self._load_bundle()
            self._load_text_model()
        except (FileNotFoundError, ValueError, Phase5Unavailable) as exc:
            return {
                "ready": False,
                "detail": str(exc),
                "bundle_count": 0,
                "text_model_ready": False,
                "media_available": False,
            }
        refs = [reference for reference in bundle.image_refs if reference]
        return {
            "ready": True,
            "detail": None,
            "bundle_count": len(bundle.ids),
            "text_model_ready": True,
            "media_available": self.media_root is not None and self.media_root.is_dir(),
            "image_reference_count": len(refs),
        }

    def _predict(self, embedding: np.ndarray) -> dict[str, object]:
        payload = self._load_text_model()
        if embedding.shape != (TEXT_DIMENSION,) or not np.isfinite(embedding).all():
            raise Phase5Unavailable("MiniLM produced an invalid text embedding")
        transformed = payload["scaler"].transform(embedding[None])
        probabilities = np.asarray(
            [model.predict_proba(transformed)[0, 1] for model in payload["emotion"].estimators_],
            dtype=np.float32,
        )
        thresholds = np.asarray(payload.get("thresholds"), dtype=np.float32)
        if thresholds.shape != (len(EMOTIONS),):
            raise Phase5Unavailable("Phase 5 text model has invalid thresholds")
        values = dict(zip(EMOTIONS, probabilities.astype(float).tolist(), strict=True))
        return {
            "model": "text",
            "emotion_probabilities": values,
            "emotion_labels": [
                label
                for label, value, threshold in zip(EMOTIONS, probabilities, thresholds, strict=True)
                if value >= threshold
            ],
            "thresholds": dict(zip(EMOTIONS, thresholds.astype(float).tolist(), strict=True)),
        }

    def recommend(self, text: str, top_k: int) -> tuple[dict[str, object], list[RankedMeme]]:
        if not text.strip():
            raise ValueError("text must not be empty")
        bundle = self._load_bundle()
        try:
            embedding = self._embedder().text(text)
        except Exception as exc:  # local-only model cache failures are surfaced as readiness errors
            raise Phase5Unavailable("MiniLM is not available in the local model cache") from exc
        prediction = self._predict(embedding)
        scores = np.asarray(
            [prediction["emotion_probabilities"][label] for label in EMOTIONS], dtype=np.float32
        )
        return prediction, bundle.rank(
            embedding, scores, top_k=top_k, semantic_weight=0.8, emotion_weight=0.2
        )

    def image_path(self, meme_id: str) -> Path | None:
        bundle = self._load_bundle()
        try:
            index = bundle.ids.index(meme_id)
        except ValueError:
            return None
        reference = bundle.image_refs[index]
        if not reference or self.media_root is None or not self.media_root.is_dir():
            return None
        # Avoid ambiguous filename collisions: a reference may resolve only once.
        if Counter(bundle.image_refs)[reference] != 1:
            return None
        root = self.media_root.resolve()
        for candidate in (root / reference, root / "images" / reference):
            try:
                resolved = candidate.resolve()
                resolved.relative_to(root)
            except ValueError:
                continue
            if resolved.is_file():
                return resolved
        return None


@lru_cache
def get_phase6_service(
    bundle_dir: str, text_model: str, media_root: str | None
) -> Phase6RecommendationService:
    return Phase6RecommendationService(
        Path(bundle_dir), Path(text_model), Path(media_root) if media_root else None
    )
