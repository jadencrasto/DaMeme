"""Portable Phase 5 semantic retrieval and emotion-aware ranking.

The bundle is deliberately independent of the legacy catalog: its ordered IDs
are stored alongside the vectors and gated scores, so FAISS positions can never
be joined to CSV row positions implicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.emotion import EMOTIONS
from app.emotion_fusion import GatedEmotionFusion
from app.expanded_emotion import (
    FEATURE_VERSION,
    IMAGE_DIMENSION,
    TEXT_DIMENSION,
    _column,
    _identifier,
    load_expansion_artifacts,
)

try:
    import faiss  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised on minimal local installs
    faiss = None


@dataclass(frozen=True)
class RankedMeme:
    meme_id: str
    image_ref: str | None
    semantic_similarity: float
    emotion_compatibility: float
    score: float


class Phase5Bundle:
    """Exact IP retrieval with a deterministic NumPy fallback."""

    def __init__(
        self,
        ids: tuple[str, ...],
        text: np.ndarray,
        emotions: np.ndarray,
        image_refs: tuple[str | None, ...],
    ):
        if len(ids) != len(set(ids)) or not ids:
            raise ValueError("bundle meme IDs must be non-empty and unique")
        self.ids, self.text, self.emotions, self.image_refs = (
            ids,
            np.asarray(text, dtype=np.float32),
            np.asarray(emotions, dtype=np.float32),
            image_refs,
        )
        if self.text.shape != (len(ids), TEXT_DIMENSION) or self.emotions.shape != (
            len(ids),
            len(EMOTIONS),
        ):
            raise ValueError("bundle feature dimensions are incompatible")
        if (
            len(image_refs) != len(ids)
            or not np.isfinite(self.text).all()
            or not np.isfinite(self.emotions).all()
        ):
            raise ValueError("bundle contains invalid values")
        if np.any(np.abs(np.linalg.norm(self.text, axis=1) - 1) > 1e-4) or np.any(
            (self.emotions < 0) | (self.emotions > 1)
        ):
            raise ValueError("bundle vectors or emotion scores are out of range")
        self._index = None
        if faiss is not None:
            self._index = faiss.IndexFlatIP(TEXT_DIMENSION)
            self._index.add(self.text)

    def candidates(self, query: np.ndarray, limit: int) -> list[tuple[int, float]]:
        if limit < 1:
            raise ValueError("top_k must be positive")
        query = np.asarray(query, dtype=np.float32)
        if query.shape != (TEXT_DIMENSION,) or not np.isfinite(query).all():
            raise ValueError("query must be a finite 384-dimensional vector")
        norm = np.linalg.norm(query)
        if norm <= 1e-12:
            raise ValueError("query must not be zero")
        normalized = query / norm
        if self._index is not None:
            scores, indexes = self._index.search(normalized[None], min(limit, len(self.ids)))
            rows = [
                (int(index), float(score))
                for index, score in zip(indexes[0], scores[0], strict=True)
                if index >= 0
            ]
        else:
            rows = list(enumerate((self.text @ normalized).astype(float).tolist()))
        return sorted(rows, key=lambda row: (-row[1], self.ids[row[0]]))[:limit]

    def rank(
        self,
        query: np.ndarray,
        user_emotions: np.ndarray,
        *,
        top_k: int = 10,
        candidate_k: int = 100,
        semantic_weight: float = 0.8,
        emotion_weight: float = 0.2,
    ) -> list[RankedMeme]:
        if semantic_weight < 0 or emotion_weight < 0 or semantic_weight + emotion_weight <= 0:
            raise ValueError("ranking weights must be non-negative with a positive total")
        user = np.asarray(user_emotions, dtype=np.float32)
        if (
            user.shape != (len(EMOTIONS),)
            or not np.isfinite(user).all()
            or np.any((user < 0) | (user > 1))
        ):
            raise ValueError("user emotion scores must be finite probabilities in class order")
        user_norm = max(float(np.linalg.norm(user)), 1e-12)
        rows = []
        total = semantic_weight + emotion_weight
        for index, similarity in self.candidates(query, candidate_k):
            meme = self.emotions[index]
            compatibility = float(
                np.dot(user, meme) / (user_norm * max(float(np.linalg.norm(meme)), 1e-12))
            )
            semantic = (similarity + 1.0) / 2.0
            score = (semantic_weight * semantic + emotion_weight * compatibility) / total
            rows.append(
                RankedMeme(
                    self.ids[index],
                    self.image_refs[index],
                    float(similarity),
                    compatibility,
                    float(score),
                )
            )
        return sorted(rows, key=lambda row: (-row.score, -row.semantic_similarity, row.meme_id))[
            :top_k
        ]

    @classmethod
    def load(cls, directory: Path) -> Phase5Bundle:
        npz, metadata = directory / "phase5_bundle.npz", directory / "phase5_metadata.json"
        if not npz.is_file() or not metadata.is_file():
            raise FileNotFoundError("Phase 5 bundle artifacts are missing")
        try:
            info = json.loads(metadata.read_text(encoding="utf-8"))
            with np.load(npz) as values:
                ids = tuple(str(value) for value in values["meme_ids"].tolist())
                text = np.asarray(values["text_embeddings"], dtype=np.float32)
                emotions = np.asarray(values["emotion_scores"], dtype=np.float32)
                references = tuple(value or None for value in values["image_refs"].tolist())
        except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Phase 5 bundle is malformed") from exc
        if info.get("feature_version") != FEATURE_VERSION or info.get("emotion_order") != list(
            EMOTIONS
        ):
            raise ValueError("Phase 5 bundle metadata is incompatible")
        return cls(ids, text, emotions, references)


def _image_ref(row: dict[str, str]) -> str | None:
    """Retain a display filename only; never publish a Drive or host path."""
    value = _column(row, "filename", "image_filename", "image_path") or ""
    name = value.replace("\\", "/").rsplit("/", 1)[-1]
    return name if name and name not in {".", ".."} else None


def build_bundle(
    base: Path, checkpoint: Path, output: Path, batch_size: int = 128
) -> dict[str, object]:
    """Run the saved gated state over prepared vectors; never retrains or re-embeds."""
    import torch

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    data = load_expansion_artifacts(base)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"gated checkpoint is missing: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("feature_version") != FEATURE_VERSION or payload.get("label_order") != list(
        EMOTIONS
    ):
        raise ValueError("gated checkpoint has incompatible feature metadata")
    thresholds = np.asarray(payload.get("thresholds"), dtype=np.float32)
    if thresholds.shape != (len(EMOTIONS),) or not np.isfinite(thresholds).all():
        raise ValueError("gated checkpoint has invalid thresholds")
    state = payload.get("best_state") or payload.get("state")
    if not isinstance(state, dict):
        raise ValueError("gated checkpoint has no model state")
    model = GatedEmotionFusion(TEXT_DIMENSION, IMAGE_DIMENSION)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError("gated checkpoint architecture is incompatible") from exc
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(data.meme_ids), batch_size):
            logits, _ = model(
                torch.from_numpy(data.text[start : start + batch_size]),
                torch.from_numpy(data.image[start : start + batch_size]),
            )
            outputs.append(torch.sigmoid(logits).numpy())
    scores = np.concatenate(outputs).astype(np.float32)
    # Only portable basenames are stored. The serving host resolves any image route itself.
    with (base / "dameme_5500_training_manifest.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        manifest = {_identifier(row): row for row in csv.DictReader(handle)}
    refs = tuple(_image_ref(manifest[meme_id]) for meme_id in data.meme_ids)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "phase5_bundle.npz",
        meme_ids=np.asarray(data.meme_ids),
        text_embeddings=data.text,
        emotion_scores=scores,
        image_refs=np.asarray([value or "" for value in refs]),
    )
    metadata = {
        "format": 1,
        "feature_version": FEATURE_VERSION,
        "emotion_order": list(EMOTIONS),
        "thresholds": thresholds.tolist(),
        "count": len(data.meme_ids),
        "text_dimension": TEXT_DIMENSION,
        "seed": 42,
    }
    (output / "phase5_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build DaMeme Phase 5 portable retrieval bundle")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(argv)
    print(
        json.dumps(build_bundle(args.base, args.checkpoint, args.output, args.batch_size), indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
