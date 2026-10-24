"""Local, exact FAISS retrieval over versioned MemeMatch embedding exports."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import faiss
import numpy as np

from app.emotion import PretrainedFeatureExtractor

RETRIEVAL_FORMAT_VERSION = 1
NormalizationTolerance = 1e-4
Modality = Literal["text", "image", "fused"]


@dataclass(frozen=True)
class RetrievalConfig:
    text_weight: float = 1.0
    image_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.text_weight < 0 or self.image_weight < 0:
            raise ValueError("fusion weights must be non-negative")
        if self.text_weight == 0 and self.image_weight == 0:
            raise ValueError("at least one fusion weight must be positive")


@dataclass(frozen=True)
class ExportedEmbeddings:
    text: np.ndarray
    image: np.ndarray
    records: list[dict[str, object]]
    metadata: dict[str, object]


@dataclass(frozen=True)
class SearchResult:
    meme_id: str
    position: int
    score: float


def _artifact_paths(prefix: Path) -> tuple[Path, Path, Path]:
    return (
        prefix.with_suffix(".embeddings.npz"),
        prefix.with_suffix(".records.json"),
        prefix.with_suffix(".metadata.json"),
    )


def _normalise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not len(values):
        raise ValueError("embedding matrix must be a non-empty two-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("embedding matrix contains non-finite values")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("embedding matrix contains zero vectors")
    return values / norms


def _validate_normalised(values: np.ndarray, label: str) -> np.ndarray:
    normalised = _normalise(values)
    if not np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=NormalizationTolerance):
        raise ValueError(f"{label} embeddings must be L2-normalized")
    return normalised


def _stable_records(records: object, count: int) -> list[dict[str, object]]:
    if not isinstance(records, list):
        raise ValueError("records artifact must be a JSON list")
    completed = [record for record in records if isinstance(record, dict) and record.get("status") == "completed"]
    if len(completed) != count:
        raise ValueError("completed record count does not match embedding count")
    try:
        ordered = sorted(completed, key=lambda record: int(record["embedding_index"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("completed records require integer embedding_index values") from exc
    if [int(record["embedding_index"]) for record in ordered] != list(range(count)):
        raise ValueError("record embedding indexes must be contiguous and start at zero")
    identifiers = [record.get("source_key") for record in ordered]
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
        raise ValueError("completed records require non-empty source_key identifiers")
    if len(set(identifiers)) != count:
        raise ValueError("record source_key identifiers must be unique")
    return ordered


def load_export(prefix: Path) -> ExportedEmbeddings:
    """Load and validate the three portable artifacts written by the Phase 5 exporter."""
    embeddings_path, records_path, metadata_path = _artifact_paths(prefix)
    missing = [str(path) for path in (embeddings_path, records_path, metadata_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing retrieval export artifacts: {', '.join(missing)}")
    try:
        with np.load(embeddings_path) as payload:
            text = np.asarray(payload["text_embeddings"], dtype=np.float32)
            image = np.asarray(payload["image_embeddings"], dtype=np.float32)
    except (KeyError, OSError, ValueError) as exc:
        raise ValueError("invalid embeddings NPZ artifact") from exc
    try:
        records = json.loads(records_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid retrieval provenance JSON") from exc
    if not isinstance(metadata, dict):
        raise ValueError("metadata artifact must be a JSON object")
    if text.shape[0] != image.shape[0]:
        raise ValueError("text and image embedding counts differ")
    text = _validate_normalised(text, "text")
    image = _validate_normalised(image, "image")
    if metadata.get("text_embedding_dimension") != text.shape[1]:
        raise ValueError("metadata text embedding dimension is incompatible")
    if metadata.get("image_embedding_dimension") != image.shape[1]:
        raise ValueError("metadata image embedding dimension is incompatible")
    if not isinstance(metadata.get("text_model"), str) or not isinstance(metadata.get("image_model"), str):
        raise ValueError("metadata model identifiers are required")
    return ExportedEmbeddings(text, image, _stable_records(records, len(text)), metadata)


def fuse_embeddings(text: np.ndarray, image: np.ndarray, config: RetrievalConfig) -> np.ndarray:
    """Build a cosine-compatible fused representation without mixing dimensions."""
    text, image = _normalise(text), _normalise(image)
    if len(text) != len(image):
        raise ValueError("text and image vectors must have matching row counts")
    values = np.concatenate((config.text_weight * text, config.image_weight * image), axis=1)
    return _normalise(values)


class RetrievalService:
    """Exact cosine retrieval with stable source-key mappings and safe query validation."""

    def __init__(self, exported: ExportedEmbeddings, config: RetrievalConfig | None = None):
        self.exported, self.config = exported, config or RetrievalConfig()
        self.vectors = {
            "text": exported.text,
            "image": exported.image,
            "fused": fuse_embeddings(exported.text, exported.image, self.config),
        }
        self.indexes: dict[Modality, faiss.IndexFlatIP] = {
            modality: self._index(values) for modality, values in self.vectors.items()
        }

    @staticmethod
    def _index(vectors: np.ndarray) -> faiss.IndexFlatIP:
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        return index

    @property
    def text_model_id(self) -> str:
        return str(self.exported.metadata["text_model"])

    @property
    def image_model_id(self) -> str:
        return str(self.exported.metadata["image_model"])

    def search(self, modality: Modality, query: np.ndarray, limit: int = 10) -> list[SearchResult]:
        if limit < 1:
            raise ValueError("search limit must be positive")
        index = self.indexes.get(modality)
        if index is None or index.ntotal == 0:
            return []
        query = np.asarray(query, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != index.d:
            raise ValueError(f"{modality} query must be a {index.d}-dimensional vector")
        if not np.isfinite(query).all():
            raise ValueError("query contains non-finite values")
        norm = float(np.linalg.norm(query))
        if norm <= 1e-12:
            raise ValueError("query must not be a zero vector")
        # Retrieve all rows before sorting so ties are deterministic across FAISS builds.
        scores, positions = index.search((query / norm)[None], index.ntotal)
        candidates = [
            SearchResult(str(self.exported.records[int(position)]["source_key"]), int(position), float(score))
            for score, position in zip(scores[0], positions[0], strict=True)
            if position >= 0
        ]
        return sorted(candidates, key=lambda item: (-item.score, item.meme_id))[:limit]

    def validate_query_model(self, modality: Modality, model_id: str) -> None:
        expected = self.text_model_id if modality == "text" else self.image_model_id
        if modality == "fused":
            raise ValueError("fused queries require both text and image model compatibility checks")
        if model_id != expected:
            raise ValueError(f"{modality} query encoder {model_id!r} is incompatible with {expected!r}")

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for modality, index in self.indexes.items():
            faiss.write_index(index, str(directory / f"{modality}.faiss"))
        manifest = {
            "format_version": RETRIEVAL_FORMAT_VERSION,
            "config": asdict(self.config),
            "metadata": self.exported.metadata,
            "records": self.exported.records,
            "dimensions": {modality: index.d for modality, index in self.indexes.items()},
            "counts": {modality: index.ntotal for modality, index in self.indexes.items()},
        }
        (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> RetrievalService:
        manifest_path = directory / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("invalid or missing retrieval index manifest") from exc
        if manifest.get("format_version") != RETRIEVAL_FORMAT_VERSION:
            raise ValueError("unsupported retrieval index format")
        metadata, records = manifest.get("metadata"), manifest.get("records")
        dimensions, counts = manifest.get("dimensions"), manifest.get("counts")
        if not isinstance(metadata, dict) or not isinstance(dimensions, dict) or not isinstance(counts, dict):
            raise ValueError("retrieval manifest is incomplete")
        count = int(counts.get("text", -1))
        stable_records = _stable_records(records, count)
        config = RetrievalConfig(**manifest.get("config", {}))
        indexes: dict[Modality, faiss.IndexFlatIP] = {}
        for modality in ("text", "image", "fused"):
            path = directory / f"{modality}.faiss"
            if not path.is_file():
                raise FileNotFoundError(f"missing {modality} FAISS index")
            index = faiss.read_index(str(path))
            if not isinstance(index, faiss.IndexFlatIP) or index.d != dimensions.get(modality):
                raise ValueError(f"invalid {modality} FAISS index")
            if index.ntotal != count or counts.get(modality) != count:
                raise ValueError(f"{modality} index count is incompatible with the record mapping")
            indexes[modality] = index
        instance = cls.__new__(cls)
        instance.exported = ExportedEmbeddings(
            np.empty((count, int(dimensions["text"])), dtype=np.float32),
            np.empty((count, int(dimensions["image"])), dtype=np.float32),
            stable_records,
            metadata,
        )
        instance.config, instance.vectors, instance.indexes = config, {}, indexes
        return instance


class TextQueryEmbedder:
    """Lazy MiniLM query encoder, constrained to the model recorded by an index."""

    def __init__(self, service: RetrievalService, device: str = "cpu") -> None:
        self.service = service
        self.extractor = PretrainedFeatureExtractor(device)

    def embed(self, content: str) -> np.ndarray:
        self.service.validate_query_model("text", self.extractor.text_model_id)
        return self.extractor.embed_texts([content])[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or query DaMeme FAISS retrieval indexes")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--export-prefix", type=Path, required=True)
    build.add_argument("--index-dir", type=Path, required=True)
    build.add_argument("--text-weight", type=float, default=1.0)
    build.add_argument("--image-weight", type=float, default=1.0)
    search = commands.add_parser("search-text")
    search.add_argument("--index-dir", type=Path, required=True)
    search.add_argument("--text", required=True)
    search.add_argument("--limit", type=int, default=10)
    search.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    if args.command == "build":
        service = RetrievalService(load_export(args.export_prefix), RetrievalConfig(args.text_weight, args.image_weight))
        service.save(args.index_dir)
        print(json.dumps({"dimensions": {key: index.d for key, index in service.indexes.items()}, "count": len(service.exported.records)}, indent=2))
        return 0
    service = RetrievalService.load(args.index_dir)
    query = TextQueryEmbedder(service, args.device).embed(args.text)
    print(json.dumps([asdict(result) for result in service.search("text", query, args.limit)], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
