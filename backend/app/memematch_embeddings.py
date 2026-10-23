"""Resumable, Colab-friendly export of MemeMatch retrieval embeddings.

This module deliberately shares Phase 4's frozen MiniLM/BLIP feature extractor.
It writes only portable NumPy and JSON artifacts; model weights and source images
remain outside the repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.emotion import PretrainedFeatureExtractor

EXPORT_FORMAT_VERSION = 1


class EmbeddingExtractor(Protocol):
    text_model_id: str
    image_model_id: str

    def embed_texts(self, contents: list[str]) -> np.ndarray: ...

    def embed_images(self, paths: list[Path]) -> np.ndarray: ...


@dataclass(frozen=True)
class ExportConfig:
    metadata_csv: Path
    dataset_root: Path
    output_prefix: Path
    batch_size: int = 16
    device: str = "cpu"
    resume: bool = True


def _paths(prefix: Path) -> tuple[Path, Path, Path]:
    return (
        prefix.with_suffix(".embeddings.npz"),
        prefix.with_suffix(".records.json"),
        prefix.with_suffix(".metadata.json"),
    )


def _safe_relative(root: Path, candidate: str) -> Path | None:
    """Return candidate only when it resolves below root."""
    path = Path(candidate)
    if path.is_absolute() or not candidate:
        return None
    resolved_root, resolved = root.resolve(), (root / path).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        return None
    return resolved


def resolve_image_path(row: dict[str, str], dataset_root: Path) -> Path | None:
    """Resolve local images without trusting Colab's absolute source paths."""
    root = dataset_root.resolve()
    filename = row.get("filename", "").strip()
    # A portable filename is preferred, including when image_path is a stale Drive path.
    if filename:
        for relative in (Path("images") / filename, Path(filename)):
            candidate = _safe_relative(root, str(relative))
            if candidate and candidate.is_file():
                return candidate
    relative = _safe_relative(root, row.get("image_path", "").strip())
    return relative if relative and relative.is_file() else None


def compose_context(row: dict[str, str]) -> str:
    """Stable retrieval text representation for local and global MemeMatch context."""
    parts = []
    if local := row.get("local_context", "").strip():
        parts.append(f"Local context: {local}")
    if global_context := row.get("global_context", "").strip():
        parts.append(f"Global context: {global_context}")
    return "\n\n".join(parts)


def _source_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_image(path: Path) -> str | None:
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        return str(exc)
    return None


class MemeMatchEmbeddingExporter:
    def __init__(self, config: ExportConfig, extractor: EmbeddingExtractor | None = None):
        if config.batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.config = config
        self.extractor = extractor or PretrainedFeatureExtractor(config.device)

    def _load_checkpoint(self, source_digest: str) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
        embeddings_path, records_path, metadata_path = _paths(self.config.output_prefix)
        if not self.config.resume or not all(path.exists() for path in _paths(self.config.output_prefix)):
            return [], np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "source_sha256": source_digest,
            "text_model": self.extractor.text_model_id,
            "image_model": self.extractor.image_model_id,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("existing export is incompatible with this source or encoder configuration")
        with np.load(embeddings_path) as data:
            text, image = data["text_embeddings"], data["image_embeddings"]
        records = json.loads(records_path.read_text(encoding="utf-8"))
        completed = sum(record["status"] == "completed" for record in records)
        if text.shape[0] != completed or image.shape[0] != completed:
            raise ValueError("export checkpoint has inconsistent record and embedding counts")
        return records, text, image

    def _write(
        self, records: list[dict[str, object]], text: np.ndarray, image: np.ndarray, source_digest: str
    ) -> None:
        embeddings_path, records_path, metadata_path = _paths(self.config.output_prefix)
        embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_embeddings = embeddings_path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp_embeddings, text_embeddings=text, image_embeddings=image)
        os.replace(tmp_embeddings, embeddings_path)
        records_path.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")
        metadata = {
            "format_version": EXPORT_FORMAT_VERSION,
            "created_at": datetime.now(UTC).isoformat(),
            "source_csv": str(self.config.metadata_csv),
            "source_sha256": source_digest,
            "dataset_root": str(self.config.dataset_root),
            "text_model": self.extractor.text_model_id,
            "image_model": self.extractor.image_model_id,
            "text_embedding_dimension": int(text.shape[1]) if text.ndim == 2 and len(text) else None,
            "image_embedding_dimension": int(image.shape[1]) if image.ndim == 2 and len(image) else None,
            "l2_normalized": True,
            "completed_records": sum(record["status"] == "completed" for record in records),
            "failed_records": sum(record["status"] != "completed" for record in records),
            "configuration": asdict(self.config),
        }
        # Path objects are not JSON values.
        metadata["configuration"] = {key: str(value) if isinstance(value, Path) else value for key, value in metadata["configuration"].items()}
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    def export(self) -> dict[str, object]:
        source_digest = _source_digest(self.config.metadata_csv)
        records, text, image = self._load_checkpoint(source_digest)
        # Failed rows are retried on the next run; completed vectors are immutable.
        records = [record for record in records if record["status"] == "completed"]
        completed_keys = {str(record["source_key"]) for record in records}
        with self.config.metadata_csv.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError("metadata CSV has no rows")
        pending: list[tuple[dict[str, object], Path]] = []
        source_keys: set[str] = set()
        for row_number, row in enumerate(rows):
            filename = row.get("filename", "").strip()
            source_key = filename or f"row-{row_number}"
            if source_key in source_keys:
                raise ValueError(f"metadata contains duplicate source key: {source_key}")
            source_keys.add(source_key)
            if source_key in completed_keys:
                continue
            record: dict[str, object] = {
                "source_key": source_key,
                "row_number": row_number,
                "filename": filename,
                "context": compose_context(row),
            }
            image_path = resolve_image_path(row, self.config.dataset_root)
            if not image_path:
                record.update(status="missing_image", error="image could not be resolved below dataset root")
                records.append(record)
                continue
            if error := _verify_image(image_path):
                record.update(status="invalid_image", error=error)
                records.append(record)
                continue
            record["image_path"] = str(image_path.relative_to(self.config.dataset_root.resolve()))
            pending.append((record, image_path))

        for start in range(0, len(pending), self.config.batch_size):
            batch = pending[start : start + self.config.batch_size]
            try:
                text_batch = self.extractor.embed_texts([str(record["context"]) for record, _ in batch])
                image_batch = self.extractor.embed_images([path for _, path in batch])
                if text_batch.shape[0] != len(batch) or image_batch.shape[0] != len(batch):
                    raise ValueError("extractor returned a batch with an unexpected row count")
                text = np.concatenate((text, text_batch.astype(np.float32)), axis=0) if len(text) else text_batch.astype(np.float32)
                image = np.concatenate((image, image_batch.astype(np.float32)), axis=0) if len(image) else image_batch.astype(np.float32)
                for offset, (record, _) in enumerate(batch):
                    record["status"] = "completed"
                    record["embedding_index"] = int(text.shape[0] - len(batch) + offset)
                    records.append(record)
            except Exception as exc:
                # A model failure is recorded per batch so a Colab run can resume and be diagnosed.
                for record, _ in batch:
                    record.update(status="embedding_failed", error=str(exc))
                    records.append(record)
            self._write(records, text, image, source_digest)
        self._write(records, text, image, source_digest)
        return {
            "completed": sum(record["status"] == "completed" for record in records),
            "failed": sum(record["status"] != "completed" for record in records),
            "artifacts": [str(path) for path in _paths(self.config.output_prefix)],
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export MemeMatch MiniLM and BLIP embeddings")
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)
    if args.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
    result = MemeMatchEmbeddingExporter(
        ExportConfig(
            metadata_csv=args.metadata_csv,
            dataset_root=args.dataset_root,
            output_prefix=args.output_prefix,
            batch_size=args.batch_size,
            device=args.device,
            resume=not args.no_resume,
        )
    ).export()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
