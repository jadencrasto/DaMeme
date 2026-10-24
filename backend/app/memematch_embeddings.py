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
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.emotion import PretrainedFeatureExtractor

EXPORT_FORMAT_VERSION = 1
TEXT_REPRESENTATION_VERSION = 2
EXPANSION_MANIFEST_VERSION = 1


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
    manifest: Path | None = None
    text_representation_version: int = TEXT_REPRESENTATION_VERSION


@dataclass(frozen=True)
class PreparedText:
    """Versioned, source-faithful text used for one retrieval embedding."""

    content: str
    fields: tuple[str, ...]
    source_columns: dict[str, str]


def _normalised_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def clean_source_text(value: str) -> str:
    """Normalize transport noise only; never correct, summarize, or infer text."""
    value = unicodedata.normalize("NFKC", value)
    value = "".join(" " if unicodedata.category(character).startswith("C") else character for character in value)
    return " ".join(value.split())


def _field(row: dict[str, str], *names: str) -> tuple[str, str] | None:
    by_header = {_normalised_header(key): (key, value) for key, value in row.items()}
    for name in names:
        source = by_header.get(name)
        if source and (value := clean_source_text(source[1])):
            return source[0], value
    return None


def _verified_template(row: dict[str, str]) -> tuple[str, str] | None:
    """Use template text only when the source explicitly marks it verified."""
    direct = _field(row, "verified_template", "verified_template_name")
    if direct:
        return direct
    verified = _field(row, "template_verified", "is_template_verified")
    if verified and verified[1].lower() in {"1", "true", "yes"}:
        return _field(row, "template_name", "template")
    return None


def stable_meme_id(row: dict[str, str]) -> str:
    """A source-stable identifier; row positions are intentionally never used."""
    identity = _field(row, "source_key", "filename", "id", "meme_id")
    if identity is None:
        raise ValueError("metadata row requires source_key, filename, id, or meme_id")
    value = identity[1].replace("\\", "/")
    if value.startswith("/") or "/../" in f"/{value}/" or value in {".", ".."}:
        raise ValueError("metadata identifier must be a safe relative source identifier")
    return value


def prepare_text(row: dict[str, str], version: int = TEXT_REPRESENTATION_VERSION) -> PreparedText:
    """Build deterministic retrieval text and declare every contributing field."""
    if version != TEXT_REPRESENTATION_VERSION:
        raise ValueError(f"unsupported text representation version: {version}")
    candidates = (
        ("cleaned_ocr", "Text in meme", _field(row, "cleaned_ocr", "ocr_text", "ocr", "local_ocr")),
        ("local_context", "Local meme context", _field(row, "local_context")),
        ("caption_or_title", "Caption or title", _field(row, "caption", "title", "meme_title")),
        ("verified_template", "Verified template", _verified_template(row)),
        # This is retained as provenance-rich visual context, not asserted meaning.
        ("global_context", "Global visual context (may be generic)", _field(row, "global_context")),
    )
    included = [(field, label, source) for field, label, source in candidates if source is not None]
    if not included:
        raise ValueError("metadata row has no usable text fields for embedding")
    return PreparedText(
        "\n\n".join(f"{label}: {source[1]}" for _, label, source in included),
        tuple(field for field, _, _ in included),
        {field: source[0] for field, _, source in included},
    )


def validate_metadata_row(
    row: dict[str, str], version: int = TEXT_REPRESENTATION_VERSION
) -> tuple[str, PreparedText]:
    """Validate the minimum source schema before any image/model work starts."""
    if not isinstance(row, dict) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in row.items()):
        raise ValueError("metadata rows must map string columns to string values")
    return stable_meme_id(row), prepare_text(row, version)


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
    """Backward-compatible access point for the current versioned representation."""
    return prepare_text(row).content


def _source_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_expansion_manifest(
    metadata_csv: Path, output: Path, *, target_count: int, source_url: str, license_status: str,
    access_notes: str = "",
) -> dict[str, object]:
    """Select a deterministic, identifier-deduplicated acquisition manifest only."""
    if target_count < 1:
        raise ValueError("target_count must be positive")
    with metadata_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("metadata CSV has no rows")
    selected, seen = [], set()
    for row_number, row in enumerate(rows):
        key = stable_meme_id(row)
        if key in seen:
            continue
        seen.add(key)
        selected.append({
            "source_key": key,
            "filename": row.get("filename", "").strip(),
            "source_path": row.get("kaggle_path", row.get("image_path", "")).strip(),
            "metadata_row": row_number,
        })
        if len(selected) == target_count:
            break
    manifest: dict[str, object] = {
        "schema_version": EXPANSION_MANIFEST_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "source_csv": str(metadata_csv),
        "source_sha256": _source_digest(metadata_csv),
        "source": {"url": source_url, "license_status": license_status, "access_notes": access_notes},
        "target_count": target_count,
        "selected_count": len(selected),
        "entries": selected,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def load_expansion_manifest(path: Path, metadata_csv: Path) -> list[dict[str, object]]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid expansion manifest") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != EXPANSION_MANIFEST_VERSION:
        raise ValueError("unsupported expansion manifest schema")
    if manifest.get("source_sha256") != _source_digest(metadata_csv):
        raise ValueError("expansion manifest does not match metadata CSV")
    source = manifest.get("source")
    entries = manifest.get("entries")
    if not isinstance(source, dict) or not source.get("url") or not source.get("license_status"):
        raise ValueError("expansion manifest requires source URL and license status")
    if not isinstance(entries, list):
        raise ValueError("expansion manifest entries must be a list")
    keys = [entry.get("source_key") for entry in entries if isinstance(entry, dict)]
    if len(keys) != len(entries) or any(not isinstance(key, str) or not key for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("expansion manifest source keys must be unique non-empty strings")
    return entries


def _manifest_rows(
    entries: list[dict[str, object]], rows: list[dict[str, str]], version: int
) -> list[tuple[int, dict[str, str], str, PreparedText]]:
    """Resolve manifest entries in manifest order and reject stale row references."""
    selected = []
    for entry in entries:
        row_number = entry.get("metadata_row")
        if not isinstance(row_number, int) or row_number < 0 or row_number >= len(rows):
            raise ValueError("expansion manifest metadata_row is invalid")
        row = rows[row_number]
        source_key, prepared = validate_metadata_row(row, version)
        if entry.get("source_key") != source_key or entry.get("filename") != row.get("filename", "").strip():
            raise ValueError("expansion manifest entry does not match metadata CSV row")
        source_path = row.get("kaggle_path", row.get("image_path", "")).strip()
        if entry.get("source_path", "") != source_path:
            raise ValueError("expansion manifest source path does not match metadata CSV row")
        selected.append((row_number, row, source_key, prepared))
    return selected


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

    def _load_checkpoint(self, input_digest: str) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
        embeddings_path, records_path, metadata_path = _paths(self.config.output_prefix)
        if not self.config.resume or not all(path.exists() for path in _paths(self.config.output_prefix)):
            return [], np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "input_sha256": input_digest,
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
        self, records: list[dict[str, object]], text: np.ndarray, image: np.ndarray, source_digest: str,
        input_digest: str,
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
            "input_sha256": input_digest,
            "dataset_root": str(self.config.dataset_root),
            "text_model": self.extractor.text_model_id,
            "image_model": self.extractor.image_model_id,
            "text_embedding_dimension": int(text.shape[1]) if text.ndim == 2 and len(text) else None,
            "image_embedding_dimension": int(image.shape[1]) if image.ndim == 2 and len(image) else None,
            "l2_normalized": True,
            "text_representation": {
                "version": self.config.text_representation_version,
                "field_order": ["cleaned_ocr", "local_context", "caption_or_title", "verified_template", "global_context"],
            },
            "completed_records": sum(record["status"] == "completed" for record in records),
            "failed_records": sum(record["status"] != "completed" for record in records),
            "configuration": asdict(self.config),
        }
        # Path objects are not JSON values.
        metadata["configuration"] = {key: str(value) if isinstance(value, Path) else value for key, value in metadata["configuration"].items()}
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    def export(self) -> dict[str, object]:
        source_digest = _source_digest(self.config.metadata_csv)
        manifest_entries = load_expansion_manifest(self.config.manifest, self.config.metadata_csv) if self.config.manifest else None
        input_digest = hashlib.sha256((source_digest + (_source_digest(self.config.manifest) if self.config.manifest else "")).encode()).hexdigest()
        records, text, image = self._load_checkpoint(input_digest)
        # Failed rows are retried on the next run; completed vectors are immutable.
        records = [record for record in records if record["status"] == "completed"]
        completed_keys = {str(record["source_key"]) for record in records}
        with self.config.metadata_csv.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError("metadata CSV has no rows")
        pending: list[tuple[dict[str, object], Path]] = []
        source_keys: set[str] = set()
        selected_rows = _manifest_rows(manifest_entries, rows, self.config.text_representation_version) if manifest_entries is not None else [
            (row_number, row, *validate_metadata_row(row, self.config.text_representation_version))
            for row_number, row in enumerate(rows)
        ]
        for row_number, row, source_key, prepared in selected_rows:
            filename = row.get("filename", "").strip()
            if source_key in source_keys:
                raise ValueError(f"metadata contains duplicate source key: {source_key}")
            source_keys.add(source_key)
            if source_key in completed_keys:
                continue
            record: dict[str, object] = {
                "source_key": source_key,
                "row_number": row_number,
                "filename": filename,
                "context": prepared.content,
                "text_representation_version": self.config.text_representation_version,
                "text_fields": list(prepared.fields),
                "text_source_columns": prepared.source_columns,
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
            self._write(records, text, image, source_digest, input_digest)
        self._write(records, text, image, source_digest, input_digest)
        return {
            "completed": sum(record["status"] == "completed" for record in records),
            "failed": sum(record["status"] != "completed" for record in records),
            "artifacts": [str(path) for path in _paths(self.config.output_prefix)],
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export MemeMatch MiniLM and BLIP embeddings")
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--manifest", type=Path, help="validated source-only expansion manifest")
    parser.add_argument("--prepare-manifest", type=Path, help="write a source-only expansion manifest and stop")
    parser.add_argument("--target-count", type=int, default=1000)
    parser.add_argument("--source-url", default="")
    parser.add_argument("--license-status", default="")
    parser.add_argument("--access-notes", default="")
    args = parser.parse_args(argv)
    if args.prepare_manifest:
        if not args.source_url or not args.license_status:
            parser.error("--prepare-manifest requires --source-url and --license-status")
        print(json.dumps(build_expansion_manifest(args.metadata_csv, args.prepare_manifest, target_count=args.target_count, source_url=args.source_url, license_status=args.license_status, access_notes=args.access_notes), indent=2))
        return 0
    if args.dataset_root is None or args.output_prefix is None:
        parser.error("embedding export requires --dataset-root and --output-prefix")
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
            manifest=args.manifest,
        )
    ).export()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
