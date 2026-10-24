import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.memematch_embeddings import (
    ExportConfig,
    MemeMatchEmbeddingExporter,
    build_expansion_manifest,
    clean_source_text,
    load_expansion_manifest,
    prepare_text,
    resolve_image_path,
    stable_meme_id,
    validate_metadata_row,
)


class FakeExtractor:
    text_model_id = "test-minilm"
    image_model_id = "test-blip"

    def __init__(self) -> None:
        self.text_calls = 0

    def embed_texts(self, values: list[str]) -> np.ndarray:
        self.text_calls += 1
        return np.array([[len(value), 1.0] for value in values], dtype=np.float32)

    def embed_images(self, paths: list[Path]) -> np.ndarray:
        return np.array([[float(index), 2.0, 3.0] for index, _ in enumerate(paths)], dtype=np.float32)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "image_path", "local_context", "global_context"])
        writer.writeheader()
        writer.writerows(rows)


def test_resolve_uses_portable_filename_and_rejects_path_escape(tmp_path: Path) -> None:
    image = tmp_path / "images" / "meme.png"
    image.parent.mkdir()
    image.write_bytes(b"x")
    assert resolve_image_path({"filename": "meme.png", "image_path": "/content/drive/stale.png"}, tmp_path) == image
    assert resolve_image_path({"filename": "../outside.png", "image_path": "../outside.png"}, tmp_path) is None


def test_export_writes_embeddings_metadata_and_resumes(tmp_path: Path) -> None:
    image = tmp_path / "images" / "one.png"
    image.parent.mkdir()
    Image.new("RGB", (4, 4), "red").save(image)
    csv_path = tmp_path / "metadata.csv"
    _write_csv(
        csv_path,
        [
            {"filename": "one.png", "image_path": "/content/drive/MyDrive/one.png", "local_context": "local", "global_context": "global"},
            {"filename": "missing.png", "image_path": "", "local_context": "missing", "global_context": ""},
        ],
    )
    config = ExportConfig(csv_path, tmp_path, tmp_path / "exports" / "memematch", batch_size=1)
    extractor = FakeExtractor()
    first = MemeMatchEmbeddingExporter(config, extractor).export()
    assert first["completed"] == 1
    assert first["failed"] == 1
    assert extractor.text_calls == 1
    with np.load(tmp_path / "exports" / "memematch.embeddings.npz") as values:
        assert values["text_embeddings"].shape == (1, 2)
        assert values["image_embeddings"].shape == (1, 3)
    metadata = json.loads((tmp_path / "exports" / "memematch.metadata.json").read_text())
    assert metadata["text_model"] == "test-minilm"
    assert metadata["text_embedding_dimension"] == 2
    resumed = FakeExtractor()
    second = MemeMatchEmbeddingExporter(config, resumed).export()
    assert second == first
    assert resumed.text_calls == 0


def test_prepared_text_is_versioned_clean_and_source_faithful() -> None:
    row = {
        "filename": "template-1.jpg",
        "ocr_text": "  I\u00a0 finally\npassed!  ",
        "local_context": "student reaction",
        "global_context": "a person indoors",
        "caption": "exam results",
        "verified_template": "Success Kid",
    }
    prepared = prepare_text(row)
    assert clean_source_text("a\x00  b\n") == "a b"
    assert prepared.fields == ("cleaned_ocr", "local_context", "caption_or_title", "verified_template", "global_context")
    assert "Text in meme: I finally passed!" in prepared.content
    assert "Global visual context (may be generic): a person indoors" in prepared.content
    assert prepared.source_columns["verified_template"] == "verified_template"
    assert stable_meme_id(row) == "template-1.jpg"
    with pytest.raises(ValueError, match="requires source_key"):
        stable_meme_id({"local_context": "no identity"})
    with pytest.raises(ValueError, match="string columns"):
        validate_metadata_row({"filename": 1})  # type: ignore[dict-item]


def test_manifest_deduplicates_stable_ids_and_validates_source_digest(tmp_path: Path) -> None:
    csv_path = tmp_path / "metadata.csv"
    _write_csv(csv_path, [
        {"filename": "same.jpg", "image_path": "", "local_context": "one", "global_context": ""},
        {"filename": "same.jpg", "image_path": "", "local_context": "duplicate", "global_context": ""},
        {"filename": "two.jpg", "image_path": "", "local_context": "two", "global_context": ""},
    ])
    manifest_path = tmp_path / "selection.json"
    manifest = build_expansion_manifest(csv_path, manifest_path, target_count=1000, source_url="https://example.test/source", license_status="reviewed")
    assert manifest["selected_count"] == 2
    loaded = load_expansion_manifest(manifest_path, csv_path)
    assert [entry["source_key"] for entry in loaded] == ["same.jpg", "two.jpg"]
    assert [entry["metadata_row"] for entry in loaded] == [0, 2]
    csv_path.write_text(csv_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        load_expansion_manifest(manifest_path, csv_path)


def test_export_uses_manifest_order_and_rejects_stale_source_path(tmp_path: Path) -> None:
    for name in ("one.png", "two.png"):
        image = tmp_path / "images" / name
        image.parent.mkdir(exist_ok=True)
        Image.new("RGB", (4, 4), "blue").save(image)
    csv_path = tmp_path / "metadata.csv"
    _write_csv(csv_path, [
        {"filename": "one.png", "image_path": "source/one.png", "local_context": "first", "global_context": ""},
        {"filename": "two.png", "image_path": "source/two.png", "local_context": "second", "global_context": ""},
    ])
    manifest_path = tmp_path / "selection.json"
    build_expansion_manifest(csv_path, manifest_path, target_count=2, source_url="https://example.test/source", license_status="reviewed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entries"].reverse()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = ExportConfig(csv_path, tmp_path, tmp_path / "exports" / "ordered", manifest=manifest_path)
    MemeMatchEmbeddingExporter(config, FakeExtractor()).export()
    records = json.loads((tmp_path / "exports" / "ordered.records.json").read_text(encoding="utf-8"))
    assert [record["source_key"] for record in records] == ["two.png", "one.png"]
    manifest["entries"][0]["source_path"] = "different.png"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="source path"):
        MemeMatchEmbeddingExporter(
            ExportConfig(csv_path, tmp_path, tmp_path / "exports" / "other", manifest=manifest_path), FakeExtractor()
        ).export()
