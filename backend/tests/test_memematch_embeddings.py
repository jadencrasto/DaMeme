import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from app.memematch_embeddings import ExportConfig, MemeMatchEmbeddingExporter, resolve_image_path


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
