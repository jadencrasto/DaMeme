from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from app.datasets.ingestion import DatasetIngester, IngestionOptions, SourceDetails


def make_image(path: Path, color: str, text: str = "") -> None:
    image = Image.new("RGB", (128, 128), color)
    ImageDraw.Draw(image).text((10, 10), text, fill="white")
    image.save(path)


@pytest.fixture
def source() -> SourceDetails:
    return SourceDetails("mood", "sample-v1", "https://example.test/mood", "pending")


def test_import_is_idempotent_and_preserves_supplied_split(
    tmp_path: Path, source: SourceDetails
) -> None:
    root = tmp_path / "images"
    root.mkdir()
    make_image(root / "one.png", "red")
    database = tmp_path / "metadata.db"
    ingester = DatasetIngester(database)
    metadata = {"one.png": {"split": "test", "emotion_label": "joy", "label_provenance": "MOOD"}}
    first = ingester.ingest(root, source, IngestionOptions(batch_size=1), metadata)
    second = ingester.ingest(root, source, IngestionOptions(batch_size=1), metadata)
    rows = ingester.connection.execute("SELECT dataset_split, emotion_label FROM memes").fetchall()
    assert (first.accepted, second.accepted, len(rows)) == (1, 0, 1)
    assert tuple(rows[0]) == ("test", "joy")
    ingester.close()


def test_exact_and_near_duplicates_are_recorded(tmp_path: Path, source: SourceDetails) -> None:
    root = tmp_path / "images"
    root.mkdir()
    make_image(root / "one.png", "red")
    (root / "copy.png").write_bytes((root / "one.png").read_bytes())
    make_image(root / "near.jpg", "red")
    ingester = DatasetIngester(tmp_path / "metadata.db")
    ingester.ingest(root, source, IngestionOptions(near_duplicate_distance=64))
    stats = ingester.statistics()
    assert stats["exact_duplicates"] == 1
    assert stats["near_duplicates"] >= 1
    ingester.close()


def test_invalid_file_records_failure_and_checkpoint_allows_resume(
    tmp_path: Path, source: SourceDetails
) -> None:
    root = tmp_path / "images"
    root.mkdir()
    (root / "bad.png").write_text("not an image", encoding="utf-8")
    make_image(root / "valid.png", "blue")
    ingester = DatasetIngester(tmp_path / "metadata.db")
    result = ingester.ingest(root, source, IngestionOptions(batch_size=1))
    assert (result.accepted, result.rejected) == (1, 1)
    assert ingester.statistics()["failures"] == 1
    ingester.close()


def test_rejects_pixels_above_limit(tmp_path: Path, source: SourceDetails) -> None:
    root = tmp_path / "images"
    root.mkdir()
    make_image(root / "large.png", "blue")
    ingester = DatasetIngester(tmp_path / "metadata.db")
    result = ingester.ingest(root, source, IngestionOptions(max_pixels=100))
    assert result.rejected == 1
    ingester.close()
