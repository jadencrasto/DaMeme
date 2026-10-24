import csv
import json

import numpy as np
import pytest

from app.recommendation_engine import RecommendationEngine, RecommendationUnavailable
from app.retrieval import ExportedEmbeddings, RetrievalService


def _write_catalog(tmp_path, source_key="one"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    metadata = tmp_path / "metadata.csv"
    with metadata.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_key", "filename", "local_joy", "template_group_heuristic"])
        writer.writeheader()
        writer.writerow({"source_key": source_key, "filename": "one.jpg", "local_joy": "0.8", "template_group_heuristic": "one"})
    exported = ExportedEmbeddings(
        np.array([[1, 0]], dtype=np.float32), np.array([[1, 0, 0]], dtype=np.float32),
        [{"status": "completed", "embedding_index": 0, "source_key": source_key}],
        {"text_model": "sentence-transformers/all-MiniLM-L6-v2", "image_model": "Salesforce/blip-image-captioning-base"},
    )
    indexes = tmp_path / "indexes"
    RetrievalService(exported).save(indexes)
    return metadata, indexes


def test_catalog_load_validates_record_metadata_alignment_and_rollback(tmp_path) -> None:
    metadata, indexes = _write_catalog(tmp_path)
    engine = RecommendationEngine(indexes, metadata, catalog_id="test", expected_count=1)
    assert engine.status() == {"ready": True, "detail": None, "meme_count": 1}
    bad_metadata, bad_indexes = _write_catalog(tmp_path / "bad", source_key="metadata-only")
    # The index uses one key while the CSV exposes another: startup remains
    # unavailable rather than serving mismatched recommendations.
    _write_catalog(tmp_path / "bad", source_key="index-key")
    bad_metadata.write_text("source_key,filename\nmetadata-only,one.jpg\n", encoding="utf-8")
    rollback = RecommendationEngine(bad_indexes, bad_metadata, catalog_id="test", expected_count=1)
    assert rollback.status()["ready"] is False
    with pytest.raises(RecommendationUnavailable):
        rollback.options()


def test_catalog_rejects_export_provenance_misalignment(tmp_path) -> None:
    metadata, indexes = _write_catalog(tmp_path)
    records = tmp_path / "export.records.json"
    export_metadata = tmp_path / "export.metadata.json"
    records.write_text(json.dumps([{"status": "completed", "embedding_index": 0, "source_key": "one"}]))
    export_metadata.write_text(json.dumps({"text_embedding_dimension": 2, "image_embedding_dimension": 3}))
    ready = RecommendationEngine(indexes, metadata, expected_count=1, export_records=records, export_metadata=export_metadata)
    assert ready.status()["ready"] is True
    records.write_text(json.dumps([{"status": "completed", "embedding_index": 0, "source_key": "wrong"}]))
    rejected = RecommendationEngine(indexes, metadata, expected_count=1, export_records=records, export_metadata=export_metadata)
    assert rejected.status()["ready"] is False
