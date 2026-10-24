import json
from pathlib import Path

import numpy as np
import pytest

from app.retrieval import RetrievalConfig, RetrievalService, fuse_embeddings, load_export


def _export(prefix: Path) -> None:
    np.savez_compressed(
        prefix.with_suffix(".embeddings.npz"),
        text_embeddings=np.array([[1, 0], [0, 1], [1, 0]], dtype=np.float32),
        image_embeddings=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
    )
    prefix.with_suffix(".records.json").write_text(
        json.dumps(
            [
                {"status": "completed", "embedding_index": 0, "source_key": "z-meme"},
                {"status": "completed", "embedding_index": 1, "source_key": "a-meme"},
                {"status": "completed", "embedding_index": 2, "source_key": "b-meme"},
            ]
        )
    )
    prefix.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "text_embedding_dimension": 2,
                "image_embedding_dimension": 3,
                "text_model": "sentence-transformers/all-MiniLM-L6-v2",
                "image_model": "Salesforce/blip-image-captioning-base",
            }
        )
    )


def test_build_search_persistence_and_stable_ties(tmp_path: Path) -> None:
    prefix = tmp_path / "sample"
    _export(prefix)
    service = RetrievalService(load_export(prefix), RetrievalConfig(2.0, 1.0))
    assert {name: index.d for name, index in service.indexes.items()} == {
        "text": 2,
        "image": 3,
        "fused": 5,
    }
    assert [result.meme_id for result in service.search("text", np.array([1, 0]), 3)] == [
        "b-meme",
        "z-meme",
        "a-meme",
    ]
    directory = tmp_path / "indexes"
    service.save(directory)
    loaded = RetrievalService.load(directory)
    assert loaded.search("image", np.array([0, 1, 0]), 1)[0].meme_id == "a-meme"
    assert loaded.search("fused", fuse_embeddings(np.array([[1, 0]], dtype=np.float32), np.array([[1, 0, 0]], dtype=np.float32), RetrievalConfig(2, 1))[0], 1)[0].meme_id == "z-meme"


def test_validation_rejects_corruption_dimensions_and_model_mismatch(tmp_path: Path) -> None:
    prefix = tmp_path / "sample"
    _export(prefix)
    service = RetrievalService(load_export(prefix))
    with pytest.raises(ValueError, match="2-dimensional"):
        service.search("text", np.array([1, 0, 0]), 1)
    with pytest.raises(ValueError, match="incompatible"):
        service.validate_query_model("text", "other-model")
    np.savez_compressed(
        prefix.with_suffix(".embeddings.npz"),
        text_embeddings=np.array([[0, 0]], dtype=np.float32),
        image_embeddings=np.array([[1, 0, 0]], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="zero vectors"):
        load_export(prefix)


def test_fusion_weights_and_artifact_requirements(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        RetrievalConfig(0, 0)
    values = fuse_embeddings(
        np.array([[3, 4]], dtype=np.float32), np.array([[0, 0, 2]], dtype=np.float32), RetrievalConfig(3, 1)
    )
    assert values.shape == (1, 5)
    assert np.allclose(np.linalg.norm(values, axis=1), 1)
    with pytest.raises(FileNotFoundError):
        load_export(tmp_path / "missing")
