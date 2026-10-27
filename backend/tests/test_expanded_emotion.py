import csv

import numpy as np
import pytest

from app.expanded_emotion import (
    IMAGE_DIMENSION,
    TEXT_DIMENSION,
    load_expansion_artifacts,
    predict_text_embedding,
)


def _write(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _artifacts(tmp_path):
    ids = ["one", "two", "three"]
    _write(
        tmp_path / "dameme_5500_model_inputs.csv",
        ["meme_id"],
        [{"meme_id": value} for value in ids],
    )
    _write(
        tmp_path / "dameme_5500_dataset_splits.csv",
        ["meme_id", "split"],
        [
            {"meme_id": value, "split": split}
            for value, split in zip(ids, ("train", "validation", "test"), strict=True)
        ],
    )
    fields = [
        "meme_id",
        "include_in_training",
        "source",
        "sha256",
        "anger",
        "fear",
        "joy",
        "sadness",
        "surprise",
        "disgust",
        "sentiment",
    ]
    _write(
        tmp_path / "dameme_5500_training_manifest.csv",
        fields,
        [
            {
                "meme_id": value,
                "include_in_training": "True",
                "source": "MOOD",
                "sha256": value,
                "anger": "1" if value == "one" else "0",
                "fear": "0",
                "joy": "1" if value == "two" else "0",
                "sadness": "0",
                "surprise": "0",
                "disgust": "0",
                "sentiment": "positive",
            }
            for value in ids
        ],
    )
    np.savez(
        tmp_path / "dameme_5500_minilm_embeddings.npz",
        meme_ids=np.asarray(ids),
        embeddings=np.ones((3, TEXT_DIMENSION), dtype=np.float32),
    )
    np.save(
        tmp_path / "dameme_5500_blip_embeddings.npy",
        np.ones((3, IMAGE_DIMENSION), dtype=np.float32),
    )


def test_load_expansion_validates_alignment_and_mood_labels(tmp_path):
    _artifacts(tmp_path)
    data = load_expansion_artifacts(tmp_path, strict_counts=False)
    assert data.text.shape == (3, TEXT_DIMENSION)
    assert data.labels[0].tolist() == [1, 0, 0, 0, 0, 0]
    assert data.mood_mask.all()


def test_load_expansion_rejects_feature_id_misalignment(tmp_path):
    _artifacts(tmp_path)
    np.savez(
        tmp_path / "dameme_5500_minilm_embeddings.npz",
        meme_ids=np.asarray(["two", "one", "three"]),
        embeddings=np.ones((3, TEXT_DIMENSION), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="exact model-input row order"):
        load_expansion_artifacts(tmp_path, strict_counts=False)


def test_text_prediction_rejects_missing_or_wrong_dimension_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError):
        predict_text_embedding(tmp_path / "missing.pkl", np.ones(TEXT_DIMENSION, dtype=np.float32))


def test_load_expansion_filters_full_manifest_to_training_candidates(tmp_path):
    candidate_count = 5469
    ids = [f"candidate-{index}" for index in range(candidate_count)]
    splits = ["train"] * 4373 + ["validation"] * 552 + ["test"] * 544
    _write(
        tmp_path / "dameme_5500_model_inputs.csv",
        ["meme_id"],
        [{"meme_id": meme_id} for meme_id in ids],
    )
    _write(
        tmp_path / "dameme_5500_dataset_splits.csv",
        ["meme_id", "split"],
        [{"meme_id": meme_id, "split": split} for meme_id, split in zip(ids, splits, strict=True)],
    )
    fields = [
        "meme_id",
        "include_in_training",
        "source",
        "sha256",
        "anger",
        "fear",
        "joy",
        "sadness",
        "surprise",
        "disgust",
        "sentiment",
    ]
    candidates = [
        {
            "meme_id": meme_id,
            "include_in_training": "True",
            "source": "MOOD",
            "sha256": meme_id,
            "anger": "0",
            "fear": "0",
            "joy": "1",
            "sadness": "0",
            "surprise": "0",
            "disgust": "0",
            "sentiment": "positive",
        }
        for meme_id in ids
    ]
    excluded_duplicates = [
        {**candidates[index], "include_in_training": "False"} for index in range(31)
    ]
    _write(tmp_path / "dameme_5500_training_manifest.csv", fields, candidates + excluded_duplicates)
    np.savez(
        tmp_path / "dameme_5500_minilm_embeddings.npz",
        meme_ids=np.asarray(ids),
        embeddings=np.ones((candidate_count, TEXT_DIMENSION), dtype=np.float32),
    )
    np.save(
        tmp_path / "dameme_5500_blip_embeddings.npy",
        np.ones((candidate_count, IMAGE_DIMENSION), dtype=np.float32),
    )

    data = load_expansion_artifacts(tmp_path)

    assert len(data.meme_ids) == candidate_count
    assert data.meme_ids == tuple(ids)
