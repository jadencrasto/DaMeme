import csv
from pathlib import Path

import numpy as np
import pytest

from app.recommendations import (
    DiversityConfig,
    MemeMatchMetadata,
    RankingWeights,
    RecommendationService,
)
from app.retrieval import ExportedEmbeddings, RetrievalService


def _metadata(path: Path) -> MemeMatchMetadata:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "image_path", "local_emotion_joy", "global_emotion_joy", "local_emotion_sadness", "local_sentiment_positive", "usage"])
        writer.writeheader()
        writer.writerows([
            {"filename": "same-template_1.jpg", "image_path": "/content/drive/stale/same-template_1.jpg", "local_emotion_joy": "0.2", "global_emotion_joy": "0.4", "local_emotion_sadness": "0.8", "local_sentiment_positive": "0.1", "usage": "sarcasm|parody"},
            {"filename": "same-template_2.jpg", "local_emotion_joy": "0.9", "global_emotion_joy": "0.7", "local_emotion_sadness": "0.1", "local_sentiment_positive": "0.9", "usage": "parody"},
            {"filename": "other.jpg", "local_emotion_joy": "0.6", "global_emotion_joy": "0.6", "local_emotion_sadness": "0.2", "local_sentiment_positive": "0.4", "usage": "absurd humor"},
        ])
    return MemeMatchMetadata.from_csv(path)


def _service(metadata: MemeMatchMetadata) -> RecommendationService:
    records = [{"source_key": key, "embedding_index": position, "status": "completed"} for position, key in enumerate(metadata.entries)]
    exported = ExportedEmbeddings(np.array([[1, 0], [0.8, 0.6], [0, 1]], dtype=np.float32), np.eye(3, dtype=np.float32), records, {"text_model": "sentence-transformers/all-MiniLM-L6-v2", "image_model": "Salesforce/blip-image-captioning-base"})
    return RecommendationService(RetrievalService(exported), metadata)


def test_emotion_text_sentiment_usage_and_diversity_ranking(tmp_path: Path) -> None:
    service = _service(_metadata(tmp_path / "metadata.csv"))
    emotion = service.recommend_emotion("joy", limit=3)
    assert [item.meme_id for item in emotion][:2] == ["same-template_2.jpg", "other.jpg"]
    combined = service.recommend_text(np.array([1, 0]), emotion="joy", sentiment="positive", weights=RankingWeights(1, 2, 2), limit=3)
    assert combined[0].meme_id == "same-template_2.jpg"
    assert "source joy probability" in combined[0].explanation
    usage = service.recommend_emotion("joy", usage_preferences=["sarcasm"], limit=3)
    assert [item.meme_id for item in usage] == ["same-template_1.jpg"]
    diverse = service.recommend_emotion("joy", diversity=DiversityConfig(max_per_template=1), limit=3)
    assert len({item.meme_id.rsplit("_", 1)[0] for item in diverse}) == len(diverse)


def test_metadata_mapping_missing_duplicates_and_invalid_weights(tmp_path: Path) -> None:
    metadata = _metadata(tmp_path / "metadata.csv")
    service = _service(metadata)
    assert service.recommend_text(np.array([1, 0]), limit=2)[0].meme_id == "same-template_1.jpg"
    with pytest.raises(ValueError, match="at least one"):
        RankingWeights(0, 0, 0)
    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text("filename,local_emotion_joy\na.jpg,0.1\na.jpg,0.2\n")
    with pytest.raises(ValueError, match="duplicate"):
        MemeMatchMetadata.from_csv(duplicate)
    assert metadata.get("unknown.jpg") is None
    image = tmp_path / "images" / "same-template_1.jpg"
    image.parent.mkdir()
    image.write_bytes(b"local")
    resolved = MemeMatchMetadata.from_csv(tmp_path / "metadata.csv", tmp_path)
    assert resolved.get("same-template_1.jpg").image_path == str(image)


def test_text_ranking_is_semantic_first_and_can_return_no_match(tmp_path: Path) -> None:
    service = _service(_metadata(tmp_path / "metadata.csv"))
    # The default semantic-first weights keep the exact semantic match ahead of
    # the joy-heavy near match; equal legacy weights would choose the latter.
    rows = service.recommend_text(np.array([1, 0]), emotion="joy", limit=3)
    assert rows[0].meme_id == "same-template_1.jpg"
    assert service.recommend_text(np.array([0.5, 0.5]), min_semantic_similarity=1.0, limit=3) == []
    with pytest.raises(ValueError, match="between -1 and 1"):
        service.recommend_text(np.array([1, 0]), min_semantic_similarity=1.1, limit=3)


def test_direct_memematch_probability_columns_are_extracted(tmp_path: Path) -> None:
    path = tmp_path / "memematch.csv"
    path.write_text(
        "filename,local_sadness,global_sadness,local_positive,global_positive\n"
        "one.jpg,0.8,0.2,0.1,0.7\n"
    )
    entry = MemeMatchMetadata.from_csv(path).get("one.jpg")
    assert entry.emotion_score("sadness") == pytest.approx(0.5)
    assert entry.sentiment_score("positive") == pytest.approx(0.4)


def test_source_key_is_preferred_over_filename_for_retrieval_joins(tmp_path: Path) -> None:
    path = tmp_path / "memematch.csv"
    path.write_text(
        "filename,source_key,local_joy,global_positive,usage\n"
        "display-name.jpg,stable-source-id,0.8,0.7,reaction|celebration\n"
    )
    metadata = MemeMatchMetadata.from_csv(path)
    assert metadata.get("stable-source-id") is not None
    assert metadata.get("display-name.jpg") is None
    entry = metadata.get("stable-source-id")
    assert entry is not None and entry.emotion_score("joy") == pytest.approx(0.8)
    assert entry.sentiment_score("positive") == pytest.approx(0.7)
    assert entry.usage == frozenset({"reaction", "celebration"})
