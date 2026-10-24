import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from app.recommendations import (
    DiversityConfig,
    MemeMatchMetadata,
    RankingWeights,
    RecommendationService,
    SafetyConfig,
    normalise_template_group,
)
from app.retrieval import ExportedEmbeddings, RetrievalService, SearchResult


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


def test_sentiment_is_available_only_for_an_explicit_sentiment_preference(tmp_path: Path) -> None:
    service = _service(_metadata(tmp_path / "metadata.csv"))
    unspecified = service.recommend_text(np.array([1, 0]), emotion="joy", limit=1)
    assert unspecified[0].sentiment_relevance is None
    requested = service.recommend_text(np.array([1, 0]), emotion="joy", sentiment="positive", limit=1)
    assert requested[0].sentiment_relevance == pytest.approx(0.1)


def test_semantic_default_wins_a_close_contextual_match_over_emotion_only_rerank(tmp_path: Path) -> None:
    service = _service(_metadata(tmp_path / "metadata.csv"))
    candidates = [SearchResult("same-template_1.jpg", 0, 0.6), SearchResult("same-template_2.jpg", 1, 0.5)]
    old = service._rank(candidates, emotion="joy", sentiment=None, usage_preferences=(), weights=RankingWeights(0.8, 0.15, 0.05), diversity=None, safety=SafetyConfig(), min_semantic_similarity=None, limit=2)
    improved = service._rank(candidates, emotion="joy", sentiment=None, usage_preferences=(), weights=RankingWeights(0.90, 0.08, 0.02), diversity=None, safety=SafetyConfig(), min_semantic_similarity=None, limit=2)
    assert old[0].meme_id == "same-template_2.jpg"
    assert improved[0].meme_id == "same-template_1.jpg"


def test_template_group_normalization_collapses_instance_variants_only(tmp_path: Path) -> None:
    path = tmp_path / "templates.csv"
    path.write_text(
        "source_key,filename,template_group_heuristic,local_joy\n"
        "wolf-one,wolf-one.jpg,23-wolves-laugh-163,0.9\n"
        "wolf-two,wolf-two.jpg,23_wolves_laugh,0.8\n"
        "other,other.jpg,24-wolves-laugh,0.7\n"
        "distinct,distinct.jpg,template-2,0.6\n"
    )
    metadata = MemeMatchMetadata.from_csv(path)
    assert normalise_template_group("23-wolves-laugh-163") == "23_wolves_laugh"
    assert metadata.get("wolf-one").template_group == metadata.get("wolf-two").template_group
    assert metadata.get("wolf-one").template_group != metadata.get("other").template_group
    assert metadata.get("distinct").template_group == "template_2"


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


def test_v2_filename_image_fallback_is_safe_when_kaggle_path_is_absolute(tmp_path: Path) -> None:
    image = tmp_path / "images" / "v2.jpg"
    image.parent.mkdir()
    image.write_bytes(b"local")
    path = tmp_path / "metadata.csv"
    path.write_text("source_key,filename,kaggle_path,local_joy\nv2,v2.jpg,/content/drive/v2.jpg,0.9\n")
    assert MemeMatchMetadata.from_csv(path, tmp_path).get("v2").image_path == str(image)


def test_safety_filter_template_metadata_and_missing_metadata_are_deterministic(tmp_path: Path) -> None:
    metadata = _metadata(tmp_path / "metadata.csv")
    entries = dict(metadata.entries)
    entries["same-template_1.jpg"] = replace(entries["same-template_1.jpg"], template_group="drake")
    entries["same-template_2.jpg"] = replace(
        entries["same-template_2.jpg"], template_group="drake", safety_text="mentions suicide"
    )
    entries["other.jpg"] = replace(entries["other.jpg"], safety_text="sexual assault reference")
    service = _service(MemeMatchMetadata(entries))
    # Source joy annotations still rerank eligible candidates, but explicit
    # self-harm text is excluded by default and can be deliberately disabled.
    safe = service.recommend_emotion("joy", diversity=DiversityConfig(), limit=3)
    assert "same-template_2.jpg" not in [item.meme_id for item in safe]
    assert "other.jpg" not in [item.meme_id for item in service.recommend_emotion("sadness", limit=3)]
    assert "other.jpg" in [item.meme_id for item in service.recommend_emotion("joy", limit=3)]
    unsafe_allowed = service.recommend_emotion("joy", safety=SafetyConfig(enabled=False), limit=3)
    assert unsafe_allowed[0].meme_id == "same-template_2.jpg"
    diverse = service.recommend_emotion("joy", safety=SafetyConfig(enabled=False), diversity=DiversityConfig(), limit=3)
    assert len([item for item in diverse if item.meme_id.startswith("same-template")]) == 1
    missing = service._rank(
        ["not-in-metadata.jpg"], emotion="joy", sentiment=None, usage_preferences=(),
        weights=RankingWeights(1, 1, 0), diversity=None, safety=SafetyConfig(),
        min_semantic_similarity=None, limit=1,
    )
    assert missing[0].meme_id == "not-in-metadata.jpg"
