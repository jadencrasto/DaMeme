import numpy as np
import pytest

from app.emotion import EMOTIONS
from app.phase5_retrieval import Phase5Bundle
from app.phase6_recommendations import Phase6RecommendationService


def _bundle(references: tuple[str | None, ...]) -> Phase5Bundle:
    text = np.zeros((2, 384), dtype=np.float32)
    text[0, 0], text[1, 1] = 1, 1
    emotions = np.zeros((2, len(EMOTIONS)), dtype=np.float32)
    emotions[:, 0] = 1
    return Phase5Bundle(("first", "second"), text, emotions, references)


def test_phase6_image_resolution_is_rooted_and_non_ambiguous(tmp_path):
    image = tmp_path / "images" / "first.jpg"
    image.parent.mkdir()
    image.write_bytes(b"image")
    service = Phase6RecommendationService(tmp_path / "bundle", tmp_path / "model.pkl", tmp_path)
    service._bundle = _bundle(("first.jpg", "second.jpg"))
    assert service.image_path("first") == image
    assert service.image_path("missing") is None


def test_phase6_image_resolution_rejects_ambiguous_references(tmp_path):
    service = Phase6RecommendationService(tmp_path / "bundle", tmp_path / "model.pkl", tmp_path)
    service._bundle = _bundle(("same.jpg", "same.jpg"))
    assert service.image_path("first") is None


def test_phase6_emotion_ranking_uses_bundle_columns_and_stable_ties(tmp_path):
    text = np.zeros((3, 384), dtype=np.float32)
    text[0, 0], text[1, 1], text[2, 2] = 1, 1, 1
    emotions = np.zeros((3, len(EMOTIONS)), dtype=np.float32)
    emotions[:, EMOTIONS.index("joy")] = (.9, .9, .2)
    emotions[:, EMOTIONS.index("sadness")] = (.1, .3, .8)
    service = Phase6RecommendationService(tmp_path / "bundle", None, None)
    service._bundle = Phase5Bundle(
        ("DM-00003", "DM-00001", "DM-00002"), text, emotions, (None, None, None)
    )

    assert [row.meme_id for row in service.recommend_emotion("joy", 3)] == [
        "DM-00001",
        "DM-00003",
        "DM-00002",
    ]
    assert [row.meme_id for row in service.recommend_emotion("sadness", 2)] == [
        "DM-00002",
        "DM-00001",
    ]
    for emotion in EMOTIONS:
        assert service.recommend_emotion(emotion, 1)
    with pytest.raises(ValueError, match="unsupported emotion"):
        service.recommend_emotion("contentment", 5)
