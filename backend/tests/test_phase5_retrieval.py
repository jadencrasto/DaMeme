import json

import numpy as np
import pytest

from app.emotion import EMOTIONS
from app.phase5_retrieval import Phase5Bundle, _image_ref


def _bundle() -> Phase5Bundle:
    text = np.zeros((3, 384), dtype=np.float32)
    text[0, 0], text[1, 1], text[2, 0] = 1, 1, 1
    emotions = np.asarray(
        [[1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0.8, 0, 0.2, 0, 0, 0]], dtype=np.float32
    )
    return Phase5Bundle(("a", "b", "c"), text, emotions, ("a.jpg", None, "c.jpg"))


def test_phase5_ranking_is_deterministic_and_bounded():
    bundle = _bundle()
    query = np.zeros(384, dtype=np.float32)
    query[0] = 1
    user = np.zeros(len(EMOTIONS), dtype=np.float32)
    user[0] = 1
    rows = bundle.rank(query, user, top_k=2, candidate_k=3)
    assert [row.meme_id for row in rows] == ["a", "c"]
    assert all(0 <= row.semantic_similarity <= 1 for row in rows)
    assert all(0 <= row.emotion_compatibility <= 1 and 0 <= row.score <= 1 for row in rows)


def test_phase5_rejects_invalid_bundle_and_ranking_inputs():
    with pytest.raises(ValueError, match="unique"):
        Phase5Bundle(
            ("a", "a"),
            np.ones((2, 384), dtype=np.float32),
            np.ones((2, 6), dtype=np.float32),
            (None, None),
        )
    with pytest.raises(ValueError, match="weights"):
        _bundle().rank(
            np.eye(1, 384, dtype=np.float32)[0],
            np.ones(6, dtype=np.float32),
            semantic_weight=0,
            emotion_weight=0,
        )


def test_phase5_bundle_load_preserves_explicit_id_mapping(tmp_path):
    bundle = _bundle()
    np.savez_compressed(
        tmp_path / "phase5_bundle.npz",
        meme_ids=np.asarray(bundle.ids),
        text_embeddings=bundle.text,
        emotion_scores=bundle.emotions,
        image_refs=np.asarray([value or "" for value in bundle.image_refs]),
    )
    (tmp_path / "phase5_metadata.json").write_text(
        json.dumps(
            {"feature_version": "dameme-5500-minilm-blipvision-v1", "emotion_order": list(EMOTIONS)}
        )
    )
    loaded = Phase5Bundle.load(tmp_path)
    assert loaded.ids == bundle.ids


def test_phase5_image_reference_is_a_filename_not_a_host_path():
    assert _image_ref({"image_path": r"C:\drive\memes\safe.jpg"}) == "safe.jpg"
    assert _image_ref({"image_path": "../../secret.jpg"}) == "secret.jpg"
