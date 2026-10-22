import json

import numpy as np

from app.emotion import EMOTIONS, deterministic_splits, mood_targets, text_features


def test_mood_labels_are_multi_label_and_sentiment_is_multiclass() -> None:
    annotation = json.dumps(
        {
            "emotions": {
                label: label if label in {"anger", "fear"} else f"not_{label}" for label in EMOTIONS
            },
            "sentiment": "negative",
        }
    )
    emotions, sentiment = mood_targets(annotation)
    assert emotions.tolist() == [1, 1, 0, 0, 0, 0]
    assert sentiment == 1


def test_grouped_splits_are_deterministic() -> None:
    rows = [{"id": 1, "sha256": "same"}, {"id": 2, "sha256": "same"}, {"id": 3, "sha256": "other"}]
    first, second = deterministic_splits(rows), deterministic_splits(rows)
    assert first == second
    assert first[1] == first[2]


def test_text_features_are_normalized_and_handle_empty_text() -> None:
    values = text_features("")
    assert values.shape == (128,)
    assert np.isfinite(values).all()
