import pytest

from app.api.v1.moods import emotions
from app.moods import MOODS, mood_query, primary_classifier_emotion, resolve_mood


def test_taxonomy_has_exactly_32_unique_ids_and_four_groups() -> None:
    assert len(MOODS) == 32
    assert len({mood.id for mood in MOODS}) == 32
    assert {mood.group for mood in MOODS} == {"Positive and uplifting", "Sad and reflective", "Stress and negative", "Anticipation and neutral"}
    assert len(emotions().moods) == 32


def test_synonyms_normalize_without_collapsing_distinct_moods() -> None:
    assert resolve_mood("worried").id == "anxiety"
    assert resolve_mood("stressed").id == "stress"
    assert resolve_mood("anxiety").id != resolve_mood("stress").id
    with pytest.raises(ValueError):
        resolve_mood("unsupported")


def test_mood_query_is_deterministic_and_preserves_the_situation() -> None:
    original = "My presentation starts in five minutes"
    assert mood_query(original, resolve_mood("anxiety")) == mood_query(original, resolve_mood("anxiety"))
    assert original in mood_query(original, resolve_mood("anxiety"))
    assert mood_query(original, None) == original
    assert primary_classifier_emotion(resolve_mood("anxiety")) == "fear"
    assert primary_classifier_emotion(resolve_mood("stress")) == "fear"
