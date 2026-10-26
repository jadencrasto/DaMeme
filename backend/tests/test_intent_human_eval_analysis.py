import pytest

from app.intent_human_eval_analysis import aggregate_ratings, ndcg_at_five


def test_graded_ndcg_and_complete_rating_validation() -> None:
    assert ndcg_at_five([3, 2], [3, 2, 0]) == 1.0
    ratings = {"evaluators": {"one": {"q:m": {"situation": 3, "tone": 2, "sendability": 1}}}}
    assert aggregate_ratings(ratings, {"q:m"})["q:m"]["tone"] == 2.0
    with pytest.raises(ValueError, match="incomplete"):
        aggregate_ratings(ratings, {"q:m", "q:missing"})
