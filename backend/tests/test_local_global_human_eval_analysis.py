import pytest

from app.local_global_human_eval_analysis import aggregate_ratings, dcg, ndcg_at_five


def test_ndcg_uses_graded_relevance_and_full_judged_candidate_set() -> None:
    assert dcg([3, 2]) > dcg([2, 3])
    assert ndcg_at_five([3, 2], [3, 2, 0]) == 1.0
    assert 0 < ndcg_at_five([2, 3], [3, 2, 0]) < 1


def test_rating_aggregation_rejects_missing_axis_or_pair() -> None:
    ratings = {"evaluators": {"one": {"q:m": {"situation": 3, "tone": 2, "sendability": 1}}}}
    assert aggregate_ratings(ratings, {"q:m"}) == {"q:m": {"situation": 3.0, "tone": 2.0, "sendability": 1.0}}
    with pytest.raises(ValueError, match="incomplete"):
        aggregate_ratings(ratings, {"q:m", "q:missing"})
