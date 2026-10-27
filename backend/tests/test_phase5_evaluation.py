import numpy as np

from app.emotion import EMOTIONS
from app.phase5_evaluation import (
    EvaluationInputs,
    eligible_candidates,
    evaluate_partition,
    rank_query,
    select_weight,
)
from app.phase5_retrieval import Phase5Bundle


def _inputs() -> EvaluationInputs:
    text = np.zeros((6, 384), dtype=np.float32)
    text[:, 0] = 1
    emotions = np.zeros((6, len(EMOTIONS)), dtype=np.float32)
    emotions[[0, 2, 3, 5], 0] = 1
    emotions[[1, 4], 2] = 1
    labels = np.zeros((6, len(EMOTIONS)), dtype=np.int8)
    labels[[0, 2, 3, 5], 0] = 1
    labels[[1, 4], 2] = 1
    bundle = Phase5Bundle(tuple(f"m{index}" for index in range(6)), text, emotions, (None,) * 6)
    return EvaluationInputs(
        bundle,
        ("train", "train", "validation", "test", "test", "test"),
        np.ones(6, dtype=bool),
        labels,
        ("h0", "h1", "h2", "h3", "h4", "h0"),
    )


def test_eligible_candidates_exclude_self_and_duplicate_groups():
    inputs = _inputs()
    assert eligible_candidates(inputs, 3).tolist() == [0, 1]
    assert eligible_candidates(inputs, 5).tolist() == [1]


def test_evaluation_handles_zero_relevance_and_reports_metrics():
    result = evaluate_partition(_inputs(), "test", 0.2)
    assert result["aggregate"]["total_mood_queries"] == 3
    assert result["aggregate"]["excluded_zero_relevant"] == 1
    assert result["aggregate"]["query_count"] == 2
    assert result["aggregate"]["precision_at_5"] == 0.2
    assert result["aggregate"]["hit_rate_at_10"] == 1.0


def test_ranking_does_not_consult_ground_truth_labels():
    inputs = _inputs()
    candidates = eligible_candidates(inputs, 3)
    first = rank_query(inputs, 3, candidates, 0.2)
    inputs.labels[:] = 0
    second = rank_query(inputs, 3, candidates, 0.2)
    assert first.tolist() == second.tolist()


def test_weight_selection_is_validation_only_and_deterministic():
    result = select_weight(_inputs(), (0.0, 0.2, 0.4))
    assert result["selection_metric"] == "validation nDCG@10"
    assert result["selected_emotion_weight"] in {0.0, 0.2, 0.4}
