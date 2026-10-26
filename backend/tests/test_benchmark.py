import json
from pathlib import Path

import pytest

from app.benchmark import (
    _BENCHMARK_HTML,
    AXES,
    FINAL_CONFIRMATION,
    POOL_FORMAT_VERSION,
    RATINGS_FORMAT_VERSION,
    benchmark_metrics,
    build_pilot,
    canonical_digest,
    load_benchmark_ratings,
    load_pilot,
    load_pool_dataset,
    load_queries,
    pilot_analysis,
    query_digest,
    rating_progress,
    save_benchmark_ratings,
    update_rating,
)

QUERY_PATH = Path(__file__).parents[1] / "benchmarks" / "phase7r_v1_queries.json"


def _pool_payload(query_path: Path) -> dict[str, object]:
    queries, query_payload = load_queries(query_path)
    query = next(item for item in queries.values() if item.split == "development")
    run = {"id": "semantic-fixture", "split": "development", "model_version": "fixture", "configuration": {}, "catalog": {}}
    pairs = [
        {"key": f"{query.id}:{name}.jpg", "query_id": query.id, "meme_id": f"{name}.jpg",
         "contributions": [{"run_id": "semantic-fixture", "rank": rank}]}
        for rank, name in enumerate(("a", "b", "c", "d", "e"), start=1)
    ]
    return {"format_version": POOL_FORMAT_VERSION, "benchmark_id": query_payload["benchmark_id"],
            "query_digest": query_digest(queries), "catalog": {}, "runs": [run], "pairs": pairs}


def _dataset(tmp_path):
    pool_path = tmp_path / "pool.json"
    pool_path.write_text(json.dumps(_pool_payload(QUERY_PATH)), encoding="utf-8")
    return load_pool_dataset(QUERY_PATH, pool_path)


def test_query_set_is_deterministic_stratified_and_complete() -> None:
    first, payload = load_queries(QUERY_PATH)
    second, _ = load_queries(QUERY_PATH)
    assert len(first) == 80
    assert query_digest(first) == query_digest(second)
    assert sum(query.split == "development" for query in first.values()) == 40
    assert payload["benchmark_id"] == "phase7r-v1"
    assert {query.category for query in first.values()} == {
        "Everyday situations", "Academic and workplace stress", "Celebrations and achievements",
        "Negative emotions and setbacks", "Sarcasm and irony", "Nostalgia and memories",
        "Meme-template references", "Reaction and response intent",
    }


def test_pool_rejects_duplicate_pairs_and_wrong_provenance(tmp_path) -> None:
    payload = _pool_payload(QUERY_PATH)
    payload["pairs"].append(payload["pairs"][0])
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_pool_dataset(QUERY_PATH, path)
    payload = _pool_payload(QUERY_PATH)
    payload["query_digest"] = "not-the-query-set"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        load_pool_dataset(QUERY_PATH, path)


def test_three_axis_ratings_persist_and_evaluators_are_isolated(tmp_path) -> None:
    dataset = _dataset(tmp_path)
    ratings_path = tmp_path / "ratings.json"
    ratings = load_benchmark_ratings(ratings_path, dataset)
    first_key = next(iter(dataset.pairs))
    update_rating(ratings, dataset, "rater-a", first_key, {"situation": 3, "tone": 2, "sendability": 1})
    update_rating(ratings, dataset, "rater-b", first_key, {"situation": 0, "tone": 1, "sendability": 2})
    save_benchmark_ratings(ratings_path, dataset, ratings)
    restored = load_benchmark_ratings(ratings_path, dataset)
    assert restored["evaluators"]["rater-a"][first_key] == {"situation": 3, "tone": 2, "sendability": 1}
    assert restored["evaluators"]["rater-b"][first_key]["situation"] == 0
    assert rating_progress(dataset, restored, "rater-a", "development") == {"completed": 1, "total": 5, "remaining": 4}
    agreement = benchmark_metrics(dataset, restored, split="development")["inter_rater_agreement"]
    assert agreement["pairs"][0]["axes"]["situation"]["overlap"] == 1
    with pytest.raises(ValueError, match="all three"):
        update_rating(restored, dataset, "rater-a", first_key, {"situation": 3})


def test_metrics_do_not_treat_missing_judgments_as_zero_and_report_axes(tmp_path) -> None:
    dataset = _dataset(tmp_path)
    ratings = {"format_version": RATINGS_FORMAT_VERSION, "pool_digest": canonical_digest(dataset.pool), "evaluators": {"rater-a": {}}}
    partial_key = next(iter(dataset.pairs))
    ratings["evaluators"]["rater-a"][partial_key] = {axis: 3 for axis in AXES}
    incomplete = benchmark_metrics(dataset, ratings, split="development")
    run = incomplete["runs"]["semantic-fixture"]
    assert run["fully_judged_queries_at_5"] == 0
    assert run["mean_ndcg_at_5"] is None
    assert incomplete["unjudged_candidate_pairs"] == 4
    for index, key in enumerate(dataset.pairs):
        ratings["evaluators"]["rater-a"][key] = {"situation": 3 if index == 0 else 0, "tone": 2, "sendability": 1}
    complete = benchmark_metrics(dataset, ratings, split="development")
    assert complete["runs"]["semantic-fixture"]["mean_precision_at_5"] == pytest.approx(0.2)
    assert complete["axis_summaries"]["tone"]["mean"] == pytest.approx(2.0)


def test_held_out_metrics_require_explicit_architecture_freeze(tmp_path) -> None:
    dataset = _dataset(tmp_path)
    ratings = {"format_version": RATINGS_FORMAT_VERSION, "pool_digest": canonical_digest(dataset.pool), "evaluators": {}}
    with pytest.raises(ValueError, match="final-confirmation"):
        benchmark_metrics(dataset, ratings, split="test")
    result = benchmark_metrics(dataset, ratings, split="test", final_confirmation=FINAL_CONFIRMATION)
    assert result["split"] == "test"


def test_stratified_pilot_is_reproducible_pool_bound_and_has_scoring_anchors(tmp_path) -> None:
    # A fixture with one query cannot supply the eight-category production pilot.
    # The checked-in pool is the production source for this focused invariant.
    production = load_pool_dataset(QUERY_PATH, QUERY_PATH.parent / "phase7r_v1_development.pool.json")
    first = tmp_path / "pilot-a.json"
    second = tmp_path / "pilot-b.json"
    assert build_pilot(production, first) == {"pairs": 25, "categories": 8}
    assert build_pilot(production, second) == {"pairs": 25, "categories": 8}
    assert first.read_bytes() == second.read_bytes()
    assert len(load_pilot(first, production)) == 25
    assert "0 = not at all; 1 = slight fit; 2 = clear fit; 3 = strong fit" in _BENCHMARK_HTML
    assert "Does it fit the concrete situation?" in _BENCHMARK_HTML


def test_pilot_analysis_reports_missing_as_insufficient_not_zero(tmp_path) -> None:
    dataset = _dataset(tmp_path)
    keys = set(dataset.pairs)
    ratings = {"format_version": RATINGS_FORMAT_VERSION, "pool_digest": canonical_digest(dataset.pool),
               "evaluators": {"pilot-a": {next(iter(keys)): {"situation": 1, "tone": 2, "sendability": 0}}}}
    result = pilot_analysis(dataset, ratings, keys, "pilot-a")
    assert result["completed_reviews"] == 1
    assert len(result["missing_pair_keys"]) == 4
    assert result["axis_averages"] == {"situation": 1.0, "tone": 2.0, "sendability": 0.0}
    assert result["benchmark_metrics"]["reliably_calculable"] is False
