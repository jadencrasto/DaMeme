from app.benchmark import BenchmarkQuery, PoolDataset, canonical_digest, load_assignment
from app.local_global_human_eval import assignment, merge_pool


def _dataset() -> PoolDataset:
    query = BenchmarkQuery("development-1", "query", None, "category", "notes", "development")
    pool = {
        "format_version": 1,
        "benchmark_id": "phase7r-v1",
        "query_digest": "digest",
        "catalog": {},
        "runs": [],
        "pairs": [
            {
                "key": "development-1:a.jpg",
                "query_id": "development-1",
                "meme_id": "a.jpg",
                "contributions": [{"run_id": "old", "rank": 1}],
            }
        ],
    }
    return PoolDataset({query.id: query}, pool)


def test_merge_deduplicates_by_stable_query_meme_and_preserves_provenance() -> None:
    comparison = {
        "experiment": "comparison",
        "queries": [
            {
                "query_id": "development-1",
                "split": "development",
                "query_embedding": {"model_id": "model"},
                "variants": {
                    variant: {
                        "final_top_10": [
                            {
                                "source_key": "a.jpg",
                                "filtered_rank": 1,
                                "raw_rank": 1,
                                "raw_score": 0.5,
                                "final_score": 0.75,
                            }
                        ]
                    }
                    for variant in ("A0", "A1", "A2", "A3-EQ", "A3-LH", "A3-RRF")
                },
            }
        ],
    }
    pool = merge_pool(_dataset(), comparison)
    assert len(pool["pairs"]) == 1
    assert {row["run_id"] for row in pool["pairs"][0]["contributions"]} == {
        "old",
        "A0",
        "A1",
        "A2",
        "A3-EQ",
        "A3-LH",
        "A3-RRF",
    }


def test_assignment_loads_only_development_pair_identities(tmp_path) -> None:
    dataset = _dataset()
    payload = assignment(dataset.pool, ["development-1:a.jpg"], "diagnostic", 1, "test")
    path = tmp_path / "assignment.json"
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")
    assert load_assignment(path, dataset) == {"development-1:a.jpg"}
    assert payload["pool_digest"] == canonical_digest(dataset.pool)
