"""Prepare a bounded, provenance-preserving Phase 7R.4E.1 evaluation pool."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.benchmark import (
    POOL_FORMAT_VERSION,
    PoolDataset,
    canonical_digest,
    load_assignment,
    load_benchmark_ratings,
    load_pool_dataset,
)

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
BENCHMARKS = BACKEND / "benchmarks"
QUERIES = BENCHMARKS / "phase7r_v1_queries.json"
BASE_POOL = BENCHMARKS / "phase7r_v1_development.pool.json"
PILOT = BENCHMARKS / "phase7r_v1_pilot_25.json"
PILOT_RATINGS = BENCHMARKS / "phase7r_v1_pilot.ratings.json"
COMPARISON = BENCHMARKS / "phase7r_v1_local_global_comparison_20261025T054251Z" / "results.json"
VARIANTS = ("A0", "A1", "A2", "A3-EQ", "A3-LH", "A3-RRF")
DIAGNOSTIC_PER_CATEGORY = 9
SEED = 20261025
SECOND_EVALUATOR_SEED = 20261026


def _save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _hash(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _comparison_contributions(comparison: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    contributions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query in comparison["queries"]:
        if query["split"] != "development":
            raise ValueError("comparison output contains a held-out query")
        query_id = query["query_id"]
        for variant in VARIANTS:
            for row in query["variants"][variant]["final_top_10"]:
                key = f"{query_id}:{row['source_key']}"
                contributions[key].append(
                    {
                        "run_id": variant,
                        "rank": row["filtered_rank"],
                        "raw_rank": row["raw_rank"],
                        "retrieval_score": row["raw_score"],
                        "final_score": row["final_score"],
                        "comparison_artifact": "phase7r.4d.2",
                    }
                )
    return contributions


def merge_pool(dataset: PoolDataset, comparison: dict[str, Any]) -> dict[str, Any]:
    """Merge final result provenance by stable query--meme identity only."""
    pairs = {pair["key"]: copy.deepcopy(pair) for pair in dataset.pool["pairs"]}
    comparison_contributions = _comparison_contributions(comparison)
    for key, rows in comparison_contributions.items():
        query_id, meme_id = key.split(":", 1)
        pair = pairs.setdefault(
            key,
            {"key": key, "query_id": query_id, "meme_id": meme_id, "contributions": []},
        )
        pair["contributions"].extend(rows)
        pair["contributions"] = sorted(
            pair["contributions"], key=lambda value: (value["run_id"], value["rank"])
        )
    runs = copy.deepcopy(dataset.pool["runs"])
    runs.extend(
        {
            "id": variant,
            "split": "development",
            "model_version": comparison["queries"][0]["query_embedding"]["model_id"],
            "configuration": {
                "comparison_experiment": comparison["experiment"],
                "final_result_limit": 10,
                "comparison_results_sha256": hashlib.sha256(
                    json.dumps(comparison, sort_keys=True).encode()
                ).hexdigest(),
            },
        }
        for variant in VARIANTS
    )
    return {
        "format_version": POOL_FORMAT_VERSION,
        "benchmark_id": dataset.pool["benchmark_id"],
        "query_digest": dataset.pool["query_digest"],
        "catalog": copy.deepcopy(dataset.pool["catalog"]),
        "runs": sorted(runs, key=lambda run: run["id"]),
        "pairs": [pairs[key] for key in sorted(pairs)],
    }


def _variant_keys(pool: dict[str, Any], variant: str) -> set[str]:
    return {
        pair["key"]
        for pair in pool["pairs"]
        if any(contribution["run_id"] == variant for contribution in pair["contributions"])
    }


def _comparison_rows(pair: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in pair["contributions"] if row["run_id"] in VARIANTS]


def diagnostic_keys(dataset: PoolDataset, rated_keys: set[str], seed: int = SEED) -> list[str]:
    """Select 9 unjudged disagreement/novelty candidates in each category."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in dataset.visible_pairs("development"):
        if pair["key"] in rated_keys:
            continue
        contributions = _comparison_rows(pair)
        if contributions:
            grouped[dataset.queries[pair["query_id"]].category].append(pair)
    chosen: list[str] = []
    for category in sorted(grouped):

        def priority(pair: dict[str, Any]) -> tuple[int, int, int, str]:
            rows = _comparison_rows(pair)
            variants = {row["run_id"] for row in rows}
            introduced = "A0" not in variants
            disagreement = len(variants) > 1
            return (
                -int(introduced),
                -int(disagreement),
                min(row["rank"] for row in rows),
                _hash(seed, pair["key"]),
            )

        selected = sorted(grouped[category], key=priority)[:DIAGNOSTIC_PER_CATEGORY]
        if len(selected) != DIAGNOSTIC_PER_CATEGORY:
            raise ValueError(f"insufficient unjudged comparison pairs in {category}")
        chosen.extend(pair["key"] for pair in selected)
    return sorted(chosen)


def comparative_keys(dataset: PoolDataset, seed: int = SEED) -> tuple[list[str], list[str]]:
    """Preregister one development query per category before examining ratings."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for query in dataset.queries.values():
        if query.split == "development":
            grouped[query.category].append(query.id)
    selected_queries = [
        min(ids, key=lambda query_id: _hash(seed, query_id)) for _, ids in sorted(grouped.items())
    ]
    selected = set(selected_queries)
    keys = {
        pair["key"]
        for pair in dataset.visible_pairs("development")
        if pair["query_id"] in selected
        and any(row["run_id"] in VARIANTS and row["rank"] <= 5 for row in pair["contributions"])
    }
    return selected_queries, sorted(keys)


def assignment(
    pool: dict[str, Any], keys: list[str], name: str, seed: int, description: str
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "assignment": name,
        "pool_digest": canonical_digest(pool),
        "split": "development",
        "pair_keys": sorted(keys),
        "presentation_seed": seed,
        "presentation_order": sorted(keys, key=lambda key: _hash(seed, key)),
        "description": description,
    }


def _ratings_seed(ratings: dict[str, Any], pool: dict[str, Any]) -> dict[str, Any]:
    valid = {pair["key"] for pair in pool["pairs"]}
    return {
        "format_version": 1,
        "pool_digest": canonical_digest(pool),
        "evaluators": {
            evaluator: {key: axes for key, axes in values.items() if key in valid}
            for evaluator, values in ratings["evaluators"].items()
        },
    }


def workload_report(
    dataset: PoolDataset,
    pool: dict[str, Any],
    ratings: dict[str, Any],
    diagnostic: list[str],
    comparative: list[str],
    selected_queries: list[str],
) -> dict[str, Any]:
    rated = (
        set().union(*(set(values) for values in ratings["evaluators"].values()))
        if ratings["evaluators"]
        else set()
    )
    variant_coverage = {}
    for variant in VARIANTS:
        keys = _variant_keys(pool, variant)
        judged = keys & rated
        variant_coverage[variant] = {
            "pooled_pairs": len(keys),
            "judged_pairs": len(judged),
            "unjudged_pairs": len(keys - rated),
        }
    per_query = Counter(
        len([pair for pair in pool["pairs"] if pair["query_id"] == query_id])
        for query_id in dataset.queries
    )
    return {
        "format_version": 1,
        "scope": "development-only; no ratings were collected or inferred",
        "merged_pool_unique_pairs": len(pool["pairs"]),
        "previously_rated_pairs": len(rated),
        "new_unjudged_pairs": len(pool["pairs"]) - len(rated),
        "per_query_pool_sizes": {
            query_id: len([pair for pair in pool["pairs"] if pair["query_id"] == query_id])
            for query_id in sorted(
                query.id for query in dataset.queries.values() if query.split == "development"
            )
        },
        "per_variant_judged_coverage": variant_coverage,
        "plan_a_diagnostic": {
            "additional_ratings": len(diagnostic),
            "selection": "9 unjudged comparison candidates per category; deterministic novelty/disagreement priority",
            "exploratory": True,
        },
        "plan_b_comparative": {
            "selected_query_ids": selected_queries,
            "selected_query_count": len(selected_queries),
            "union_top_five_pairs": len(comparative),
            "previously_rated_within_assignment": len(set(comparative) & rated),
            "exact_additional_ratings": len(set(comparative) - rated),
            "selection": "one development query per category by fixed pre-registered hash; union of each variant's final top five",
        },
        "second_evaluator": {
            "optional": True,
            "same_identity_count": len(diagnostic),
            "presentation_seed": SECOND_EVALUATOR_SEED,
        },
        "validation": {
            "duplicate_query_meme_identities": len({pair["key"] for pair in pool["pairs"]})
            != len(pool["pairs"]),
            "held_out_pairs_present": any(
                dataset.queries[pair["query_id"]].split != "development" for pair in pool["pairs"]
            ),
            "unjudged_pairs_scored_as_irrelevant": False,
            "pool_size_distribution": dict(sorted(per_query.items())),
        },
    }


def run(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing experiment directory: {output}")
    base = load_pool_dataset(QUERIES, BASE_POOL)
    existing_ratings = load_benchmark_ratings(PILOT_RATINGS, base)
    comparison = json.loads(COMPARISON.read_text(encoding="utf-8"))
    if comparison["compatibility"]["status"] != "PASS":
        raise ValueError("A0 compatibility must pass before pooling")
    pool = merge_pool(base, comparison)
    output.mkdir(parents=True)
    pool_path = output / "development.pool.json"
    _save(pool_path, pool)
    merged = load_pool_dataset(QUERIES, pool_path)
    seed_ratings = _ratings_seed(existing_ratings, pool)
    _save(output / "ratings.seed.json", seed_ratings)
    rated = (
        set().union(*(set(values) for values in seed_ratings["evaluators"].values()))
        if seed_ratings["evaluators"]
        else set()
    )
    diagnostic = diagnostic_keys(merged, rated)
    selected_queries, comparative = comparative_keys(merged)
    diagnostic_assignment = assignment(
        pool,
        diagnostic,
        "phase7r.4e.1-diagnostic",
        SEED,
        "Exploratory balanced diagnostic assignment",
    )
    comparative_assignment = assignment(
        pool, comparative, "phase7r.4e.1-comparative", SEED, "Preregistered comparative assignment"
    )
    second_assignment = assignment(
        pool,
        diagnostic,
        "phase7r.4e.1-diagnostic-second-evaluator",
        SECOND_EVALUATOR_SEED,
        "Optional overlap with randomized presentation",
    )
    _save(output / "diagnostic.assignment.json", diagnostic_assignment)
    _save(output / "comparative.assignment.json", comparative_assignment)
    _save(output / "second_evaluator_diagnostic.assignment.json", second_assignment)
    # Validate generated, immutable-format inputs through the same loaders the UI uses.
    load_benchmark_ratings(output / "ratings.seed.json", merged)
    if load_assignment(output / "diagnostic.assignment.json", merged) != set(diagnostic):
        raise ValueError("diagnostic assignment did not preserve pair identities")
    if load_assignment(output / "comparative.assignment.json", merged) != set(comparative):
        raise ValueError("comparative assignment did not preserve pair identities")
    report = workload_report(merged, pool, seed_ratings, diagnostic, comparative, selected_queries)
    _save(output / "workload.report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare Phase 7R.4E.1 human-evaluation materials")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = (
        args.output
        or BENCHMARKS / f"phase7r_v1_human_eval_preparation_{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    )
    report = run(output)
    print(f"Merged pool: {report['merged_pool_unique_pairs']} pairs")
    print(
        f"Existing judgments: {report['previously_rated_pairs']}; new unjudged: {report['new_unjudged_pairs']}"
    )
    print(
        f"Plan A diagnostic: {report['plan_a_diagnostic']['additional_ratings']} additional ratings"
    )
    print(
        f"Plan B comparative: {report['plan_b_comparative']['exact_additional_ratings']} additional ratings"
    )
    print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
