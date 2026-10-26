"""Read-only final analysis for the Phase 7R.5C intent evaluation."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.benchmark import load_assignment, load_benchmark_ratings, load_pool_dataset

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
BENCHMARKS = BACKEND / "benchmarks"
EVALUATION = BENCHMARKS / "phase7r_v1_intent_human_eval_20261025T065752Z"
QUERIES = BENCHMARKS / "phase7r_v1_queries.json"
POOL = EVALUATION / "development.pool.json"
ASSIGNMENT = EVALUATION / "complete_comparative.assignment.json"
VARIANTS = ("A0", "A3-LH", "A3-LH-INTENT")
AXES = ("situation", "tone", "sendability")


def dcg(values: list[float]) -> float:
    return sum(value / math.log2(rank + 1) for rank, value in enumerate(values, start=1))


def ndcg_at_five(values: list[float], judged_values: list[float]) -> float:
    ideal = dcg(sorted(judged_values, reverse=True)[:5])
    return dcg(values[:5]) / ideal if ideal else 0.0


def find_completed_ratings(directory: Path, dataset: Any, assignment: set[str]) -> Path:
    candidates = []
    for path in directory.glob("ratings*.json"):
        ratings = load_benchmark_ratings(path, dataset)
        completed = (
            set().union(*(set(values) for values in ratings["evaluators"].values()))
            if ratings["evaluators"]
            else set()
        )
        if assignment.issubset(completed):
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError(
            "no ratings*.json file in the evaluation directory completes every assigned identity"
        )
    if len(candidates) > 1:
        raise ValueError(
            f"multiple completed ratings files found: {', '.join(str(path) for path in candidates)}"
        )
    return candidates[0]


def aggregate_ratings(ratings: dict[str, Any], keys: set[str]) -> dict[str, dict[str, float]]:
    values = {key: {axis: [] for axis in AXES} for key in keys}
    for evaluator in ratings["evaluators"].values():
        for key, row in evaluator.items():
            if key in values:
                for axis in AXES:
                    values[key][axis].append(row[axis])
    missing = sorted(key for key, axes in values.items() if any(not axes[axis] for axis in AXES))
    if missing:
        raise ValueError(f"incomplete ratings: {', '.join(missing)}")
    return {
        key: {axis: sum(axis_values) / len(axis_values) for axis, axis_values in axes.items()}
        for key, axes in values.items()
    }


def _top_five(dataset: Any, query_id: str, variant: str) -> list[str]:
    values = []
    for pair in dataset.visible_pairs("development"):
        if pair["query_id"] != query_id:
            continue
        for contribution in pair["contributions"]:
            if contribution["run_id"] == variant and contribution["rank"] <= 5:
                values.append((contribution["rank"], pair["key"]))
    return [key for _, key in sorted(values)]


def metrics(dataset: Any, assignment: set[str], ratings: dict[str, Any]) -> dict[str, Any]:
    values = aggregate_ratings(ratings, assignment)
    query_ids = sorted({dataset.pairs[key]["query_id"] for key in assignment})
    report: dict[str, Any] = {"query_ids": query_ids, "variants": {}}
    for variant in VARIANTS:
        per_query = {}
        for query_id in query_ids:
            ranking = _top_five(dataset, query_id, variant)
            if len(ranking) != 5 or any(key not in assignment for key in ranking):
                raise ValueError(f"incomplete top five provenance for {variant} {query_id}")
            judged = [
                values[key] for key in assignment if dataset.pairs[key]["query_id"] == query_id
            ]
            per_query[query_id] = {
                "ranking": ranking,
                "metrics": {
                    axis: {
                        "ndcg_at_5": ndcg_at_five(
                            [values[key][axis] for key in ranking], [row[axis] for row in judged]
                        ),
                        "precision_at_5": sum(values[key][axis] >= 2 for key in ranking) / 5,
                        "mean_top_5_rating": sum(values[key][axis] for key in ranking) / 5,
                        "irrelevant_top_5_count": sum(values[key][axis] <= 1 for key in ranking)
                        if axis == "situation"
                        else None,
                    }
                    for axis in AXES
                },
            }
        aggregate = {
            axis: {
                metric: sum(per_query[query]["metrics"][axis][metric] for query in query_ids)
                / len(query_ids)
                for metric in ("ndcg_at_5", "precision_at_5", "mean_top_5_rating")
            }
            for axis in AXES
        }
        aggregate["irrelevant_top_5_recommendations"] = sum(
            per_query[query]["metrics"]["situation"]["irrelevant_top_5_count"]
            for query in query_ids
        )
        report["variants"][variant] = {"aggregate": aggregate, "per_query": per_query}
    baseline = report["variants"]["A0"]["per_query"]
    for variant in VARIANTS:
        report["variants"][variant]["per_query_difference_vs_a0"] = {
            query: {
                axis: {
                    metric: report["variants"][variant]["per_query"][query]["metrics"][axis][metric]
                    - baseline[query]["metrics"][axis][metric]
                    for metric in ("ndcg_at_5", "precision_at_5", "mean_top_5_rating")
                }
                for axis in AXES
            }
            for query in query_ids
        }
    return report


def failure_analysis(
    dataset: Any, metric_report: dict[str, Any], ratings: dict[str, Any]
) -> dict[str, Any]:
    keys = {
        key
        for variant in metric_report["variants"].values()
        for query in variant["per_query"].values()
        for key in query["ranking"]
    }
    values = aggregate_ratings(ratings, keys)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen = Counter()
    for variant, variant_values in metric_report["variants"].items():
        for query_id, query in variant_values["per_query"].items():
            for rank, key in enumerate(query["ranking"], start=1):
                value = values[key]
                row = {
                    "variant": variant,
                    "query_id": query_id,
                    "source_key": dataset.pairs[key]["meme_id"],
                    "rank": rank,
                    **value,
                }
                seen[row["source_key"]] += 1
                if value["tone"] >= 2 and value["situation"] <= 1:
                    groups["good_tone_poor_situation"].append(row)
                if value["situation"] >= 2 and value["sendability"] <= 1:
                    groups["good_situation_poor_sendability"].append(row)
                if query_id == "reaction-02" and value["situation"] <= 1:
                    groups["reaction_intent_low_situation"].append(row)
                if value["situation"] == 3 and value["tone"] >= 2 and value["sendability"] >= 2:
                    groups["strong_recommendations"].append(row)
    groups["frequently_repeated_identities"] = [
        {"source_key": key, "top_five_appearances": count}
        for key, count in seen.most_common()
        if count > 1
    ]
    return {
        name: sorted(
            rows,
            key=lambda row: (row.get("query_id", ""), row.get("rank", 0), row.get("variant", "")),
        )
        if name != "frequently_repeated_identities"
        else rows
        for name, rows in groups.items()
    }


def run(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite analysis directory: {output}")
    dataset = load_pool_dataset(QUERIES, POOL)
    assignment = load_assignment(ASSIGNMENT, dataset)
    ratings_path = find_completed_ratings(EVALUATION, dataset, assignment)
    ratings = load_benchmark_ratings(ratings_path, dataset)
    values = aggregate_ratings(ratings, assignment)
    metric_report = metrics(dataset, assignment, ratings)
    report = {
        "format_version": 1,
        "experiment": "phase7r.5d-final-intent-human-evaluation-analysis",
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "eight preregistered fresh development queries only; read-only rating analysis",
        "ratings_path": str(ratings_path),
        "rating_validation": {
            "status": "PASS",
            "assigned_pairs": len(assignment),
            "complete_pairs": len(values),
            "missing_pairs": [],
            "evaluators": sorted(ratings["evaluators"]),
            "unjudged_assigned_pairs": 0,
            "stable_identity_format": all(
                key == f"{dataset.pairs[key]['query_id']}:{dataset.pairs[key]['meme_id']}"
                for key in assignment
            ),
        },
        "metrics": metric_report,
        "failures": failure_analysis(dataset, metric_report, ratings),
        "uncertainty": "Eight development queries and one evaluator support descriptive comparison only; no statistical superiority or architecture winner is claimed.",
    }
    output.mkdir(parents=True)
    (output / "final_intent_evaluation_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze completed Phase 7R.5C ratings")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = (
        args.output
        or BENCHMARKS / f"phase7r_v1_intent_human_eval_analysis_{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    )
    report = run(output)
    print(
        f"PASS | {report['rating_validation']['complete_pairs']}/{report['rating_validation']['assigned_pairs']} complete rating pairs"
    )
    for variant, values in report["metrics"]["variants"].items():
        print(f"{variant}: situation nDCG@5={values['aggregate']['situation']['ndcg_at_5']:.3f}")
    print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
