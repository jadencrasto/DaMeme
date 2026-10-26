"""Read-only Phase 7R.4E.2 analysis of completed comparative human ratings."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.benchmark import load_assignment, load_benchmark_ratings, load_pool_dataset
from app.recommendations import MemeMatchMetadata

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
BENCHMARKS = BACKEND / "benchmarks"
PREPARATION = BENCHMARKS / "phase7r_v1_human_eval_preparation_20261025T055104Z"
QUERIES = BENCHMARKS / "phase7r_v1_queries.json"
POOL = PREPARATION / "development.pool.json"
ASSIGNMENT = PREPARATION / "comparative.assignment.json"
RATINGS = PREPARATION / "ratings.seed.json"
ORIGINAL_PILOT_RATINGS = BENCHMARKS / "phase7r_v1_pilot.ratings.json"
COMPARISON = BENCHMARKS / "phase7r_v1_local_global_comparison_20261025T054251Z" / "results.json"
CATALOG = ROOT / "data" / "retrieval" / "catalogs" / "memematch_1000_v2"
VARIANTS = ("A0", "A1", "A2", "A3-EQ", "A3-LH", "A3-RRF")
AXES = ("situation", "tone", "sendability")


def dcg(values: list[float]) -> float:
    return sum(value / math.log2(position + 1) for position, value in enumerate(values, start=1))


def ndcg_at_five(values: list[float], judged_values: list[float]) -> float:
    ideal = dcg(sorted(judged_values, reverse=True)[:5])
    return dcg(values[:5]) / ideal if ideal else 0.0


def _mean_axes(ratings: dict[str, Any], keys: set[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for axis in AXES:
        values = [
            entry[axis]
            for evaluator in ratings["evaluators"].values()
            for key, entry in evaluator.items()
            if key in keys
        ]
        result[axis] = sum(values) / len(values) if values else 0.0
    return result


def aggregate_ratings(ratings: dict[str, Any], keys: set[str]) -> dict[str, dict[str, float]]:
    aggregated: dict[str, dict[str, list[int]]] = {key: {axis: [] for axis in AXES} for key in keys}
    for evaluator in ratings["evaluators"].values():
        for key, entry in evaluator.items():
            if key in aggregated:
                for axis in AXES:
                    aggregated[key][axis].append(entry[axis])
    missing = sorted(
        key for key, axes in aggregated.items() if any(not axes[axis] for axis in AXES)
    )
    if missing:
        raise ValueError(f"incomplete ratings for: {', '.join(missing)}")
    return {
        key: {axis: sum(values) / len(values) for axis, values in axes.items()}
        for key, axes in aggregated.items()
    }


def validate_completed_ratings(
    dataset: Any, assignment_keys: set[str], ratings: dict[str, Any], original_pilot: dict[str, Any]
) -> dict[str, Any]:
    aggregate_ratings(ratings, assignment_keys)
    all_pairs = set(dataset.pairs)
    invalid = sorted(
        key
        for evaluator in ratings["evaluators"].values()
        for key in evaluator
        if key not in all_pairs
    )
    pilot_preserved = all(
        ratings["evaluators"].get(evaluator, {}).get(key) == axes
        for evaluator, values in original_pilot["evaluators"].items()
        for key, axes in values.items()
    )
    per_evaluator = {
        evaluator: len(set(values) & assignment_keys)
        for evaluator, values in ratings["evaluators"].items()
    }
    return {
        "status": "PASS" if not invalid and pilot_preserved else "FAIL",
        "required_pairs": len(assignment_keys),
        "complete_pairs": len(assignment_keys),
        "missing_pairs": [],
        "invalid_rating_keys": invalid,
        "stable_identity_format": all(
            key == f"{dataset.pairs[key]['query_id']}:{dataset.pairs[key]['meme_id']}"
            for key in assignment_keys
        ),
        "pilot_ratings_preserved": pilot_preserved,
        "assignment_ratings_by_evaluator": per_evaluator,
        "rating_axes": list(AXES),
    }


def _variant_top_five(dataset: Any, query_id: str, variant: str) -> list[str]:
    rows = []
    for pair in dataset.visible_pairs("development"):
        if pair["query_id"] != query_id:
            continue
        for contribution in pair["contributions"]:
            if contribution["run_id"] == variant and contribution["rank"] <= 5:
                rows.append((contribution["rank"], pair["key"]))
    return [key for _, key in sorted(rows)]


def ranking_metrics(
    dataset: Any, assignment_keys: set[str], ratings: dict[str, Any]
) -> dict[str, Any]:
    values = aggregate_ratings(ratings, assignment_keys)
    query_ids = sorted({dataset.pairs[key]["query_id"] for key in assignment_keys})
    per_variant: dict[str, Any] = {}
    for variant in VARIANTS:
        per_query: dict[str, Any] = {}
        for query_id in query_ids:
            ranking = _variant_top_five(dataset, query_id, variant)
            if len(ranking) != 5 or any(key not in assignment_keys for key in ranking):
                raise ValueError(f"{variant} lacks a fully judged top five for {query_id}")
            judged = [
                values[key] for key in assignment_keys if dataset.pairs[key]["query_id"] == query_id
            ]
            axes = {
                axis: {
                    "ndcg_at_5": ndcg_at_five(
                        [values[key][axis] for key in ranking], [item[axis] for item in judged]
                    ),
                    "precision_at_5": sum(values[key][axis] >= 2 for key in ranking) / 5,
                    "mean_top_5_rating": sum(values[key][axis] for key in ranking) / 5,
                }
                for axis in AXES
            }
            per_query[query_id] = {"ranking": ranking, "metrics": axes}
        aggregate = {
            axis: {
                metric: sum(per_query[query]["metrics"][axis][metric] for query in query_ids)
                / len(query_ids)
                for metric in ("ndcg_at_5", "precision_at_5", "mean_top_5_rating")
            }
            for axis in AXES
        }
        per_variant[variant] = {"macro_metrics": aggregate, "per_query": per_query}
    baseline = per_variant["A0"]["per_query"]
    for variant in VARIANTS:
        per_variant[variant]["per_query_difference_vs_a0"] = {
            query_id: {
                axis: {
                    metric: per_variant[variant]["per_query"][query_id]["metrics"][axis][metric]
                    - baseline[query_id]["metrics"][axis][metric]
                    for metric in ("ndcg_at_5", "precision_at_5", "mean_top_5_rating")
                }
                for axis in AXES
            }
            for query_id in query_ids
        }
    return {"query_ids": query_ids, "metrics": per_variant}


def relevance_failures(
    dataset: Any, metric_data: dict[str, Any], ratings: dict[str, Any]
) -> dict[str, Any]:
    keys = {
        key
        for variant in metric_data["metrics"].values()
        for query in variant["per_query"].values()
        for key in query["ranking"]
    }
    values = aggregate_ratings(ratings, keys)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for variant, variant_data in metric_data["metrics"].items():
        for query_id, query in variant_data["per_query"].items():
            for rank, key in enumerate(query["ranking"], start=1):
                row = {
                    "variant": variant,
                    "query_id": query_id,
                    "source_key": dataset.pairs[key]["meme_id"],
                    "rank": rank,
                    **values[key],
                }
                if row["situation"] <= 1:
                    groups["low_situation_top_five"].append(row)
                if row["tone"] >= 2 and row["situation"] <= 1:
                    groups["good_tone_poor_situation"].append(row)
                if row["tone"] >= 2 and row["situation"] >= 2 and row["sendability"] <= 1:
                    groups["contextually_appropriate_not_sendable"].append(row)
                if row["situation"] == 3:
                    groups["strongly_relevant"].append(row)
    return {
        name: sorted(
            rows, key=lambda row: (row["rank"], row["query_id"], row["variant"], row["source_key"])
        )
        for name, rows in groups.items()
    }


def repetition_audit(
    comparison: dict[str, Any], dataset: Any, assignment_keys: set[str]
) -> dict[str, Any]:
    metadata = MemeMatchMetadata.from_csv(
        CATALOG / "memematch_1000_metadata_v2.csv", CATALOG / "images"
    )
    audit: dict[str, Any] = {"variants": {}, "ui_query_repetition": {}}
    for variant in VARIANTS:
        occurrences: dict[int, list[tuple[str, str, str]]] = {5: [], 10: []}
        raw_template_duplicates = final_template_duplicates = 0
        within_list_meme_duplicates = 0
        for query in comparison["queries"]:
            final = query["variants"][variant]["final_top_10"]
            raw = query["variants"][variant]["raw_candidates"][:10]
            for cutoff in occurrences:
                occurrences[cutoff].extend(
                    (
                        row["source_key"],
                        query["query_id"],
                        dataset.queries[query["query_id"]].category,
                    )
                    for row in final[:cutoff]
                )
            raw_templates = [
                metadata.get(row["source_key"]).template_group or row["source_key"] for row in raw
            ]
            final_templates = [
                metadata.get(row["source_key"]).template_group or row["source_key"] for row in final
            ]
            raw_template_duplicates += len(raw_templates) - len(set(raw_templates))
            final_template_duplicates += len(final_templates) - len(set(final_templates))
            within_list_meme_duplicates += len(final) - len({row["source_key"] for row in final})
        cutoff_result = {}
        for cutoff, rows in occurrences.items():
            meme_counts = Counter(row[0] for row in rows)
            template_counts = Counter(
                (metadata.get(key).template_group if metadata.get(key) else None) or key
                for key, _, _ in rows
            )
            categories = defaultdict(set)
            for key, _, category in rows:
                categories[key].add(category)
            repeated_cross_category = {
                key: count
                for key, count in meme_counts.items()
                if count > 1 and len(categories[key]) > 1
            }
            cutoff_result[f"top_{cutoff}"] = {
                "total_result_slots": len(rows),
                "unique_memes": len(meme_counts),
                "unique_templates": len(template_counts),
                "cross_query_repetition_rate": 1 - len(meme_counts) / len(rows),
                "most_frequent_memes": meme_counts.most_common(20),
                "most_frequent_templates": template_counts.most_common(20),
                "memes_repeated_across_categories": repeated_cross_category,
            }
        audit["variants"][variant] = {
            **cutoff_result,
            "within_result_list_duplicate_memes": within_list_meme_duplicates,
            "raw_top_10_extra_template_slots_before_diversity": raw_template_duplicates,
            "final_top_10_extra_template_slots_after_diversity": final_template_duplicates,
        }
    audit["ui_query_repetition"] = {
        "assignment_pair_tasks_by_query": dict(
            sorted(Counter(dataset.pairs[key]["query_id"] for key in assignment_keys).items())
        ),
        "interpretation": "Repeated query presentation is an evaluation-UI consequence of showing multiple query--meme pairs, not repeated meme retrieval within a list.",
    }
    return audit


def compact_summary(report: dict[str, Any]) -> dict[str, Any]:
    def compact_cutoff(values: dict[str, Any]) -> dict[str, Any]:
        return {
            **{key: values[key] for key in ("unique_memes", "unique_templates", "cross_query_repetition_rate")},
            "memes_repeated_across_categories": len(values["memes_repeated_across_categories"]),
        }

    return {
        "rating_validation": report["rating_validation"],
        "macro_metrics": {
            variant: data["macro_metrics"]
            for variant, data in report["ranking_metrics"]["metrics"].items()
        },
        "failure_counts": {
            name: len(rows) for name, rows in report["relevance_failures"].items()
        },
        "repetition": {
            variant: {
                "top_5": compact_cutoff(audit["top_5"]),
                "top_10": compact_cutoff(audit["top_10"]),
                "raw_top_10_extra_template_slots_before_diversity": audit["raw_top_10_extra_template_slots_before_diversity"],
                "final_top_10_extra_template_slots_after_diversity": audit["final_top_10_extra_template_slots_after_diversity"],
            }
            for variant, audit in report["repetition_audit"]["variants"].items()
        },
        "ui_query_repetition": report["repetition_audit"]["ui_query_repetition"],
        "uncertainty": report["uncertainty"],
    }


def query_processing_audit(
    comparison: dict[str, Any], metric_data: dict[str, Any]
) -> dict[str, Any]:
    selected = set(metric_data["query_ids"])
    rows = []
    for query in comparison["queries"]:
        if query["query_id"] in selected:
            rows.append(
                {
                    "query_id": query["query_id"],
                    "original_query": query["message"],
                    "mood": query["mood"],
                    "mood_expanded_query": query["effective_query"],
                    "retrieval_query_sent_to_minilm": query["effective_query"],
                    "expansion_added": query["effective_query"] != query["message"],
                }
            )
    return {
        "selected_queries": rows,
        "ranking_logic": "semantic score is ranked with default text weights (semantic 0.90, primary mood emotion 0.08, sentiment 0.02 when supplied); this run supplies no sentiment. Source-text safety filtering precedes one-per-template diversity filtering.",
        "finding": "The audit records deterministic mood-description suffixes exactly. It does not infer whether a suffix is semantically unrelated from ratings alone; that requires a separately designed rewrite-fidelity judgment.",
    }


def run(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing analysis directory: {output}")
    dataset = load_pool_dataset(QUERIES, POOL)
    assignment_keys = load_assignment(ASSIGNMENT, dataset)
    ratings = load_benchmark_ratings(RATINGS, dataset)
    original_pilot = json.loads(ORIGINAL_PILOT_RATINGS.read_text(encoding="utf-8"))
    validation = validate_completed_ratings(dataset, assignment_keys, ratings, original_pilot)
    if validation["status"] != "PASS":
        raise ValueError(f"rating validation failed: {validation}")
    metrics = ranking_metrics(dataset, assignment_keys, ratings)
    comparison = json.loads(COMPARISON.read_text(encoding="utf-8"))
    report = {
        "format_version": 1,
        "experiment": "phase7r.4e.2-human-evaluation-analysis",
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "eight preregistered development queries only; no held-out evaluation",
        "rating_validation": validation,
        "ranking_metrics": metrics,
        "relevance_failures": relevance_failures(dataset, metrics, ratings),
        "repetition_audit": repetition_audit(comparison, dataset, assignment_keys),
        "query_processing_audit": query_processing_audit(comparison, metrics),
        "uncertainty": "Eight development queries and one primary evaluator are insufficient for statistical superiority or architecture selection.",
    }
    output.mkdir(parents=True)
    (output / "evaluation.metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (output / "analysis.summary.json").write_text(
        json.dumps(compact_summary(report), indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze completed Phase 7R.4E.2 ratings")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = (
        args.output
        or BENCHMARKS / f"phase7r_v1_human_eval_analysis_{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    )
    report = run(output)
    print(
        f"{report['rating_validation']['status']} | {report['rating_validation']['complete_pairs']}/{report['rating_validation']['required_pairs']} complete rating pairs"
    )
    for variant, values in report["ranking_metrics"]["metrics"].items():
        print(
            f"{variant}: situation nDCG@5={values['macro_metrics']['situation']['ndcg_at_5']:.3f}"
        )
    print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
