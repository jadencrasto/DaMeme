"""Read-only impact audit for the frozen Phase 7R.7A reranking output."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
BENCHMARKS = BACKEND / "benchmarks"
QUERIES = BENCHMARKS / "phase7r_v1_queries.json"
SOURCE = BENCHMARKS / "phase7r_v1_situation_reranking_20261026T000001Z" / "comparison.json"
POOL = BENCHMARKS / "phase7r_v1_multimodal_human_eval_20261025T075000Z" / "blinded.pool.json"
RATINGS = BENCHMARKS / "phase7r_v1_multimodal_human_eval_20261025T075000Z" / "ratings.multimodal-evaluator-1.json"
METADATA = ROOT / "data" / "retrieval" / "catalogs" / "memematch_1000_v2" / "memematch_1000_metadata_v2.csv"
AXES = ("situation", "tone", "sendability")
PAIRS = (("R-T", "A3-LH"), ("R-V", "M1"))
AUDIT_GENERIC_TERMS = frozenset(("made", "make", "made", "thing", "things", "format", "post"))


def top_five_impact(baseline: list[str], reranked: list[str]) -> dict[str, Any]:
    """Separate top-five membership changes from unchanged-set reordering."""
    before, after = baseline[:5], reranked[:5]
    entering = [key for key in after if key not in before]
    leaving = [key for key in before if key not in after]
    rank_changes = [
        {"source_key": key, "baseline_rank": before.index(key) + 1, "reranked_rank": after.index(key) + 1}
        for key in after
        if key in before and before.index(key) != after.index(key)
    ]
    return {
        "top_five_order_changed": before != after,
        "top_five_membership_changed": bool(entering or leaving),
        "entering": entering,
        "leaving": leaving,
        "overlap": len(set(before) & set(after)) / 5,
        "within_top_five_swaps": rank_changes,
        "rank_one_changed": before[0] != after[0],
    }


def _dcg(values: list[float]) -> float:
    return sum((2**value - 1) / math.log2(position + 2) for position, value in enumerate(values))


def judged_metrics(
    ranking: list[str], all_pairs: list[str], ratings: dict[str, dict[str, int]]
) -> dict[str, Any] | None:
    """Use nDCG@5/P@5 only when the full comparison pool is rated for each axis."""
    if len(ranking) < 5:
        return None
    metrics: dict[str, Any] = {}
    for axis in AXES:
        if any(key not in ratings or axis not in ratings[key] for key in [*all_pairs, *ranking[:5]]):
            return None
        values = [float(ratings[key][axis]) for key in ranking[:5]]
        ideal = sorted((float(ratings[key][axis]) for key in all_pairs), reverse=True)[:5]
        denominator = _dcg(ideal)
        metrics[axis] = {
            "ndcg_at_5": _dcg(values) / denominator if denominator else 0.0,
            "precision_at_5": sum(value >= 2 for value in values) / 5,
        }
    return metrics


def _metadata() -> dict[str, dict[str, str]]:
    with METADATA.open(encoding="utf-8-sig", newline="") as handle:
        return {row["source_key"]: row for row in csv.DictReader(handle)}


def _audit_evidence(rows: list[dict[str, Any]], metadata: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    changed = []
    for row in rows:
        for reranked, baseline in PAIRS:
            baseline_by_key = {
                item["source_key"]: rank
                for rank, item in enumerate(row["variants"][baseline]["final_top_10"], start=1)
            }
            for item in row["variants"][reranked]["final_top_10"]:
                if not item["rank_displacement"]:
                    continue
                adjustment = item["situation_adjustment"]
                key = item["source_key"]
                source = metadata[key]
                changed.append(
                    {
                        "variant": reranked,
                        "baseline": baseline,
                        "query_id": row["query_id"],
                        "original_query": row["original_query"],
                        "source_key": key,
                        "baseline_rank": baseline_by_key[key],
                        "reranked_rank": item["reranked_rank"],
                        "rank_displacement": item["rank_displacement"],
                        "matched_terms": adjustment["matched_terms"],
                        "negated_terms": adjustment["negated_terms"],
                        "reason": adjustment["reason"],
                        "contains_audit_generic_term": bool(set(adjustment["matched_terms"]) & AUDIT_GENERIC_TERMS),
                        "source_evidence": {
                            name: source.get(name, "")
                            for name in (
                                "local_Extracted Text + Title",
                                "local_context",
                                "global_context",
                                "global_Caption",
                                "usage",
                            )
                        },
                    }
                )
    # One deterministic sample of each movement direction, spread across variants.
    return sorted(changed, key=lambda item: (item["variant"], item["query_id"], item["source_key"]))


def _reranker_comparison(rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare frozen reranker versions without consulting ratings."""
    reference_by_id = {str(row["query_id"]): row for row in reference_rows}
    if {str(row["query_id"]) for row in rows} != set(reference_by_id):
        raise ValueError("reranker comparison requires matching query identities")
    result: dict[str, Any] = {}
    for reranked, _ in PAIRS:
        events = []
        for row in rows:
            original = [item["source_key"] for item in reference_by_id[row["query_id"]]["variants"][reranked]["final_top_10"]]
            revised = [item["source_key"] for item in row["variants"][reranked]["final_top_10"]]
            events.append(top_five_impact(original, revised))
        result[reranked] = {
            "queries_with_order_change_from_original_reranker": sum(event["top_five_order_changed"] for event in events),
            "queries_with_membership_change_from_original_reranker": sum(event["top_five_membership_changed"] for event in events),
            "mean_top_five_overlap_with_original_reranker": sum(event["overlap"] for event in events) / len(events),
            "rank_one_changes_from_original_reranker": sum(event["rank_one_changed"] for event in events),
        }
    return result


def run(output: Path, *, source_path: Path = SOURCE, reference_path: Path | None = None) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing audit directory: {output}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    rows = source["queries"]
    pool_payload = json.loads(POOL.read_text(encoding="utf-8"))
    if pool_payload.get("format_version") != 1 or pool_payload.get("split") != "development":
        raise ValueError("unexpected multimodal development pool")
    pool_pairs = pool_payload.get("pairs")
    if not isinstance(pool_pairs, list):
        raise ValueError("multimodal pool has no pairs")
    ratings_payload = json.loads(RATINGS.read_text(encoding="utf-8"))
    if ratings_payload.get("format_version") != 1 or not isinstance(ratings_payload.get("evaluators"), dict):
        raise ValueError("unexpected multimodal ratings")
    evaluators = list(ratings_payload["evaluators"].values())
    ratings = {
        key: {axis: round(sum(entry[key][axis] for entry in evaluators if key in entry) / sum(key in entry for entry in evaluators), 6) for axis in AXES}
        for key in (str(pair["key"]) for pair in pool_pairs)
        if any(key in entry for entry in evaluators)
    }
    pairs_by_query: dict[str, list[str]] = {}
    for pair in pool_pairs:
        pairs_by_query.setdefault(str(pair["query_id"]), []).append(str(pair["key"]))

    report: dict[str, Any] = {
        "format_version": 1,
        "experiment": "phase7r.7b-situation-reranking-impact-audit",
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "read-only development analysis of frozen Phase 7R.7A output",
        "provenance": {"reranking_comparison": str(source_path), "pool": str(POOL), "ratings": str(RATINGS)},
        "variants": {},
    }
    for reranked, baseline in PAIRS:
        events, displacements, changed_top5_keys = [], [], set()
        judged_comparisons = []
        low_quality_remaining = []
        for row in rows:
            before = [item["source_key"] for item in row["variants"][baseline]["final_top_10"]]
            after_rows = row["variants"][reranked]["final_top_10"]
            after = [item["source_key"] for item in after_rows]
            impact = top_five_impact(before, after)
            changed_rows = [item for item in after_rows if item["rank_displacement"] and item["reranked_rank"] <= 5]
            changed_top5_keys.update(f"{row['query_id']}:{item['source_key']}" for item in changed_rows)
            displacements.extend(abs(int(item["rank_displacement"])) for item in after_rows if item["rank_displacement"])
            query_pairs = pairs_by_query.get(row["query_id"], [])
            baseline_metrics = judged_metrics([f"{row['query_id']}:{key}" for key in before], query_pairs, ratings)
            reranked_metrics = judged_metrics([f"{row['query_id']}:{key}" for key in after], query_pairs, ratings)
            if baseline_metrics is not None and reranked_metrics is not None:
                judged_comparisons.append({"query_id": row["query_id"], "baseline": baseline_metrics, "reranked": reranked_metrics})
            for key in after[:5]:
                rating = ratings.get(f"{row['query_id']}:{key}")
                if rating is not None and rating["situation"] <= 1:
                    low_quality_remaining.append({"query_id": row["query_id"], "source_key": key, "ratings": rating})
            events.append({"query_id": row["query_id"], **impact})
        report["variants"][reranked] = {
            "baseline": baseline,
            "top_five": {
                "queries_with_any_rank_order_change": sum(event["top_five_order_changed"] for event in events),
                "queries_with_membership_change": sum(event["top_five_membership_changed"] for event in events),
                "total_entering_candidates": sum(len(event["entering"]) for event in events),
                "total_leaving_candidates": sum(len(event["leaving"]) for event in events),
                "mean_overlap": sum(event["overlap"] for event in events) / len(events),
                "within_top_five_rank_changes": sum(len(event["within_top_five_swaps"]) for event in events),
                "within_top_five_swap_events": sum(
                    len(event["within_top_five_swaps"]) // 2 for event in events
                ),
                "queries_with_rank_one_change": sum(event["rank_one_changed"] for event in events),
                "events": events,
            },
            "rank_displacement": {
                "maximum": max(displacements, default=0),
                "mean_among_changed_candidates": sum(displacements) / len(displacements) if displacements else 0.0,
            },
            "existing_judgments": {
                "changed_top_five_pairs": len(changed_top5_keys),
                "changed_top_five_pairs_already_judged": sum(key in ratings for key in changed_top5_keys),
                "fully_covered_queries_for_descriptive_metrics": len(judged_comparisons),
                "descriptive_comparisons": judged_comparisons,
                "previously_rated_low_situation_remaining_in_top_five": low_quality_remaining,
            },
        }
    audit = _audit_evidence(rows, _metadata())
    fallback_queries = []
    for row in rows:
        for reranked, _ in PAIRS:
            entries = row["variants"][reranked]["final_top_10"]
            if all(not item["situation_adjustment"]["matched_terms"] for item in entries):
                fallback_queries.append(
                    {
                        "variant": reranked,
                        "query_id": row["query_id"],
                        "original_query": row["original_query"],
                        "reason": entries[0]["situation_adjustment"]["reason"],
                        "order_unchanged": all(not item["rank_displacement"] for item in entries),
                    }
                )
    report["heuristic_evidence_audit"] = {
        "all_changed_candidates": audit,
        "matched_term_counts_on_changed_candidates": dict(
            Counter(term for item in audit for term in item["matched_terms"])
        ),
        "changed_candidates_with_audit_generic_terms": sum(item["contains_audit_generic_term"] for item in audit),
        "changed_candidates_with_negated_query_terms": sum(bool(item["negated_terms"]) for item in audit),
        "no_match_final_candidate_sets": fallback_queries,
    }
    if reference_path is not None:
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        report["comparison_with_original_reranker"] = {
            "original_reranker_comparison": str(reference_path),
            "variants": _reranker_comparison(rows, reference["queries"]),
        }
    output.mkdir(parents=True)
    (output / "top_five_impact.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "qualitative_audit.json").write_text(
        json.dumps(report["heuristic_evidence_audit"], indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit frozen Phase 7R.7A top-five impact")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args(argv)
    output = args.output or BENCHMARKS / f"phase7r_v1_situation_reranking_impact_{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    report = run(output, source_path=args.source, reference_path=args.reference)
    for name, value in report["variants"].items():
        top_five = value["top_five"]
        print(f"{name}: order={top_five['queries_with_any_rank_order_change']}; membership={top_five['queries_with_membership_change']}")
    print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
