"""Prepare blinded, development-only human evaluation for Phase 7R.5C."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.benchmark import (
    POOL_FORMAT_VERSION,
    canonical_digest,
    load_assignment,
    load_pool_dataset,
)

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
BENCHMARKS = BACKEND / "benchmarks"
QUERIES = BENCHMARKS / "phase7r_v1_queries.json"
COMPARISON_DIR = BENCHMARKS / "phase7r_v1_intent_comparison_20261025T065302Z"
COMPARISON = COMPARISON_DIR / "comparison.results.json"
EXPORT = COMPARISON_DIR / "fresh_human_evaluation_candidates.json"
SOURCE_POOL = (
    BENCHMARKS / "phase7r_v1_human_eval_preparation_20261025T055104Z" / "development.pool.json"
)
SEED = 20261025
VARIANTS = ("A0", "A3-LH", "A3-LH-INTENT")


def _save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _assignment(
    pool: dict[str, Any], keys: set[str], name: str, description: str
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "assignment": name,
        "pool_digest": canonical_digest(pool),
        "split": "development",
        "pair_keys": sorted(keys),
        "presentation_seed": SEED,
        "description": description,
    }


def _pool_from_export(source_pool: dict[str, Any], export: dict[str, Any]) -> dict[str, Any]:
    pairs = []
    for row in export["pairs"]:
        contributions = [
            {
                "run_id": item["variant"],
                "rank": item["rank"],
                "raw_rank": item["raw_rank"],
                "raw_score": item["raw_score"],
                "final_score": item["final_score"],
            }
            for item in row["contributors"]
        ]
        pairs.append(
            {
                "key": row["key"],
                "query_id": row["query_id"],
                "meme_id": row["source_key"],
                "contributions": sorted(
                    contributions, key=lambda item: (item["run_id"], item["rank"])
                ),
            }
        )
    runs = [
        {
            "id": variant,
            "split": "development",
            "model_version": "sentence-transformers/all-MiniLM-L6-v2",
            "configuration": {"experiment": "phase7r.5b-conservative-intent-retrieval"},
        }
        for variant in VARIANTS
    ]
    return {
        "format_version": POOL_FORMAT_VERSION,
        "benchmark_id": source_pool["benchmark_id"],
        "query_digest": source_pool["query_digest"],
        "catalog": source_pool["catalog"],
        "runs": runs,
        "pairs": sorted(pairs, key=lambda pair: pair["key"]),
    }


def _variant_top_five(record: dict[str, Any], variant: str) -> set[str]:
    return {
        item["source_key"]
        for item in record["variants"][variant]["final_top_10"]
        if item["filtered_rank"] <= 5
    }


def prepare() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_pool = json.loads(SOURCE_POOL.read_text(encoding="utf-8"))
    export = json.loads(EXPORT.read_text(encoding="utf-8"))
    comparison = json.loads(COMPARISON.read_text(encoding="utf-8"))
    query_ids = export["query_ids"]
    if len(query_ids) != 8 or len(set(query_ids)) != 8:
        raise ValueError("fresh export must contain exactly eight unique queries")
    comparison_by_id = {row["query_id"]: row for row in comparison["queries"]}
    selected = [comparison_by_id[query_id] for query_id in query_ids]
    if any(row["intent"]["confidence"] not in {"high", "low", "medium"} for row in selected):
        raise ValueError("intent confidence is invalid")
    pool = _pool_from_export(source_pool, export)
    export_keys = {row["key"] for row in export["pairs"]}
    complete_keys, diagnostic_keys, details = set(), set(), []
    for row in selected:
        by_variant = {variant: _variant_top_five(row, variant) for variant in VARIANTS}
        complete = set().union(*by_variant.values())
        a3 = by_variant["A3-LH"]
        intent = by_variant["A3-LH-INTENT"]
        changed = a3.symmetric_difference(intent)
        intent_only = {
            item["source_key"] for item in row["variants"]["A3-LH-INTENT"]["final_top_10"]
        } - {item["source_key"] for item in row["variants"]["A3-LH"]["final_top_10"]}
        complete_keys.update(f"{row['query_id']}:{key}" for key in complete)
        diagnostic_keys.update(f"{row['query_id']}:{key}" for key in changed | intent_only)
        details.append(
            {
                "query_id": row["query_id"],
                "category": row["category"],
                "intent_confidence": row["intent"]["confidence"],
                "intent_labels": row["intent"]["intent_labels"],
                "a3_intent_identical_top_five": not changed,
                "changed_top_five_source_keys": sorted(changed),
                "intent_only_top_ten_source_keys": sorted(intent_only),
            }
        )
    if not complete_keys.issubset(export_keys) or not diagnostic_keys.issubset(export_keys):
        raise ValueError("assignment keys must be present in the fresh export")
    report = {
        "format_version": 1,
        "experiment": "phase7r.5c-focused-intent-human-evaluation",
        "scope": "fresh development-only identities; blinded UI; no ratings collected",
        "fresh_query_ids": query_ids,
        "fresh_query_categories": dict(Counter(row["category"] for row in selected)),
        "intent_affected_query_ids": [
            row["query_id"] for row in selected if row["intent"]["confidence"] == "high"
        ],
        "identical_top_five_query_ids": [
            row["query_id"] for row in details if row["a3_intent_identical_top_five"]
        ],
        "query_changes": details,
        "export_pair_count": len(export_keys),
        "diagnostic_rating_workload": len(diagnostic_keys),
        "complete_comparative_rating_workload": len(complete_keys),
        "full_export_rating_workload": len(export_keys),
        "full_export_justified": False,
        "full_export_rationale": "The focused question requires complete top-five unions. Rating all exported top-ten candidates would increase workload without being necessary for a paired top-five comparison.",
        "validation": {
            "duplicate_export_keys": len(export_keys) != len(export["pairs"]),
            "held_out_queries_present": False,
            "provenance_variants": list(VARIANTS),
            "existing_rating_reuse_count": 0,
        },
    }
    diagnostic = _assignment(
        pool,
        diagnostic_keys,
        "phase7r.5c-diagnostic",
        "Exploratory, selection-biased: changed A3-LH/intent candidates and intent-only top-ten candidates.",
    )
    complete = _assignment(
        pool,
        complete_keys,
        "phase7r.5c-complete-comparative",
        "Complete union of blinded top-five identities for A0, A3-LH, and A3-LH-INTENT across all preregistered fresh queries.",
    )
    return pool, diagnostic, complete, report


def run(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite preparation directory: {output}")
    pool, diagnostic, complete, report = prepare()
    output.mkdir(parents=True)
    _save(output / "development.pool.json", pool)
    _save(output / "diagnostic.assignment.json", diagnostic)
    _save(output / "complete_comparative.assignment.json", complete)
    _save(
        output / "ratings.seed.json",
        {"format_version": 1, "pool_digest": canonical_digest(pool), "evaluators": {}},
    )
    _save(output / "workload.report.json", report)
    dataset = load_pool_dataset(QUERIES, output / "development.pool.json")
    if (
        load_assignment(output / "diagnostic.assignment.json", dataset)
        != set(diagnostic["pair_keys"])
        or load_assignment(output / "complete_comparative.assignment.json", dataset)
        != set(complete["pair_keys"])
    ):
        raise ValueError("generated assignments are not loadable from generated pool")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare Phase 7R.5C intent-aware human evaluation"
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = (
        args.output
        or BENCHMARKS / f"phase7r_v1_intent_human_eval_{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    )
    report = run(output)
    print(f"Affected queries: {len(report['intent_affected_query_ids'])}/8")
    print(
        f"Diagnostic workload: {report['diagnostic_rating_workload']}; complete workload: {report['complete_comparative_rating_workload']}"
    )
    print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
