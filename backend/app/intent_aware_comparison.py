"""Offline A0/A3-LH/conservative-intent retrieval comparison for Phase 7R.5B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.benchmark import load_assignment, load_benchmark_ratings, load_pool_dataset, load_queries
from app.emotion import PretrainedFeatureExtractor
from app.moods import primary_classifier_emotion, resolve_mood
from app.query_intent import RULE_VERSION, interpret_query
from app.recommendations import (
    DiversityConfig,
    MemeMatchMetadata,
    RecommendationService,
    SafetyConfig,
)
from app.retrieval import RetrievalService, SearchResult

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
BENCHMARKS = BACKEND / "benchmarks"
QUERIES = BENCHMARKS / "phase7r_v1_queries.json"
COMPARISON = BENCHMARKS / "phase7r_v1_local_global_comparison_20261025T054251Z" / "results.json"
INDEXES = BENCHMARKS / "phase7r_v1_local_global_indexes"
CATALOG = ROOT / "data" / "retrieval" / "catalogs" / "memematch_1000_v2"
PREPARATION = BENCHMARKS / "phase7r_v1_human_eval_preparation_20261025T055104Z"
POOL = PREPARATION / "development.pool.json"
ASSIGNMENT = PREPARATION / "comparative.assignment.json"
RATINGS = PREPARATION / "ratings.seed.json"
CANDIDATE_DEPTH = 100
RESULT_LIMIT = 10
SEED = 20261025
VARIANTS = ("A0", "A3-LH", "A3-LH-INTENT")


def _hash(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def _ordered(scores: dict[str, float], limit: int) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]


def _scores(index: faiss.Index, query: np.ndarray, keys: list[str]) -> dict[str, float]:
    scores, positions = index.search(query.reshape(1, -1).astype(np.float32), index.ntotal)
    return {
        keys[int(position)]: float(score)
        for score, position in zip(scores[0], positions[0], strict=True)
        if position >= 0
    }


def fuse_local_heavy(
    local: dict[str, float], global_: dict[str, float], available: set[str]
) -> dict[str, float]:
    if set(local) != set(global_):
        raise ValueError("local/global source keys must match")
    return {
        key: 0.7 * local[key] + 0.3 * global_[key] if key in available else global_[key]
        for key in global_
    }


def retrieval_configuration() -> dict[str, Any]:
    return {
        "candidate_depth": CANDIDATE_DEPTH,
        "safety": "existing SafetyConfig",
        "diversity": "existing one-per-template DiversityConfig",
        "tie_break": "score then source key",
    }


def _rank(service: RecommendationService, query: Any, scores: dict[str, float]) -> dict[str, Any]:
    ordered = _ordered(scores, CANDIDATE_DEPTH)
    candidates = [SearchResult(key, rank, score) for rank, (key, score) in enumerate(ordered)]
    emotion = primary_classifier_emotion(resolve_mood(query.mood))
    safety = SafetyConfig()
    excluded = [
        row.meme_id
        for row in candidates
        if (metadata := service.metadata.get(row.meme_id)) is not None
        and safety.excludes(metadata.safety_text, emotion)
    ]
    ranked = service.rank_candidates(
        candidates, emotion=emotion, safety=safety, limit=CANDIDATE_DEPTH
    )
    final = service.apply_diversity(ranked, DiversityConfig(max_per_template=1), RESULT_LIMIT)
    raw_rank = {key: rank for rank, (key, _) in enumerate(ordered, start=1)}
    return {
        "raw_candidates": [
            {"source_key": key, "raw_rank": rank, "raw_score": score}
            for rank, (key, score) in enumerate(ordered, start=1)
        ],
        "final_top_10": [
            {
                "source_key": item.meme_id,
                "filtered_rank": rank,
                "raw_rank": raw_rank[item.meme_id],
                "raw_score": item.semantic_similarity,
                "final_score": item.final_score,
            }
            for rank, item in enumerate(final, start=1)
        ],
        "safety": {"excluded_source_keys": excluded, "after_safety": len(ranked)},
        "template_diversity": {
            "distinct_template_keys": len(
                {
                    (
                        service.metadata.get(item.meme_id).template_group
                        if service.metadata.get(item.meme_id)
                        else None
                    )
                    or item.meme_id
                    for item in final
                }
            )
        },
    }


def _rating_values(ratings: dict[str, Any], key: str) -> dict[str, float] | None:
    values = [
        ratings_by_evaluator[key]
        for ratings_by_evaluator in ratings["evaluators"].values()
        if key in ratings_by_evaluator
    ]
    if not values:
        return None
    return {
        axis: sum(value[axis] for value in values) / len(values)
        for axis in ("situation", "tone", "sendability")
    }


def _fresh_preregistered_queries(queries: dict[str, Any], rated_query_ids: set[str]) -> list[str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for query in queries.values():
        if query.split == "development" and query.id not in rated_query_ids:
            grouped[query.category].append(query.id)
    return [min(ids, key=_hash) for _, ids in sorted(grouped.items())]


def run(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing experiment directory: {output}")
    queries, _ = load_queries(QUERIES)
    development = {key: query for key, query in queries.items() if query.split == "development"}
    if len(development) != 40:
        raise ValueError("expected exactly 40 development queries")
    comparison = json.loads(COMPARISON.read_text(encoding="utf-8"))
    comparison_by_id = {row["query_id"]: row for row in comparison["queries"]}
    if set(comparison_by_id) != set(development):
        raise ValueError("comparison source must contain exactly development identities")
    with (INDEXES / "manifest.json").open(encoding="utf-8") as source:
        index_manifest = json.load(source)
    keys = list(index_manifest["source_keys"])
    local_available = {
        key
        for key, available in zip(keys, index_manifest["availability"]["local"], strict=True)
        if available
    }
    service = RecommendationService(
        RetrievalService.load(CATALOG / "indexes"),
        MemeMatchMetadata.from_csv(CATALOG / "memematch_1000_metadata_v2.csv", CATALOG / "images"),
    )
    local_index = faiss.read_index(str(INDEXES / "local.faiss"))
    global_index = faiss.read_index(str(INDEXES / "global.faiss"))
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    encoder = PretrainedFeatureExtractor("cpu")
    dataset = load_pool_dataset(QUERIES, POOL)
    assignment = load_assignment(ASSIGNMENT, dataset)
    ratings = load_benchmark_ratings(RATINGS, dataset)
    rated_query_ids = {dataset.pairs[key]["query_id"] for key in assignment}
    records = []
    for query_id, query in sorted(development.items()):
        source = comparison_by_id[query_id]
        base_representation = source["effective_query"]
        intent = interpret_query(query.message, base_representation, query.mood)
        started = time.perf_counter()
        base_embedding = encoder.embed_texts([base_representation])[0]
        base_encoding_ms = (time.perf_counter() - started) * 1000
        intent_encoding_ms = 0.0
        if intent.retrieval_representation == base_representation:
            intent_embedding = base_embedding
        else:
            started = time.perf_counter()
            intent_embedding = encoder.embed_texts([intent.retrieval_representation])[0]
            intent_encoding_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        a0_scores = {
            item.meme_id: item.score
            for item in service.retrieval.search("text", base_embedding, CANDIDATE_DEPTH)
        }
        a0 = _rank(service, query, a0_scores)
        a0_latency_ms = (time.perf_counter() - started) * 1000 + base_encoding_ms
        started = time.perf_counter()
        base_lh = fuse_local_heavy(
            _scores(local_index, base_embedding, keys),
            _scores(global_index, base_embedding, keys),
            local_available,
        )
        a3 = _rank(service, query, base_lh)
        a3_latency_ms = (time.perf_counter() - started) * 1000 + base_encoding_ms
        started = time.perf_counter()
        intent_lh = fuse_local_heavy(
            _scores(local_index, intent_embedding, keys),
            _scores(global_index, intent_embedding, keys),
            local_available,
        )
        intent_ranked = _rank(service, query, intent_lh)
        intent_latency_ms = (
            (time.perf_counter() - started) * 1000 + base_encoding_ms + intent_encoding_ms
        )
        records.append(
            {
                "query_id": query_id,
                "category": query.category,
                "original_query": query.message,
                "selected_mood": query.mood,
                "base_retrieval_representation": base_representation,
                "intent": intent.to_dict(),
                "variants": {
                    "A0": {**a0, "latency_ms": a0_latency_ms},
                    "A3-LH": {**a3, "latency_ms": a3_latency_ms},
                    "A3-LH-INTENT": {**intent_ranked, "latency_ms": intent_latency_ms},
                },
            }
        )
    report = {
        "format_version": 1,
        "experiment": "phase7r.5b-conservative-intent-retrieval",
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "development-only; no session penalty",
        "interpreter_rule_version": RULE_VERSION,
        "retrieval_configuration": retrieval_configuration(),
        "queries": records,
    }
    report["summary"] = _summary(records, assignment, ratings, dataset, development)
    export = _human_export(records, _fresh_preregistered_queries(development, rated_query_ids))
    output.mkdir(parents=True)
    (output / "comparison.results.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(report["summary"], indent=2) + "\n", encoding="utf-8"
    )
    (output / "fresh_human_evaluation_candidates.json").write_text(
        json.dumps(export, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _summary(
    records: list[dict[str, Any]],
    assignment: set[str],
    ratings: dict[str, Any],
    dataset: Any,
    development: dict[str, Any],
) -> dict[str, Any]:
    coverage = Counter(row["intent"]["confidence"] for row in records)
    a3_sets = {
        row["query_id"]: [entry["source_key"] for entry in row["variants"]["A3-LH"]["final_top_10"]]
        for row in records
    }
    intent_sets = {
        row["query_id"]: [
            entry["source_key"] for entry in row["variants"]["A3-LH-INTENT"]["final_top_10"]
        ]
        for row in records
    }
    result: dict[str, Any] = {
        "intent_coverage": dict(coverage),
        "fallback_count": sum(value for key, value in coverage.items() if key != "high"),
        "intent_added_count": coverage["high"],
        "variants": {},
    }
    for variant in VARIANTS:
        finals = [
            entry["source_key"]
            for row in records
            for entry in row["variants"][variant]["final_top_10"]
        ]
        latencies = [row["variants"][variant]["latency_ms"] for row in records]
        rated = [
            _rating_values(ratings, f"{row['query_id']}:{key}")
            for row in records
            if row["query_id"] in {dataset.pairs[pair]["query_id"] for pair in assignment}
            for key in [entry["source_key"] for entry in row["variants"][variant]["final_top_10"]]
        ]
        judged = [value for value in rated if value is not None]
        result["variants"][variant] = {
            "unique_final_memes": len(set(finals)),
            "cross_query_repetition_rate": 1 - len(set(finals)) / len(finals),
            "median_latency_ms": float(np.median(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "rated_slot_count": len(judged),
            "unjudged_slot_count": len(rated) - len(judged),
            "mean_available_human_rating": {
                axis: sum(value[axis] for value in judged) / len(judged) if judged else None
                for axis in ("situation", "tone", "sendability")
            },
        }
    overlaps5 = [len(set(intent_sets[key][:5]) & set(a3_sets[key][:5])) / 5 for key in development]
    overlaps10 = [len(set(intent_sets[key]) & set(a3_sets[key])) / 10 for key in development]
    result["intent_vs_a3_lh"] = {
        "mean_top_5_overlap": float(np.mean(overlaps5)),
        "mean_top_10_overlap": float(np.mean(overlaps10)),
        "intent_only_final_candidates": len(
            set().union(*map(set, intent_sets.values())) - set().union(*map(set, a3_sets.values()))
        ),
    }
    return result


def _human_export(records: list[dict[str, Any]], query_ids: list[str]) -> dict[str, Any]:
    pairs: dict[str, dict[str, Any]] = {}
    for row in records:
        if row["query_id"] not in query_ids:
            continue
        for variant in VARIANTS:
            for item in row["variants"][variant]["final_top_10"]:
                key = f"{row['query_id']}:{item['source_key']}"
                pair = pairs.setdefault(
                    key,
                    {
                        "key": key,
                        "query_id": row["query_id"],
                        "source_key": item["source_key"],
                        "contributors": [],
                    },
                )
                pair["contributors"].append(
                    {
                        "variant": variant,
                        "rank": item["filtered_rank"],
                        "raw_rank": item["raw_rank"],
                        "raw_score": item["raw_score"],
                        "final_score": item["final_score"],
                    }
                )
    return {
        "format_version": 1,
        "selection": "one fresh development query per category by fixed hash; excludes all eight previously rated comparative query identities",
        "query_ids": query_ids,
        "pairs": [pairs[key] for key in sorted(pairs)],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Phase 7R.5B conservative intent-aware comparison"
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = (
        args.output
        or BENCHMARKS / f"phase7r_v1_intent_comparison_{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    )
    report = run(output)
    print(
        f"Intent coverage: {report['summary']['intent_coverage']}; fallback={report['summary']['fallback_count']}"
    )
    print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
