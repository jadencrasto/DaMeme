"""Offline session-aware repetition-control ablation for Phase 7R.4F."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict, deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.benchmark import load_assignment, load_benchmark_ratings, load_pool_dataset, load_queries
from app.moods import primary_classifier_emotion, resolve_mood
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
PREPARATION = BENCHMARKS / "phase7r_v1_human_eval_preparation_20261025T055104Z"
POOL = PREPARATION / "development.pool.json"
ASSIGNMENT = PREPARATION / "comparative.assignment.json"
RATINGS = PREPARATION / "ratings.seed.json"
CATALOG = ROOT / "data" / "retrieval" / "catalogs" / "memematch_1000_v2"
VARIANTS = ("A0", "A3-LH")
POLICIES = {"none": 0, "previous_1": 1, "previous_3": 3}
PENALTY = 0.05
RESULT_LIMIT = 10
SEED = 20261025


def _hash(value: str) -> str:
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def build_sessions(queries: dict[str, Any]) -> list[dict[str, Any]]:
    """Cover all development queries in related, unrelated, and repeated sessions."""
    development = [query for query in queries.values() if query.split == "development"]
    by_category: dict[str, list[str]] = defaultdict(list)
    for query in development:
        by_category[query.category].append(query.id)
    ordered_categories = sorted(by_category)
    ordered = {category: sorted(ids, key=_hash) for category, ids in by_category.items()}
    sessions = [
        {"id": f"related-{index:02d}", "type": "related", "query_ids": ordered[category]}
        for index, category in enumerate(ordered_categories, start=1)
    ]
    sessions.extend(
        {
            "id": f"unrelated-{position:02d}",
            "type": "unrelated",
            "query_ids": [ordered[category][position] for category in ordered_categories],
        }
        for position in range(len(next(iter(ordered.values()))))
    )
    sessions.extend(
        {
            "id": f"repeated-{query.id}",
            "type": "repeated",
            "query_ids": [query.id, query.id, query.id],
        }
        for query in sorted(development, key=lambda query: _hash(query.id))
    )
    return sessions


def rerank_with_session_penalty(
    ranked: list[Any], recently_shown: set[str], penalty: float = PENALTY
) -> list[Any]:
    """Softly demote, never remove, candidates shown in the simulated session."""
    if penalty < 0:
        raise ValueError("penalty must be non-negative")
    return sorted(
        ranked,
        key=lambda item: (
            -(item.final_score - (penalty if item.meme_id in recently_shown else 0.0)),
            item.meme_id,
        ),
    )


def _rated_values(ratings: dict[str, Any], key: str) -> dict[str, float] | None:
    entries = [values[key] for values in ratings["evaluators"].values() if key in values]
    if not entries:
        return None
    return {
        axis: sum(entry[axis] for entry in entries) / len(entries)
        for axis in ("situation", "tone", "sendability")
    }


def _candidate_ranking(
    service: RecommendationService, query: Any, source: dict[str, Any]
) -> list[Any]:
    candidates = [
        SearchResult(row["source_key"], row["raw_rank"], row["raw_score"])
        for row in source["raw_candidates"]
    ]
    return service.rank_candidates(
        candidates,
        emotion=primary_classifier_emotion(resolve_mood(query.mood)),
        safety=SafetyConfig(),
        limit=len(candidates),
    )


def _finalize(service: RecommendationService, ranked: list[Any]) -> list[Any]:
    return service.apply_diversity(ranked, DiversityConfig(max_per_template=1), RESULT_LIMIT)


def _event(
    service: RecommendationService,
    query: Any,
    source: dict[str, Any],
    recently_shown: set[str],
    policy: str,
    ratings: dict[str, Any],
) -> dict[str, Any]:
    baseline_ranked = _candidate_ranking(service, query, source)
    reranked = (
        rerank_with_session_penalty(baseline_ranked, recently_shown)
        if policy != "none"
        else baseline_ranked
    )
    final = _finalize(service, reranked)
    baseline_final = _finalize(service, baseline_ranked)
    keys = [item.meme_id for item in final]
    baseline_keys = [item.meme_id for item in baseline_final]
    ratings_by_key = {
        f"{query.id}:{key}": _rated_values(ratings, f"{query.id}:{key}") for key in keys
    }
    judged = {key: value for key, value in ratings_by_key.items() if value is not None}
    return {
        "query_id": query.id,
        "shown_source_keys": keys,
        "baseline_source_keys": baseline_keys,
        "top_5_overlap_with_unmodified": len(set(keys[:5]) & set(baseline_keys[:5])) / 5,
        "top_10_overlap_with_unmodified": len(set(keys) & set(baseline_keys)) / 10,
        "displaced_original_top_5": [key for key in baseline_keys[:5] if key not in keys[:5]],
        "penalized_candidates_shown": sorted(set(keys) & recently_shown),
        "human_ratings": judged,
        "unjudged_shown_count": RESULT_LIMIT - len(judged),
    }


def simulate(
    queries: dict[str, Any], comparison: dict[str, Any], ratings: dict[str, Any]
) -> dict[str, Any]:
    by_query = {row["query_id"]: row for row in comparison["queries"]}
    metadata = MemeMatchMetadata.from_csv(
        CATALOG / "memematch_1000_metadata_v2.csv", CATALOG / "images"
    )
    service = RecommendationService(RetrievalService.load(CATALOG / "indexes"), metadata)
    sessions = build_sessions(queries)
    runs: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for policy, history_size in POLICIES.items():
            for session in sessions:
                history: deque[set[str]] = deque(maxlen=history_size)
                events = []
                for query_id in session["query_ids"]:
                    recently_shown = set().union(*history) if history else set()
                    event = _event(
                        service,
                        queries[query_id],
                        by_query[query_id]["variants"][variant],
                        recently_shown,
                        policy,
                        ratings,
                    )
                    events.append(event)
                    if history_size:
                        history.append(set(event["shown_source_keys"]))
                runs.append(
                    {"variant": variant, "policy": policy, "session": session, "events": events}
                )
    return {"sessions": sessions, "runs": runs}


def summarize(simulation: dict[str, Any]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in simulation["runs"]:
        grouped[(run["variant"], run["policy"])].append(run)
    summary: dict[str, Any] = {}
    for (variant, policy), runs in grouped.items():
        session_metrics, overlaps5, overlaps10, displaced, judged_axes, unjudged = (
            [],
            [],
            [],
            0,
            defaultdict(list),
            0,
        )
        for run in runs:
            shown = [key for event in run["events"] for key in event["shown_source_keys"]]
            session_metrics.append(
                {
                    "session_id": run["session"]["id"],
                    "type": run["session"]["type"],
                    "unique_memes": len(set(shown)),
                    "slots": len(shown),
                    "repeated_meme_rate": 1 - len(set(shown)) / len(shown),
                }
            )
            for event in run["events"]:
                overlaps5.append(event["top_5_overlap_with_unmodified"])
                overlaps10.append(event["top_10_overlap_with_unmodified"])
                displaced += len(event["displaced_original_top_5"])
                unjudged += event["unjudged_shown_count"]
                for values in event["human_ratings"].values():
                    for axis, value in values.items():
                        judged_axes[axis].append(value)
        summary[f"{variant}:{policy}"] = {
            "mean_unique_memes_per_session": sum(item["unique_memes"] for item in session_metrics)
            / len(session_metrics),
            "mean_session_repeated_meme_rate": sum(
                item["repeated_meme_rate"] for item in session_metrics
            )
            / len(session_metrics),
            "top_5_overlap_with_unmodified": sum(overlaps5) / len(overlaps5),
            "top_10_overlap_with_unmodified": sum(overlaps10) / len(overlaps10),
            "displaced_original_top_5_count": displaced,
            "human_rated_slot_count": sum(len(values) for values in judged_axes.values()) // 3,
            "mean_available_human_rating": {
                axis: sum(values) / len(values) if values else None
                for axis, values in judged_axes.items()
            },
            "unjudged_shown_slot_count": unjudged,
            "per_session": session_metrics,
        }
    return summary


def run(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing experiment directory: {output}")
    queries, _ = load_queries(QUERIES)
    dataset = load_pool_dataset(QUERIES, POOL)
    assignment_keys = load_assignment(ASSIGNMENT, dataset)
    ratings = load_benchmark_ratings(RATINGS, dataset)
    if len(assignment_keys) != 103:
        raise ValueError("expected completed 103-pair comparative assignment")
    comparison = json.loads(COMPARISON.read_text(encoding="utf-8"))
    simulation = simulate(queries, comparison, ratings)
    report = {
        "format_version": 1,
        "experiment": "phase7r.4f-offline-session-repetition-ablation",
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "development-only simulated sessions; no production behavior changed",
        "policy": {
            "penalty": PENALTY,
            "definitions": {
                "none": "unmodified final reranking",
                "previous_1": "soft penalty when shown in the immediately previous search",
                "previous_3": "soft penalty when shown in any of the prior three searches",
            },
        },
        "definitions": {
            "within_list_duplicates": "duplicate source keys in one final list",
            "cross_query_repetition": "same meme across independent benchmark searches",
            "session_repetition": "same meme shown again in the bounded simulated session",
        },
        "simulation": simulation,
    }
    report["summary"] = summarize(simulation)
    output.mkdir(parents=True)
    (output / "session_simulation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(report["summary"], indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Phase 7R.4F offline repetition-control ablation"
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = (
        args.output
        or BENCHMARKS / f"phase7r_v1_session_repetition_ablation_{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    )
    report = run(output)
    for key, values in report["summary"].items():
        print(
            f"{key}: session_repeat_rate={values['mean_session_repeated_meme_rate']:.3f}, top5_overlap={values['top_5_overlap_with_unmodified']:.3f}"
        )
    print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
