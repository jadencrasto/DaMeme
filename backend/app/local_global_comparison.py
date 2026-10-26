"""Development-only A0--A3 offline retrieval comparison for Phase 7R.4D.2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from app.benchmark import BenchmarkQuery, load_queries
from app.emotion import PretrainedFeatureExtractor
from app.hybrid_retrieval import reciprocal_rank_fusion
from app.moods import mood_query, primary_classifier_emotion, resolve_mood
from app.recommendations import (
    DiversityConfig,
    MemeMatchMetadata,
    RecommendationService,
    SafetyConfig,
)
from app.retrieval import RetrievalService, SearchResult

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
CATALOG = ROOT / "data" / "retrieval" / "catalogs" / "memematch_1000_v2"
ABLATION = BACKEND / "benchmarks" / "phase7r_v1_local_global_indexes"
QUERIES = BACKEND / "benchmarks" / "phase7r_v1_queries.json"
EXPECTED_DIMENSION = 384
EXPECTED_COUNT = 1000
CANDIDATE_DEPTH = 100
RESULT_LIMIT = 10
RRF_K = 60


@dataclass(frozen=True)
class Candidate:
    source_key: str
    score: float
    local_score: float | None
    global_score: float | None
    local_rank: int | None
    global_rank: int | None


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered(scores: dict[str, float], limit: int) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]


def fuse_scores(
    local_scores: dict[str, float],
    global_scores: dict[str, float],
    local_available: set[str],
    local_weight: float,
) -> dict[str, float]:
    """Fuse aligned modalities, falling back to global-only for absent local rows."""
    if not 0.0 <= local_weight <= 1.0:
        raise ValueError("local_weight must be between zero and one")
    if set(local_scores) != set(global_scores):
        raise ValueError("local and global scores must have identical source keys")
    return {
        key: local_weight * local_scores[key] + (1.0 - local_weight) * global_scores[key]
        if key in local_available
        else global_scores[key]
        for key in global_scores
    }


def rrf_scores(
    local_ranking: list[str], global_ranking: list[str], k: int = RRF_K
) -> dict[str, float]:
    """Return deterministic RRF scores with the same de-duplication semantics as production."""
    fused = reciprocal_rank_fusion(
        local_ranking, global_ranking, k=k, limit=len(set(local_ranking) | set(global_ranking))
    )
    local_positions = {
        key: position for position, key in enumerate(dict.fromkeys(local_ranking), start=1)
    }
    global_positions = {
        key: position for position, key in enumerate(dict.fromkeys(global_ranking), start=1)
    }
    return {
        key: (1.0 / (k + local_positions[key]) if key in local_positions else 0.0)
        + (1.0 / (k + global_positions[key]) if key in global_positions else 0.0)
        for key in fused
    }


def development_queries(queries: Iterable[BenchmarkQuery]) -> list[BenchmarkQuery]:
    return [query for query in queries if query.split == "development"]


def check_source_key_alignment(a0_keys: list[str], ablation_keys: list[str]) -> None:
    if a0_keys != ablation_keys:
        raise ValueError("A0 source-key order does not exactly match the A1/A2 manifest")


def compatibility_report(
    a0: RetrievalService, manifest: dict[str, Any], metadata_csv: Path
) -> dict[str, Any]:
    a0_keys = [str(record["source_key"]) for record in a0.exported.records]
    ablation_keys = [str(key) for key in manifest["source_keys"]]
    checks = {
        "source_key_identity_and_order": a0_keys == ablation_keys,
        "expected_catalog_count": len(a0_keys) == EXPECTED_COUNT,
        "query_encoder_model": (
            a0.text_model_id == manifest["model_id"] == PretrainedFeatureExtractor.text_model_id
        ),
        "query_encoder_dimension": a0.indexes["text"].d
        == manifest["dimensions"]
        == EXPECTED_DIMENSION,
        "query_encoder_configuration": manifest["encoder"]
        == {
            "pooling": "attention-mask mean",
            "max_length": 256,
            "normalization": "l2",
            "device": "cpu",
        },
        "a0_text_representation": a0.exported.metadata.get("text_representation")
        == {
            "version": 2,
            "field_order": [
                "cleaned_ocr",
                "local_context",
                "caption_or_title",
                "verified_template",
                "global_context",
            ],
        },
        "source_csv_sha256": (
            _digest(metadata_csv)
            == str(manifest["source_sha256"])
            == str(a0.exported.metadata.get("source_sha256"))
        ),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "a0_source_key_count": len(a0_keys),
        "ablation_source_key_count": len(ablation_keys),
        "query_preprocessing": "app.moods.mood_query(message, resolve_mood(mood)); shared unchanged across all variants",
        "mood_expansion": "shared app.moods mood descriptions; safety emotion is primary_classifier_emotion(mood)",
    }


def _scores(index: faiss.Index, query: np.ndarray, keys: list[str]) -> dict[str, float]:
    scores, positions = index.search(
        np.asarray(query, dtype=np.float32).reshape(1, -1), index.ntotal
    )
    return {
        keys[int(position)]: float(score)
        for score, position in zip(scores[0], positions[0], strict=True)
        if position >= 0
    }


def _candidate_rows(
    scores: dict[str, float],
    local_scores: dict[str, float],
    global_scores: dict[str, float],
    local_ranks: dict[str, int],
    global_ranks: dict[str, int],
) -> list[Candidate]:
    return [
        Candidate(
            key,
            score,
            local_scores.get(key),
            global_scores.get(key),
            local_ranks.get(key),
            global_ranks.get(key),
        )
        for key, score in _ordered(scores, CANDIDATE_DEPTH)
    ]


def _serialize_candidates(candidates: list[Candidate]) -> list[dict[str, Any]]:
    return [
        {
            "source_key": candidate.source_key,
            "raw_rank": rank,
            "raw_score": candidate.score,
            "local_score": candidate.local_score,
            "global_score": candidate.global_score,
            "local_rank": candidate.local_rank,
            "global_rank": candidate.global_rank,
        }
        for rank, candidate in enumerate(candidates, start=1)
    ]


def _post_filter(
    service: RecommendationService, candidates: list[Candidate], emotion: str | None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    search = [
        SearchResult(candidate.source_key, rank, candidate.score)
        for rank, candidate in enumerate(candidates)
    ]
    safety = SafetyConfig()
    safety_excluded = [
        candidate.source_key
        for candidate in candidates
        if (metadata := service.metadata.get(candidate.source_key)) is not None
        and safety.excludes(metadata.safety_text, emotion)
    ]
    ranked = service.rank_candidates(search, emotion=emotion, safety=safety, limit=CANDIDATE_DEPTH)
    final = service.apply_diversity(ranked, DiversityConfig(max_per_template=1), RESULT_LIMIT)
    raw_ranks = {candidate.source_key: rank for rank, candidate in enumerate(candidates, start=1)}
    filtered = [
        {
            "source_key": item.meme_id,
            "filtered_rank": rank,
            "raw_rank": raw_ranks[item.meme_id],
            "raw_score": item.semantic_similarity,
            "final_score": item.final_score,
        }
        for rank, item in enumerate(final, start=1)
    ]
    template_keys = [
        (
            service.metadata.get(item.meme_id).template_group
            if service.metadata.get(item.meme_id)
            else None
        )
        or item.meme_id
        for item in final
    ]
    return filtered, {
        "retrieved": len(candidates),
        "safety_excluded": safety_excluded,
        "after_safety": len(ranked),
        "after_template_diversity": len(final),
        "filtering_reduced_result_count": len(final) < RESULT_LIMIT,
        "template_diversity": {
            "distinct_template_keys": len(set(template_keys)),
            "template_keys": template_keys,
        },
    }


def run(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing experiment directory: {output}")
    with (ABLATION / "manifest.json").open(encoding="utf-8") as source:
        manifest: dict[str, Any] = json.load(source)
    a0 = RetrievalService.load(CATALOG / "indexes")
    compatibility = compatibility_report(a0, manifest, CATALOG / "memematch_1000_metadata_v2.csv")
    if compatibility["status"] != "PASS":
        raise ValueError(f"A0 compatibility failed: {compatibility['checks']}")
    check_source_key_alignment(
        [str(record["source_key"]) for record in a0.exported.records], list(manifest["source_keys"])
    )

    local_index = faiss.read_index(str(ABLATION / "local.faiss"))
    global_index = faiss.read_index(str(ABLATION / "global.faiss"))
    keys = list(manifest["source_keys"])
    availability = np.asarray(manifest["availability"]["local"], dtype=bool)
    local_available = {
        key for key, is_available in zip(keys, availability, strict=True) if is_available
    }
    metadata = MemeMatchMetadata.from_csv(
        CATALOG / "memematch_1000_metadata_v2.csv", CATALOG / "images"
    )
    service = RecommendationService(a0, metadata)
    queries = development_queries(load_queries(QUERIES)[0].values())
    if len(queries) != 40:
        raise ValueError(
            f"development-only comparison requires exactly 40 queries, got {len(queries)}"
        )
    if any(query.split != "development" for query in queries):
        raise ValueError("held-out queries reached development comparison")

    # This offline experiment must not download model weights as a side effect.
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    extractor = PretrainedFeatureExtractor("cpu")
    output.mkdir(parents=True)
    results: list[dict[str, Any]] = []
    for query in queries:
        effective_query = mood_query(query.message, resolve_mood(query.mood))
        started = time.perf_counter()
        embedding = extractor.embed_texts([effective_query])[0]
        embedding_latency_ms = (time.perf_counter() - started) * 1000
        identity = hashlib.sha256(np.asarray(embedding, dtype=np.float32).tobytes()).hexdigest()
        a0_started = time.perf_counter()
        a0_scores = {row.meme_id: row.score for row in a0.search("text", embedding, EXPECTED_COUNT)}
        local_scores_all = _scores(local_index, embedding, keys)
        global_scores = _scores(global_index, embedding, keys)
        local_scores = {
            key: score for key, score in local_scores_all.items() if key in local_available
        }
        local_ranking = [key for key, _ in _ordered(local_scores, len(local_scores))]
        global_ranking = [key for key, _ in _ordered(global_scores, len(global_scores))]
        local_ranks = {key: rank for rank, key in enumerate(local_ranking, start=1)}
        global_ranks = {key: rank for rank, key in enumerate(global_ranking, start=1)}
        variants = {
            "A0": a0_scores,
            "A1": local_scores,
            "A2": global_scores,
            "A3-EQ": fuse_scores(local_scores_all, global_scores, local_available, 0.5),
            "A3-LH": fuse_scores(local_scores_all, global_scores, local_available, 0.7),
            "A3-RRF": rrf_scores(local_ranking, global_ranking),
        }
        query_result = {
            "query_id": query.id,
            "split": query.split,
            "message": query.message,
            "mood": query.mood,
            "effective_query": effective_query,
            "query_embedding": {
                "model_id": extractor.text_model_id,
                "sha256": identity,
                "l2_norm": float(np.linalg.norm(embedding)),
                "latency_ms": embedding_latency_ms,
            },
            "variants": {},
        }
        for name, scores in variants.items():
            variant_started = time.perf_counter()
            candidates = _candidate_rows(
                scores, local_scores_all, global_scores, local_ranks, global_ranks
            )
            final, filtering = _post_filter(
                service, candidates, primary_classifier_emotion(resolve_mood(query.mood))
            )
            query_result["variants"][name] = {
                "raw_candidates": _serialize_candidates(candidates),
                "final_top_10": final,
                "candidate_counts": filtering,
                "latency_ms": (time.perf_counter() - variant_started) * 1000,
            }
        query_result["a0_score_collection_latency_ms"] = (time.perf_counter() - a0_started) * 1000
        results.append(query_result)

    summary = _summary(results)
    report = {
        "format_version": 1,
        "experiment": "phase7r.4d.2-development-local-global-comparison",
        "created_at": datetime.now(UTC).isoformat(),
        "scope": {
            "split": "development",
            "query_count": len(results),
            "candidate_depth": CANDIDATE_DEPTH,
            "result_limit": RESULT_LIMIT,
        },
        "compatibility": compatibility,
        "missing_modality_policy": "A1 excludes the four unavailable local rows. A3 score fusions renormalize available local/global weights; unavailable-local rows use their global score only. A3-RRF receives only the global rank for unavailable-local rows.",
        "queries": results,
        "summary": summary,
    }
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return report


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values), percentile))


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    variants = ("A0", "A1", "A2", "A3-EQ", "A3-LH", "A3-RRF")
    summary: dict[str, Any] = {"variants": {}, "missing_local": {"unavailable_rows": 4}}
    a0_sets = {
        row["query_id"]: [item["source_key"] for item in row["variants"]["A0"]["final_top_10"]]
        for row in results
    }
    for name in variants:
        overlaps5, overlaps10, latencies, counts, candidates = [], [], [], [], set()
        for row in results:
            final = [item["source_key"] for item in row["variants"][name]["final_top_10"]]
            baseline = a0_sets[row["query_id"]]
            overlaps5.append(len(set(final[:5]) & set(baseline[:5])) / 5)
            overlaps10.append(len(set(final) & set(baseline)) / 10)
            latencies.append(float(row["variants"][name]["latency_ms"]))
            counts.append(len(final))
            candidates.update(final)
        summary["variants"][name] = {
            "mean_top_5_overlap_with_a0": float(np.mean(overlaps5)),
            "mean_top_10_overlap_with_a0": float(np.mean(overlaps10)),
            "distinct_final_candidates": len(candidates),
            "distinct_candidates_introduced_vs_a0": len(
                candidates - set().union(*map(set, a0_sets.values()))
            ),
            "median_latency_ms": float(np.median(latencies)),
            "p95_latency_ms": _percentile(latencies, 95),
            "per_query_result_counts": counts,
            "queries_below_top_10_after_filtering": sum(count < RESULT_LIMIT for count in counts),
        }
    missing_hits = Counter()
    for row in results:
        for name in ("A2", "A3-EQ", "A3-LH", "A3-RRF"):
            keys = [item["source_key"] for item in row["variants"][name]["final_top_10"]]
            local_ranks = {
                item["source_key"]: item["local_rank"]
                for item in row["variants"][name]["raw_candidates"]
            }
            missing_hits[name] += sum(
                key in local_ranks and local_ranks[key] is None for key in keys
            )
    summary["missing_local"]["top_10_results_with_no_local_vector"] = dict(missing_hits)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase 7R.4D.2 on development queries only")
    parser.add_argument(
        "--output", type=Path, default=None, help="new output directory; refuses to overwrite"
    )
    args = parser.parse_args(argv)
    output = (
        args.output
        or BACKEND
        / "benchmarks"
        / f"phase7r_v1_local_global_comparison_{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    )
    report = run(output)
    print(f"{report['compatibility']['status']} | A0 compatibility")
    for name, values in report["summary"]["variants"].items():
        print(
            f"{name}: overlap@5={values['mean_top_5_overlap_with_a0']:.3f}, overlap@10={values['mean_top_10_overlap_with_a0']:.3f}, median_ms={values['median_latency_ms']:.3f}"
        )
    print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
