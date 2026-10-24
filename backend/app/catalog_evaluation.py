"""Deterministic, human-review-oriented comparison runner for local catalogs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from app.core.config import Settings
from app.recommendation_engine import RecommendationEngine
from app.recommendations import RankingWeights

EVALUATION_QUERIES = (
    ("I finally passed my exam", "joy"),
    ("I am exhausted after studying all night", "sadness"),
    ("My presentation starts in five minutes", "fear"),
    ("My plans were ruined at the last minute", "anger"),
    ("That was completely unexpected", "surprise"),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export five fixed query results for human catalog review")
    parser.add_argument("--catalog", choices=("memematch_281", "memematch_1000_v2"), default="memematch_281")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--candidate-limit", type=int, default=50)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--include-unsafe", action="store_true")
    parser.add_argument("--max-per-template", type=int, default=1)
    parser.add_argument("--modes", default="semantic,hybrid,hybrid_cross_encoder")
    args = parser.parse_args(argv)
    settings = Settings(recommendation_catalog=args.catalog)
    catalog = settings.active_catalog()
    engine = RecommendationEngine(
        catalog.index_dir, catalog.metadata_csv, catalog.image_root,
        catalog_id=catalog.catalog_id, expected_count=catalog.expected_count,
        expected_dimensions=catalog.expected_dimensions,
        export_records=catalog.export_records, export_metadata=catalog.export_metadata,
        query_device=args.device,
    )
    service = engine._load()
    modes = args.modes.split(",")
    if any(mode not in {"semantic", "hybrid", "hybrid_cross_encoder"} for mode in modes):
        parser.error("--modes must contain semantic, hybrid, or hybrid_cross_encoder")
    report = {"catalog": catalog.catalog_id, "queries": []}
    for query, emotion in EVALUATION_QUERIES:
        outputs = {}
        for mode in modes:
            results = engine.recommend_text(query, emotion=emotion, sentiment=None, usage=[], weights=RankingWeights(0.90, 0.08, 0.02), candidate_limit=args.candidate_limit, min_semantic_similarity=None, diversity_max_per_template=args.max_per_template, exclude_unsafe=not args.include_unsafe, retrieval_mode=mode, bm25_candidate_limit=args.candidate_limit, cross_encoder_limit=min(args.candidate_limit, 50), limit=args.limit)
            outputs[mode] = {
                "retrieval_latency_ms": engine.last_retrieval_latency_ms,
                "cross_encoder_latency_ms": engine.last_cross_encoder_latency_ms,
                "warning": engine.last_retrieval_warning,
                "retrieval_candidate_count": engine.last_retrieval_candidate_count,
                "post_safety_candidate_count": engine.last_safe_candidate_count,
                "safety_filter_enabled": not args.include_unsafe,
                "results": [
                    {**asdict(item), "template_group": (metadata.template_group if (metadata := service.metadata.get(item.meme_id)) else None), "local_context": (metadata.local_context if metadata else None), "global_context": (metadata.global_context if metadata else None)}
                    for item in results
                ],
            }
        report["queries"].append({"query": query, "emotion": emotion, "modes": outputs})
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
