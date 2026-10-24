"""Focused diagnostic: mood changes are observable, not proof of relevance."""

from __future__ import annotations

import argparse
import json

from app.core.config import Settings
from app.moods import mood_query, resolve_mood
from app.recommendation_engine import RecommendationEngine
from app.recommendations import RankingWeights

SCENARIOS = (
    ("My presentation starts in five minutes", "anxiety", "excitement"),
    ("Everyone saw my mistake", "embarrassment", "pride"),
    ("I found an old photo album", "nostalgia", "amusement"),
    ("The deadline is tomorrow", "calm", "stress"),
    ("I do not understand these instructions", "confusion", "curiosity"),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare deterministic mood-aware retrieval scenarios")
    parser.add_argument("--catalog", choices=("memematch_281", "memematch_1000_v2"), default="memematch_1000_v2")
    parser.add_argument("--mode", choices=("semantic", "hybrid", "hybrid_cross_encoder"), default="semantic")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args(argv)
    catalog = Settings(recommendation_catalog=args.catalog).active_catalog()
    engine = RecommendationEngine(catalog.index_dir, catalog.metadata_csv, catalog.image_root, catalog_id=catalog.catalog_id, expected_count=catalog.expected_count, expected_dimensions=catalog.expected_dimensions, export_records=catalog.export_records, export_metadata=catalog.export_metadata)
    report = {"catalog": args.catalog, "retrieval_mode": args.mode, "scenarios": []}
    for query, first, second in SCENARIOS:
        rows = []
        for mood in (first, second):
            results = engine.recommend_text(query, emotion=None, mood=mood, sentiment=None, usage=[], weights=RankingWeights(.90, .08, .02), candidate_limit=50, bm25_candidate_limit=50, cross_encoder_limit=20, min_semantic_similarity=None, diversity_max_per_template=1, exclude_unsafe=True, retrieval_mode=args.mode, limit=args.limit)
            rows.append({"mood": mood, "effective_query": mood_query(query, resolve_mood(mood)), "top_ids": [item.meme_id for item in results], "candidate_count": engine.last_retrieval_candidate_count, "retrieval_latency_ms": engine.last_retrieval_latency_ms, "cross_encoder_latency_ms": engine.last_cross_encoder_latency_ms, "warning": engine.last_retrieval_warning})
        report["scenarios"].append({"query": query, "comparisons": rows, "ranking_changed": rows[0]["top_ids"] != rows[1]["top_ids"]})
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
