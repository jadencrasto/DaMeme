"""Print observable reranking changes and component-level diagnostics."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from app.recommendations import (
    DiversityConfig,
    MemeMatchMetadata,
    RankingWeights,
    RecommendationService,
    filename_template_key,
)
from app.retrieval import RetrievalService, TextQueryEmbedder


def _diversity(items: list[object]) -> int:
    return len({filename_template_key(item.meme_id) for item in items})


def _weights(value: str) -> tuple[str, RankingWeights]:
    try:
        name, raw = value.split(":", 1)
        semantic, emotion, sentiment = (float(part) for part in raw.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("weight config must be name:semantic,emotion,sentiment") from exc
    return name, RankingWeights(semantic, emotion, sentiment)


def _comparison(
    service: RecommendationService, query: np.ndarray, emotion: str, sentiment: str | None,
    weights: RankingWeights, candidate_limit: int, limit: int, diversity: DiversityConfig | None,
) -> dict[str, object]:
    candidates = service.retrieval.search("text", query, candidate_limit)
    semantic = service.recommend_text(query, weights=RankingWeights(1, 0, 0), diversity=diversity, candidate_limit=candidate_limit, limit=limit)
    combined = service.recommend_text(query, emotion=emotion, sentiment=sentiment, weights=weights, diversity=diversity, candidate_limit=candidate_limit, limit=limit)
    semantic_ids, combined_ids = [item.meme_id for item in semantic], [item.meme_id for item in combined]
    return {
        "candidate_limit": candidate_limit,
        "weights": asdict(weights),
        "candidate_metadata_coverage": sum(service.metadata.get(row.meme_id) is not None for row in candidates) / len(candidates) if candidates else 0.0,
        "top_k_overlap": len(set(semantic_ids) & set(combined_ids)) / limit if limit else 0.0,
        "order_changed": semantic_ids != combined_ids,
        "template_diversity": {"semantic": _diversity(semantic), "combined": _diversity(combined)},
    }


def _table(
    service: RecommendationService, query: np.ndarray, emotion: str, sentiment: str | None,
    weights: RankingWeights, candidate_limit: int, table_limit: int, diversity: DiversityConfig | None,
) -> list[dict[str, object]]:
    candidates = service.retrieval.search("text", query, candidate_limit)
    before = {candidate.meme_id: rank for rank, candidate in enumerate(candidates, start=1)}
    combined = service.recommend_text(query, emotion=emotion, sentiment=sentiment, weights=weights, diversity=diversity, candidate_limit=candidate_limit, limit=candidate_limit)
    return [
        {
            "meme_id": item.meme_id,
            "semantic_similarity": item.semantic_similarity,
            "emotion_relevance": item.emotion_relevance,
            "sentiment_relevance": item.sentiment_relevance,
            "final_combined_score": item.final_score,
            "rank_before_reranking": before[item.meme_id],
            "rank_after_reranking": after_rank,
        }
        for after_rank, item in enumerate(combined[:table_limit], start=1)
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose metadata-aware reranking")
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    query = parser.add_mutually_exclusive_group(required=True)
    query.add_argument("--query-vector", type=Path)
    query.add_argument("--query-text")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--emotion", required=True, choices=("anger", "fear", "joy", "sadness", "surprise", "disgust"))
    parser.add_argument("--sentiment", choices=("positive", "negative", "neutral"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--candidate-limits", default="10,50,100,281")
    parser.add_argument("--weight-config", action="append", type=_weights)
    parser.add_argument("--diagnostic-config", default="balanced")
    parser.add_argument("--max-per-template", type=int)
    args = parser.parse_args(argv)
    pools = [int(value) for value in args.candidate_limits.split(",")]
    if any(value < 1 for value in pools):
        parser.error("candidate limits must be positive")
    metadata = MemeMatchMetadata.from_csv(args.metadata_csv)
    service = RecommendationService(RetrievalService.load(args.index_dir), metadata)
    vector = np.load(args.query_vector) if args.query_vector else TextQueryEmbedder(service.retrieval, args.device).embed(args.query_text)
    configs = args.weight_config or [
        ("semantic-only", RankingWeights(1, 0, 0)),
        ("balanced", RankingWeights(1, 1, 1 if args.sentiment else 0)),
        ("emotion-prioritized", RankingWeights(1, 2, 1 if args.sentiment else 0)),
    ]
    selected = next((weights for name, weights in configs if name == args.diagnostic_config), None)
    if selected is None:
        parser.error("--diagnostic-config must name a supplied weight configuration")
    diversity = DiversityConfig(args.max_per_template) if args.max_per_template else None
    comparisons = [
        {"configuration": name, **_comparison(service, vector, args.emotion, args.sentiment, weights, pool, args.limit, diversity)}
        for name, weights in configs for pool in pools
    ]
    emotion_only = service.recommend_emotion(args.emotion, sentiment=args.sentiment, diversity=diversity, limit=args.limit)
    print(json.dumps({
        "emotion": args.emotion,
        "sentiment": args.sentiment,
        "diversity_cap": args.max_per_template,
        "emotion_only_template_diversity": _diversity(emotion_only),
        "comparisons": comparisons,
        "diagnostic_table": _table(service, vector, args.emotion, args.sentiment, selected, max(pools), args.limit, diversity),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
