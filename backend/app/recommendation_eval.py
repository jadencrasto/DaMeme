"""Report observable ranking changes without claiming human relevance metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from app.recommendations import (
    MemeMatchMetadata,
    RankingWeights,
    RecommendationService,
    filename_template_key,
)
from app.retrieval import RetrievalService, TextQueryEmbedder


def _diversity(items: list[object]) -> int:
    return len({filename_template_key(item.meme_id) for item in items})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare observable recommendation ranking properties")
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path, required=True)
    query = parser.add_mutually_exclusive_group(required=True)
    query.add_argument("--query-vector", type=Path, help=".npy 384-dimensional text query")
    query.add_argument("--query-text", help="text embedded with the configured MiniLM encoder")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--emotion", required=True, choices=("anger", "fear", "joy", "sadness", "surprise", "disgust"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--candidate-limit", type=int, default=100)
    parser.add_argument("--baseline-weights", default="1,1,1", help="original semantic,emotion,sentiment weights")
    parser.add_argument("--improved-weights", default="0.8,0.15,0.05", help="semantic-first weights")
    parser.add_argument("--min-semantic-similarity", type=float, default=None)
    parser.add_argument("--max-per-template", type=int, default=None)
    parser.add_argument("--export-csv", type=Path, help="write paired results with blank manual-judgment columns")
    args = parser.parse_args(argv)
    metadata = MemeMatchMetadata.from_csv(args.metadata_csv)
    service = RecommendationService(RetrievalService.load(args.index_dir), metadata)
    query_vector = (
        np.load(args.query_vector)
        if args.query_vector
        else TextQueryEmbedder(service.retrieval, args.device).embed(args.query_text)
    )
    def parse_weights(value: str) -> RankingWeights:
        try:
            return RankingWeights(*(float(part) for part in value.split(",")))
        except (TypeError, ValueError) as exc:
            parser.error("weights must be semantic,emotion,sentiment")
            raise AssertionError from exc

    baseline_weights, improved_weights = parse_weights(args.baseline_weights), parse_weights(args.improved_weights)
    from app.recommendations import DiversityConfig
    diversity = DiversityConfig(args.max_per_template) if args.max_per_template else None
    semantic = service.recommend_text(query_vector, weights=RankingWeights(1, 0, 0), candidate_limit=args.candidate_limit, diversity=diversity, min_semantic_similarity=args.min_semantic_similarity, limit=args.limit)
    baseline = service.recommend_text(query_vector, emotion=args.emotion, weights=baseline_weights, candidate_limit=args.candidate_limit, diversity=diversity, min_semantic_similarity=args.min_semantic_similarity, limit=args.limit)
    combined = service.recommend_text(query_vector, emotion=args.emotion, weights=improved_weights, candidate_limit=args.candidate_limit, diversity=diversity, min_semantic_similarity=args.min_semantic_similarity, limit=args.limit)
    emotion = service.recommend_emotion(args.emotion, weights=RankingWeights(0, 1, 0), limit=args.limit)
    semantic_ids, baseline_ids, combined_ids = [item.meme_id for item in semantic], [item.meme_id for item in baseline], [item.meme_id for item in combined]
    report = {
        "configuration": {"baseline_weights": args.baseline_weights, "improved_weights": args.improved_weights, "min_semantic_similarity": args.min_semantic_similarity, "max_per_template": args.max_per_template},
        "semantic_candidate_coverage": len(semantic) / args.limit if args.limit else 0.0,
        "baseline_to_improved_top_k_overlap": len(set(baseline_ids) & set(combined_ids)) / args.limit if args.limit else 0.0,
        "baseline_to_improved_order_changed": baseline_ids != combined_ids,
        "semantic_to_combined_top_k_overlap": len(set(semantic_ids) & set(combined_ids)) / args.limit if args.limit else 0.0,
        "semantic_to_combined_order_changed": semantic_ids != combined_ids,
        "template_diversity": {"semantic": _diversity(semantic), "emotion": _diversity(emotion), "baseline": _diversity(baseline), "improved": _diversity(combined)},
        "counts": {"semantic": len(semantic), "emotion": len(emotion), "baseline": len(baseline), "improved": len(combined)},
    }
    if args.export_csv:
        rows = []
        for configuration, items in (("baseline", baseline), ("improved", combined)):
            for rank, item in enumerate(items, start=1):
                rows.append({"configuration": configuration, "rank": rank, "meme_id": item.meme_id, "semantic_similarity": item.semantic_similarity, "emotion_relevance": item.emotion_relevance, "final_score": item.final_score, "template_key": filename_template_key(item.meme_id), "situational_relevance": "", "emotion_appropriateness": "", "diversity_note": "", "reviewer_notes": ""})
        args.export_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.export_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["configuration", "rank", "meme_id", "semantic_similarity", "emotion_relevance", "final_score", "template_key", "situational_relevance", "emotion_appropriateness", "diversity_note", "reviewer_notes"])
            writer.writeheader()
            writer.writerows(rows)
        report["review_export"] = str(args.export_csv)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
