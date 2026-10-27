"""Leakage-resistant offline evaluation for Phase 5 retrieval.

Ground-truth MOOD labels are used only to decide relevance after a ranking is
produced. Query emotion vectors always come from the GATED bundle predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.emotion import EMOTIONS
from app.expanded_emotion import (
    _column,
    _identifier,
    _include_in_training,
    load_expansion_artifacts,
)
from app.phase5_retrieval import Phase5Bundle

K_VALUES = (5, 10)


@dataclass(frozen=True)
class EvaluationInputs:
    bundle: Phase5Bundle
    splits: tuple[str, ...]
    mood: np.ndarray
    labels: np.ndarray
    hashes: tuple[str | None, ...]


def _hashes(base: Path, ids: tuple[str, ...]) -> tuple[str | None, ...]:
    with (base / "dameme_5500_training_manifest.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        rows = [row for row in csv.DictReader(handle) if _include_in_training(row)]
    by_id = {
        _identifier(row): _column(row, "perceptual_hash", "phash", "sha256") or None for row in rows
    }
    if set(by_id) != set(ids):
        raise ValueError("manifest candidate IDs do not align with the Phase 5 bundle")
    return tuple(by_id[meme_id] for meme_id in ids)


def load_inputs(base: Path, bundle_dir: Path) -> EvaluationInputs:
    data = load_expansion_artifacts(base)
    bundle = Phase5Bundle.load(bundle_dir)
    if bundle.ids != data.meme_ids:
        raise ValueError("bundle meme IDs must exactly match model-input ordering")
    return EvaluationInputs(
        bundle, data.splits, data.mood_mask, data.labels, _hashes(base, bundle.ids)
    )


def eligible_candidates(
    inputs: EvaluationInputs, query_index: int, *, candidate_split: str = "train"
) -> np.ndarray:
    """Return a fixed pool excluding self and matching hash group when present."""
    query_hash = inputs.hashes[query_index]
    eligible = np.asarray([split == candidate_split for split in inputs.splits])
    eligible[query_index] = False
    if query_hash:
        eligible &= np.asarray([value != query_hash for value in inputs.hashes])
    return np.flatnonzero(eligible)


def rank_query(
    inputs: EvaluationInputs, query_index: int, candidates: np.ndarray, emotion_weight: float
) -> np.ndarray:
    """Rank a fixed candidate pool without consulting labels."""
    if not 0 <= emotion_weight <= 1:
        raise ValueError("emotion weight must be between zero and one")
    query = inputs.bundle.text[query_index]
    semantic = (inputs.bundle.text[candidates] @ query + 1.0) / 2.0
    user = inputs.bundle.emotions[query_index]
    compatibility = inputs.bundle.emotions[candidates] @ user
    compatibility /= max(float(np.linalg.norm(user)), 1e-12)
    compatibility /= np.maximum(np.linalg.norm(inputs.bundle.emotions[candidates], axis=1), 1e-12)
    score = (1.0 - emotion_weight) * semantic + emotion_weight * compatibility
    return candidates[
        np.lexsort(
            (np.asarray([inputs.bundle.ids[index] for index in candidates]), -semantic, -score)
        )
    ]


def relevance(inputs: EvaluationInputs, query_index: int, candidates: np.ndarray) -> np.ndarray:
    """Binary relevance: both MOOD records share at least one positive emotion."""
    if not inputs.mood[query_index]:
        raise ValueError("only MOOD rows may be evaluation queries")
    return (
        inputs.mood[candidates]
        & np.any(inputs.labels[candidates] & inputs.labels[query_index], axis=1)
    ).astype(np.int8)


def _query_metrics(ranked_relevance: np.ndarray, relevant_count: int, k: int) -> dict[str, float]:
    top = ranked_relevance[:k]
    hits = int(top.sum())
    precision = hits / k
    recall = hits / relevant_count
    discounts = 1.0 / np.log2(np.arange(len(top)) + 2)
    dcg = float(np.sum(top * discounts))
    ideal = float(np.sum(discounts[: min(relevant_count, k)]))
    return {
        f"precision_at_{k}": precision,
        f"recall_at_{k}": recall,
        f"hit_rate_at_{k}": float(hits > 0),
        f"ndcg_at_{k}": dcg / ideal if ideal else 0.0,
    }


def evaluate_partition(
    inputs: EvaluationInputs, split: str, emotion_weight: float
) -> dict[str, object]:
    query_indexes = [
        index for index, value in enumerate(inputs.splits) if value == split and inputs.mood[index]
    ]
    rows: list[dict[str, object]] = []
    excluded_zero = 0
    for query_index in query_indexes:
        candidates = eligible_candidates(inputs, query_index)
        candidate_relevance = relevance(inputs, query_index, candidates)
        relevant_count = int(candidate_relevance.sum())
        if not relevant_count:
            excluded_zero += 1
            continue
        ranked = rank_query(inputs, query_index, candidates, emotion_weight)
        positions = {int(index): position for position, index in enumerate(candidates)}
        ranked_relevance = np.asarray(
            [candidate_relevance[positions[int(index)]] for index in ranked], dtype=np.int8
        )
        row: dict[str, object] = {
            "query_id": inputs.bundle.ids[query_index],
            "relevant_candidates": relevant_count,
            "emotions": [
                label
                for position, label in enumerate(EMOTIONS)
                if inputs.labels[query_index, position]
            ],
        }
        for k in K_VALUES:
            row.update(_query_metrics(ranked_relevance, relevant_count, k))
        rows.append(row)
    aggregate: dict[str, float | int] = {
        "query_count": len(rows),
        "excluded_zero_relevant": excluded_zero,
        "total_mood_queries": len(query_indexes),
    }
    for k in K_VALUES:
        for metric in ("precision", "recall", "hit_rate", "ndcg"):
            key = f"{metric}_at_{k}"
            aggregate[key] = float(np.mean([row[key] for row in rows])) if rows else 0.0
    per_emotion: dict[str, dict[str, float | int]] = {}
    for emotion in EMOTIONS:
        subset = [row for row in rows if emotion in row["emotions"]]
        per_emotion[emotion] = {
            "query_count": len(subset),
            **{
                f"ndcg_at_{k}": float(np.mean([row[f"ndcg_at_{k}"] for row in subset]))
                if subset
                else 0.0
                for k in K_VALUES
            },
        }
    return {"aggregate": aggregate, "per_emotion": per_emotion, "queries": rows}


def select_weight(inputs: EvaluationInputs, weights: tuple[float, ...]) -> dict[str, object]:
    if not weights or any(weight < 0 or weight > 1 for weight in weights):
        raise ValueError("weights must be non-empty values between zero and one")
    validation = {
        str(weight): evaluate_partition(inputs, "validation", weight) for weight in weights
    }
    selected = max(
        weights, key=lambda weight: (validation[str(weight)]["aggregate"]["ndcg_at_10"], -weight)
    )
    return {
        "selection_metric": "validation nDCG@10",
        "weights": validation,
        "selected_emotion_weight": selected,
    }


def evaluate(
    inputs: EvaluationInputs, weights: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4)
) -> dict[str, object]:
    selection = select_weight(inputs, weights)
    selected = float(selection["selected_emotion_weight"])
    test = {
        "semantic_only": evaluate_partition(inputs, "test", 0.0),
        "baseline_0_2": evaluate_partition(inputs, "test", 0.2),
        "selected": evaluate_partition(inputs, "test", selected),
    }
    baseline = test["semantic_only"]["aggregate"]
    comparison = {
        name: {
            f"{metric}_pp_vs_semantic": 100 * (report["aggregate"][metric] - baseline[metric])
            for metric in (
                "precision_at_5",
                "precision_at_10",
                "recall_at_5",
                "recall_at_10",
                "hit_rate_at_5",
                "hit_rate_at_10",
                "ndcg_at_5",
                "ndcg_at_10",
            )
        }
        for name, report in test.items()
        if name != "semantic_only"
    }
    return {
        "protocol": {
            "candidate_pool": "train split only; self and same hash group excluded",
            "query_split": "held-out MOOD test",
            "relevance": "shared positive MOOD emotion label",
            "query_emotion_vector": "GATED prediction only",
        },
        "validation_selection": selection,
        "test": test,
        "comparison": comparison,
    }


def _csv_rows(report: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for method, details in report["test"].items():
        aggregate = details["aggregate"]
        rows.append({"scope": "aggregate", "method": method, "emotion": "all", **aggregate})
        for emotion, values in details["per_emotion"].items():
            rows.append({"scope": "per_emotion", "method": method, "emotion": emotion, **values})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Phase 5 retrieval without retraining or rebuilding embeddings"
    )
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--emotion-weights", default="0,0.1,0.2,0.3,0.4")
    args = parser.parse_args(argv)
    weights = tuple(float(value) for value in args.emotion_weights.split(","))
    report = evaluate(load_inputs(args.base, args.bundle), weights)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "phase5_evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    rows = _csv_rows(report)
    with (args.output_dir / "phase5_evaluation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "selected_emotion_weight": report["validation_selection"][
                    "selected_emotion_weight"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
