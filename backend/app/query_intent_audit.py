"""Development-only Phase 7R.5A query-intent research audit."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
BENCHMARKS = BACKEND / "benchmarks"
QUERIES = BENCHMARKS / "phase7r_v1_queries.json"
COMPARISON = BENCHMARKS / "phase7r_v1_local_global_comparison_20261025T054251Z" / "results.json"
EVALUATION = (
    BENCHMARKS / "phase7r_v1_human_eval_analysis_20261025T063017Z" / "evaluation.metrics.json"
)
TAXONOMY_PATH = BENCHMARKS / "phase7r_v1_query_intent_taxonomy_v1.json"


TAXONOMY: dict[str, Any] = {
    "format_version": 1,
    "name": "phase7r-query-intent-taxonomy-v1",
    "scope": "deterministic, user-text-only interpretation for research; not a classifier or production feature",
    "slots": {
        "situation_event": "verbatim or minimally normalized event/situation phrase from the original query",
        "communicative_intent": "one or more constrained intent labels, or unknown",
        "desired_tone": "explicit tone cue, user-selected mood, or unknown",
        "send_as_response": "yes/no/unknown; true only for an explicit response/send/reaction request",
        "ambiguity": "none/low/medium/high with explicit reasons",
        "confidence": "high/medium/low based on explicit lexical evidence, never inferred world knowledge",
    },
    "communicative_intents": [
        {"id": "celebration", "definition": "marking an achievement, good news, or success"},
        {
            "id": "frustration_setback",
            "definition": "an inconvenience, blocked effort, loss, or setback",
        },
        {
            "id": "disbelief_confusion",
            "definition": "surprise, disbelief, or not understanding what happened",
        },
        {
            "id": "sarcasm_irony",
            "definition": "a mismatch between literal wording and intended irony",
        },
        {
            "id": "comfort_empathy",
            "definition": "supporting or acknowledging another person's difficult experience",
        },
        {
            "id": "embarrassment_self_deprecation",
            "definition": "an awkward mistake or self-directed embarrassment",
        },
        {
            "id": "relatable_humor",
            "definition": "sharing a familiar everyday experience for recognition or humor",
        },
        {
            "id": "conversational_reaction",
            "definition": "a short response to send, react with, or reply to another message",
        },
    ],
    "non_inference_rules": [
        "Preserve the original query verbatim as the primary retrieval representation.",
        "Only add labels supported by explicit query wording or an explicit user-selected mood.",
        "Do not invent actors, causes, events, sarcasm, or interpersonal intent.",
        "Keep multiple plausible intent labels rather than selecting one when evidence conflicts.",
        "Use original-query-only retrieval when confidence is low or ambiguity is high.",
    ],
}


def development_queries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("queries")
    if not isinstance(rows, list):
        raise ValueError("query artifact requires a queries list")
    development = [
        row for row in rows if isinstance(row, dict) and row.get("split") == "development"
    ]
    if len(development) != 40:
        raise ValueError(f"expected exactly 40 development queries, got {len(development)}")
    return development


def _failure_summary(evaluation: dict[str, Any]) -> dict[str, Any]:
    failures = evaluation["relevance_failures"]
    groups = (
        "low_situation_top_five",
        "good_tone_poor_situation",
        "contextually_appropriate_not_sendable",
        "strongly_relevant",
    )
    return {
        group: {"count": len(failures.get(group, [])), "examples": failures.get(group, [])[:5]}
        for group in groups
    }


def _repetition_summary(evaluation: dict[str, Any]) -> dict[str, Any]:
    variants = evaluation["repetition_audit"]["variants"]
    return {
        variant: {
            "top_5_cross_query_repetition_rate": data["top_5"]["cross_query_repetition_rate"],
            "top_5_repeated_across_categories": list(
                data["top_5"]["memes_repeated_across_categories"].items()
            )[:10],
            "top_5_most_frequent_memes": data["top_5"]["most_frequent_memes"][:10],
        }
        for variant, data in variants.items()
    }


def build_report() -> dict[str, Any]:
    query_payload = json.loads(QUERIES.read_text(encoding="utf-8"))
    development = development_queries(query_payload)
    comparison = json.loads(COMPARISON.read_text(encoding="utf-8"))
    comparison_by_id = {row["query_id"]: row for row in comparison["queries"]}
    if {row["id"] for row in development} != set(comparison_by_id):
        raise ValueError("comparison queries do not exactly match development query identities")
    evaluation = json.loads(EVALUATION.read_text(encoding="utf-8"))
    return {
        "format_version": 1,
        "experiment": "phase7r.5a-development-query-intent-audit",
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "development-only query wording audit; no held-out query text included",
        "query_count": len(development),
        "category_counts": dict(sorted(Counter(row["category"] for row in development).items())),
        "mood_counts": dict(
            sorted(Counter(row.get("mood") or "none" for row in development).items())
        ),
        "queries": [
            {
                "query_id": row["id"],
                "category": row["category"],
                "original_query": row["message"],
                "selected_mood": row.get("mood"),
                "mood_expanded_query": comparison_by_id[row["id"]]["effective_query"],
                "retrieval_query_sent_to_minilm": comparison_by_id[row["id"]]["effective_query"],
            }
            for row in development
        ],
        "observed_failure_evidence": _failure_summary(evaluation),
        "repetition_evidence": _repetition_summary(evaluation),
        "interpreter_specification": {
            "input": ["original_query", "optional_user_selected_mood"],
            "output": [
                "original_query",
                "explicit_cues",
                "intent_labels",
                "desired_tone",
                "send_as_response",
                "ambiguity",
                "confidence",
                "short_retrieval_variants",
            ],
            "confidence_rules": {
                "high": "explicit situation plus an explicit intent/reaction cue",
                "medium": "explicit situation with only mood or general reaction evidence",
                "low": "underspecified, conflicting, or indirect wording; use original-query-only retrieval",
            },
            "short_retrieval_variants": {
                "default": "original query only",
                "supported": "original query plus compact labels such as 'Reaction intent: celebration.' only when explicit",
                "forbidden": "generated event details, inferred sarcasm, or unconstrained paraphrases",
            },
        },
        "phase7r_5b_plan": {
            "representations": [
                "A0 existing combined",
                "A3-LH existing local/global",
                "A3-LH plus conservative intent representation",
            ],
            "development_protocol": "Define rules before a fresh development evaluation set; do not tune from the eight already-rated comparative queries.",
            "candidate_depth": 100,
            "controls": [
                "existing safety filter",
                "one-per-template diversity",
                "stable source-key tie-breaking",
            ],
            "measurements": [
                "p50/p95 retrieval latency",
                "top-5/top-10 overlap",
                "candidate novelty",
                "per-query result count",
                "session repetition separately",
            ],
            "human_evaluation": "Use a preregistered, fresh development subset with paired identity-level ratings; retain the current eight-query set for descriptive regression checks only.",
        },
    }


def run(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit directory: {output}")
    report = build_report()
    output.mkdir(parents=True)
    (output / "development_query_intent_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    TAXONOMY_PATH.write_text(json.dumps(TAXONOMY, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create the Phase 7R.5A development-only query-intent audit"
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = (
        args.output
        or BENCHMARKS / f"phase7r_v1_query_intent_audit_{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    )
    report = run(output)
    print(f"PASS | audited {report['query_count']} development queries")
    print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
