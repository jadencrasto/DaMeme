"""Development-only, lexical situation reranking for Phase 7R.7C.

The module deliberately operates on saved, already-filtered baseline candidates.
It does not alter retrieval, catalog data, safety policy, or template diversity.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
BENCHMARKS = BACKEND / "benchmarks"
QUERIES = BENCHMARKS / "phase7r_v1_queries.json"
FOUR_WAY = BENCHMARKS / "phase7r_v1_four_way_comparison_20261025T074000Z" / "results.json"
CATALOG = ROOT / "data" / "retrieval" / "catalogs" / "memematch_1000_v2"
METADATA = CATALOG / "memematch_1000_metadata_v2.csv"
RATINGS = BENCHMARKS / "phase7r_v1_multimodal_human_eval_20261025T075000Z" / "ratings.multimodal-evaluator-1.json"
RESULT_LIMIT = 10
RULE_VERSION = "phase7r.7c.1"
MAX_PROMOTION = 1
VARIANTS = ("A0", "A3-LH", "M1", "R-T", "R-V")

TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
GENERIC_TERMS = frozenset(
    "a an and are as at be because but by can could do for from get got have how i if in is it just"
    " like me my of oh on or so that the then this to was we what when with you your today thing"
    " things feeling feel feels mood happy sad angry upset stressed stress tired funny meme reaction"
    " work working school class day really very about after before now someone something anything"
    " will would should could needed need needs group person people friend friends last says been looking"
    " format template popular self time days completely different thought made make".split()
)
AMBIGUOUS_MARKERS = ("not sure", "i guess", "maybe", "kind of", "sort of", "whatever", "something")
NEGATION = frozenset(("no", "not", "never", "without", "dont", "don't", "cant", "can't"))
NEGATION_FILLERS = frozenset(("a", "an", "the", "any", "my", "your", "this", "that"))


@dataclass(frozen=True)
class SituationEvidence:
    """A conservative lexical result; zero means retain the baseline order."""

    matched_terms: tuple[str, ...]
    negated_terms: tuple[str, ...]
    reason: str

    @property
    def is_strong(self) -> bool:
        return bool(self.matched_terms)


def _tokens(value: str) -> list[str]:
    return TOKEN.findall(value.lower())


def query_concepts(query: str) -> tuple[set[str], set[str], str | None]:
    """Extract explicit, non-mood lexical concepts without semantic expansion."""
    lowered = query.lower()
    if any(marker in lowered for marker in AMBIGUOUS_MARKERS):
        return set(), set(), "ambiguous_query"
    tokens = _tokens(query)
    negated: set[str] = set()
    for index, token in enumerate(tokens[:-1]):
        if token in NEGATION:
            following = next(
                (value for value in tokens[index + 1 :] if value not in NEGATION_FILLERS), None
            )
            if following:
                negated.add(following)
    concepts = {
        token
        for token in tokens
        if len(token) >= 4 and token not in GENERIC_TERMS and token not in negated
    }
    if not concepts:
        return concepts, negated, "no_explicit_situation_concept"
    return concepts, negated, None


def situation_match(query: str, candidate_text: str) -> SituationEvidence:
    """Return only directly evidenced lexical matches, never inferred concepts."""
    concepts, negated, fallback = query_concepts(query)
    if fallback:
        return SituationEvidence((), tuple(sorted(negated)), fallback)
    evidence = set(_tokens(candidate_text))
    matched = tuple(sorted(concepts.intersection(evidence)))
    return SituationEvidence(
        matched,
        tuple(sorted(negated)),
        "explicit_lexical_situation_match" if matched else "weak_or_absent_situation_evidence",
    )


def rerank_candidates(
    candidates: list[dict[str, Any]], *, query: str, candidate_text: dict[str, str], max_promotion: int = MAX_PROMOTION
) -> list[dict[str, Any]]:
    """Stable bounded promotion over an unchanged, pre-filtered candidate set."""
    if max_promotion < 0:
        raise ValueError("max_promotion must be non-negative")
    audited = []
    for original_rank, candidate in enumerate(candidates, start=1):
        key = str(candidate["source_key"])
        evidence = situation_match(query, candidate_text.get(key, ""))
        target = max(1, original_rank - max_promotion) if evidence.is_strong else original_rank
        audited.append((target, original_rank, candidate, evidence))
    ordered = sorted(audited, key=lambda row: (row[0], row[1], str(row[2]["source_key"])))
    result = []
    for reranked_rank, (_, original_rank, candidate, evidence) in enumerate(ordered, start=1):
        row = dict(candidate)
        row.update(
            {
                "baseline_final_rank": original_rank,
                "reranked_rank": reranked_rank,
                "rank_displacement": original_rank - reranked_rank,
                "situation_adjustment": {
                    "max_promotion": max_promotion,
                    "matched_terms": list(evidence.matched_terms),
                    "negated_terms": list(evidence.negated_terms),
                    "reason": evidence.reason,
                },
            }
        )
        result.append(row)
    if [row["source_key"] for row in result] != [row["source_key"] for row in candidates] and any(
        abs(int(row["rank_displacement"])) > max_promotion for row in result
    ):
        raise AssertionError("bounded reranking exceeded maximum movement")
    return result


EVIDENCE_FIELDS = (
    "local_Extracted Text + Title",
    "local_context",
    "global_context",
    "global_Caption",
)


def candidate_evidence_text(row: dict[str, str]) -> str:
    """Return source content only; usage and template labels are excluded."""
    return " ".join(row.get(name, "") or "" for name in EVIDENCE_FIELDS)


def _candidate_texts(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        result = {}
        for row in rows:
            key = row["source_key"]
            result[key] = candidate_evidence_text(row)
    return result


def _keys(rows: list[dict[str, Any]]) -> list[str]:
    return [str(row["source_key"]) for row in rows]


def _latency_ms(variant: dict[str, Any]) -> float:
    """Read saved baseline timing without pretending the measurements are identical."""
    for field in ("latency_ms", "measured_openclip_query_latency_ms"):
        if field in variant:
            return float(variant[field])
    raise ValueError("variant has no recorded latency")


def _variant_summary(records: list[dict[str, Any]], name: str) -> dict[str, Any]:
    rankings = [_keys(row["variants"][name]["final_top_10"]) for row in records]
    a0 = [_keys(row["variants"]["A0"]["final_top_10"]) for row in records]
    all_keys = [key for ranking in rankings for key in ranking]
    return {
        "unique_final_memes": len(set(all_keys)),
        "cross_query_repetition_rate": 1 - len(set(all_keys)) / len(all_keys),
        "mean_top_5_overlap_with_a0": sum(
            len(set(ranking[:5]) & set(baseline[:5])) / 5
            for ranking, baseline in zip(rankings, a0, strict=True)
        ) / len(records),
        "mean_top_10_overlap_with_a0": sum(
            len(set(ranking) & set(baseline)) / 10
            for ranking, baseline in zip(rankings, a0, strict=True)
        ) / len(records),
        "median_latency_ms": sorted(
            _latency_ms(row["variants"][name]) for row in records
        )[len(records) // 2],
    }


def _diagnostics(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"variants": {}}
    baseline_for = {"R-T": "A3-LH", "R-V": "M1"}
    for name, baseline in baseline_for.items():
        changes = [
            item
            for row in records
            for item in row["variants"][name]["final_top_10"]
            if item["rank_displacement"]
        ]
        parity = all(
            set(_keys(row["variants"][name]["final_top_10"]))
            == set(_keys(row["variants"][baseline]["final_top_10"]))
            for row in records
        )
        result["variants"][name] = {
            "baseline": baseline,
            "promotions": sum(item["rank_displacement"] > 0 for item in changes),
            "demotions": sum(item["rank_displacement"] < 0 for item in changes),
            "maximum_absolute_rank_displacement": max(
                (abs(item["rank_displacement"]) for item in changes), default=0
            ),
            "queries_with_any_rank_change": len(
                {row["query_id"] for row in records if any(item["rank_displacement"] for item in row["variants"][name]["final_top_10"])}
            ),
            "safety_and_template_candidate_set_parity": parity,
            "weak_evidence_fallback_queries": sum(
                all(not item["situation_adjustment"]["matched_terms"] for item in row["variants"][name]["final_top_10"])
                for row in records
            ),
        }
    return result


def _descriptive_ratings(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Report only observed legacy ratings; absent query--meme pairs stay absent."""
    payload = json.loads(RATINGS.read_text(encoding="utf-8"))
    evaluators = payload.get("evaluators", {})
    observed = [value for value in evaluators.values() if isinstance(value, dict)]
    result: dict[str, Any] = {"rating_file": str(RATINGS), "runtime_feature": False, "variants": {}}
    for name in VARIANTS:
        values: dict[str, list[float]] = {axis: [] for axis in ("situation", "tone", "sendability")}
        values_at_5: dict[str, list[float]] = {axis: [] for axis in values}
        for row in records:
            for rank, item in enumerate(row["variants"][name]["final_top_10"], start=1):
                key = f"{row['query_id']}:{item['source_key']}"
                ratings = [evaluator[key] for evaluator in observed if key in evaluator]
                if not ratings:
                    continue
                for axis in values:
                    mean = sum(float(rating[axis]) for rating in ratings) / len(ratings)
                    values[axis].append(mean)
                    if rank <= 5:
                        values_at_5[axis].append(mean)
        result["variants"][name] = {
            "observed_rated_slots": len(values["situation"]),
            "mean_observed_rating": {axis: sum(rows) / len(rows) if rows else None for axis, rows in values.items()},
            "top_5_mean_observed_rating": {axis: sum(rows) / len(rows) if rows else None for axis, rows in values_at_5.items()},
        }
    return result


def run(output: Path) -> dict[str, Any]:
    """Create the saved-candidate development diagnostic, refusing overwrite."""
    from app.benchmark import load_queries

    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing experiment directory: {output}")
    queries, _ = load_queries(QUERIES)
    source = json.loads(FOUR_WAY.read_text(encoding="utf-8"))
    source_rows = {str(row["query_id"]): row for row in source["queries"]}
    development = {query.id: query for query in queries.values() if query.split == "development"}
    if set(source_rows) != set(development) or len(development) != 40:
        raise ValueError("saved comparison must contain exactly the 40 development queries")
    text = _candidate_texts(METADATA)
    records = []
    for query_id in sorted(development):
        query, source_row = development[query_id], source_rows[query_id]
        if source_row.get("split") != "development":
            raise ValueError("held-out query reached reranking prototype")
        variants = {name: dict(source_row["variants"][name]) for name in ("A0", "A3-LH", "M1")}
        for name in ("A0", "A3-LH", "M1"):
            candidates = variants[name]["final_top_10"]
            if len(candidates) != RESULT_LIMIT or len(set(_keys(candidates))) != RESULT_LIMIT:
                raise ValueError(f"{query_id} {name} does not have ten unique filtered candidates")
            if any(key not in text for key in _keys(candidates)):
                raise ValueError(f"{query_id} {name} has candidate without catalog metadata")
        for reranked, baseline in (("R-T", "A3-LH"), ("R-V", "M1")):
            started = time.perf_counter()
            final = rerank_candidates(variants[baseline]["final_top_10"], query=query.message, candidate_text=text)
            variants[reranked] = {
                "final_top_10": final,
                "baseline_variant": baseline,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "candidate_pool": "unchanged baseline final_top_10 after existing safety and one-per-template diversity",
            }
        records.append(
            {
                "query_id": query_id,
                "category": query.category,
                "original_query": query.message,
                "selected_mood": query.mood,
                "effective_query": source_row["effective_query"],
                "variants": variants,
            }
        )
    report = {
        "format_version": 1,
        "experiment": "phase7r.7c.1-final-lexical-reranking-prototype",
        "created_at": datetime.now(UTC).isoformat(),
        "scope": "development-only, saved candidates only; no ratings are runtime features",
        "provenance": {
            "saved_comparison": str(FOUR_WAY),
            "catalog_metadata": str(METADATA),
            "catalog_identity": source.get("provenance", {}),
            "source_key_mapping_verified": True,
            "query_representation": "saved effective_query retained for baseline provenance; original query only drives lexical reranking",
        },
        "heuristic": {
            "rule_version": RULE_VERSION,
            "type": "exact token overlap between explicit original-query concepts and source OCR/local context/global context/BLIP caption",
            "excluded_evidence_fields": ["usage", "template_group_heuristic"],
            "generic_or_mood_terms": "excluded, including function and broad relational terms",
            "negation": "negated following token is never a positive feature",
            "ambiguity": "ambiguous queries fall back unchanged",
            "max_promotion": MAX_PROMOTION,
            "no_human_ratings_as_runtime_features": True,
        },
        "repetition": "No session-level exposure control is applied; it remains a separate later experiment.",
        "queries": records,
    }
    report["summary"] = {
        "variants": {name: _variant_summary(records, name) for name in VARIANTS},
        "rank_changes": _diagnostics(records),
        "descriptive_existing_ratings": _descriptive_ratings(records),
    }
    output.mkdir(parents=True)
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(report["summary"], indent=2) + "\n", encoding="utf-8")
    (output / "rank_changes.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase 7R.7A saved-candidate reranking")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    output = args.output or BENCHMARKS / f"phase7r_v1_situation_reranking_{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    report = run(output)
    for name, values in report["summary"]["variants"].items():
        print(f"{name}: overlap@5={values['mean_top_5_overlap_with_a0']:.3f}; median_ms={values['median_latency_ms']:.3f}")
    print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
