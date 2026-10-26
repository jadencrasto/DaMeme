"""Offline, development-only A0/A3-LH/M1/M2 comparison from saved candidates."""
# ruff: noqa: E701, E702, F401
from __future__ import annotations

import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.local_global_comparison import CANDIDATE_DEPTH, Candidate, _post_filter
from app.moods import primary_classifier_emotion, resolve_mood
from app.recommendations import MemeMatchMetadata, RecommendationService
from app.retrieval import RetrievalService

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
CATALOG = ROOT / "data" / "retrieval" / "catalogs" / "memematch_1000_v2"
TEXT = BACKEND / "benchmarks" / "phase7r_v1_local_global_comparison_20261025T054251Z" / "results.json"
M1 = BACKEND / "benchmarks" / "phase7r_v1_openclip_m1_raw.json"
K = 60


def rrf(a3: list[dict[str, Any]], m1: list[dict[str, Any]], k: int = K) -> list[dict[str, Any]]:
    """Deduplicate depth-limited rankings and return a stable rank-only fusion."""
    merged: dict[str, dict[str, Any]] = {}
    for label, rows in (("a3", a3), ("m1", m1)):
        seen: set[str] = set()
        for row in rows:
            key, rank = str(row["source_key"]), int(row["raw_rank"])
            if key in seen:
                continue
            seen.add(key)
            value = merged.setdefault(key, {"source_key": key, "a3_rank": None, "m1_rank": None})
            value[f"{label}_rank"] = rank
    for value in merged.values():
        value["rrf_score"] = sum(1 / (k + value[name]) for name in ("a3_rank", "m1_rank") if value[name])
    return sorted(merged.values(), key=lambda value: (-value["rrf_score"], value["source_key"]))[:CANDIDATE_DEPTH]


def _m1_candidates(rows: list[dict[str, Any]]) -> list[Candidate]:
    return [Candidate(str(row["source_key"]), float(row["raw_score"]), None, None, None, None) for row in rows]


def _m2_candidates(rows: list[dict[str, Any]]) -> list[Candidate]:
    return [Candidate(str(row["source_key"]), float(row["rrf_score"]), None, None, row["a3_rank"], row["m1_rank"]) for row in rows]


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    variants = ("A0", "A3-LH", "M1", "M2")
    final = {name: [[item["source_key"] for item in row["variants"][name]["final_top_10"]] for row in rows] for name in variants}
    a0, a3 = final["A0"], final["A3-LH"]
    result: dict[str, Any] = {"variants": {}}
    for name in variants:
        all_keys = set().union(*map(set, final[name]))
        repetition = 1 - len(all_keys) / sum(len(value) for value in final[name])
        result["variants"][name] = {"unique_final_memes": len(all_keys), "cross_query_repetition_rate": repetition, "mean_top5_overlap_a0": sum(len(set(value[:5]) & set(base[:5])) / 5 for value, base in zip(final[name], a0, strict=True)) / len(rows), "mean_top10_overlap_a0": sum(len(set(value) & set(base)) / 10 for value, base in zip(final[name], a0, strict=True)) / len(rows), "mean_top5_overlap_a3_lh": sum(len(set(value[:5]) & set(base[:5])) / 5 for value, base in zip(final[name], a3, strict=True)) / len(rows), "new_final_candidates_vs_a0": len(all_keys - set().union(*map(set, a0))), "queries_below_10": sum(len(value) < 10 for value in final[name])}
    return result


def run(output: Path) -> dict[str, Any]:
    if output.exists(): raise FileExistsError(output)
    text = json.loads(TEXT.read_text(encoding="utf-8")); m1 = json.loads(M1.read_text(encoding="utf-8"))
    text_rows = {row["query_id"]: row for row in text["queries"]}; m1_rows = {row["query_id"]: row for row in m1["queries"]}
    if len(text_rows) != len(m1_rows) != 40 or set(text_rows) != set(m1_rows): raise ValueError("expected matching 40 development query ids")
    a0 = RetrievalService.load(CATALOG / "indexes"); metadata = MemeMatchMetadata.from_csv(CATALOG / "memematch_1000_metadata_v2.csv", CATALOG / "images"); service = RecommendationService(a0, metadata)
    rows=[]
    for query_id in sorted(text_rows):
        base, aligned = text_rows[query_id], m1_rows[query_id]
        if base["split"] != "development" or base["effective_query"] != aligned["effective_query"]: raise ValueError("query split or representation mismatch")
        a3raw=base["variants"]["A3-LH"]["raw_candidates"]; m1raw=aligned["candidates"]
        if len(a3raw) != len(m1raw) != 100 or len({x["source_key"] for x in m1raw}) != 100: raise ValueError("raw depth-100 candidates invalid")
        fusion=rrf(a3raw,m1raw); emotion=primary_classifier_emotion(resolve_mood(base["mood"])); started=time.perf_counter(); m1final,m1counts=_post_filter(service,_m1_candidates(m1raw),emotion); m2final,m2counts=_post_filter(service,_m2_candidates(fusion),emotion)
        rows.append({"query_id":query_id,"split":"development","effective_query":base["effective_query"],"mood":base["mood"],"variants":{"A0":base["variants"]["A0"],"A3-LH":base["variants"]["A3-LH"],"M1":{"raw_candidates":m1raw,"final_top_10":m1final,"candidate_counts":m1counts,"measured_openclip_query_latency_ms":aligned["latency_ms"]},"M2":{"raw_candidates":fusion,"final_top_10":m2final,"candidate_counts":m2counts,"cached_candidate_fusion_filter_latency_ms":(time.perf_counter()-started)*1000}}})
    report={"format_version":1,"experiment":"phase7r.6b.3-four-way-development-comparison","created_at":datetime.now(UTC).isoformat(),"scope":{"split":"development","query_count":40,"candidate_depth":100,"rrf_k":60},"provenance":{"m1_raw":str(M1),"text_baseline":str(TEXT),"openclip_model":m1["model"],"pretrained":m1["pretrained"]},"query_handling":"All variants use the recorded mood-expanded effective query; M1 is the matching OpenCLIP text tower. M2 fuses ranks only.","queries":rows,"summary":_summary(rows)}
    output.mkdir(parents=True); (output/"results.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8"); (output/"summary.json").write_text(json.dumps(report["summary"],indent=2)+"\n",encoding="utf-8"); return report

