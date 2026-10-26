"""Create launch-only UI schema adapters without changing frozen evaluation artifacts."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.benchmark import canonical_digest, load_queries, query_digest


ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "backend" / "benchmarks" / "phase7r_v1_multimodal_human_eval_20261025T075000Z"
QUERIES = ROOT / "backend" / "benchmarks" / "phase7r_v1_queries.json"


def main() -> None:
    query_rows, queries = load_queries(QUERIES)
    blinded = json.loads((DIRECTORY / "blinded.pool.json").read_text(encoding="utf-8"))
    frozen = json.loads((DIRECTORY / "complete_comparative.assignment.json").read_text(encoding="utf-8"))
    pool = {
        "format_version": 1,
        "runs": [{"id": "blinded", "split": "development", "query_results": {}}],
        "benchmark_id": queries["benchmark_id"],
        "query_digest": query_digest(query_rows),
        "pairs": [
            {**pair, "contributions": [{"run_id": "blinded", "rank": 1}]}
            for pair in blinded["pairs"]
        ],
    }
    path = DIRECTORY / "ui.pool.json"
    path.write_text(json.dumps(pool, indent=2) + "\n", encoding="utf-8")
    assignment = {"format_version": 1, "pool_digest": canonical_digest(pool), "split": "development", "pair_keys": frozen["pair_keys"]}
    (DIRECTORY / "ui.assignment.json").write_text(json.dumps(assignment, indent=2) + "\n", encoding="utf-8")
    (DIRECTORY / "ratings.multimodal-evaluator-1.json").write_text(json.dumps({"format_version": 1, "pool_digest": canonical_digest(pool), "evaluators": {}}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
