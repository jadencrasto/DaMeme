"""Make real Phase 6 HTTP requests; expected outputs are never hardcoded."""

from __future__ import annotations

import argparse
import json

import httpx

SENTENCES = {
    "joy": "I finally passed all my exams and I'm so happy! Everything is going great today.",
    "anger": "I've been working on this project for hours and it keeps crashing. I'm so frustrated!",
    "sadness": "I really miss my old friends. We barely talk anymore, and I feel lonely sometimes.",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test a running DaMeme Phase 6 API")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)
    base = args.base_url.rstrip("/")
    with httpx.Client(timeout=30) as client:
        readiness = client.get(f"{base}/api/v1/recommendations/phase5/readiness")
        readiness.raise_for_status()
        if not readiness.json().get("ready"):
            raise RuntimeError(f"recommendation system is not ready: {readiness.json()}")
        results = {}
        for name, text in SENTENCES.items():
            response = client.post(
                f"{base}/api/v1/recommendations/text", json={"text": text, "top_k": args.top_k}
            )
            response.raise_for_status()
            results[name] = response.json()
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
