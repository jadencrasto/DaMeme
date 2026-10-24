"""Local, blinded human relevance review for a catalog-evaluation JSON export."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mimetypes
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

FORMAT_VERSION = 1
RATING_SCALE = {0, 1, 2, 3}


@dataclass(frozen=True)
class EvaluationItem:
    key: str
    query_id: str
    query: str
    meme_id: str


@dataclass(frozen=True)
class EvaluationDataset:
    digest: str
    items: dict[str, EvaluationItem]
    rankings: dict[str, dict[str, list[str]]]

    @property
    def required_keys(self) -> set[str]:
        return {key for rankings in self.rankings.values() for ids in rankings.values() for key in ids}


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_export(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to load evaluation export: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("queries"), list):
        raise ValueError("evaluation export requires a queries list")
    return payload


def build_dataset(payload: dict[str, Any], *, cutoff: int = 5) -> EvaluationDataset:
    """Create stable, deduplicated query--meme rating tasks from top-k rankings."""
    if cutoff < 1:
        raise ValueError("cutoff must be positive")
    items: dict[str, EvaluationItem] = {}
    rankings: dict[str, dict[str, list[str]]] = {}
    digest_payload: list[dict[str, object]] = []
    for query_index, query_row in enumerate(payload.get("queries", [])):
        if not isinstance(query_row, dict) or not isinstance(query_row.get("query"), str):
            raise ValueError("evaluation query rows require query text")
        query = query_row["query"]
        query_id = f"q{query_index}-{_digest(query)[:12]}"
        modes = query_row.get("modes")
        if not isinstance(modes, dict):
            raise ValueError("evaluation query rows require modes")
        rankings[query_id] = {}
        digest_modes: dict[str, list[str]] = {}
        for mode, mode_row in sorted(modes.items()):
            results = mode_row.get("results") if isinstance(mode_row, dict) else None
            if not isinstance(results, list) or len(results) < cutoff:
                raise ValueError(f"{query!r} {mode!r} has fewer than {cutoff} results")
            keys: list[str] = []
            for result in results[:cutoff]:
                meme_id = result.get("meme_id") if isinstance(result, dict) else None
                if not isinstance(meme_id, str) or not meme_id:
                    raise ValueError(f"{query!r} {mode!r} has a result without meme_id")
                key = f"{query_id}:{meme_id}"
                items.setdefault(key, EvaluationItem(key, query_id, query, meme_id))
                keys.append(key)
            rankings[query_id][mode] = keys
            digest_modes[mode] = keys
        digest_payload.append({"query_id": query_id, "query": query, "modes": digest_modes})
    if not items:
        raise ValueError("evaluation export contains no rating candidates")
    return EvaluationDataset(_digest(digest_payload), items, rankings)


def load_ratings(path: Path, dataset: EvaluationDataset) -> dict[str, int]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to load ratings file: {path}") from exc
    if not isinstance(payload, dict) or payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported ratings file format")
    if payload.get("evaluation_digest") != dataset.digest:
        raise ValueError("ratings file belongs to a different evaluation export")
    raw_ratings = payload.get("ratings", {})
    if not isinstance(raw_ratings, dict):
        raise ValueError("ratings file has invalid ratings")
    ratings = {key: value for key, value in raw_ratings.items() if key in dataset.items}
    if any(isinstance(value, bool) or not isinstance(value, int) or value not in RATING_SCALE for value in ratings.values()):
        raise ValueError("ratings must be integers from 0 through 3")
    return ratings


def save_ratings(path: Path, dataset: EvaluationDataset, ratings: dict[str, int]) -> None:
    if any(
        key not in dataset.items
        or isinstance(value, bool)
        or not isinstance(value, int)
        or value not in RATING_SCALE
        for key, value in ratings.items()
    ):
        raise ValueError("ratings must target evaluation candidates and use the 0--3 scale")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "evaluation_digest": dataset.digest,
        "ratings": dict(sorted(ratings.items())),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def progress(dataset: EvaluationDataset, ratings: dict[str, int]) -> dict[str, int]:
    required = dataset.required_keys
    completed = len(required.intersection(ratings))
    return {"completed": completed, "total": len(required), "remaining": len(required) - completed}


def _dcg(values: list[int]) -> float:
    return sum((2**rating - 1) / math.log2(position + 2) for position, rating in enumerate(values))


def metrics(dataset: EvaluationDataset, ratings: dict[str, int], *, cutoff: int = 5) -> dict[str, object]:
    if cutoff != 5:
        raise ValueError("Phase 7A.1 reports Precision@5 and nDCG@5 only")
    status = progress(dataset, ratings)
    if status["remaining"]:
        return {"complete": False, "progress": status, "missing_rating_keys": sorted(dataset.required_keys - ratings.keys())}
    by_mode: dict[str, list[dict[str, float]]] = defaultdict(list)
    for query_id, mode_rows in sorted(dataset.rankings.items()):
        for mode, keys in sorted(mode_rows.items()):
            values = [ratings[key] for key in keys]
            precision = sum(rating >= 2 for rating in values) / cutoff
            ideal = sorted(values, reverse=True)
            denominator = _dcg(ideal)
            ndcg = _dcg(values) / denominator if denominator else 0.0
            by_mode[mode].append({"precision_at_5": precision, "ndcg_at_5": ndcg, "query_id": query_id})
    modes: dict[str, object] = {}
    for mode, rows in sorted(by_mode.items()):
        modes[mode] = {
            "queries": rows,
            "mean_precision_at_5": sum(row["precision_at_5"] for row in rows) / len(rows),
            "mean_ndcg_at_5": sum(row["ndcg_at_5"] for row in rows) / len(rows),
        }
    return {"complete": True, "progress": status, "modes": modes}


def _image_path(image_root: Path, meme_id: str) -> Path | None:
    root = image_root.resolve()
    filename = Path(meme_id).name
    for candidate in (root / "images" / filename, root / filename):
        try:
            resolved = candidate.resolve()
            resolved.relative_to(root)
        except ValueError:
            continue
        if resolved.is_file():
            return resolved
    return None


class RatingUpdate(BaseModel):
    key: str
    rating: int = Field(ge=0, le=3)


def create_app(dataset: EvaluationDataset, ratings_path: Path, image_root: Path) -> FastAPI:
    ratings = load_ratings(ratings_path, dataset)
    app = FastAPI(title="DaMeme Human Relevance Evaluation", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _HTML

    @app.get("/api/state")
    def state() -> dict[str, object]:
        # Do not expose source ranking mode, rank, or model scores to raters.
        return {
            "progress": progress(dataset, ratings),
            "items": [
                {
                    "key": item.key,
                    "query": item.query,
                    "filename": item.meme_id,
                    "rating": ratings.get(item.key),
                    "image_url": f"/images/{quote(item.key, safe='')}",
                }
                for item in dataset.items.values()
            ],
        }

    @app.post("/api/ratings")
    def update_rating(update: RatingUpdate) -> dict[str, object]:
        if update.key not in dataset.items:
            raise HTTPException(404, "unknown evaluation candidate")
        ratings[update.key] = update.rating
        save_ratings(ratings_path, dataset, ratings)
        return {"progress": progress(dataset, ratings)}

    @app.get("/images/{key}")
    def image(key: str) -> FileResponse:
        item = dataset.items.get(key)
        path = _image_path(image_root, item.meme_id) if item else None
        if path is None:
            raise HTTPException(404, "image is not available under the configured catalog root")
        return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0])

    return app


_HTML = """<!doctype html><html><head><meta charset=utf-8><title>DaMeme relevance review</title>
<style>body{font:16px system-ui;margin:2rem;max-width:56rem}#items{display:grid;gap:1rem}.item{border:1px solid #bbb;padding:1rem}.item img{display:block;max-width:100%;max-height:32rem;object-fit:contain;margin:.75rem 0}.buttons button{margin-right:.5rem;padding:.4rem .7rem}.selected{background:#174ea6;color:white}small{color:#555}</style>
</head><body><h1>DaMeme relevance review</h1><p id=progress></p><p>Rate how relevant each meme is to its query. Scores, source mode, and rank are hidden.</p><div id=items></div>
<script>const labels=['0 — Irrelevant','1 — Slightly relevant','2 — Relevant','3 — Highly relevant'];
async function load(){const state=await (await fetch('/api/state')).json();document.querySelector('#progress').textContent=`Progress: ${state.progress.completed} / ${state.progress.total}`;const target=document.querySelector('#items');target.innerHTML='';for(const item of state.items){const node=document.createElement('section');node.className='item';node.innerHTML=`<strong>Query:</strong> ${escape(item.query)}<br><small>Filename: ${escape(item.filename)}</small><img src="${item.image_url}" alt="${escape(item.filename)}"><div class=buttons></div>`;const buttons=node.querySelector('.buttons');labels.forEach((label,rating)=>{const button=document.createElement('button');button.textContent=label;if(item.rating===rating)button.className='selected';button.onclick=async()=>{await fetch('/api/ratings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key:item.key,rating})});load()};buttons.append(button)});target.append(node)}}function escape(text){const span=document.createElement('span');span.textContent=text;return span.innerHTML}load();</script></body></html>"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run or report a blinded local DaMeme relevance evaluation")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--evaluation", type=Path, required=True)
    serve.add_argument("--ratings", type=Path, required=True)
    serve.add_argument("--catalog-root", type=Path, required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8090)
    report = commands.add_parser("report")
    report.add_argument("--evaluation", type=Path, required=True)
    report.add_argument("--ratings", type=Path, required=True)
    report.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    dataset = build_dataset(load_export(args.evaluation))
    if args.command == "serve":
        uvicorn.run(create_app(dataset, args.ratings, args.catalog_root), host=args.host, port=args.port)
        return 0
    result = metrics(dataset, load_ratings(args.ratings, dataset))
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
