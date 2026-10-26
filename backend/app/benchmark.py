"""Versioned, human-rated benchmark utilities for contextual meme retrieval.

The module deliberately keeps benchmark judgments separate from production ranking.
It never changes catalog artifacts and treats an absent judgment as unknown.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mimetypes
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from app.core.config import Settings
from app.recommendation_engine import RecommendationEngine
from app.recommendations import RankingWeights

QUERY_FORMAT_VERSION = 1
POOL_FORMAT_VERSION = 1
RATINGS_FORMAT_VERSION = 1
PILOT_FORMAT_VERSION = 1
RATING_SCALE = {0, 1, 2, 3}
AXES = ("situation", "tone", "sendability")
SPLITS = ("development", "test")
FINAL_CONFIRMATION = "ARCHITECTURE_FROZEN"


def canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class BenchmarkQuery:
    id: str
    message: str
    mood: str | None
    category: str
    notes: str
    split: Literal["development", "test"]


def deterministic_split(query_id: str, category: str, seed: int) -> Literal["development", "test"]:
    """Return a stable split; validation also enforces five per category."""
    digest = hashlib.sha256(f"{seed}:{category}:{query_id}".encode()).hexdigest()
    return "development" if int(digest, 16) % 2 == 0 else "test"


def expected_assignments(rows: list[dict[str, Any]], seed: int) -> dict[str, str]:
    """Stratify deterministically: the lower five digests in each category are development."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["category"])].append(row)
    assignments: dict[str, str] = {}
    for category, entries in grouped.items():
        ordered = sorted(
            entries,
            key=lambda row: hashlib.sha256(
                f"{seed}:{_category_token(category)}:{row['id']}".encode()
            ).hexdigest(),
        )
        for position, row in enumerate(ordered):
            assignments[str(row["id"])] = "development" if position < 5 else "test"
    return assignments


def _category_token(category: str) -> str:
    """Stable short category identifiers used for v1 split hashing."""
    tokens = {
        "Everyday situations": "everyday",
        "Academic and workplace stress": "academic",
        "Celebrations and achievements": "celebration",
        "Negative emotions and setbacks": "setback",
        "Sarcasm and irony": "sarcasm",
        "Nostalgia and memories": "nostalgia",
        "Meme-template references": "template",
        "Reaction and response intent": "reaction",
    }
    return tokens.get(category, category)


def load_queries(path: Path) -> tuple[dict[str, BenchmarkQuery], dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to load benchmark queries: {path}") from exc
    if not isinstance(payload, dict) or payload.get("format_version") != QUERY_FORMAT_VERSION:
        raise ValueError("unsupported benchmark query format")
    rows, seed = payload.get("queries"), payload.get("split_seed")
    if not isinstance(rows, list) or not isinstance(seed, int):
        raise ValueError("benchmark queries require queries and split_seed")
    if len(rows) != 80:
        raise ValueError("benchmark v1 requires exactly 80 queries")
    expected = expected_assignments(rows, seed)
    queries: dict[str, BenchmarkQuery] = {}
    categories: dict[str, int] = defaultdict(int)
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("benchmark query rows must be objects")
        query_id = row.get("id")
        message, category, notes, split, mood = (
            row.get("message"), row.get("category"), row.get("context_notes"), row.get("split"), row.get("mood"),
        )
        if (
            not isinstance(query_id, str) or not query_id or query_id in queries
            or not isinstance(message, str) or not message.strip()
            or not isinstance(category, str) or not category
            or not isinstance(notes, str) or not notes.strip()
            or split not in SPLITS or (mood is not None and not isinstance(mood, str))
        ):
            raise ValueError("benchmark query row is invalid")
        if split != expected[query_id]:
            raise ValueError(f"query {query_id} does not match deterministic stratified split")
        queries[query_id] = BenchmarkQuery(query_id, message, mood, category, notes, split)
        categories[category] += 1
    if len(categories) != 8 or any(count != 10 for count in categories.values()):
        raise ValueError("benchmark v1 requires eight categories with ten queries each")
    if sum(query.split == "development" for query in queries.values()) != 40:
        raise ValueError("benchmark v1 requires a 40/40 split")
    return queries, payload


def query_digest(queries: dict[str, BenchmarkQuery]) -> str:
    return canonical_digest([
        {"id": query.id, "message": query.message, "mood": query.mood, "category": query.category,
         "notes": query.notes, "split": query.split}
        for query in sorted(queries.values(), key=lambda item: item.id)
    ])


def _catalog_provenance(settings: Settings) -> dict[str, Any]:
    catalog = settings.active_catalog()
    paths = {
        "metadata_csv": catalog.metadata_csv,
        "index_manifest": catalog.index_dir / "manifest.json",
        "export_metadata": catalog.export_metadata,
    }
    missing = [str(path) for path in paths.values() if path is None or not Path(path).is_file()]
    if missing:
        raise ValueError(f"catalog provenance is incomplete: {', '.join(missing)}")
    return {
        "catalog_id": catalog.catalog_id,
        "expected_count": catalog.expected_count,
        "metadata_csv_sha256": file_digest(catalog.metadata_csv),
        "index_manifest_sha256": file_digest(catalog.index_dir / "manifest.json"),
        "export_metadata_sha256": file_digest(catalog.export_metadata),
    }


def _load_pool(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"format_version": POOL_FORMAT_VERSION, "runs": [], "pairs": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to load candidate pool: {path}") from exc
    if not isinstance(payload, dict) or payload.get("format_version") != POOL_FORMAT_VERSION:
        raise ValueError("unsupported candidate pool format")
    if not isinstance(payload.get("runs"), list) or not isinstance(payload.get("pairs"), list):
        raise ValueError("candidate pool requires runs and pairs")
    return payload


def _save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _allow_split(split: str, final_confirmation: str | None) -> None:
    if split not in SPLITS:
        raise ValueError("split must be development or test")
    if split == "test" and final_confirmation != FINAL_CONFIRMATION:
        raise ValueError("test pooling/reporting requires --final-confirmation ARCHITECTURE_FROZEN")


def add_semantic_pool(
    queries: dict[str, BenchmarkQuery], query_payload: dict[str, Any], output: Path, *, split: str,
    limit: int, settings: Settings, final_confirmation: str | None = None,
) -> dict[str, int]:
    """Append one deterministic semantic run, preserving all prior pair judgments."""
    _allow_split(split, final_confirmation)
    if limit < 1 or limit > 50:
        raise ValueError("candidate limit must be between 1 and 50")
    selected = [query for query in queries.values() if query.split == split]
    if not selected:
        raise ValueError("no queries match the requested split")
    provenance = _catalog_provenance(settings)
    catalog = settings.active_catalog()
    engine = RecommendationEngine(
        catalog.index_dir, catalog.metadata_csv, catalog.image_root, catalog_id=catalog.catalog_id,
        expected_count=catalog.expected_count, expected_dimensions=catalog.expected_dimensions,
        export_records=catalog.export_records, export_metadata=catalog.export_metadata,
        cross_encoder_model=settings.recommendation_cross_encoder_model,
    )
    config = {
        "retrieval_mode": "semantic", "candidate_limit": limit, "bm25_candidate_limit": limit,
        "cross_encoder_limit": 0, "weights": {"semantic": 0.90, "emotion": 0.08, "sentiment": 0.02},
        "min_semantic_similarity": None, "max_per_template": 1, "exclude_unsafe": True,
        "query_encoder": "sentence-transformers/all-MiniLM-L6-v2",
    }
    run_id = "semantic-" + canonical_digest({"benchmark": query_payload.get("benchmark_id"), "split": split,
                                                "query_digest": query_digest(queries), "catalog": provenance,
                                                "configuration": config})[:16]
    results: dict[str, list[str]] = {}
    for query in sorted(selected, key=lambda item: item.id):
        rows = engine.recommend_text(
            query.message, emotion=None, sentiment=None, usage=[], weights=RankingWeights(.90, .08, .02),
            candidate_limit=limit, min_semantic_similarity=None, diversity_max_per_template=1,
            exclude_unsafe=True, retrieval_mode="semantic", bm25_candidate_limit=limit,
            cross_encoder_limit=1, mood=query.mood, limit=limit,
        )
        results[query.id] = [row.meme_id for row in rows]
    pool = _load_pool(output)
    digest = query_digest(queries)
    if pool.get("benchmark_id") not in (None, query_payload.get("benchmark_id")) or pool.get("query_digest") not in (None, digest):
        raise ValueError("candidate pool belongs to a different benchmark query set")
    if pool.get("catalog") not in (None, provenance):
        raise ValueError("candidate pool catalog provenance differs; use a new pool file")
    runs = [row for row in pool["runs"] if row.get("id") != run_id]
    runs.append({"id": run_id, "split": split, "model_version": config["query_encoder"], "configuration": config,
                 "catalog": provenance, "query_results": results})
    pairs: dict[str, dict[str, Any]] = {}
    for row in pool["pairs"]:
        if not isinstance(row, dict) or not isinstance(row.get("key"), str):
            raise ValueError("candidate pool has an invalid pair")
        pairs[row["key"]] = row
    for query_id, meme_ids in results.items():
        for rank, meme_id in enumerate(meme_ids, start=1):
            key = f"{query_id}:{meme_id}"
            pair = pairs.setdefault(key, {"key": key, "query_id": query_id, "meme_id": meme_id, "contributions": []})
            contributions = [item for item in pair["contributions"] if item.get("run_id") != run_id]
            contributions.append({"run_id": run_id, "rank": rank})
            pair["contributions"] = sorted(contributions, key=lambda item: (item["run_id"], item["rank"]))
    payload = {"format_version": POOL_FORMAT_VERSION, "benchmark_id": query_payload.get("benchmark_id"),
               "query_digest": digest, "catalog": provenance, "runs": sorted(runs, key=lambda row: row["id"]),
               "pairs": [pairs[key] for key in sorted(pairs)]}
    _save_json(output, payload)
    return {"runs": len(runs), "pairs": len(pairs), "new_run_candidates": sum(len(values) for values in results.values())}


@dataclass(frozen=True)
class PoolDataset:
    queries: dict[str, BenchmarkQuery]
    pool: dict[str, Any]

    @property
    def pairs(self) -> dict[str, dict[str, Any]]:
        return {str(pair["key"]): pair for pair in self.pool["pairs"]}

    def visible_pairs(self, split: str) -> list[dict[str, Any]]:
        return [pair for pair in self.pool["pairs"] if self.queries[pair["query_id"]].split == split]


def load_pool_dataset(query_path: Path, pool_path: Path) -> PoolDataset:
    queries, query_payload = load_queries(query_path)
    pool = _load_pool(pool_path)
    if pool.get("benchmark_id") != query_payload.get("benchmark_id") or pool.get("query_digest") != query_digest(queries):
        raise ValueError("candidate pool does not match benchmark queries")
    seen: set[str] = set()
    for pair in pool["pairs"]:
        if not isinstance(pair, dict) or not all(isinstance(pair.get(name), str) and pair[name] for name in ("key", "query_id", "meme_id")):
            raise ValueError("candidate pool has an invalid pair")
        if pair["key"] != f"{pair['query_id']}:{pair['meme_id']}" or pair["key"] in seen or pair["query_id"] not in queries:
            raise ValueError("candidate pool has duplicate or invalid query--meme pairs")
        if not isinstance(pair.get("contributions"), list) or not pair["contributions"]:
            raise ValueError("candidate pool pair requires contributing runs")
        seen.add(pair["key"])
    return PoolDataset(queries, pool)


def build_pilot(dataset: PoolDataset, output: Path, *, seed: int = 20261025) -> dict[str, int]:
    """Create a 25-pair stratified development assignment without modifying the pool."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in dataset.visible_pairs("development"):
        grouped[dataset.queries[pair["query_id"]].category].append(pair)
    if len(grouped) != 8 or any(len(pairs) < 3 for pairs in grouped.values()):
        raise ValueError("pilot requires at least three development pairs in each category")
    def ranking_key(pair: dict[str, Any]) -> str:
        return hashlib.sha256(f"{seed}:{pair['key']}".encode()).hexdigest()

    selected = [pair for category in sorted(grouped) for pair in sorted(grouped[category], key=ranking_key)[:3]]
    selected_keys = {pair["key"] for pair in selected}
    remaining = [pair for pair in dataset.visible_pairs("development") if pair["key"] not in selected_keys]
    selected.append(min(remaining, key=ranking_key))
    payload = {"format_version": PILOT_FORMAT_VERSION, "seed": seed, "pool_digest": canonical_digest(dataset.pool),
               "split": "development", "pair_keys": sorted(pair["key"] for pair in selected)}
    _save_json(output, payload)
    return {"pairs": len(selected), "categories": len({dataset.queries[pair["query_id"]].category for pair in selected})}


def load_pilot(path: Path, dataset: PoolDataset) -> set[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to load pilot assignment: {path}") from exc
    keys = payload.get("pair_keys") if isinstance(payload, dict) else None
    if (not isinstance(payload, dict) or payload.get("format_version") != PILOT_FORMAT_VERSION
            or payload.get("pool_digest") != canonical_digest(dataset.pool) or payload.get("split") != "development"
            or not isinstance(keys, list) or len(keys) != 25 or len(set(keys)) != 25):
        raise ValueError("invalid pilot assignment")
    development = {pair["key"] for pair in dataset.visible_pairs("development")}
    if any(not isinstance(key, str) or key not in development for key in keys):
        raise ValueError("pilot contains a non-development pool pair")
    return set(keys)


def load_assignment(path: Path, dataset: PoolDataset) -> set[str]:
    """Load a bounded development assignment for the existing blinded UI."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to load evaluation assignment: {path}") from exc
    keys = payload.get("pair_keys") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("format_version") != 1
        or payload.get("pool_digest") != canonical_digest(dataset.pool)
        or payload.get("split") != "development"
        or not isinstance(keys, list)
        or not keys
        or len(set(keys)) != len(keys)
    ):
        raise ValueError("invalid evaluation assignment")
    development = {pair["key"] for pair in dataset.visible_pairs("development")}
    if any(not isinstance(key, str) or key not in development for key in keys):
        raise ValueError("assignment contains a non-development pool pair")
    return set(keys)


def _empty_ratings(dataset: PoolDataset) -> dict[str, Any]:
    return {"format_version": RATINGS_FORMAT_VERSION, "pool_digest": canonical_digest(dataset.pool), "evaluators": {}}


def load_benchmark_ratings(path: Path, dataset: PoolDataset) -> dict[str, Any]:
    if not path.exists():
        return _empty_ratings(dataset)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to load benchmark ratings: {path}") from exc
    if not isinstance(payload, dict) or payload.get("format_version") != RATINGS_FORMAT_VERSION:
        raise ValueError("unsupported benchmark ratings format")
    if payload.get("pool_digest") != canonical_digest(dataset.pool):
        raise ValueError("ratings belong to a different candidate pool")
    evaluators = payload.get("evaluators")
    if not isinstance(evaluators, dict):
        raise ValueError("ratings require evaluators")
    valid_pairs = set(dataset.pairs)
    for evaluator, ratings in evaluators.items():
        if not isinstance(evaluator, str) or not evaluator.strip() or not isinstance(ratings, dict):
            raise ValueError("invalid evaluator ratings")
        for key, axes in ratings.items():
            if key not in valid_pairs or not _valid_axes(axes):
                raise ValueError("invalid benchmark rating")
    return payload


def _valid_axes(axes: object) -> bool:
    return isinstance(axes, dict) and set(axes) == set(AXES) and all(
        isinstance(axes.get(axis), int) and not isinstance(axes.get(axis), bool) and axes[axis] in RATING_SCALE
        for axis in AXES
    )


def save_benchmark_ratings(path: Path, dataset: PoolDataset, ratings: dict[str, Any]) -> None:
    # Re-validate through the normal loader contract before replacing the file.
    probe = path.with_suffix(path.suffix + ".validate")
    _save_json(probe, ratings)
    try:
        load_benchmark_ratings(probe, dataset)
    finally:
        if probe.exists():
            probe.unlink()
    _save_json(path, ratings)


def update_rating(ratings: dict[str, Any], dataset: PoolDataset, evaluator: str, key: str, axes: dict[str, int]) -> None:
    if not evaluator.strip() or key not in dataset.pairs or not _valid_axes(axes):
        raise ValueError("rating must have a known candidate, evaluator, and all three 0--3 axes")
    ratings["evaluators"].setdefault(evaluator, {})[key] = dict(axes)


def rating_progress(dataset: PoolDataset, ratings: dict[str, Any], evaluator: str, split: str) -> dict[str, int]:
    keys = {pair["key"] for pair in dataset.visible_pairs(split)}
    completed = len(keys.intersection(ratings["evaluators"].get(evaluator, {})))
    return {"completed": completed, "total": len(keys), "remaining": len(keys) - completed}


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


class AxisUpdate(BaseModel):
    key: str
    situation: int = Field(ge=0, le=3)
    tone: int = Field(ge=0, le=3)
    sendability: int = Field(ge=0, le=3)


def create_benchmark_app(
    dataset: PoolDataset, ratings_path: Path, image_root: Path, *, evaluator: str, split: str, seed: int,
    max_items: int | None = None, pilot_keys: set[str] | None = None,
) -> FastAPI:
    _allow_split(split, None) if split == "development" else _allow_split(split, FINAL_CONFIRMATION)
    ratings = load_benchmark_ratings(ratings_path, dataset)
    app = FastAPI(title="DaMeme Benchmark Review", docs_url=None, redoc_url=None)

    def ordered_pairs() -> list[dict[str, Any]]:
        pairs = dataset.visible_pairs(split)
        if pilot_keys is not None:
            pairs = [pair for pair in pairs if pair["key"] in pilot_keys]
        # Seed-only ordering lets a second evaluator deliberately review the same bounded subset.
        ordered = sorted(pairs, key=lambda item: hashlib.sha256(f"{seed}:{item['key']}".encode()).hexdigest())
        return ordered[:max_items] if max_items is not None else ordered

    review_keys = {pair["key"] for pair in ordered_pairs()}

    def review_progress() -> dict[str, int]:
        completed = len(review_keys.intersection(ratings["evaluators"].get(evaluator, {})))
        return {"completed": completed, "total": len(review_keys), "remaining": len(review_keys) - completed}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _BENCHMARK_HTML

    @app.get("/api/state")
    def state() -> dict[str, object]:
        own = ratings["evaluators"].get(evaluator, {})
        return {"progress": review_progress(), "axes": AXES,
                "items": [{"key": pair["key"], "query": dataset.queries[pair["query_id"]].message,
                            "filename": pair["meme_id"], "rating": own.get(pair["key"]),
                            "image_url": f"/images/{quote(pair['key'], safe='')}"} for pair in ordered_pairs()]}

    @app.post("/api/ratings")
    def submit(update: AxisUpdate) -> dict[str, object]:
        try:
            if update.key not in review_keys:
                raise ValueError("candidate is outside this evaluator's assigned review subset")
            update_rating(ratings, dataset, evaluator, update.key, update.model_dump(exclude={"key"}))
            save_benchmark_ratings(ratings_path, dataset, ratings)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"progress": review_progress()}

    @app.get("/images/{key}")
    def image(key: str) -> FileResponse:
        pair = dataset.pairs.get(key)
        path = _image_path(image_root, str(pair["meme_id"])) if pair else None
        if path is None:
            raise HTTPException(404, "image is not available under the configured catalog root")
        return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0])

    return app


_BENCHMARK_HTML = """<!doctype html><html><head><meta charset=utf-8><title>DaMeme benchmark review</title>
<style>body{font:16px system-ui;margin:2rem;max-width:58rem}.item{border:1px solid #bbb;padding:1rem;margin:1rem 0}.item img{display:block;max-width:100%;max-height:30rem;object-fit:contain;margin:.75rem 0}.axis{margin:.7rem 0}.axis button{margin-right:.4rem;padding:.35rem .55rem}.selected{background:#174ea6;color:#fff}small{color:#555}</style></head><body><h1>DaMeme benchmark review</h1><p id=progress></p><p>Rate each axis from 0 (not at all) to 3 (strongly). Source mode, rank, scores, and other evaluators’ ratings are hidden.</p><div id=items></div>
<script>const labels=['0','1','2','3']; const names={situation:'Situation relevance',tone:'Tone/mood appropriateness',sendability:'Sendability / reaction usefulness'}; const descriptions={situation:'Does it fit the concrete situation?',tone:'Does it fit the selected or implied tone?',sendability:'Would it be useful to send as this reaction?'}; const anchors='0 = not at all; 1 = slight fit; 2 = clear fit; 3 = strong fit';
async function load(){const state=await (await fetch('/api/state')).json();document.querySelector('#progress').textContent=`Progress: ${state.progress.completed} / ${state.progress.total}`;const target=document.querySelector('#items');target.innerHTML='';for(const item of state.items){const node=document.createElement('section');node.className='item';node.innerHTML=`<strong>Query:</strong> ${esc(item.query)}<br><small>Filename: ${esc(item.filename)}</small><img src="${item.image_url}" alt="${esc(item.filename)}">`;const values={...(item.rating||{})};for(const axis of state.axes){const row=document.createElement('div');row.className='axis';row.innerHTML=`<strong>${names[axis]}:</strong> <small>${descriptions[axis]}</small><br><small>${anchors}</small><br>`;labels.forEach(value=>{const button=document.createElement('button');button.textContent=value;if(values[axis]===Number(value))button.className='selected';button.onclick=async()=>{values[axis]=Number(value);if(state.axes.every(name=>Number.isInteger(values[name]))){await fetch('/api/ratings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key:item.key,...values})});load()}};row.append(button)});node.append(row)}target.append(node)}}function esc(value){const span=document.createElement('span');span.textContent=value;return span.innerHTML}load();</script></body></html>"""


def _dcg(values: list[float]) -> float:
    return sum((2**value - 1) / math.log2(position + 2) for position, value in enumerate(values))


def _pair_scores(ratings: dict[str, Any], key: str) -> dict[str, float]:
    by_axis: dict[str, list[int]] = defaultdict(list)
    for evaluator_ratings in ratings["evaluators"].values():
        if key in evaluator_ratings:
            for axis in AXES:
                by_axis[axis].append(evaluator_ratings[key][axis])
    return {axis: sum(values) / len(values) for axis, values in by_axis.items() if values}


def _run_rankings(dataset: PoolDataset, split: str) -> dict[str, dict[str, list[str]]]:
    rows: dict[str, dict[str, list[tuple[int, str]]]] = defaultdict(lambda: defaultdict(list))
    for pair in dataset.visible_pairs(split):
        for contribution in pair["contributions"]:
            rows[str(contribution["run_id"])][str(pair["query_id"])].append((int(contribution["rank"]), str(pair["key"])))
    return {run: {query: [key for _, key in sorted(values)] for query, values in queries.items()} for run, queries in rows.items()}


def _query_metrics(
    keys: list[str], scores: dict[str, dict[str, float]], cutoff: int, all_query_keys: list[str]
) -> dict[str, float] | None:
    selected = keys[:cutoff]
    if (
        len(selected) < cutoff
        or any(key not in scores or "situation" not in scores[key] for key in selected)
        or any(key not in scores or "situation" not in scores[key] for key in all_query_keys)
    ):
        return None
    values = [scores[key]["situation"] for key in selected]
    ideal = sorted((scores[key]["situation"] for key in all_query_keys), reverse=True)[:cutoff]
    denominator = _dcg(ideal)
    return {"precision": sum(value >= 2 for value in values) / cutoff,
            "ndcg": _dcg(values) / denominator if denominator else 0.0}


def _bootstrap(values: list[float], seed: int = 7, samples: int = 1000) -> list[float] | None:
    if len(values) < 10:
        return None
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(samples))
    return [means[int(.025 * (samples - 1))], means[int(.975 * (samples - 1))]]


def _agreement(ratings: dict[str, Any]) -> dict[str, object]:
    evaluators = sorted(ratings["evaluators"])
    pairs = []
    for first_index, first in enumerate(evaluators):
        for second in evaluators[first_index + 1:]:
            overlap = sorted(set(ratings["evaluators"][first]).intersection(ratings["evaluators"][second]))
            axes: dict[str, object] = {}
            for axis in AXES:
                values = [(ratings["evaluators"][first][key][axis], ratings["evaluators"][second][key][axis]) for key in overlap]
                exact = sum(left == right for left, right in values) / len(values) if values else None
                axes[axis] = {"overlap": len(values), "exact_agreement": exact}
            pairs.append({"evaluators": [first, second], "axes": axes})
    return {"pairs": pairs}


def benchmark_metrics(dataset: PoolDataset, ratings: dict[str, Any], *, split: str, final_confirmation: str | None = None) -> dict[str, Any]:
    _allow_split(split, final_confirmation)
    scores = {key: _pair_scores(ratings, key) for key in dataset.pairs}
    rankings = _run_rankings(dataset, split)
    candidates_by_query: dict[str, list[str]] = defaultdict(list)
    for pair in dataset.visible_pairs(split):
        candidates_by_query[str(pair["query_id"])].append(str(pair["key"]))
    reports: dict[str, Any] = {}
    for run_id, by_query in rankings.items():
        query_rows = []
        for query_id, keys in sorted(by_query.items()):
            row: dict[str, Any] = {"query_id": query_id, "category": dataset.queries[query_id].category,
                                   "judged_at_5": _query_metrics(keys, scores, 5, candidates_by_query[query_id]),
                                   "judged_at_10": _query_metrics(keys, scores, 10, candidates_by_query[query_id])}
            query_rows.append(row)
        complete5 = [row for row in query_rows if row["judged_at_5"] is not None]
        complete10 = [row for row in query_rows if row["judged_at_10"] is not None]
        categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in complete5:
            categories[row["category"]].append(row)
        reports[run_id] = {
            "fully_judged_queries_at_5": len(complete5), "fully_judged_queries_at_10": len(complete10),
            "mean_precision_at_5": sum(row["judged_at_5"]["precision"] for row in complete5) / len(complete5) if complete5 else None,
            "mean_ndcg_at_5": sum(row["judged_at_5"]["ndcg"] for row in complete5) / len(complete5) if complete5 else None,
            "mean_ndcg_at_10": sum(row["judged_at_10"]["ndcg"] for row in complete10) / len(complete10) if complete10 else None,
            "ndcg_at_5_bootstrap_95_ci": _bootstrap([row["judged_at_5"]["ndcg"] for row in complete5]),
            "per_category": {category: {"fully_judged_queries_at_5": len(rows),
                              "mean_precision_at_5": sum(row["judged_at_5"]["precision"] for row in rows) / len(rows),
                              "mean_ndcg_at_5": sum(row["judged_at_5"]["ndcg"] for row in rows) / len(rows)} for category, rows in sorted(categories.items())},
            "queries": query_rows,
        }
    axis_values: dict[str, list[float]] = defaultdict(list)
    for value in scores.values():
        for axis, score in value.items():
            axis_values[axis].append(score)
    visible = dataset.visible_pairs(split)
    return {
        "primary_metric": "macro mean nDCG@5 on situation relevance, averaged over fully judged queries; per-pair scores are mean across available evaluators",
        "split": split, "candidate_pairs": len(visible),
        "judged_pair_coverage": sum(bool(scores[pair["key"]]) for pair in visible) / len(visible) if visible else 0.0,
        "unjudged_candidate_pairs": sum(not scores[pair["key"]] for pair in visible),
        "axis_summaries": {axis: {"rated_pairs": len(values), "mean": sum(values) / len(values) if values else None} for axis, values in sorted(axis_values.items())},
        "runs": reports, "inter_rater_agreement": _agreement(ratings),
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _pearson(left: list[int], right: list[int]) -> float | None:
    if len(left) < 2:
        return None
    left_mean, right_mean = sum(left) / len(left), sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    denominator = math.sqrt(sum((x - left_mean) ** 2 for x in left) * sum((y - right_mean) ** 2 for y in right))
    return numerator / denominator if denominator else None


def pilot_analysis(
    dataset: PoolDataset, ratings: dict[str, Any], pilot_keys: set[str], evaluator: str
) -> dict[str, Any]:
    """Summarize one evaluator's completed pilot without inferring qualitative error labels."""
    evaluator_ratings = ratings["evaluators"].get(evaluator, {})
    completed = {key: evaluator_ratings[key] for key in pilot_keys if key in evaluator_ratings}
    missing = sorted(pilot_keys - completed.keys())
    axis_values = {axis: [axes[axis] for axes in completed.values()] for axis in AXES}
    categories: dict[str, list[tuple[str, dict[str, int]]]] = defaultdict(list)
    for key, axes in completed.items():
        pair = dataset.pairs[key]
        categories[dataset.queries[pair["query_id"]].category].append((key, axes))
    lowest = []
    for key, axes in sorted(completed.items(), key=lambda item: (item[1]["situation"], sum(item[1].values()), item[0]))[:5]:
        pair = dataset.pairs[key]
        lowest.append({"query_id": pair["query_id"], "query": dataset.queries[pair["query_id"]].message,
                       "meme_id": pair["meme_id"], "ratings": axes})
    situation, sendability = axis_values["situation"], axis_values["sendability"]
    metric_report = benchmark_metrics(dataset, ratings, split="development")
    reliable = any(
        run["fully_judged_queries_at_5"] > 0 for run in metric_report["runs"].values()
    )
    return {
        "format_version": 1, "pilot_pairs": len(pilot_keys), "evaluator": evaluator,
        "completed_reviews": len(completed), "missing_pair_keys": missing,
        "axis_averages": {axis: _mean(axis_values[axis]) for axis in AXES},
        "axis_distributions": {axis: {str(score): axis_values[axis].count(score) for score in sorted(RATING_SCALE)} for axis in AXES},
        "per_category": {category: {"completed_reviews": len(rows),
                          "axis_averages": {axis: _mean([axes[axis] for _, axes in rows]) for axis in AXES}}
                         for category, rows in sorted(categories.items())},
        "lowest_situation_examples": lowest,
        "situation_sendability": {"paired_reviews": len(situation), "pearson_correlation": _pearson(situation, sendability),
                                    "joint_distribution": {f"{axes['situation']},{axes['sendability']}": sum(1 for value in completed.values() if value['situation'] == axes['situation'] and value['sendability'] == axes['sendability']) for axes in completed.values()}},
        "benchmark_metrics": {"reliably_calculable": reliable,
                              "reason": None if reliable else "No development query has every pooled candidate judged; missing judgments are not treated as zero.",
                              "fully_judged_queries_at_5": {run_id: run["fully_judged_queries_at_5"] for run_id, run in metric_report["runs"].items()}},
        "qualitative_error_taxonomy": {"status": "annotation proposal only; numerical scores do not uniquely identify an error category",
            "labels": ["completely_irrelevant", "semantically_related_wrong_situation", "correct_situation_wrong_emotional_tone", "relevant_not_useful_as_reaction", "unclear_meme_meaning", "other"],
            "future_annotation": {"error_label": "one optional controlled-vocabulary value", "rationale": "optional short free-text explanation", "confidence": "optional low/medium/high"}},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DaMeme Phase 7R benchmark tools")
    commands = parser.add_subparsers(dest="command", required=True)
    pool = commands.add_parser("pool-semantic", help="append development semantic candidates")
    pool.add_argument("--queries", type=Path, required=True)
    pool.add_argument("--output", type=Path, required=True)
    pool.add_argument("--split", choices=SPLITS, default="development")
    pool.add_argument("--limit", type=int, default=10)
    pool.add_argument("--catalog", choices=("memematch_281", "memematch_1000_v2"), default="memematch_1000_v2")
    pool.add_argument("--final-confirmation")
    pilot = commands.add_parser("pilot", help="create a stratified 25-pair development pilot")
    pilot.add_argument("--queries", type=Path, required=True)
    pilot.add_argument("--pool", type=Path, required=True)
    pilot.add_argument("--output", type=Path, required=True)
    pilot.add_argument("--seed", type=int, default=20261025)
    analyze = commands.add_parser("analyze-pilot", help="write a read-only evaluator pilot summary")
    analyze.add_argument("--queries", type=Path, required=True)
    analyze.add_argument("--pool", type=Path, required=True)
    analyze.add_argument("--pilot", type=Path, required=True)
    analyze.add_argument("--ratings", type=Path, required=True)
    analyze.add_argument("--evaluator", required=True)
    analyze.add_argument("--output", type=Path, required=True)
    serve = commands.add_parser("serve", help="serve a blinded three-axis review")
    serve.add_argument("--queries", type=Path, required=True)
    serve.add_argument("--pool", type=Path, required=True)
    serve.add_argument("--ratings", type=Path, required=True)
    serve.add_argument("--catalog-root", type=Path, required=True)
    serve.add_argument("--evaluator", required=True)
    serve.add_argument("--split", choices=SPLITS, default="development")
    serve.add_argument("--seed", type=int, default=20261025)
    serve.add_argument("--max-items", type=int, help="deterministic bounded subset for an evaluator")
    serve.add_argument("--pilot", type=Path, help="a validated 25-pair development pilot assignment")
    serve.add_argument("--assignment", type=Path, help="bounded development assignment; cannot be combined with --pilot")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8091)
    serve.add_argument("--final-confirmation")
    report = commands.add_parser("report", help="report complete-judgment metrics only")
    report.add_argument("--queries", type=Path, required=True)
    report.add_argument("--pool", type=Path, required=True)
    report.add_argument("--ratings", type=Path, required=True)
    report.add_argument("--split", choices=SPLITS, default="development")
    report.add_argument("--output", type=Path)
    report.add_argument("--final-confirmation")
    args = parser.parse_args(argv)
    dataset = load_pool_dataset(args.queries, args.pool) if args.command != "pool-semantic" else None
    if args.command == "pool-semantic":
        queries, payload = load_queries(args.queries)
        result = add_semantic_pool(queries, payload, args.output, split=args.split, limit=args.limit,
                                   settings=Settings(recommendation_catalog=args.catalog), final_confirmation=args.final_confirmation)
        print(json.dumps(result, indent=2))
        return 0
    assert dataset is not None
    if args.command == "pilot":
        print(json.dumps(build_pilot(dataset, args.output, seed=args.seed), indent=2))
        return 0
    if args.command == "analyze-pilot":
        result = pilot_analysis(dataset, load_benchmark_ratings(args.ratings, dataset), load_pilot(args.pilot, dataset), args.evaluator)
        _save_json(args.output, result)
        print(json.dumps({"output": str(args.output), "completed_reviews": result["completed_reviews"]}, indent=2))
        return 0
    if args.command == "serve":
        _allow_split(args.split, args.final_confirmation)
        if args.max_items is not None and args.max_items < 1:
            parser.error("--max-items must be positive")
        if args.pilot and args.assignment:
            parser.error("--pilot and --assignment cannot be combined")
        pilot_keys = load_pilot(args.pilot, dataset) if args.pilot else (
            load_assignment(args.assignment, dataset) if args.assignment else None
        )
        uvicorn.run(create_benchmark_app(dataset, args.ratings, args.catalog_root, evaluator=args.evaluator,
                                         split=args.split, seed=args.seed, max_items=args.max_items,
                                         pilot_keys=pilot_keys), host=args.host, port=args.port)
        return 0
    result = benchmark_metrics(dataset, load_benchmark_ratings(args.ratings, dataset), split=args.split,
                               final_confirmation=args.final_confirmation)
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        _save_json(args.output, result)
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
