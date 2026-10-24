"""CPU-friendly BM25, RRF, and optional offline cross-encoder reranking."""

from __future__ import annotations

import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from app.retrieval import RetrievalService

RetrievalMode = Literal["semantic", "hybrid", "hybrid_cross_encoder"]
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(value: str) -> list[str]:
    return TOKEN_PATTERN.findall(value.lower())


@dataclass(frozen=True)
class KeywordResult:
    meme_id: str
    score: float


class BM25Index:
    """Small deterministic BM25 implementation; no persisted artifacts required."""

    def __init__(self, documents: dict[str, str], *, k1: float = 1.2, b: float = 0.75):
        self.documents = {key: tokenize(value) for key, value in documents.items() if tokenize(value)}
        self.k1, self.b = k1, b
        self.lengths = {key: len(tokens) for key, tokens in self.documents.items()}
        self.average_length = sum(self.lengths.values()) / len(self.lengths) if self.lengths else 0.0
        self.postings: dict[str, dict[str, int]] = defaultdict(dict)
        for meme_id, tokens in self.documents.items():
            for token, count in Counter(tokens).items():
                self.postings[token][meme_id] = count

    def search(self, query: str, limit: int) -> list[KeywordResult]:
        if limit < 1 or not self.documents:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores: dict[str, float] = defaultdict(float)
        document_count = len(self.documents)
        for token in set(query_tokens):
            postings = self.postings.get(token, {})
            if not postings:
                continue
            inverse_frequency = math.log(1 + (document_count - len(postings) + 0.5) / (len(postings) + 0.5))
            for meme_id, frequency in postings.items():
                length_norm = 1 - self.b + self.b * self.lengths[meme_id] / self.average_length
                scores[meme_id] += inverse_frequency * frequency * (self.k1 + 1) / (frequency + self.k1 * length_norm)
        return [KeywordResult(meme_id, score) for meme_id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def reciprocal_rank_fusion(*rankings: list[str], k: int = 60, limit: int = 50) -> list[str]:
    if k < 1 or limit < 1:
        raise ValueError("RRF k and limit must be positive")
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        # A source must not amplify an item merely because it accidentally
        # supplied it twice. Preserve the first position deterministically.
        for rank, meme_id in enumerate(dict.fromkeys(ranking), start=1):
            scores[meme_id] += 1 / (k + rank)
    return [meme_id for meme_id, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]]


@dataclass(frozen=True)
class HybridResult:
    candidates: list[Any]
    documents: dict[str, str]
    retrieval_latency_ms: float
    warning: str | None = None


class HybridRetriever:
    def __init__(self, retrieval: RetrievalService, documents: dict[str, str]):
        record_ids = {str(record["source_key"]) for record in retrieval.exported.records}
        if set(documents) - record_ids:
            raise ValueError("BM25 documents contain identifiers absent from retrieval records")
        self.retrieval, self.documents = retrieval, documents
        self.bm25 = BM25Index(documents)

    def search(self, mode: RetrievalMode, query: str, vector, *, semantic_limit: int, bm25_limit: int) -> HybridResult:
        started = time.perf_counter()
        semantic = self.retrieval.search("text", vector, semantic_limit)
        if mode == "semantic":
            return HybridResult(semantic, {row.meme_id: self.documents.get(row.meme_id, "") for row in semantic}, (time.perf_counter() - started) * 1000)
        keyword = self.bm25.search(query, bm25_limit)
        # Preserve true cosine similarity for Phase 6 scoring, while RRF supplies
        # only rank-based candidate selection/order—not a fake comparable score.
        all_semantic = {row.meme_id: row for row in self.retrieval.search("text", vector, len(self.retrieval.exported.records))}
        fused_ids = reciprocal_rank_fusion([row.meme_id for row in semantic], [row.meme_id for row in keyword], limit=max(semantic_limit, bm25_limit))
        candidates = [all_semantic[meme_id] for meme_id in fused_ids if meme_id in all_semantic]
        return HybridResult(candidates, {meme_id: self.documents.get(meme_id, "") for meme_id in fused_ids}, (time.perf_counter() - started) * 1000)


class CrossEncoderUnavailable(RuntimeError):
    pass


class OfflineCrossEncoder:
    """Loads only already-cached model files; normal request handling never downloads."""

    def __init__(self, model_id: str = "cross-encoder/ms-marco-MiniLM-L6-v2", device: str = "cpu"):
        self.model_id, self.device = model_id, device
        self._tokenizer = self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, local_files_only=True)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_id, local_files_only=True).to(self.device).eval()
            self._torch = torch
        except Exception as exc:  # Model/cache errors are intentionally non-fatal fallback signals.
            raise CrossEncoderUnavailable(f"cross-encoder is unavailable locally: {exc}") from exc

    def rerank(self, query: str, candidates: list[str], documents: dict[str, str], limit: int) -> tuple[list[str], float]:
        if not query.strip() or limit < 1:
            return candidates, 0.0
        shortlist = [meme_id for meme_id in dict.fromkeys(candidates) if documents.get(meme_id)][:limit]
        if not shortlist:
            return candidates, 0.0
        self._load()
        started = time.perf_counter()
        encoded = self._tokenizer([(query, documents[meme_id]) for meme_id in shortlist], padding=True, truncation=True, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self._torch.no_grad():
            scores = self._model(**encoded).logits.squeeze(-1).detach().cpu().tolist()
        reranked = [meme_id for meme_id, _ in sorted(zip(shortlist, scores, strict=True), key=lambda item: (-item[1], item[0]))]
        reranked_ids = set(reranked)
        tail = [meme_id for meme_id in dict.fromkeys(candidates) if meme_id not in reranked_ids]
        return reranked + tail, (time.perf_counter() - started) * 1000


def cross_encoder_with_fallback(
    reranker: OfflineCrossEncoder, query: str, candidates: list[str], documents: dict[str, str], limit: int
) -> tuple[list[str], float, str | None]:
    try:
        ordered, latency = reranker.rerank(query, candidates, documents, limit)
        return ordered, latency, None
    except CrossEncoderUnavailable as exc:
        return candidates, 0.0, str(exc)
