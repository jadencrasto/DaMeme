from dataclasses import dataclass

import numpy as np

from app.hybrid_retrieval import (
    BM25Index,
    CrossEncoderUnavailable,
    HybridRetriever,
    OfflineCrossEncoder,
    cross_encoder_with_fallback,
    reciprocal_rank_fusion,
)


@dataclass(frozen=True)
class Candidate:
    meme_id: str
    score: float


class FakeRetrieval:
    def __init__(self):
        self.exported = type("Exported", (), {"records": [{"source_key": "semantic"}, {"source_key": "keyword"}, {"source_key": "both"}]})()

    def search(self, _modality, _vector, limit):
        rows = [Candidate("semantic", 1.0), Candidate("both", 0.8), Candidate("keyword", 0.2)]
        return rows[:limit]


def _retrieval() -> FakeRetrieval:
    return FakeRetrieval()


def test_bm25_empty_query_rrf_deduplication_and_alignment() -> None:
    bm25 = BM25Index({"one": "exam passed celebration", "two": "exam anxiety", "empty": ""})
    assert bm25.search("", 10) == []
    assert [item.meme_id for item in bm25.search("passed exam", 10)] == ["one", "two"]
    assert "empty" not in bm25.documents
    assert reciprocal_rank_fusion(["a", "a", "b"], ["b", "c", "a"], limit=3) == ["b", "a", "c"]
    retrieval = _retrieval()
    hybrid = HybridRetriever(retrieval, {"semantic": "generic", "keyword": "passed exam", "both": "exam passed"})
    rows = hybrid.search("hybrid", "passed exam", np.array([1, 0]), semantic_limit=2, bm25_limit=2)
    assert len({item.meme_id for item in rows.candidates}) == len(rows.candidates)
    try:
        HybridRetriever(retrieval, {"unknown": "not aligned"})
    except ValueError as exc:
        assert "identifiers" in str(exc)
    else:
        raise AssertionError("unaligned documents must fail")


def test_cross_encoder_fallback_is_offline_and_preserves_candidates(monkeypatch) -> None:
    reranker = OfflineCrossEncoder()

    def unavailable():
        raise CrossEncoderUnavailable("cross-encoder is unavailable locally")

    monkeypatch.setattr(reranker, "_load", unavailable)
    ordered, latency, warning = cross_encoder_with_fallback(reranker, "exam", ["a", "b"], {"a": "exam", "b": "other"}, 2)
    assert ordered == ["a", "b"]
    assert latency == 0.0
    assert warning == "cross-encoder is unavailable locally"


def test_cross_encoder_skips_missing_candidate_text_without_loading_a_model() -> None:
    reranker = OfflineCrossEncoder()
    assert reranker.rerank("exam", ["missing"], {}, 20) == (["missing"], 0.0)
