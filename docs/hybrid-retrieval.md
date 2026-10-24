# Phase 7A hybrid retrieval

The default API mode remains `semantic`, using the existing FAISS MiniLM text
index. No embedding, index, metadata, or image artifact is modified.

## Modes

- `semantic`: existing FAISS cosine retrieval.
- `hybrid`: FAISS candidates plus in-memory BM25 over the export record text
  representation, fused with Reciprocal Rank Fusion (RRF, `k=60`).
- `hybrid_cross_encoder`: hybrid retrieval followed by an optional local
  cross-encoder score of at most 20 candidates by default. Its rank order is
  RRF-fused with the existing semantic-first metadata rank order.

BM25 is created in memory from the retrieval records' v2 text contexts, which
already contain cleaned OCR/local context and selected source text. It does not
index usage probabilities or arbitrary CSV metadata. RRF is rank-based and is
used only to choose the candidate set; its score is never averaged with cosine,
BM25, or cross-encoder scores. Phase 6 semantic/emotion/sentiment ranking then
uses the original cosine similarity. Cross-encoder logits establish only a rank
order for the bounded, already-safe metadata-ranked shortlist, which is
rank-fused with the semantic-first order rather than numerically blended.

Safety filtering runs before the final cross-encoder/diversity selection.

## API configuration

`GET /api/v1/recommendations/text` accepts these additive query parameters:

```text
retrieval_mode=semantic|hybrid|hybrid_cross_encoder
candidate_limit=50
bm25_candidate_limit=50
cross_encoder_limit=20
```

The API response schema is unchanged. `semantic` remains the default and can be
used as immediate rollback: `retrieval_mode=semantic`.

To return the service to the original catalog as well, restart it after:

```powershell
$env:DAMEME_RECOMMENDATION_CATALOG = "memematch_281"
```

## Explicit offline cross-encoder cache

`transformers` and `torch` are already project development dependencies; Phase
7A adds no package dependency. Normal API requests use `local_files_only=True`
and fall back to hybrid RRF with a runtime warning if the model is absent.

To explicitly install development dependencies and cache the candidate model:

```powershell
Set-Location backend
uv sync --group dev
$env:HF_HOME = "$PWD\.model-cache"
uv run python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; model='cross-encoder/ms-marco-MiniLM-L6-v2'; AutoTokenizer.from_pretrained(model); AutoModelForSequenceClassification.from_pretrained(model)"
```

This cache command is an intentional network download and should be run only
when approved. CPU is the default runtime. Set
`DAMEME_RECOMMENDATION_CROSS_ENCODER_MODEL` to choose a compatible cached model.
For a server that should use that cache without hub access, set the same cache
location before launching it:

```powershell
$env:HF_HOME = "$PWD\.model-cache"
$env:HF_HUB_OFFLINE = "1"
uv run uvicorn app.main:app --reload
```

## Benchmark the fixed review queries

```powershell
Set-Location backend
uv run python -m app.catalog_evaluation --catalog memematch_1000_v2 --device cpu --modes semantic,hybrid,hybrid_cross_encoder
```

The JSON reports retrieval latency and cross-encoder latency separately, top
result IDs, context, and ranking scores. It is for human relevance review; a
higher model score is not evidence of better relevance. Later labeled evaluation
should calculate Precision@10 and nDCG@10.

To compare only the original path, use:

```powershell
uv run python -m app.catalog_evaluation --catalog memematch_281 --device cpu --modes semantic
```
