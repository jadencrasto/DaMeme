# Emotion-aware recommendation ranking

Phase 5's local ranking layer reranks text FAISS candidates or ranks the full
metadata catalog for an emotion-only request. It joins by the export record's
stable `source_key`/filename—not CSV row number.

MemeMatch local/global emotion and sentiment values are source-provided
probabilities. DaMeme averages available local/global values for the requested
label. These values are not Phase 4 classifier outputs, are not calibrated across
sources, and must not be interpreted as interchangeable model probabilities.

For text recommendations, cosine similarity is mapped from `[-1, 1]` to `[0, 1]`.
Emotion and sentiment components are already probabilities in `[0, 1]`. Active
configured weights are normalized to sum to one, making each final score an
interpretable weighted average. Missing optional metadata contributes no component
and does not discard a semantic candidate unless usage filtering was requested.

The API's text defaults are semantic-first: `semantic_weight=0.8`,
`emotion_weight=0.15`, and `sentiment_weight=0.05`. All three remain query
parameters for experiments. Emotion-only browsing deliberately has no semantic
component. `min_semantic_similarity` is an optional raw cosine cutoff in `[-1,1]`:
when supplied, an empty response means no retrieved candidate met the caller's
threshold, rather than an unrelated fallback. It has no asserted universal default
because this small source corpus has no labeled relevance threshold. Set
`max_per_template=1` to apply the deterministic filename heuristic described below.

Usage preferences match normalized pipe/comma/semicolon-separated source categories.
Optional diversity caps results per a configurable filename heuristic that removes
only a trailing `_number` or `-number`. It is a convenience grouping, not a claim
of definitive meme-template identity; callers can supply a different key function.
The metadata loader does not use a CSV `image_path` unless callers explicitly pass
a local image root; it then accepts only files beneath that root, with portable
filename fallbacks. Stale Colab absolute paths are therefore not assumed to exist.

## Reproducible comparison

Use a text query (embedded lazily with the existing MiniLM encoder), or pass a
previously saved 384-dimensional `.npy` query vector, then run:

```powershell
Set-Location backend
uv run python -m app.recommendation_eval `
  --index-dir ..\data\retrieval\memematch_281_indexes `
  --metadata-csv C:\path\to\memematch_281_metadata.csv `
  --query-text "I am exhausted after studying all night" `
  --emotion sadness `
  --limit 10
```

For an already generated vector, replace `--query-text` with
`--query-vector C:\path\to\query.npy`.

The report intentionally contains only observable properties: returned-candidate
coverage, semantic-to-combined top-k overlap/order changes, and heuristic template
diversity. It does not claim human relevance metrics without relevance judgments.

Add `--export-csv .\review.csv` to create paired `baseline` (`1,1,1`) and
`improved` (`0.8,0.15,0.05`) top-k rows. The CSV intentionally leaves
`situational_relevance`, `emotion_appropriateness`, `diversity_note`, and
`reviewer_notes` blank for human review. It reports rank/order/diversity only;
accuracy, precision, and recall require completed labeled judgments.

## Reranking diagnostics

The unified MemeMatch CSV uses direct source columns such as `local_sadness`,
`global_sadness`, `local_positive`, and `global_positive`. The loader supports
these columns as well as explicitly named emotion/sentiment probability columns.
They are source annotations, not Phase 4 classifier outputs.

Use the diagnostics command to print the semantic score, source emotion/sentiment
relevance, final score, and before/after ranks for each returned item. It compares
semantic-only, balanced (`1:1:0`), and emotion-prioritized (`1:2:0`) weights over
candidate pools of 10, 50, 100, and 281 by default.

```powershell
Set-Location backend
uv run python -m app.recommendation_diagnostics `
  --index-dir ..\data\retrieval\memematch_281_indexes `
  --metadata-csv ..\data\retrieval\memematch_281\memematch_281_metadata.csv `
  --query-text "I am exhausted after studying all night" `
  --emotion sadness `
  --limit 10

uv run python -m app.recommendation_diagnostics `
  --index-dir ..\data\retrieval\memematch_281_indexes `
  --metadata-csv ..\data\retrieval\memematch_281\memematch_281_metadata.csv `
  --query-text "I finally passed my difficult exam" `
  --emotion joy `
  --limit 10
```

For offline reproduction, use `--query-vector path\to\query.npy` with a saved
384-dimensional MiniLM query vector. Add `--max-per-template 1` only when a
heuristic filename-based diversity cap is desired; it is disabled by default and
does not assert visual-template identity.
