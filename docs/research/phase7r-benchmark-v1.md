# Phase 7R.3 benchmark v1

## Purpose and scope

Benchmark v1 evaluates contextual meme recommendation quality before new retrieval models are optimized. It is an offline research tool; it does not alter the recommendation API, rebuild indexes, download models, or acquire data. The versioned query artifact is [`backend/benchmarks/phase7r_v1_queries.json`](../../backend/benchmarks/phase7r_v1_queries.json), and the implementation is [`backend/app/benchmark.py`](../../backend/app/benchmark.py).

The query set has 80 Codex-authored prompts. They were written independently of catalog results and must not be described as independently human-authored. There are ten prompts in each of eight categories: everyday situations; academic/workplace stress; celebrations; negative setbacks; sarcasm/irony; nostalgia; template-format intent; and reaction/response intent. Each has a stable ID, optional supported selected mood, contextual note, and deterministic split assignment.

## Query split and methodology

The split seed is `20261025`. Within each category, query IDs are SHA-256 sorted using that seed; the first five form development and the remaining five form held-out test. This produces 40 development and 40 test prompts, with five per category in each split. Loader validation rejects changes that invalidate the split, category counts, IDs, or prompt structure.

Prompts aim to vary subject matter, affect, specificity, response intent, and template-like language without quoting catalog captions. They are not a representative sample of all meme requests: they are English-only, author-generated, limited to eight editorial categories, and may encode the prompt author's cultural assumptions. They should be augmented through a documented human-authored query collection process before broader claims are made.

## Candidate pooling and provenance

`pool-semantic` generates a bounded initial pool from the existing semantic baseline. Its default is ten candidates for each development query: nominally 400 query–meme pairs before any within-query duplicate removal. The generated pool is a local, provenance-bearing research artifact: regenerate it when the approved catalog or query encoder changes rather than treating it as portable across environments.

The JSON pool is append-only in meaning: every pair has a stable `<query-id>:<meme-id>` key and a list of contributing run IDs/ranks. Adding a future run retains prior pairs and ratings, deduplicates query–meme pairs, and records every contributor. A pool manifest contains the benchmark/query digest, catalog ID/count and SHA-256 values for the CSV, FAISS manifest, and embedding metadata; each run records retrieval mode, query encoder/model version, weights, candidate limits, diversity/safety settings, and rankings. A changed catalog provenance or query digest requires a new pool rather than silently mixing results.

Unjudged candidates are unknown, never irrelevant. Reported ranking metrics require every pooled candidate for a query to be judged, so ideal DCG is computed from the full judged pool rather than assuming missing pairs are zero.

## Human review

The local review server shows only the query, filename, and actual catalog image. It hides retrieval mode, rank, scores, contributing runs, and all other evaluators’ data. Each evaluator uses a required ID; ratings are namespaced by evaluator in one persistent JSON file. A three-axis 0–3 rating is saved only after all axes have values, and later selections edit that evaluator's entry.

Axes are:

- **Situation relevance:** does this meme fit the concrete situation/request?
- **Tone/mood appropriateness:** does it fit the selected or implied affect?
- **Sendability / reaction usefulness:** would it be useful to send as the requested reaction?

Presentation order is a deterministic SHA-256 ordering of seed and pair key. Thus a resumed evaluator sees the same order, and two evaluators using the same seed plus `--max-items` review the same reproducible overlap. A manageable second-rater protocol is to rate a pre-agreed overlapping subset of 40–80 development pairs, then inspect the agreement output before increasing coverage.

## Metrics and protection of held-out queries

The primary metric is **macro mean nDCG@5 on situation relevance**, averaging each fully judged query equally. When several evaluators rated a pair, the pair relevance is their mean situation score. Precision@5 uses situation score >=2 as relevant. nDCG@10 is emitted only for runs that have ten candidates and fully judged query pools. Reports also include category-level P@5/nDCG@5, means/counts for all three axes, candidate-pair coverage, unjudged pair count, pairwise exact inter-rater agreement by axis, and a deterministic query-level bootstrap 95% interval for nDCG@5 when at least ten fully judged queries exist.

Development is the default for pooling, serving, and reporting. Test pooling/serving/reporting requires `--final-confirmation ARCHITECTURE_FROZEN`. This is a workflow guard rather than cryptographic access control: a user who has filesystem access can inspect artifacts or pass the flag. Keep test pool/ratings outside ordinary experiment directories and do not run the final command until model selection is frozen.

## Staged workflow and exact Windows PowerShell commands

Run from the repository root. The pool command is intentionally offline: it will fail rather than download a missing query model.

```powershell
Set-Location C:\Users\jason\OneDrive\Desktop\DaMeme\backend
$env:HF_HUB_OFFLINE = "1"
uv run python -m app.benchmark pool-semantic `
  --queries .\benchmarks\phase7r_v1_queries.json `
  --output .\benchmarks\phase7r_v1_development.pool.json `
  --catalog memematch_1000_v2 --split development --limit 10
```

Stage 1 creates the query artifact (already checked in). Stage 2 creates the bounded 40-query semantic pool. Stage 3 collects development ratings, initially from one evaluator:

```powershell
uv run python -m app.benchmark serve `
  --queries .\benchmarks\phase7r_v1_queries.json `
  --pool .\benchmarks\phase7r_v1_development.pool.json `
  --ratings .\benchmarks\phase7r_v1_development.ratings.json `
  --catalog-root ..\data\retrieval\catalogs\memematch_1000_v2 `
  --evaluator evaluator-a --split development --seed 20261025
```

Start a second server for `evaluator-b` (typically on `--port 8092`) against the same ratings file. Add `--max-items 60` to give that evaluator a manageable, deterministic overlapping subset. The server exposes only that evaluator's ratings. Report development status/metrics with:

```powershell
uv run python -m app.benchmark report `
  --queries .\benchmarks\phase7r_v1_queries.json `
  --pool .\benchmarks\phase7r_v1_development.pool.json `
  --ratings .\benchmarks\phase7r_v1_development.ratings.json `
  --split development --output .\benchmarks\phase7r_v1_development.metrics.json
```

Stage 4 appends candidates from a future model using the same pair/contribution schema; it must validate the same benchmark and catalog provenance or create a new pool. Stage 5 is the only point at which to pool and evaluate held-out prompts:

```powershell
$env:HF_HUB_OFFLINE = "1"
uv run python -m app.benchmark pool-semantic `
  --queries .\benchmarks\phase7r_v1_queries.json `
  --output .\benchmarks\phase7r_v1_test.pool.json `
  --catalog memematch_1000_v2 --split test --limit 10 `
  --final-confirmation ARCHITECTURE_FROZEN

uv run python -m app.benchmark report `
  --queries .\benchmarks\phase7r_v1_queries.json `
  --pool .\benchmarks\phase7r_v1_test.pool.json `
  --ratings .\benchmarks\phase7r_v1_test.ratings.json `
  --split test --final-confirmation ARCHITECTURE_FROZEN
```

## Limitations

This benchmark has no fabricated ratings and no accuracy claim. Its initial semantic pool does not measure recall outside its pooled candidates, evaluator IDs are labels rather than authentication, images must exist locally to review them, and score aggregation is not a substitute for adjudication. The test guard cannot stop intentional data leakage. Report any model comparison with candidate-pool coverage, number of complete queries, evaluator overlap, catalog/model checksums, and the fact that this is a bounded human study.
