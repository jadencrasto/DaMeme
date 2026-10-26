# Phase 7R.3A benchmark pilot and readiness audit

## Readiness decision

**Conditionally ready for a bounded development pilot; not ready for claims about retrieval quality or held-out evaluation.** The benchmark has persistent, blinded, multi-axis rating mechanics and a provenance-bound development pool. Its primary remaining limitation is evidence: no human pilot ratings exist, and the pool does not record why a particular request returned fewer than ten candidates.

This audit did not modify the 80-query artifact, the 319-pair development pool, the test split, production recommendation behavior, catalog artifacts, or ratings. The only implementation improvement is clearer 0–3 anchors in the reviewer UI and an independent pilot-assignment artifact.

## Mechanics verification

| Requirement | Status | Evidence |
| --- | --- | --- |
| Persistent ratings, evaluator IDs, resume/edit | Verified | Ratings are namespaced under `evaluators[evaluator_id][pair_key]`, validated against the pool digest, and atomic-replaced on each completed three-axis update ([`benchmark.py`](../../backend/app/benchmark.py)). |
| Blind randomized presentation | Verified | State exposes only query, filename, image URL, and the current evaluator's own rating; order is deterministic SHA-256 over seed and pair key. Mode, rank, scores, contributor runs, and other evaluator ratings are omitted. |
| Three independent 0–3 axes | Verified | Situation, tone, and sendability are required together by `AxisUpdate`/`_valid_axes`; the UI saves only after all three values have been selected. |
| Candidate deduplication | Verified | Pool keys are `<query-id>:<meme-id>`; loader rejects duplicate keys, and a new run appends contributions to the same pair identity. |
| Missing judgments | Verified | Unjudged pairs are reported as coverage/gaps; ranking metrics are `null` until all pooled candidates for a query have situation ratings. Missing values are never converted to zero. |
| Development/test isolation | Verified with limits | Test pool/serve/report operations require `--final-confirmation ARCHITECTURE_FROZEN`. This is an intentional workflow guard, not access control. |
| Catalog/source provenance | Verified | The pool records SHA-256 digests for metadata CSV, FAISS manifest, and export metadata. Ratings bind to a canonical digest of the full pool. |
| Future retrieval contributors | Partly verified | Pool schema supports multiple run IDs and contributions per stable pair without deleting existing pairs. `pool-semantic` implements the first contributor; a future retrieval command must use the same merge/provenance contract. |

## Development-pool audit

The checked-in local pool has **40** development queries and **319** unique query–meme pairs. At ten requested candidates/query, the nominal maximum is **400**, leaving **81** unfilled slots.

### Candidate-count distribution

| Returned candidates | Queries | Query IDs |
| ---: | ---: | --- |
| 10 | 6 | academic-07, celebration-07, reaction-02, reaction-06, sarcasm-08, template-01 |
| 9 | 11 | academic-08, celebration-06, everyday-02, everyday-08, nostalgia-03, nostalgia-10, reaction-07, sarcasm-05, setback-03, setback-07, template-04 |
| 8 | 10 | academic-01, academic-09, celebration-02, nostalgia-01, nostalgia-06, reaction-01, sarcasm-09, setback-08, template-06, template-07 |
| 7 | 8 | celebration-03, celebration-05, everyday-04, everyday-10, nostalgia-08, reaction-05, setback-05, setback-06 |
| 6 | 1 | everyday-07 |
| 5 | 3 | academic-02, sarcasm-10, template-02 |
| 3 | 1 | sarcasm-01 |

### Causes of the 81 missing slots

**Verified from recorded outputs:** The pool has one semantic run, 40 `query_results` arrays, and exactly 319 pair records; each array's length equals the final recorded candidate count. There is no query–meme deduplication loss within a query: those arrays contain unique IDs. Cross-query duplicate IDs are intentionally separate pairs because the query differs.

**Verified from code:** The pooling call requests ten semantic candidates but enables `exclude_unsafe=True` and `max_per_template=1`. The engine applies metadata safety filtering and then template diversity to the ranked candidates; either can return fewer than the requested limit ([`recommendation_engine.py`](../../backend/app/recommendation_engine.py), [`recommendations.py`](../../backend/app/recommendations.py)). No raw cosine threshold or usage filter is set for this run.

**Not individually attributable from existing artifacts:** The pool stores final result IDs/ranks, not pre-safety, post-safety, or pre-diversity counts, excluded IDs, template keys, or filter reasons. Therefore it is not valid to state how many of the 81 slots were caused by safety versus diversity for any query. A plausible explanation is that either filter reduced a 10-item semantic candidate set; it remains a hypothesis until a future non-mutating diagnostics record is produced. The particularly short `sarcasm-01` result (3) is an observation, not proof of a content or model defect.

## 25-pair pilot

`app.benchmark pilot` creates a separate assignment with three SHA-256 seed-selected development pairs from each of the eight categories (24 total), then one globally seed-selected remaining development pair. The assignment binds to the full-pool digest, contains exactly 25 unique development pair keys, and does not modify the pool or ratings.

Pilot workload: **25 pair reviews × 3 axes = 75 selections per evaluator**. A second evaluator using the same pilot and seed produces a fully overlapping 25-pair agreement sample (another 75 selections). This is intentionally bounded and is not sufficient for benchmark metrics or confidence intervals.

## Exact Windows PowerShell commands

Create the pilot assignment (offline; no models are loaded):

```powershell
Set-Location C:\Users\jason\OneDrive\Desktop\DaMeme\backend
uv run python -m app.benchmark pilot `
  --queries .\benchmarks\phase7r_v1_queries.json `
  --pool .\benchmarks\phase7r_v1_development.pool.json `
  --output .\benchmarks\phase7r_v1_pilot_25.json `
  --seed 20261025
```

Launch the pilot for one evaluator. This creates a ratings file only when the evaluator actually submits a rating; the command itself creates no ratings.

```powershell
uv run python -m app.benchmark serve `
  --queries .\benchmarks\phase7r_v1_queries.json `
  --pool .\benchmarks\phase7r_v1_development.pool.json `
  --pilot .\benchmarks\phase7r_v1_pilot_25.json `
  --ratings .\benchmarks\phase7r_v1_pilot.ratings.json `
  --catalog-root ..\data\retrieval\catalogs\memematch_1000_v2 `
  --evaluator pilot-a --split development --seed 20261025
```

For overlap, start the same command with `--evaluator pilot-b --port 8092`. Do not run a report as a quality claim after this pilot; use it only to inspect completion, axis distribution, and agreement once both evaluators have opted in.

## Focused validation

Tests cover deterministic split/query validation, pool duplicate/provenance rejection, rating persistence and evaluator separation, missing-judgment metric behavior, held-out guard, reproducible 25-pair pilot validation, and scoring-anchor presence. A test does not substitute for a human factors review of the full browser flow.
