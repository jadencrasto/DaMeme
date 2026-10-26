# Phase 7R.5C focused intent-aware human-evaluation preparation

This preparation creates no ratings and changes no retrieval behavior. It uses a new, development-only pool and ratings seed, so prior human-evaluation artifacts remain unchanged. The machine-readable [workload report](../../backend/benchmarks/phase7r_v1_intent_human_eval_20261025T065752Z/workload.report.json) and assignments are in [`backend/benchmarks/phase7r_v1_intent_human_eval_20261025T065752Z`](../../backend/benchmarks/phase7r_v1_intent_human_eval_20261025T065752Z).

## Validation and affected-query audit

The fresh export has exactly eight development queries, one from each benchmark category, with 108 unique stable query--meme identities. It excludes all previously rated comparative query identities and contains no held-out queries. Every pair retains contributor variant, filtered rank, raw rank, raw score, and final score provenance.

Only **one of eight** fresh queries receives a high-confidence intent interpretation: `reaction-02`. Its A3-LH and A3-LH-INTENT rankings differ through three changed or intent-only candidates. The other seven fresh queries have low-confidence fallback and identical A3-LH/A3-LH-INTENT top-five identities.

## Assignments

- `diagnostic.assignment.json`: **3 ratings**, limited to `reaction-02` changed or intent-only identities. It is explicitly exploratory and selection-biased; do not use it for overall comparative nDCG.
- `complete_comparative.assignment.json`: **49 ratings**, the complete union of the top-five identities from A0, A3-LH, and A3-LH-INTENT across all eight preregistered fresh queries. It is blinded: the existing UI exposes only query, image, and the evaluator's own three-axis rating; contributor identity and rank remain hidden.

No valid prior rating matches these fresh query identities, so the new `ratings.seed.json` starts empty. The UI requires situation, tone, and sendability together for every saved identity.

Rating all 108 exported top-ten pairs is **not justified** for this focused question. The 49-pair top-five union is sufficient for a complete paired top-five comparison and avoids a larger, less targeted burden.

## UI command

```powershell
Set-Location C:\Users\jason\OneDrive\Desktop\DaMeme\backend

rtk uv run python -m app.benchmark serve `
  --queries .\benchmarks\phase7r_v1_queries.json `
  --pool .\benchmarks\phase7r_v1_intent_human_eval_20261025T065752Z\development.pool.json `
  --ratings .\benchmarks\phase7r_v1_intent_human_eval_20261025T065752Z\ratings.seed.json `
  --catalog-root ..\data\retrieval\catalogs\memematch_1000_v2 `
  --evaluator intent-evaluator-1 `
  --split development `
  --assignment .\benchmarks\phase7r_v1_intent_human_eval_20261025T065752Z\complete_comparative.assignment.json `
  --seed 20261025
```

No architecture or retrieval winner follows until the complete blinded workload is rated and analyzed.
