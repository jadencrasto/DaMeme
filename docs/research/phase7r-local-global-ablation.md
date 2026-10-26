# Phase 7R.4 local/global ablation checkpoint

## Completed field-overlap audit

The source-backed overlap report is [`backend/benchmarks/phase7r_v1_field_overlap.json`](../../backend/benchmarks/phase7r_v1_field_overlap.json). It covers all 1,000 v2 source rows and records the metadata CSV SHA-256 `f4e24bce9a0d33596f4a9a2499b1b2f38fcbd16bd48714b2d6406842542710d4`.

| Comparison | Exact | Containment | Partial | Distinct | Missing |
| --- | ---: | ---: | ---: | ---: | ---: |
| `local_context` vs `local_Extracted Text + Title` | 996 | 0 | 0 | 0 | 4 |
| `global_context` vs `global_Caption` | 1,000 | 0 | 0 | 0 | 0 |

This is a verified normalization-based audit. The four local missing values are missing in both fields. Therefore A4 is **not justified**: the two unselected fields are aliases of already-selected fields, not new content. Adding them would duplicate every populated source string.

## Controlled representation definitions

- **A0:** existing v2 combined text: OCR/local context, title/caption when recognized, verified template, and global context.
- **A1:** local-only: recognized local OCR, local context, and local title fields.
- **A2:** global-only: global context and only explicitly verified template fields.
- **A3:** independently embed A1/A2, retain stable source-key order, and compare fixed equal-weight, local-heavy, and RRF fusion without using pilot ratings for tuning.

The source-faithful scoped construction is implemented in [`backend/app/local_global_ablation.py`](../../backend/app/local_global_ablation.py). It explicitly deduplicates identical text within a scope and does not infer captions or template descriptions.

## Phase 7R.4D.2 development-only retrieval comparison

The offline comparison in [`backend/benchmarks/phase7r_v1_local_global_comparison_20261025T054251Z/results.json`](../../backend/benchmarks/phase7r_v1_local_global_comparison_20261025T054251Z/results.json) used exactly the 40 development queries, a fixed 100-candidate downstream depth, shared query embeddings, the existing source-text safety filter, and the existing one-per-template diversity filter. The A0 compatibility gate passed: ordered source keys, the 1,000-row/384D shape, MiniLM model identifier, encoder settings, and source CSV SHA-256 all matched. The shared query-side processing is `mood_query(message, resolve_mood(mood))` and the mood-derived safety emotion is unchanged.

| Variant | Mean overlap@5 with A0 | Mean overlap@10 with A0 | New final candidates vs A0 | Median / p95 ms |
| --- | ---: | ---: | ---: | ---: |
| A0 | 1.000 | 1.000 | 0 | 9.84 / 14.35 |
| A1 local | 0.460 | 0.510 | 63 | 10.30 / 12.34 |
| A2 global | 0.185 | 0.190 | 94 | 10.15 / 14.95 |
| A3-EQ | 0.600 | 0.585 | 45 | 10.36 / 15.26 |
| A3-LH | 0.635 | 0.613 | 46 | 10.29 / 15.56 |
| A3-RRF | 0.400 | 0.432 | 85 | 9.99 / 14.34 |

Every query retained ten results after safety and template-diversity filtering. A1 excludes the four unavailable local rows. For score fusion, an unavailable-local row receives its global score rather than an artificial zero local similarity; in RRF it receives only its global rank contribution. Of final top-10 results, such rows occurred once each for A3-EQ and A3-LH, and not for A2 or A3-RRF.

These are retrieval-distribution and operational measurements, not relevance results. The overlap, novelty, and latency figures do **not** establish that any variant is better; human relevance judgments remain required. This run does not merge benchmark candidates, alter the 319-pair pool or pilot ratings, use held-out queries, or establish production readiness.

## Remaining bounded work

The field audit establishes that A4 should not be built. Benchmark-pool contributor merging and relevance assessment remain intentionally out of scope for this development-only comparison.

## Reproduce the audit

```powershell
Set-Location C:\Users\jason\OneDrive\Desktop\DaMeme\backend
uv run python -m app.local_global_ablation `
  --metadata-csv ..\data\retrieval\catalogs\memematch_1000_v2\memematch_1000_metadata_v2.csv `
  --output .\benchmarks\phase7r_v1_field_overlap.json
```

No model, dataset, catalog index, rating, or held-out query is accessed by this command.
