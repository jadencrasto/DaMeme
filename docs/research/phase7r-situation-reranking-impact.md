# Phase 7R.7B situation reranking impact audit

## Scope

This is a read-only development analysis of the frozen Phase 7R.7A comparison over 40 development queries. It does not rerun retrieval, alter the reranker, access held-out queries, write ratings, or make a production change. The machine-readable report is [`top_five_impact.json`](../../backend/benchmarks/phase7r_v1_situation_reranking_impact_20261026T000003Z/top_five_impact.json), with a source-backed [`qualitative_audit.md`](../../backend/benchmarks/phase7r_v1_situation_reranking_impact_20261026T000003Z/qualitative_audit.md).

## Top-five impact

| Comparison | Queries with top-five order change | Queries with top-five membership change | Entering / leaving | Mean top-five overlap | Internal rank changes | Rank-1 changes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R-T vs A3-LH | 18 / 40 | 7 / 40 | 7 / 7 | .965 | 12 | 0 |
| R-V vs M1 | 8 / 40 | 2 / 40 | 2 / 2 | .990 | 6 | 0 |

All observed candidate movement is one position: mean displacement among changed candidates is 1.0 and maximum displacement is 1 for both variants. The candidate set remains identical to its baseline at top ten by design. Thus, R-T has a limited user-visible membership effect and R-V has a very small one; most changes are internal top-five ordering. Neither variant changes the first recommendation.

## Existing ratings

The audit joins ratings only on exact query--meme keys in the completed eight-query, development-only multimodal pool. Missing keys remain missing.

| Comparison | Changed top-five query--meme pairs | Already judged | Fully covered queries for nDCG@5 / P@5 |
| --- | ---: | ---: | ---: |
| R-T vs A3-LH | 31 | 5 | 7 |
| R-V vs M1 | 14 | 2 | 7 |

For the seven completely covered R-T queries, situation nDCG@5 declined on `academic-09` (.786 to .754), while the other measured situation rankings were unchanged. `nostalgia-01` retained situation/tone ordering but its sendability nDCG@5 declined (.608 to .563). For R-V, only `nostalgia-01` changed among the fully covered rankings, increasing situation nDCG@5 slightly (.541 to .549) and tone nDCG@5 slightly (.492 to .499); P@5 did not change. These are descriptive reuses of a small pre-existing set, not parameter-selection evidence.

Previously rated low-situation recommendations remain: 11 R-T top-five entries and 10 R-V entries have observed situation scores of 0 or 1. Examples include the `academic-09:23-wolves-laugh-80.jpg` score of 0 in both variants, plus `celebration-07:job-interview.jpg` (R-T, 0) and `celebration-07:Advice_Dog_1.jpg` (R-V, 0). Reranking did not remove these failures.

## Heuristic evidence audit

The full source-backed audit records every changed candidate. Of 52 promotions, 10 were triggered by `format`, a generic metadata term; it is the most frequent matched token. Other visibly weak triggers include `will`, `last`, `needed`, and `group`. This is an observed lexical property, not a claim that every such promotion is irrelevant.

There are plausible lexical matches, for example `passed`/`test` for the driving-test query. But those words can still describe a different setting in source OCR. OCR and broad usage labels therefore provide both evidence and false-positive triggers. The ambiguous `template-01` query correctly remained unchanged. Negated terms are not positive features, but other retained terms can still trigger movement in negated queries. Thirteen R-T and twenty R-V lists had no match in their already filtered top-ten reranking set and correctly retained their baseline order. This does not establish that the original depth-100 retrieval pool lacked a better situation match.

## Go/no-go

Do not create a new category-balanced blinded evaluation for this frozen prototype. The impact is primarily internal swaps, no rank-one changes occur, top-five membership changes are small, and the lexical audit exposes generic metadata/OCR triggers. A broad evaluation would have little power to separate the reranker from its baselines while conflating those known heuristic issues.

If a human check is needed before a future redesign, use a small preregistered, paired, rank-sensitive evaluation focused on the 18 R-T and 8 R-V queries with changed top-five order. It should rate baseline and reranked ordering blind, retain missing ratings as missing, and be explicitly descriptive. The preferred next engineering step is a non-rating-driven lexical-specification review that separates template-format vocabulary and generic function words from situation concepts, followed by a new frozen reranking run and a fresh impact audit.
