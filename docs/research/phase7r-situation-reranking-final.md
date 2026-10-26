# Phase 7R.7C final lexical reranker cleanup

## Scope and frozen variant

This is the single, specification-driven final revision of the development-only lexical reranker. It reuses the same 40 development queries and saved A3-LH/M1 candidate lists as Phase 7R.7A. It does not use ratings as runtime features, change retrieval, download a model, modify the catalog, access held-out queries, or change production behavior.

The frozen `phase7r.7c.1` artifact is [`comparison.json`](../../backend/benchmarks/phase7r_v1_situation_reranking_7c_20261026T000001Z/comparison.json). Its read-only impact report is [`top_five_impact.json`](../../backend/benchmarks/phase7r_v1_situation_reranking_impact_7c_20261026T000001Z/top_five_impact.json).

R-T and R-V retain their names, but the artifact records `rule_version: phase7r.7c.1`. Both operate only on the saved, post-safety, one-per-template top ten. Candidate membership, safety outcomes, and template diversity are unchanged. Tie-breaking remains original rank then source key, and the revised maximum promotion is one rank.

## Revised evidence specification

Allowed candidate evidence fields are source `local_Extracted Text + Title`, `local_context`, `global_context`, and `global_Caption`. These are OCR/title and source-provided textual context/caption fields only.

The reranker now excludes `usage` and `template_group_heuristic` entirely, so labels such as “Popular Meme Format” cannot trigger a situation match. It continues to require exact overlap with an original-query token of at least four characters, but excludes mood vocabulary, function words, broad relational terms, and format/template vocabulary. The revised exclusion set explicitly covers the previously observed weak terms including `will`, `needed`/`need`, `group`, `format`, `template`, `person`, `friend`, `last`, and `says`. There is no semantic expansion or inferred context.

Negated content remains ineligible as a positive feature. Ambiguity markers retain the unchanged-order fallback. A list with no direct lexical evidence also retains its baseline order.

## Impact results

| Variant | Top-five order changes | Membership changes | Rank-1 changes | Mean top-five overlap with baseline | Weak-evidence fallbacks | Added median latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R-T vs A3-LH | 0 / 40 | 0 / 40 | 0 | 1.000 | 23 | .150 ms |
| R-V vs M1 | 0 / 40 | 0 / 40 | 0 | 1.000 | 29 | .140 ms |

Coverage and cross-query repetition are consequently identical to the baselines: A3-LH/R-T has 153 unique final memes and 61.8% repetition; M1/R-V has 165 and 58.8%. Safety/template candidate-set parity is true for all queries.

The original prototype had 52 promotions and trigger frequencies headed by `format` (10 promotions), alongside `group`, `will`, and `needed`. The revised output has zero promotions, zero demotions, and an empty trigger-token frequency table. Compared with the original reranker, it reverses every original visible difference: R-T differs from original R-T on 18 top-five orderings and 7 memberships; R-V differs on 8 and 2. Neither comparison changes rank one.

Existing ratings were loaded only for the descriptive impact report. Since revised rankings equal baseline rankings, their exact-pair rating coverage and descriptive nDCG@5/P@5 values are unchanged; missing values remain missing. They did not influence this specification or any parameter.

## Conclusion

The revised specification eliminates the documented weak triggers, but it no longer makes meaningful top-five changes. There is therefore insufficient evidence or user-visible impact to justify a new human evaluation. Archive this lexical reranker as a completed negative experimental result rather than retaining it as an active experimental component or beginning another tuning cycle.

The next step is to preserve the frozen artifacts and stop work on this heuristic. Any future situation-relevance work should begin as a separately authorized design rather than extending this lexical rule set.
