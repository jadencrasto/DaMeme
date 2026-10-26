# Phase 7R.7A situation-aware reranking prototype

## Scope

This is a development-only, offline ordering experiment over the 40 development queries. It reuses the saved Phase 7R.6B.4 four-way comparison and the unchanged `memematch_1000_v2` metadata. It does not train or download a model, modify the catalog, invoke held-out queries, alter ratings, or change production retrieval.

The reference and experimental baselines are A0, A3-LH, and M1. M2 is deliberately excluded: the prior human-evaluation evidence did not support its existing RRF fusion as a quality improvement.

## Provenance and safety boundary

For every development query, the implementation verifies the saved source has exactly ten unique, catalog-mapped, safety-filtered and one-per-template candidates for A0, A3-LH, and M1. It retains the saved mood-expanded effective query as baseline provenance. The reranker uses only the original user wording for its situation signal, and reads source OCR/title, local context, global context, BLIP/global caption, usage metadata, and template-group metadata as candidate evidence.

R-T reranks A3-LH's saved final ten candidates. R-V reranks M1's saved final ten candidates. This intentionally leaves the candidate membership, safety result, and template-diversity result unchanged. It is a post-filter ordering experiment rather than a new retrieval or filter.

## Heuristic

The situation signal is exact lexical overlap only. It extracts non-generic query tokens of four or more characters, ignores selected mood language and common conversational terms, then promotes a candidate only when one of those explicit tokens occurs in its source-backed candidate evidence. There is no synonym expansion, learned score, inferred scenario, filename-specific rule, query-ID rule, or human-rating feature.

Negation is conservative: the content token after a negator is never used as a positive match. Queries with ambiguity markers such as “maybe”, “not sure”, or “something” fall back to the unmodified baseline order. Any query with no direct match also falls back unchanged.

An explicit match receives a target rank at most two positions above its baseline rank; target-rank ties resolve by original rank and then source key. This is deterministic and auditable. Each output row stores its baseline and reranked rank, displacement, matched and negated terms, and reason. The observed displacement was at most one position because of the saved candidate ordering.

Lexical matching is intentionally limited: OCR errors, morphology, paraphrases, captions that omit a relevant fact, and generic-but-important concepts can cause false negatives. Conversely, an exact token can be coincidental rather than a genuine situational fit. Results are a candidate for fresh blinded evaluation, not evidence that lexical matching improves relevance.

## Development diagnostics

The machine-readable output is [`comparison.json`](../../backend/benchmarks/phase7r_v1_situation_reranking_20261026T000001Z/comparison.json), with [`rank_changes.json`](../../backend/benchmarks/phase7r_v1_situation_reranking_20261026T000001Z/rank_changes.json) retaining the per-query audit trail.

| Variant | Top-5 overlap vs A0 | Top-10 overlap vs A0 | Unique final memes | Cross-query repetition | Median saved/processing latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0 | 1.000 | 1.000 | 172 | 57.0% | 10.043 ms |
| A3-LH | .635 | .612 | 153 | 61.8% | 10.361 ms |
| M1 | .155 | .155 | 165 | 58.8% | 8.096 ms |
| R-T | .630 | .612 | 153 | 61.8% | .210 ms |
| R-V | .155 | .155 | 165 | 58.8% | .198 ms |

R-T made 35 promotions and 35 corresponding demotions across 23 queries. R-V made 17 promotions and 17 corresponding demotions across 13 queries. Thirteen R-T queries and twenty R-V queries had no direct situation evidence and therefore retained their exact baseline ordering. Both variants have exact candidate-set parity with their baseline, so safety and template-diversity membership are unchanged.

The baseline latencies are historical saved measurements and are not directly comparable across model families. R-T/R-V times measure only the added in-memory lexical ordering work. Existing ratings appear only in the comparison as descriptive observed-value aggregates: 47/51/44 of the A0/A3-LH/M1 output slots have an exact prior rating identity. R-T's top-five observed ratings are therefore not a tuning target, and R-V's are only a reordered view of the same candidate identities. Missing ratings remain absent rather than being imputed or rewritten.

## Repetition

No session-level exposure penalty is combined with R-T or R-V. A later repetition experiment should pre-register sessions (including related, unrelated, and repeat-query sequences), apply a bounded exposure policy after an independently frozen relevance ordering, compare novelty and repeat rate against an unmodified session control, and use a blinded relevance evaluation. That keeps exposure diversity separate from per-query situation relevance.

## Interpretation

The prototype is mechanically suitable for a fresh blinded human evaluation: it is deterministic, bounded, provenance-checked, safety/template-preserving, and keeps an audit trail. It is not ready for architecture selection or production use. A blinded assignment should be separately preregistered and must not treat prior missing ratings as zeros.
