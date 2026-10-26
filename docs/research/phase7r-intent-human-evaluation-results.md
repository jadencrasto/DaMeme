# Phase 7R.5D final intent-aware human-evaluation analysis

The completed fresh development assignment is fully rated: **49/49** stable query--meme identities have situation relevance, tone fit, and sendability scores from `intent-evaluator-1`. The saved evaluator artifact was located at [`ratings.seed.json`](../../backend/benchmarks/phase7r_v1_intent_human_eval_20261025T065752Z/ratings.seed.json), which now contains the completed ratings. It matches the assignment pool digest and has no invalid or missing assignment identity. No missing score was converted to zero.

The versioned [machine-readable metrics](../../backend/benchmarks/phase7r_v1_intent_human_eval_analysis_20261025T071213Z/final_intent_evaluation_metrics.json) include all per-query paired differences versus A0. The concise identity-level [failure analysis](../../backend/benchmarks/phase7r_v1_intent_human_eval_analysis_20261025T071213Z/failure-analysis.md) preserves examples without inventing failure mechanisms.

## Descriptive eight-query aggregate

Each cell is `nDCG@5 / P@5 / mean rating`; P@5 counts human scores of at least 2. nDCG is computed against the fully rated per-query top-five union.

| Variant | Situation | Tone | Sendability | Situation-irrelevant top-five entries |
| --- | --- | --- | --- | ---: |
| A0 | 0.842 / 0.675 / 2.075 | 0.861 / 0.750 / 2.275 | 0.844 / 0.675 / 2.000 | 13 |
| A3-LH | 0.857 / 0.750 / 2.150 | 0.856 / 0.800 / 2.300 | 0.857 / 0.725 / 2.100 | 10 |
| A3-LH-INTENT | 0.845 / 0.725 / 2.100 | 0.845 / 0.775 / 2.250 | 0.845 / 0.700 / 2.050 | 11 |

These are descriptive values from eight development queries and one evaluator. They do not establish statistical superiority or choose a production architecture.

## Changed-query result

The conservative suffix changed only `reaction-02`. On that query, A3-LH has situation nDCG@5 0.970, P@5 1.0, and mean situation 2.6; A3-LH-INTENT has 0.873, 0.8, and 2.2. This is evidence against retaining the current suffix unchanged, not evidence against all future intent-aware work.

## Conclusion

- Retain **A0** as the fixed reference baseline.
- Continue evaluating **A3-LH** as the experimental retrieval candidate; it has the strongest descriptive situation and sendability aggregates in this focused study.
- Retire the **current intent-suffix formulation** from further evaluation unchanged. It affects only one fresh query and performs worse than A3-LH on that changed query.

Future work should remain bounded: separately test direct response/send cues with a fresh preregistered sample, preserve original-query fallback, and review recurring identity exposure as a distinct session-control issue. Do not train a reranker or change production behavior from this result.
