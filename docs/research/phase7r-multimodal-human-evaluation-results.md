# Phase 7R.6C.2 multimodal human evaluation

All 92 frozen query--meme identities across eight development queries were rated completely by the primary evaluator. Ratings were validated as integer 0--3 values on situation relevance, tone fit, and sendability; no missing value was imputed.

| Variant | Situation nDCG / P@5 / mean | Tone nDCG / P@5 / mean | Sendability nDCG / P@5 / mean | Situation 0--1 |
| --- | --- | --- | --- | ---: |
| A0 | .671 / .675 / 1.925 | .679 / .675 / 2.000 | .604 / .575 / 1.725 | 13 |
| A3-LH | .608 / .725 / 1.925 | .641 / .700 / 2.025 | .565 / .650 / 1.800 | 11 |
| M1 | .648 / .750 / 2.025 | .659 / .825 / 2.100 | .611 / .625 / 1.850 | 10 |
| M2 | .590 / .675 / 1.850 | .590 / .700 / 1.900 | .574 / .600 / 1.750 | 13 |

M1 has the strongest mean situation, tone, and sendability ratings, as well as the fewest low-situation top-five entries. A0 has the strongest situation nDCG; M1 has the strongest sendability nDCG. M2 does not improve on A3-LH in these descriptive aggregates.

This is one evaluator and eight development queries, so it does not establish statistical superiority or select a production architecture. OpenCLIP remains worth a focused follow-up because its M1 result improves several descriptive rating measures, while rank fusion is not supported in its current form. Phase 7R.7 should investigate conservative visual-candidate gating and situation-specific reranking, then use a fresh blinded sample; it should also address repeated broadly applicable meme identities. No training or production change follows from this experiment.
