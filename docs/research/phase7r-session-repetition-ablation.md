# Phase 7R.4F offline session-repetition ablation

This experiment is an offline simulation only. It neither changes production ranking nor writes ratings, models, catalogs, or held-out artifacts. The full per-session record is [`session_simulation.json`](../../backend/benchmarks/phase7r_v1_session_repetition_ablation_20261025T063550Z/session_simulation.json), with a compact [`summary.json`](../../backend/benchmarks/phase7r_v1_session_repetition_ablation_20261025T063550Z/summary.json).

## Definitions and simulation

Three distinct phenomena are measured separately:

- **Within-list duplicates:** duplicate source keys in one result list. These remain absent.
- **Cross-query repetition:** the same meme returned by independent benchmark searches. This audit motivated the work but is not directly blocked.
- **Session repetition:** a meme re-shown after an earlier search in the same simulated session. This is the intervention target.

The deterministic simulation uses only the 40 development queries and seed `20261025`: eight category-homogeneous related-query sessions, five mixed-category unrelated sessions, and 40 repeated-query sessions (each query shown three times). This creates 200 search events. A0 and A3-LH are evaluated with no penalty, a 0.05 soft penalty for memes shown in the immediately previous search, and the same soft penalty for any meme shown in the previous three searches. Penalties are applied only to the post-safety, final candidate reranking; candidates are never excluded permanently, and one-per-template diversity remains in place.

## Repetition and ranking trade-off

| Variant / policy | Mean unique memes per session | Session repeat rate | Top-5 / top-10 overlap with unmodified | Displaced original top-five items |
| --- | ---: | ---: | ---: | ---: |
| A0 / none | 18.85 | 56.53% | 1.000 / 1.000 | 0 |
| A0 / previous 1 | 27.04 | 30.78% | 0.748 / 0.777 | 252 |
| A0 / previous 3 | 35.53 | 5.18% | 0.536 / 0.547 | 464 |
| A3-LH / none | 18.75 | 56.80% | 1.000 / 1.000 | 0 |
| A3-LH / previous 1 | 27.15 | 30.43% | 0.750 / 0.770 | 250 |
| A3-LH / previous 3 | 35.75 | 4.47% | 0.540 / 0.541 | 460 |

The one-search policy roughly halves session repetition while preserving about three quarters of the unmodified top-five identities. The three-search policy nearly eliminates simulated repetition, but it changes about 46% of top-five membership and displaces 460 original top-five items for A3-LH. It is consequently too aggressive to advance without more evidence.

## Available human-rated slots

The simulation visits 2,000 displayed slots for each variant/policy configuration; most are unjudged. Ratings are reported only where the existing 103-pair study contains a judgment, never as zero for other slots.

| A3-LH policy | Judged displayed slots | Mean situation | Mean tone | Mean sendability | Unjudged slots |
| --- | ---: | ---: | ---: | ---: | ---: |
| None | 350 | 1.836 | 1.950 | 1.793 | 1,650 |
| Previous 1 | 285 | 1.846 | 1.965 | 1.804 | 1,715 |
| Previous 3 | 218 | 1.853 | 1.986 | 1.819 | 1,782 |

These are not paired relevance estimates: the penalties change which previously rated identities appear and repeated session appearances give some identities greater weight. They therefore cannot establish that either penalty preserves or improves relevance.

## Recommendation and limitations

The conservative **previous-one-search soft penalty** deserves a further, preregistered offline study because it materially reduces simulated re-exposure while retaining substantially more of the original ranking than the three-search alternative. It is not ready for production.

Limitations: sessions are synthetic, the 0.05 penalty is an uncalibrated research setting, the 103 ratings cover only eight queries, human-rated coverage differs after reranking, and novelty is not a quality measure. A next study should collect balanced judgments of baseline-versus-penalized outcomes in session sequences before any serving change. No architecture winner, trained reranker, or global suppression rule follows from this experiment.
