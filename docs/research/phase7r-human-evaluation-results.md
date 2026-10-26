# Phase 7R.4E.2 human-evaluation and repetition analysis

This is a read-only analysis of the preregistered comparative development assignment. The full machine-readable record is [`evaluation.metrics.json`](../../backend/benchmarks/phase7r_v1_human_eval_analysis_20261025T063017Z/evaluation.metrics.json), with a compact companion [`analysis.summary.json`](../../backend/benchmarks/phase7r_v1_human_eval_analysis_20261025T063017Z/analysis.summary.json). It does not alter ratings, retrieval, query processing, models, or held-out data.

## Rating validation

All **103/103** required stable query--meme identities are completely rated on situation relevance, tone appropriateness, and sendability. Four identities retain their earlier pilot judgments and 99 additional identities are rated by the comparative evaluator. The copied pilot records exactly match the original pilot ratings; no invalid rating key, missing axis, or duplicate pool identity was found.

## Eight-query comparative metrics

The table gives macro averages over the eight preregistered development queries. Each cell is `nDCG@5 / P@5 / mean rating`, using the 0--3 human score for that axis; P@5 counts scores of at least 2. nDCG's ideal ranking is calculated from the fully judged assignment union for that query, not from unjudged candidates.

| Variant | Situation | Tone | Sendability |
| --- | --- | --- | --- |
| A0 | 0.630 / 0.600 / 1.800 | 0.668 / 0.725 / 1.963 | 0.637 / 0.600 / 1.763 |
| A1 | 0.684 / 0.625 / 1.888 | 0.693 / 0.700 / 1.975 | 0.670 / 0.600 / 1.813 |
| A2 | 0.622 / 0.575 / 1.700 | 0.715 / 0.650 / 2.050 | 0.625 / 0.575 / 1.650 |
| A3-EQ | 0.677 / 0.575 / 1.863 | 0.713 / 0.675 / 1.988 | 0.681 / 0.575 / 1.813 |
| A3-LH | 0.687 / 0.650 / 1.963 | 0.723 / 0.750 / 2.100 | 0.686 / 0.625 / 1.888 |
| A3-RRF | 0.652 / 0.650 / 1.863 | 0.714 / 0.725 / 2.075 | 0.659 / 0.625 / 1.813 |

Per-query deltas versus A0 are retained in the machine-readable report. They are descriptive only: eight development queries and one primary evaluator are not enough for uncertainty intervals, statistical superiority claims, or architecture selection.

## Relevance failures and successes

Across the six top-five lists, there are 92 entries with situation score 0 or 1, 21 with tone at least 2 but situation at most 1, and three with situation/tone at least 2 but sendability at most 1. There are also 85 strongly relevant (situation 3) entries, so the observed issue is not universal.

Examples are identity-level observations, not inferred failure explanations:

- `celebration-02` / A0 rank 1 / `A_train_hitting_a_school_bus_1.jpg`: situation 0.5, tone 1, sendability 0.5.
- `celebration-02` / A3-EQ rank 1 / `Anakin_Padme_4_Panel_2.jpg`: situation 0, tone 1, sendability 0.
- `nostalgia-03` / A0 rank 1 / `Afraid_To_Ask_Andy_6.jpg`: situation 1, tone 2, sendability 1. The same identity also appears under A3-EQ with those scores.
- `template-07` / A1 rank 2 / `tf2-heavy-i-fear-no-man.jpg`: situation 2, tone 2, sendability 1; it appears under A3-RRF at rank 4 with the same scores.
- A strongly relevant example is `academic-01` / A2 rank 1 / `fffffffuuuuuuuuuuuu.jpg`: 3 on all axes.

## Repetition audit

Across all 40 development queries, final top-five cross-query repetition is substantial: A0 50.5%, A1 53.0%, A2 51.5%, A3-EQ 57.5%, A3-LH 55.5%, and A3-RRF 54.5%. This is the fraction of 200 result slots not occupied by a unique meme. Each variant also has repeated memes spanning distinct query categories (30--38 unique repeated identities in top five). Final top-ten repetition ranges from 55.8% to 62.0%.

Within one final list, no source-key duplicate occurred. The existing one-per-template diversity filter also removed every within-list template duplicate: raw top-ten candidate lists had 66--231 extra template slots by variant, while final lists had zero. Therefore the remaining issue is cross-query concentration, not a failure of within-list template diversity.

The evaluator's repeated-query experience is separate from retrieval repetition: the comparative UI necessarily showed 10--17 pair tasks for each of its eight fixed queries because it presents a rating task for every candidate identity. The UI task counts are preserved in the summary; they are not repeated searches.

## Query-processing audit

For all eight evaluated queries, the exact MiniLM input equals the recorded mood-expanded query. The current form is `Situation: <original query>` plus `Selected mood context: <fixed mood description>`. For example, both an elevator delay and an exam surprise receive the same frustration suffix, `blocked effort or repeated difficulty.` The final ranking combines semantic similarity with the default text weights (semantic 0.90, primary mood emotion 0.08, no supplied sentiment), then applies source-text safety filtering and one-per-template diversity.

The implementation audit cannot determine from numerical relevance ratings alone that any individual mood suffix is semantically unrelated; it records the exact strings rather than inventing such explanations. It does show that the expansion is always additional wording for these eight queries, so rewrite fidelity should be tested directly before changing it.

## Next experiment

The most evidence-backed next experiment is an **offline cross-query repetition-control ablation**: apply a transparent, bounded exposure penalty or cap only after normal retrieval, measure its effect on the same human-rated identities plus a new, separately planned development set, and report situation/tone/sendability alongside coverage. This targets the observed 51--58% cross-query repetition without conflating it with within-list diversity, which already works.

Do not train a reranker yet. The current 103 pairs cover only eight queries and are inadequate for a supervised model. A conservative expansion ablation and a larger, more diverse evaluator set are sensible subsequent studies, but neither is needed to establish the next controlled repetition experiment.
