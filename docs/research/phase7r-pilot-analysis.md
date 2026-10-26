# Phase 7R.3B human pilot analysis

## Scope and completion

This analysis reads the completed `phase7r_v1_pilot.ratings.json` for evaluator `pilot-a`, the immutable 25-pair pilot assignment, and the 319-pair development pool. It does not create or edit a rating, regenerate the catalog, touch the held-out test split, or alter serving behavior. The machine-readable companion is [`backend/benchmarks/phase7r_v1_pilot.analysis.json`](../../backend/benchmarks/phase7r_v1_pilot.analysis.json).

`pilot-a` completed **25/25** assigned pair reviews. Every completed review contains the required three 0–3 ratings; there are no missing pilot pair keys.

## Numerical summary

| Axis | Mean | Distribution 0 / 1 / 2 / 3 |
| --- | ---: | --- |
| Situation relevance | 1.96 | 1 / 6 / 11 / 7 |
| Tone/mood appropriateness | 2.08 | 1 / 4 / 12 / 8 |
| Sendability / reaction usefulness | 1.88 | 2 / 6 / 10 / 7 |

The pilot's situation–sendability Pearson correlation is **0.795** across 25 ratings. This is descriptive only: it is one evaluator, a purposefully selected 25-pair subset, ordinal 0–3 scales, and does not establish causality. It is consistent with the evaluator's observation that a poor situational match often also reduces usefulness as a reaction.

### Per-category averages

| Category | Reviews | Situation | Tone | Sendability |
| --- | ---: | ---: | ---: | ---: |
| Academic and workplace stress | 4 | 2.50 | 2.75 | 2.50 |
| Celebrations and achievements | 3 | 1.67 | 1.67 | 2.00 |
| Everyday situations | 3 | 1.67 | 2.33 | 1.67 |
| Meme-template references | 3 | 1.67 | 1.67 | 1.67 |
| Negative emotions and setbacks | 3 | 1.67 | 2.00 | 1.00 |
| Nostalgia and memories | 3 | 2.33 | 2.00 | 2.67 |
| Reaction and response intent | 3 | 2.00 | 2.00 | 1.67 |
| Sarcasm and irony | 3 | 2.00 | 2.00 | 1.67 |

These category values are not comparative performance claims: three or four observations per category are far too few, and the pilot selection is deliberately stratified rather than representative.

## Lowest situation-rated pilot examples

| Query | Meme ID | Situation | Tone | Sendability |
| --- | --- | ---: | ---: | ---: |
| I passed my driving test on the first try. | `A_train_hitting_a_school_bus_1.jpg` | 0 | 0 | 0 |
| What should I send back when my friend says they got the concert tickets? | `Always_Has_Been_9.jpg` | 1 | 1 | 0 |
| It is nice that the printer only jams when there is a line behind me. | `Aaaaand_Its_Gone_3.jpg` | 1 | 1 | 1 |
| The one thing I needed is out of stock everywhere. | `10-guy-137.jpg` | 1 | 1 | 1 |
| I need a meme for when one person is calm and everyone else is panicking. | `10-guy-137.jpg` | 1 | 1 | 1 |

The final two rows show the same meme appearing under two different queries and receiving low ratings in both instances. That is a useful investigation target, not proof of why it was retrieved.

## Can the benchmark metrics be calculated?

No. The pilot covers 25 of 319 development pairs, and no development query has every pooled candidate judged. Therefore Precision@5, nDCG@5, and nDCG@10 are **insufficient coverage**, not zero. The analysis command reports zero fully judged queries for the semantic run and does not emit misleading aggregate metric values. Complete query pools are required because the implementation computes ideal DCG only from fully judged candidates.

## Qualitative error taxonomy

The evaluator's two observations motivate these optional annotations for future reviews:

- `completely_irrelevant`
- `semantically_related_wrong_situation`
- `correct_situation_wrong_emotional_tone`
- `relevant_not_useful_as_reaction`
- `unclear_meme_meaning`
- `other`

Do **not** backfill these labels from numerical ratings. A `1/1/1` triple could plausibly represent several of these mechanisms; a low sendability score could be a comprehension, social-context, humor, or personal-preference issue. Future review records should optionally capture one controlled `error_label`, a short `rationale`, and a low/medium/high confidence field after the numerical axes are completed.

## Hypotheses for Phase 7R.4 local/global-context experiments

1. **Local context improves situation specificity.** Separating local/OCR situational text from global visual context will improve situation relevance for prompts where a generic image or broad theme is semantically related but does not match the requested event.
2. **Global context can retain tone while harming event match.** Tone is slightly higher than situation in this pilot (2.08 vs. 1.96). Test whether a local-heavy candidate set plus a global-aware reranker preserves tone without promoting the wrong situation.
3. **Scope-aware reranking can reduce repeated generic mismatches.** Compare a one-vector baseline with a local-context score, a global-context score, and a transparent learned-or-fixed fusion; inspect recurrent low-rated meme/query patterns before optimizing aggregate metrics.
4. **Reaction intent needs its own signal.** Sendability is the lowest mean axis (1.88). Test whether prompts framed as replies benefit from context fields and usage categories differently from descriptive situation prompts.

Each experiment should use the same catalog, query set, provenance, and development pool policy; add candidates rather than replacing the existing pool. The next experiment should collect qualitative error labels for at least the low-situation pilot examples before claiming a mechanism.

## Readiness for Phase 7R.4

**Ready to implement a bounded, offline local/global representation ablation on development only.** It is not ready for architecture selection, held-out evaluation, or an accuracy claim. First preserve the existing semantic run, add the new run as a contributor, collect enough complete per-query development judgments for metrics, and recruit an overlapping second evaluator before interpreting small differences.

## Reproduce the analysis

```powershell
Set-Location C:\Users\jason\OneDrive\Desktop\DaMeme\backend
uv run python -m app.benchmark analyze-pilot `
  --queries .\benchmarks\phase7r_v1_queries.json `
  --pool .\benchmarks\phase7r_v1_development.pool.json `
  --pilot .\benchmarks\phase7r_v1_pilot_25.json `
  --ratings .\benchmarks\phase7r_v1_pilot.ratings.json `
  --evaluator pilot-a `
  --output .\benchmarks\phase7r_v1_pilot.analysis.json
```
