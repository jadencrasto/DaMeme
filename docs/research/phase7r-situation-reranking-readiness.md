# Phase 7R.7 situation-reranking readiness

## Evidence-backed failure taxonomy

The ratings support situation mismatch as the principal observable problem: `academic-09:23-wolves-laugh-80.jpg` scored situation/tone/sendability `0/0/0` after retrieval by A3-LH, M1, and M2. `celebration-07:job-interview.jpg` similarly scored `0/0/0` for A3-LH and M2. These scores establish mismatch, but do not establish a causal visual or semantic mechanism.

Tone without usable situation is also present: `academic-09:Afraid_To_Ask_Andy_Closeup_11.jpg` (M1) scored `1/2/1`; `everyday-10:where-does-it-hurt.png` (A0 and M2) scored `1/2/1`; and `template-02:20_minute_adventure_rick_morty_13.jpg` (M1) scored `1/2/1`. These are candidates for query-specific ordering rather than a rating-derived explanation of content.

Low sendability accompanies some low-situation results, such as `template-02:Always_Has_Been_11.jpg` (M1, `1/0/0`). Repetition remains a separate exposure issue: earlier four-way output shows repeated identities, but frequency alone is not a relevance judgment.

## Reranking readiness

| Approach | Cost | Expected benefit | Evidence link |
| --- | --- | --- |
| Query-specific situation-aware reranking over the existing depth-100 candidates | Low/medium | High | Repeated 0/0/0 and 1/2/1 candidates show that rank ordering can be wrong even when a broadly fitting tone is present. |
| Conservative candidate filter for explicit situation mismatch cues | Low | Medium | May remove the clearest failures; must be evaluated for false exclusions. |
| Session repetition control | Low | Medium | Existing repetition audits show repeated exposure, but it does not solve a single-query mismatch. |
| Catalog expansion | High | Unknown | Only justified when an inspected candidate set lacks plausible alternatives; current ratings alone cannot establish this. |

## Bounded Phase 7R.7 proposal

Do not train a model. Pre-register a deterministic candidate-ordering experiment over existing A0/M1 candidate unions: retain safety and template diversity, add only explicit query-situation features, and compare unchanged ranking against one conservative reranker on fresh development queries. Measure ranking movement, retained coverage, repetition, latency, and a small blinded human-evaluation union. Do not change production behavior.

## Limitations

Ratings come from one evaluator and small development subsets. They describe outcomes, not causal explanations; no held-out query, architecture winner, or catalog-coverage conclusion follows.
