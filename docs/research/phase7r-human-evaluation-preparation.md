# Phase 7R.4E.1 pooled human-evaluation preparation

This preparation is development-only and collects no judgments. The versioned materials are in [`backend/benchmarks/phase7r_v1_human_eval_preparation_20261025T055104Z`](../../backend/benchmarks/phase7r_v1_human_eval_preparation_20261025T055104Z). They merge the immutable 319-pair development pool with each Phase 7R.4D.2 final top-ten result, deduplicated exclusively by stable `(query_id, source_key)` identity.

The merged pool contains 1,133 unique pairs. The existing pilot supplied 25 judgments, copied unchanged to `ratings.seed.json` using the new pool digest; 1,108 pairs remain unjudged. No absent judgment is represented as a score or as irrelevance. Every pair retains prior source-run provenance plus contributor ID, filtered rank, raw rank, and raw/final retrieval scores for A0, A1, A2, A3-EQ, A3-LH, and A3-RRF.

| Contributor | Pooled pairs | Previously judged | Unjudged |
| --- | ---: | ---: | ---: |
| A0 | 400 | 17 | 383 |
| A1 | 400 | 11 | 389 |
| A2 | 400 | 3 | 397 |
| A3-EQ | 400 | 9 | 391 |
| A3-LH | 400 | 13 | 387 |
| A3-RRF | 400 | 5 | 395 |

## Proposed assignments

Plan A is the practical recommendation: an explicitly exploratory 72-pair diagnostic assignment, with nine unjudged candidates per category. It deterministically prioritizes candidates introduced by non-A0 variants and retrieval disagreements, while preserving category balance. It is not suitable for selecting an architecture winner.

Plan B is the preregistered comparative design: one development query per category, selected by a fixed hash before inspecting ratings. It fully covers the union of all six variants' final top-five results. That is 103 unique pairs: four already rated and exactly 99 additional ratings. It supports fair paired comparisons for that fixed subset but is a larger first commitment.

`second_evaluator_diagnostic.assignment.json` provides an optional second-evaluator overlap over the same 72 identities. The existing UI randomizes it using seed `20261026`; no second evaluator is assumed.

Both assignment files are validated by the same loader used by the benchmark UI. They exclude held-out queries, contain no duplicate identities, and retain no fabricated or inferred judgments.
