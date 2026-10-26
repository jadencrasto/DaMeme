# Phase 7R.5B conservative intent-aware retrieval experiment

This is an offline, development-only comparison. It does not change production recommendation behavior, catalog artifacts, ratings, models, or held-out evaluation. Results are in [`comparison.results.json`](../../backend/benchmarks/phase7r_v1_intent_comparison_20261025T065302Z/comparison.results.json), the compact [`summary.json`](../../backend/benchmarks/phase7r_v1_intent_comparison_20261025T065302Z/summary.json), and the fresh [human-evaluation candidate export](../../backend/benchmarks/phase7r_v1_intent_comparison_20261025T065302Z/fresh_human_evaluation_candidates.json).

## Interpreter coverage and fallback

The deterministic versioned interpreter (`phase7r-query-intent-v1`) detected exactly one explicit, unambiguous intent label for **10/40** development queries (25%). The remaining **30/40** queries (75%) retained the existing mood-expanded representation unchanged. It never adds a label for low-confidence or ambiguous wording.

For a high-confidence query, the only addition is one short suffix such as `Intent: celebration.` The original query, selected mood, exact evidence spans, ambiguity reasons, and final retrieval representation are all recorded per query. Mood preprocessing remains the existing recorded representation; no event, actor, sarcasm, or emotion is generated from unsupported wording.

## Retrieval parity

All variants use candidate depth 100, the same MiniLM encoder, the existing safety filter, one-per-template diversity, and deterministic score/source-key ordering. No session-repetition penalty is active. The intent variant differs from A3-LH only by the query representation on the 10 high-confidence cases.

| Variant | Unique final memes | Cross-query repetition | Median / p95 latency (ms) |
| --- | ---: | ---: | ---: |
| A0 | 172 | 57.0% | 19.75 / 34.96 |
| A3-LH | 153 | 61.75% | 19.31 / 34.93 |
| A3-LH-INTENT | 152 | 62.0% | 21.26 / 38.98 |

A3-LH-INTENT has 95.0% mean top-five overlap and 95.5% mean top-ten overlap with A3-LH, with four final candidates not present in the A3-LH final union. The small latency increase includes a second query encoding only when an intent label is added.

## Existing human ratings: descriptive regression coverage only

On the eight previously rated query identities, the available displayed-slot means are:

| Variant | Rated slots | Situation | Tone | Sendability |
| --- | ---: | ---: | ---: | ---: |
| A0 | 54 | 1.796 | 1.981 | 1.769 |
| A3-LH | 60 | 1.825 | 1.925 | 1.758 |
| A3-LH-INTENT | 57 | 1.816 | 1.921 | 1.746 |

These are not paired quality estimates: each variant can show different identities, and unjudged candidates are not scored as irrelevant. They cannot select a winner or tune the rules.

## Fresh evaluation plan

The export contains **108** stable query--meme identities from a fixed-hash, fresh eight-query subset—one unused development query per category: `academic-08`, `celebration-05`, `everyday-02`, `template-06`, `setback-03`, `nostalgia-10`, `reaction-02`, and `sarcasm-05`. It excludes all eight previously rated comparative query identities and preserves per-variant rank and score provenance. No ratings were fabricated or modified.

## Limitations

The rules were intentionally conservative, so three quarters of queries fall back unchanged. The evaluation set is still small, novelty does not prove relevance, and the fresh exported set requires blinded human judgments before assessing whether the intent representation helps situation relevance or sendability. No architecture winner, production rollout, model training, or catalog expansion follows from this experiment.
