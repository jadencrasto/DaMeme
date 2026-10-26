# Phase 7R.5A query-intent research audit

This phase is a specification and audit only. It does not change retrieval, production recommendation behavior, models, datasets, ratings, or held-out evaluation. The machine-readable artifacts are [`phase7r_v1_query_intent_taxonomy_v1.json`](../../backend/benchmarks/phase7r_v1_query_intent_taxonomy_v1.json) and the [development-only audit](../../backend/benchmarks/phase7r_v1_query_intent_audit_20261025T063955Z/development_query_intent_audit.json).

## Observed development-query patterns

The audit includes exactly 40 development queries across eight benchmark categories and deliberately omits held-out query text. The current retrieval input is always the original situation wrapped as `Situation: ...` plus a fixed selected-mood description. Consequently, the user wording remains present, but the same mood description can accompany distinct situations.

The query wording combines at least four compatible forms: a narrated event, a request for a reaction, an implied affect, and occasionally a direct conversational-send use case. None guarantees the others. A system should therefore retain the situation separately from the requested communicative use, rather than replacing it with a generic mood label. Ambiguity is expected for short, mood-heavy, or indirect prompts and must remain explicit.

The recorded human evidence is mixed rather than diagnostic of a single cause: 92 experimental top-five entries have situation relevance 0--1; 21 have tone at least 2 but situation at most 1; and 85 entries have situation relevance 3. Repeated meme identities also occur across distinct categories. These observations motivate an intent audit, but do not prove that query understanding, rather than catalog coverage or representation/ranking behavior, caused any individual failure.

## Small intent taxonomy

The proposed taxonomy has eight labels: `celebration`, `frustration_setback`, `disbelief_confusion`, `sarcasm_irony`, `comfort_empathy`, `embarrassment_self_deprecation`, `relatable_humor`, and `conversational_reaction`.

Each interpreter result keeps separate fields for:

- The verbatim original query and supported situation/event phrase.
- One or more communicative-intent labels, or `unknown`.
- Desired tone from explicit wording and optional selected mood.
- `send_as_response` as `yes`, `no`, or `unknown`.
- Explicit ambiguity reasons and a high/medium/low confidence level.

This is deliberately not an exhaustive emotion ontology. Sarcasm, event causes, interpersonal roles, and unstated emotion are never inferred merely because a meme result would benefit from them.

## Conservative interpreter specification

Phase 7R.5B should be deterministic and CPU-only. It accepts the original query and optional selected mood, extracts only lexical cues defined in a checked-in rule table, and emits a structured record plus at most one short additional retrieval form. Its default retrieval text remains the original query.

- **High confidence:** explicit situation and an explicit reaction/intent cue.
- **Medium confidence:** explicit situation with only mood or general reaction evidence.
- **Low confidence or high ambiguity:** original-query-only retrieval; record `unknown` rather than guessing.

The only permitted augmentation is compact and attributable, for example `Reaction intent: celebration.` Unsupported events, generated backstories, inferred sarcasm, and free-form paraphrases are forbidden. This prevents the interpreter from overthinking a concise request.

## Evidence-bound failure audit

The audit preserves identity-level examples rather than assigning causal labels. It includes `Afraid_To_Ask_Andy_6.jpg` for `nostalgia-03` with tone 2 but situation/sendability 1, a contextually adequate but low-sendability `tf2-heavy-i-fear-no-man.jpg` example, strongly relevant `academic-01` examples, and the recorded repeated identities across categories. These should remain error-review prompts, not training labels or rule targets.

## Phase 7R.5B implementation scope

Implement only an offline, versioned comparison of:

1. A0 combined semantic baseline.
2. Existing A3-LH local/global baseline.
3. A3-LH with the conservative intent representation.

Predefine a candidate depth of 100, retain the current safety filter, one-per-template diversity, and stable source-key tie-breaking. Record p50/p95 latency, top-five/top-ten overlap, candidate novelty, per-query result count, and session repetition separately.

Rules must be frozen before a fresh preregistered development evaluation subset is assembled. The eight already-rated comparative queries can be reported only as descriptive regression checks, not used to select or optimize intent rules. Human evaluation must use paired identity-level judgments on the fresh subset; no training, reranker, catalog expansion, or production change is in scope.
