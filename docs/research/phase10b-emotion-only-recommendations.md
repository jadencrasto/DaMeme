# Phase 10B: emotion-only recommendations

**Status:** implemented for the existing 281-item A0 product catalog. No retrieval variant, catalog selection, model, dataset, rating, or evaluation-query artifact was changed.

## Root cause

Mood-only searches failed before reaching retrieval for two independent reasons:

1. `RecommendationExperience` disabled submission and returned early when `text.trim()` was empty.
2. `GET /api/v1/recommendations/text` declared `text` as required with `min_length=1`.

The existing `/recommendations/emotion` route only accepts one of six classifier emotions. It cannot represent the 32 user-facing mood choices and did not carry the A0 semantic candidate pool or its exclusion support.

## Supported request modes

The existing A0 text route now accepts these mutually compatible inputs:

| Mode | `text` | `mood` | A0 query representation |
| --- | --- | --- | --- |
| Situation only | required | absent | The supplied situation, unchanged. |
| Situation + mood | required | present | Existing deterministic `Situation: … / Selected mood context: …` representation. |
| Emotion-only | absent or whitespace | required | Fixed selected-mood label, description, and taxonomy synonyms. |

Both fields empty return HTTP 422 with `Provide a situation or select a mood`. The fixed mood-only representation is derived solely from the documented 32-mood taxonomy; it does not invent a situation or predict user emotion. Existing text-only requests preserve their prior effective query and ranking inputs.

The selected mood continues to map deterministically to its strongest existing six-emotion metadata weight for A0's emotion component. The semantic component embeds the fixed mood description. Safety filtering, candidate exclusions, template diversity, deterministic tie ordering, and exhaustion behavior remain in the same A0 path.

## Frontend behavior

- A mood makes the recommendation button available even when the situation field is blank.
- Mood-only requests omit the empty `text` query parameter and send `mood=<id>`.
- Recent searches can contain a labelled `Mood-only search · <mood>` entry and revisiting one creates a fresh request/batch identity.
- “Show me more” retains the successful active search, including a mood-only search, and repeats its stable IDs as `exclude_meme_id` parameters.
- Feedback stores the request ID and the mood associated with the exact result batch, not a subsequently edited control value.
- Both-empty input is prevented in the UI and still validated by the API.

## Limits and safeguards

The 32 moods are a user-interface taxonomy mapped onto six legacy classifier emotions and optional positive/negative/neutral sentiment metadata. The mapping is many-to-one and is not a calibrated emotion classifier. Mood-only results are therefore broad affective matches, not evidence that a meme accurately represents a person’s state or a relevance guarantee. No confidence values are exposed, no research ratings were used, and no held-out queries were accessed for tuning.

M1 OpenCLIP, M2/RRF, and the archived lexical reranker remain outside the product path.

## Validation

- Added coverage for all 32 mood identifiers, deterministic mood-only representations, API validation, exclusion forwarding, and client request serialization.
- Added frontend coverage for mood-only submission and session-history display.
- Existing exploration coverage continues to verify exclusion serialization and exhausted results.

Validation completed: Ruff passed for changed backend files; focused backend tests passed (54); frontend ESLint passed; frontend tests passed (14 active, 4 pre-existing skipped); TypeScript/Vite production build passed; and `git diff --check` passed.
