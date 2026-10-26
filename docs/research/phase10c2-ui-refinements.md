# Phase 10C.2: UI refinements and recommendation count

## UI changes

The mood picker now displays only existing mood names. Quick chips, selected/pressed states, search, all 32 taxonomy entries, optional deselection in situation mode, and required selection in mood-only mode remain intact. Group/descriptive strings are no longer rendered; taxonomy descriptions remain in backend data for the deterministic mood-only A0 representation.

Recommendation cards retain Download, Share, Useful, and Not relevant. Download and feedback controls use compact secondary-action sizing; Share retains its larger existing visual treatment. Mobile sizing restores a 2.5rem minimum control height.

## Recommendation-count behavior

The former visible “Show me more” action is replaced by an **Show up to** selector with 10 (default), 20, and 50. The selected number is sent as the existing `limit` query parameter in one A0 request. The gallery shows the actual returned count and, when it is lower than the selected maximum, explains that fewer safe, varied matches were available.

The selected number is a maximum, never a quota. No follow-up query is made to fill a visual grid, and no duplicate appending is introduced. Situation-only, situation-plus-mood, and mood-only requests all use the same bounded route, request identity, feedback association, session history, and request-version guard.

## Backend limits and eligibility review

`/api/v1/recommendations/text` already validates `limit` from 1 through 50. Its default semantic candidate pool is 50 and `candidate_limit` must not be lower than `limit`; therefore a request for 50 remains bounded to the existing top-50 semantic pool rather than widening retrieval merely to fill cards.

The engine applies safety during candidate ranking before diversity selection. Existing template diversity is applied afterwards with the existing one-per-template configuration. These controls remain in force for 10, 20, and 50. Exclusions are still accepted by the route, although the product no longer presents a follow-up paging action.

There is no separately validated relevance cutoff on this product path. This refinement does not invent one or claim that every high-count result is equally relevant. Larger selected maxima can expose lower-ranked members of the same bounded candidate pool; users may receive fewer than requested after safety and diversity. No held-out queries or research ratings were used.

## Validation and remaining checks

Frontend tests cover the default/changed count serialization, a returned count lower than requested, hidden picker group text, all-32 search behavior, existing feedback identity, image actions, error/loading behavior, and discovery modes. Relevant existing A0 tests cover safety, template diversity, deterministic ordering, and route bounds.

Manual browser follow-up should verify 20/50 result presentation against the local 281-image catalog, card-control wrapping on a narrow touch device, and real download/share behavior. Browser tests cannot establish native share sheets or download prompts.
