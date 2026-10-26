# Phase 9D: feedback analytics and exploration

## Contracts and boundaries

`GET /api/v1/feedback/analytics?limit=20&offset=0` provides bounded local diagnostic aggregates and recent product-feedback rows. It is restricted to loopback clients (`127.0.0.1`, `::1`, and test client) and never reads research ratings. Empty data returns zero counts, `useful_percentage: null`, and an empty history. Product feedback is local directional input, not an unbiased measure of model accuracy.

The text recommendation route accepts repeated `exclude_meme_id` query parameters. IDs are removed only after the existing A0 retrieval, safety, and metadata ranking steps and before existing diversity selection. This preserves deterministic A0 ordering for the remaining candidates and never weakens safety to fill a page. An empty result is the explicit exhausted state.

## Client storage

Recent situation/mood pairs are stored only in `sessionStorage`, capped at eight entries, deduplicated, and clearable through the history utility. Storage failures are ignored safely; no blobs, accounts, or unnecessary personal data are stored.

## Completed Phase 9D.1 wiring

The active recommendation experience now records every displayed meme ID for the active situation/mood search. **Show me more** resends that fixed query and mood with repeated `exclude_meme_id` parameters, appends only unseen results, and disables while a request is active. An empty follow-up is shown as the safe, explicit `No more safe matches` state. A request-version guard ignores stale response bodies after a newer search starts. New searches and history revisits reset seen IDs, result batches, and feedback state; each batch gets its own request ID, which is passed to feedback submission for its cards.

The visible Recent searches panel lists bounded session-local entries, revisits one as a new request, and clears its browser-session history with an explicit button. It never stores images or feedback records.

Frontend validation for this wiring: ESLint passed; Vitest has 12 passing tests and 4 skipped legacy App tests; TypeScript/Vite production build passed; `git diff --check` passed. The skipped tests covered the replaced, minified legacy view; the focused recommendation-experience tests cover exclusions/exhaustion and history clear.

## Phase 9E manual checklist

Start locally with the Phase 9B commands, then verify A0 recommendations, image loading, Show-me-more exclusions/exhaustion when wired into the visible session controls, inspection, download/share, feedback, loopback-only analytics, and mobile keyboard flow. Browser tests do not establish real-device sharing or layout behavior.
