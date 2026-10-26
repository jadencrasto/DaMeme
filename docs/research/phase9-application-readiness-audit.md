# Phase 9A: application readiness audit

**Date:** 2026-10-26  
**Scope:** read-only inspection and local validation after Phase 8B.1. This audit changes no retrieval, catalog, ratings, model, or application behaviour.

## Decision

DaMeme has a working, local end-to-end **281-item reference experience** once its existing image root is configured. The browser can submit a situation and optional mood, the backend performs MiniLM retrieval and metadata-aware ranking, and the UI can render returned local images.

It is not yet a polished end-user application. The immediate blocker is configuration: the checked-in default leaves `DAMEME_RECOMMENDATION_IMAGE_ROOT` unset, so all otherwise valid default-catalog results report `image_available: false`. The 1,000-item catalog is present as an experimental/versioned artifact, but is not a safe replacement for the reference path: its expected exports and local images were not found in the configured layout, and catalog activation requires an explicit authorization decision.

## What works now

### Backend

| Capability | Implemented behaviour | Readiness |
| --- | --- | --- |
| Health | `GET /api/v1/health` responds without loading recommendation artifacts. | Ready. |
| Text recommendation | `GET /api/v1/recommendations/text` accepts 1--1,000 characters; default semantic retrieval gets 50 MiniLM text candidates, applies metadata ranking, safety, and one-per-template diversity, and returns up to 10. | Works with the local 281 reference artifacts and cached MiniLM. |
| Emotion-only browsing | `GET /api/v1/recommendations/emotion` ranks catalog metadata by the selected source emotion/sentiment and applies the same safety/diversity rules. | Works as a separate browse path. |
| Options/status | `/recommendations/options` and `/recommendations/status` allow the UI to discover filters and artifact readiness. Local status returned `ready: true`, `meme_count: 281`. | Ready. |
| Mood | A selected taxonomy mood augments the *query text* with its documented description and, when no explicit six-label preference is selected, maps to the mood's primary classifier emotion for metadata ranking. | Integrated and explainable; it is not a live emotion prediction of the user. |
| Images | `/api/v1/memes/{meme_id}/image` safely resolves a file below the configured root and returns 404 when absent. With `data/retrieval/memematch_281` set as the root, `Always_Has_Been_9.jpg` resolves locally. | Endpoint works; default configuration disables it. |
| Safety/diversity | Default `exclude_unsafe=true` applies source-text pattern exclusions, with more conservative exclusions for vulnerable emotion requests. Default `max_per_template=1` applies metadata/template heuristics after ranking. | Implemented, deterministic, and retained in all experiment paths. |

The retrieval engine validates metadata/index count, source-key alignment, dimensions, and optional export records/metadata before declaring itself ready. It lazily loads and caches either a service or its first load error. Missing/unreadable/incompatible artifacts become a user-visible 503 from recommendation routes and `ready: false` from status. Invalid request parameters are 422; a missing image is 404.

The first semantic request has no explicit server-side deadline or request cancellation. It loads MiniLM through Transformers on demand. This environment successfully completed an offline (`HF_HUB_OFFLINE=1`) retrieval, so the required checkpoint is cached locally; the application itself does not set `local_files_only`, meaning a cache miss could otherwise attempt a download. There is no startup preflight that verifies MiniLM, image availability, or a full sample request.

The SQLite schema and migration utilities exist for datasets, analysis, and emotion experiments, but the API's health/startup path does not initialize the database. The core recommendation flow reads FAISS/CSV artifacts and does not require SQLite.

### Experimental versus reference retrieval

The serving default is the existing combined-text MiniLM/FAISS path, **A0**. The public text endpoint also exposes `hybrid` BM25+RRF and `hybrid_cross_encoder` request modes, but neither has a product-selection gate and the latter can rely on a separately cached cross-encoder. These are experimental/developer controls, not a production default.

**M1** is the offline OpenCLIP visual-retrieval candidate; it has an experimental index and development-only comparison artifacts but is not connected to the serving recommendation engine. **M2** is offline rank-only RRF of A3-LH and M1 and is excluded from adoption. The lexical situation reranker is an archived negative result and is not on the online request path. Preliminary human results and overlap/novelty measurements do not establish an architecture winner.

Any future experiment must run under an explicit variant identifier, fixed catalog-generation checksum, frozen development-only query set, and separately stored output/feedback. It must not alter the default route, client default, or existing rating artifacts. Promotion requires an explicit decision after a blinded, preregistered evaluation; there must be no silent fallback from A0 to M1/M2.

### Frontend

The React/Vite single page is wired to the API, not a mock: it loads status/options/moods, submits text or emotion-only requests, renders result cards, appends the API base URL to relative image routes, displays image fallback placeholders, and has loading/error/retry states. It includes mood search and selection, optional emotion/sentiment/usage filters, a result-count selector, health indicator, responsive grid styling, labelled form controls, focus-visible styles, live loading/error announcements, and image alt text.

Missing product wiring/functionality:

- No feedback/rating control or API exists for normal users. Existing human-evaluation/rating tooling is development-only and must remain untouched.
- No result inspection view beyond the inline card; no image dialog, keyboard-driven expanded inspection, or source/provenance details.
- The UI exposes research-facing filters and an emotion-only mode that compete with the primary "describe a situation" workflow.
- There is only hash-link navigation, no history/share/new-set control, and no explicit empty-result explanation beyond the initial prompt.
- Mobile layout has a narrow breakpoint, but the large mood/filter set needs a focused responsive and keyboard audit. Tabs set ARIA roles/selection but do not implement the usual keyboard tab behaviour or `aria-controls` relationships.

## API/frontend integration checklist

| Check | Current result | Phase 9B action |
| --- | --- | --- |
| API reachable/CORS for Vite | GET-only CORS allows the current browser reads. | Retain GET; add only narrowly required methods if feedback is added. |
| Status before submit | Wired; prevents submit when artifacts report unavailable. | Show the actionable status detail and configured-catalog name. |
| Situation + optional mood | Wired to text API; mood reaches the backend. | Make this the sole primary form and keep advanced controls collapsed. |
| Returned recommendations | Wired and typed. | Surface concise relevance explanation, rank, and an accessible inspect action. |
| Image URL | Route is wired but default image root is unset. | Add a documented preflight/configuration requirement; do not make remote image fetches. |
| Errors/loading | Wired, including retry. | Distinguish artifact/configuration, offline model, timeout, and no-results messages. |
| Feedback | Missing. | Add a separate product-feedback record/API, isolated from benchmark ratings. |
| Another set | Users can edit/resubmit. | Add a clear “Try another set” action that resubmits without changing the retrieval variant. |
| Accessibility | Good baseline labels, focus, live messages, and fallback. | Complete tab semantics; test keyboard, contrast, mobile, and image-dialog focus handling. |

## Minimal polished workflow

1. The landing page asks for a situation in one prominent text field.
2. The user may choose one optional mood from a compact, searchable picker; advanced metadata filters stay collapsed.
3. Submit shows a short progress state and requests A0 only.
4. A responsive image-first result grid shows local images, a plain-language “why this matched” summary, and an inspect affordance.
5. Inspecting a recommendation opens an accessible dialog with the full image, metadata/provenance-safe explanation, and lightweight useful feedback (for example: helpful / not a fit plus an optional reason).
6. “Try another set” requests the next result set through an explicit, reproducible mechanism designed in Phase 9B; it must not silently switch models or experiment variants.

Webcam support is deliberately outside this workflow.

## Prioritized Phase 9B scope

1. **Runtime readiness and image preflight (P0).** Add a documented, testable startup/readiness check covering selected catalog, index/metadata alignment, cached MiniLM availability, configured image root, and sampled local image routes. Improve the status payload and UI message; retain the existing response contracts where feasible.
2. **Focused recommendation experience (P0).** Simplify the main view around situation + optional mood + A0 results; keep research filters out of the default path. Add empty-state, configuration-error, and no-match copy without a rewrite.
3. **Image inspection and accessibility (P1).** Add an accessible dialog/card expansion, keyboard/focus behaviour, mobile verification, and source-safe explanatory metadata.
4. **Product feedback boundary (P1).** Design and implement a new feedback store and POST API with schema/version/provenance, rate/abuse limits appropriate to local development, and tests. It must never update or reinterpret development human-evaluation ratings.
5. **Explicit experiment boundary (P1).** Introduce an internal/development-only variant runner or feature flag whose default is pinned A0; log catalog/variant hashes and never expose M1/M2 as implicit fallback. Do not change production retrieval in this work.

Out of scope: catalog ingestion/activation, model download or training, reranking tuning, held-out queries, existing rating modification, webcam work, and a full frontend rewrite.

## Missing dependencies or artifacts

| Item | Status | Consequence |
| --- | --- | --- |
| `DAMEME_RECOMMENDATION_IMAGE_ROOT` | Unset in the checked-in runtime defaults. | Default results have no image URLs despite local 281 images existing. |
| Repository `.env` | Not present. | Runtime relies on defaults/environment; setup is not self-describing. |
| MiniLM checkpoint | Present in this local environment, verified by an offline retrieval; not preflighted by the app. | A new machine may otherwise try to download it unless started offline. |
| `memematch_1000_v2` exports expected by `active_catalog()` | Not found at `exports/memematch_1000_v2.records.json` and `.metadata.json`. | The configured 1K switch should fail validation. |
| 1K local images | Not found in the catalog directory layout during this audit. | Even a valid 1K index would not render images. |
| Product feedback API/store | Absent. | Workflow cannot collect end-user feedback. |
| Request timeout/cancellation | Absent in backend/client API wrapper. | Slow cold inference has limited recovery/control. |

## Exact local startup commands

Run these in two PowerShell windows from the repository root. They explicitly use only existing local artifacts and prevent model downloads. The application remains on the 281-item reference catalog.

```powershell
# Window 1: backend
Set-Location .\backend
$env:HF_HUB_OFFLINE = "1"
$env:DAMEME_RECOMMENDATION_IMAGE_ROOT = (Join-Path (Resolve-Path ..) "data\retrieval\memematch_281")
uv sync --group dev --python 3.11
uv run uvicorn app.main:app --reload --port 8000
```

```powershell
# Window 2: frontend
Set-Location .\frontend
npm ci
npm run dev
```

Then open `http://localhost:5173`. Before using the UI, verify the service in a third shell:

```powershell
Invoke-RestMethod "http://localhost:8000/api/v1/recommendations/status"
Invoke-RestMethod "http://localhost:8000/api/v1/recommendations/text?text=surviving%20a%20Monday%20meeting&limit=5"
```

Do not set `DAMEME_RECOMMENDATION_CATALOG=memematch_1000_v2` for this readiness path. It is an unapproved experimental catalog with missing runtime artifacts identified above.

## Validation performed

- Backend dependency sync completed without downloading models.
- Backend: `105 passed` (`pytest -q`).
- Backend Ruff: one failure, pre-existing import ordering in `backend/scripts/prepare_multimodal_ui_adapter.py:7` (`I001`); no audit-phase edit was made.
- Frontend: lint passed, production build passed, and `8` tests passed.
- `git diff --check` passed before this documentation addition.
- Runtime: default engine status was ready with 281 records; an offline MiniLM query returned `Always_Has_Been_9.jpg`. With the image root omitted that result was unavailable; with the existing 281 root supplied its image path resolved.

