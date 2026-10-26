# Phase 9B: product integration

Phase 9B keeps the 281-item **A0** MiniLM/FAISS recommendation path as the only product retrieval variant. M1 OpenCLIP remains an offline development experiment; M2 RRF and the archived lexical reranker have no serving or frontend control.

## Product flow

1. Describe a situation and optionally choose a mood.
2. The browser makes the existing A0 text request with a 15-second abort timeout; the backend applies its own 15-second response bound.
3. Results retain the existing safety, ranking, and one-per-template diversity behaviour and render through the existing local image route.
4. Selecting an image opens an accessible responsive preview: the close control receives focus, Escape closes it, and image failures remain non-fatal.
5. A user can mark each recommendation Useful or Not relevant. Feedback is written once per request/meme pair to `product_feedback`; it is never read by ranking code or research-rating tooling.

## Startup preflight

`GET /api/v1/recommendations/status` now validates the selected catalog's retrieval artifacts, the configured image root and resolvable images, local cached MiniLM configuration, and SQLite availability/migrations. It returns actionable `detail` text and keeps the recommendation UI disabled when prerequisites are missing.

The local 281 catalog is configured with relative paths. No username-specific or absolute path is required. `.env.example` documents the required image root and offline timeout settings.

```powershell
# Window 1, repository root
Set-Location .\backend
$env:HF_HUB_OFFLINE = "1"
$env:DAMEME_RECOMMENDATION_IMAGE_ROOT = "../data/retrieval/memematch_281"
uv sync --group dev --python 3.11
uv run uvicorn app.main:app --reload --port 8000
```

```powershell
# Window 2, repository root
Set-Location .\frontend
npm ci
npm run dev
```

Open `http://localhost:5173`, then check `http://localhost:8000/api/v1/recommendations/status`. Do not set `DAMEME_RECOMMENDATION_CATALOG=memematch_1000_v2`.

## Feedback isolation

`product_feedback` records request ID, stable meme ID, selected mood, fixed `a0` retrieval variant, value, and timestamp. The unique `(request_id, meme_id)` constraint rejects accidental duplicates. The table is separate from benchmark/human-evaluation schemas; Phase 9B introduces no training, reranking, or rating mutation.
