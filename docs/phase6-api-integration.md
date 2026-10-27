# Phase 6 API integration

Phase 6 serves the portable Phase 5 bundle through `POST /api/v1/recommendations/text` without changing the frontend or legacy GET endpoint. The service lazily caches the bundle, text classifier, and CPU MiniLM encoder once per process. It ranks with the validated fixed formula: `0.8 * ((cosine + 1) / 2) + 0.2 * emotion-cosine-compatibility`.

Copy the directory containing `phase5_bundle.npz` and `phase5_metadata.json`, the selected `dameme_5500_text_seed42.pkl`, and the actual meme images from Drive/Colab to a local folder. Do not copy Drive paths into configuration.

```powershell
$env:DAMEME_PHASE5_BUNDLE_DIR = 'C:\DaMeme-artifacts\phase5_bundle'
$env:DAMEME_EXPANDED_EMOTION_TEXT_MODEL = 'C:\DaMeme-artifacts\dameme_5500_text_seed42.pkl'
$env:DAMEME_PHASE5_MEDIA_ROOT = 'C:\DaMeme-artifacts\images'
Set-Location C:\Users\jason\OneDrive\Desktop\DaMeme\backend
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In another PowerShell window:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/recommendations/phase5/readiness
uv run python scripts/phase6_smoke_test.py --base-url http://127.0.0.1:8000 --top-k 5
```

The readiness endpoint distinguishes a running app from loaded portable artifacts and reports only counts/status—not filesystem paths. Image URLs are emitted only when a filename reference resolves once below the configured media root. The resolver rejects traversal and ambiguous filename collisions. Missing media produces `image_available: false` and no URL; it is never reported as success.

`POST /api/v1/recommendations/text` accepts `{ "text": "I passed my exams", "top_k": 5 }` and returns text-model probabilities plus ranked IDs, semantic similarity, emotion compatibility, combined score, and safe local image URLs. Empty text and invalid `top_k` are rejected by request validation. Missing/corrupt local artifacts return 503.

The first request loads MiniLM from the local Hugging Face cache and can be slower. Model artifacts are trusted local deployment inputs; do not point environment variables at untrusted pickle files. Images remain optional, and no human-quality claim follows from automatic retrieval scores.
