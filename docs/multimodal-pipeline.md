# Multimodal pipeline

Phase 3 stores OCR and visual captions independently in `multimodal_analyses`; source records and annotations remain unchanged. A configuration digest plus image SHA-256 makes completed analyses cacheable and safely invalidates records when an image/provider configuration changes. Combined context is deterministic: `Text: … Visual: …`.

EasyOCR returns ordered text, boxes, and confidence. BLIP base produces a short visible-content description only; it does not explain meme intent, sentiment, or humor. Providers load lazily on first inference and run sequentially, defaulting to CPU.

API: `GET /api/v1/models/multimodal`, `GET /api/v1/memes/{meme_id}/analysis`, and `POST /api/v1/memes/{meme_id}/analyze`. Models are never loaded at startup.

## Phase 3.2 local research smoke run

With the private MOOD sample imported into `data/processed/phase32-research.db`, process five records sequentially on CPU:

```powershell
Set-Location C:\Users\jason\OneDrive\Desktop\DaMeme\backend
uv run python -m app.multimodal_cli analyze-batch --database ..\data\processed\phase32-research.db --limit 5 --device cpu
uv run python -m app.multimodal_cli analyze-batch --database ..\data\processed\phase32-research.db --limit 5 --device cpu
```

The first run completed five real analyses. The second reported five `skipped` records, demonstrating reuse of the SQLite analysis cache. Model output is descriptive only; source emotion/sentiment annotations remain in the immutable ingestion metadata and are not predicted by this phase.
