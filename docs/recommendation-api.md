# Recommendation API

The recommendation routes use existing local FAISS indexes and MemeMatch metadata.
They load lazily: health, multimodal, and Phase 4 emotion routes remain available
when recommendation assets are missing. Text recommendation lazily loads MiniLM;
emotion-only recommendation does not load MiniLM or BLIP.

## Configuration

Set these variables in `backend/.env` or the PowerShell session:

```powershell
$env:DAMEME_RECOMMENDATION_INDEX_DIR = "..\data\retrieval\memematch_281_indexes"
$env:DAMEME_RECOMMENDATION_METADATA_CSV = "..\data\retrieval\memematch_281\memematch_281_metadata.csv"
# Optional: a local directory containing copied image files. No images are downloaded.
$env:DAMEME_RECOMMENDATION_IMAGE_ROOT = "C:\DaMeme\MemeMatch"
```

The image root is optional. When configured, the metadata loader accepts only
existing portable filename paths beneath that root; old Google Drive/Colab paths
are never exposed as URLs.

## Start and call

```powershell
Set-Location backend
uv sync --group dev
uv run uvicorn app.main:app --reload --port 8000
```

```powershell
Invoke-RestMethod "http://localhost:8000/api/v1/recommendations/status"
Invoke-RestMethod "http://localhost:8000/api/v1/recommendations/options"
Invoke-RestMethod "http://localhost:8000/api/v1/recommendations/text?text=I%20am%20exhausted%20after%20studying&emotion=sadness&limit=5&candidate_limit=100&semantic_weight=0.8&emotion_weight=0.15&sentiment_weight=0.05&min_semantic_similarity=0.2&max_per_template=1"
Invoke-RestMethod "http://localhost:8000/api/v1/recommendations/emotion?emotion=joy&limit=5"
```

Routes:

- `GET /api/v1/recommendations/status`
- `GET /api/v1/recommendations/options`
- `GET /api/v1/recommendations/text`
- `GET /api/v1/recommendations/emotion`
- `GET /api/v1/memes/{meme_id}/image`

The image route returns 404 until a verified local image is available. Returned
recommendations include an image URL only under that condition.

Text requests default to semantic-first weights (`0.8`, `0.15`, `0.05` for
semantic/emotion/sentiment). Override those values deliberately when comparing
rankers. `min_semantic_similarity` is optional and uses raw cosine similarity in
`[-1, 1]`; when it filters every candidate the route returns `[]`, never a
semantic fallback. `max_per_template` is optional on both text and emotion routes
and applies the documented deterministic filename heuristic; it is not a claim
that filenames identify visual templates perfectly.
