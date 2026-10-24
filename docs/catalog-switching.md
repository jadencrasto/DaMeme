# Local catalog switching

DaMeme defaults to the existing `memematch_281` catalog. The 1,000-meme catalog
is opt-in and is never downloaded, generated, or activated by this code.

## Required local layout

Copy approved artifacts without changing their contents into:

```text
data/retrieval/catalogs/memematch_1000_v2/
  images/
  memematch_1000_metadata_v2.csv
  memematch_1000_v2.manifest.json
  exports/
    memematch_1000_v2.records.json
    memematch_1000_v2.metadata.json
  indexes/
    manifest.json
    text.faiss
    image.faiss
    fused.faiss
```

The export records/metadata and acquisition manifest are retained for provenance;
the running API uses the index manifest, CSV, and image directory. The loader
checks index-record count, CSV count, source-key alignment, and fused dimensions
on first use. It does not mutate any artifact.

## Switch and rollback

From `backend` in PowerShell:

```powershell
$env:DAMEME_RECOMMENDATION_CATALOG = "memematch_1000_v2"
# Optional when catalogs live elsewhere:
# $env:DAMEME_RECOMMENDATION_CATALOG_ROOT = "C:\DaMeme\data\retrieval\catalogs"
uv run uvicorn app.main:app --reload --port 8000
```

Return to the original catalog:

```powershell
$env:DAMEME_RECOMMENDATION_CATALOG = "memematch_281"
uv run uvicorn app.main:app --reload --port 8000
```

## Fixed-query human review

The evaluator runs five fixed text/emotion prompts and prints JSON containing top
results, similarity/ranking scores, local/global context, template group, and
safety-filtered candidate IDs:

```powershell
Set-Location backend
uv run python -m app.catalog_evaluation --catalog memematch_281 --device cpu
uv run python -m app.catalog_evaluation --catalog memematch_1000_v2 --device cpu
```

It is a review aid, not an accuracy or relevance measurement. It requires the
MiniLM query encoder to be locally available and does not alter indexes or images.
