# FAISS retrieval foundation

The Phase 5 retrieval module reads the three embedding-export artifacts and builds
exact cosine-similarity indexes with `faiss.IndexFlatIP`. It keeps text (MiniLM,
384 dimensions) and image (BLIP vision, 768 dimensions) indexes separate. A fused
index has 1,152 dimensions: each modality is independently normalized, weighted,
concatenated, then normalized again.

Indexes retain the exporter's `embedding_index` to `source_key` mapping. The
manifest includes the encoder identifiers, dimensions, embedding provenance, and
fusion weights. A text query always targets the text index; fused search requires
both compatible text and image query vectors.

## Install

From `backend/`, install the development environment (including CPU FAISS):

```powershell
uv sync --group dev
```

`faiss-cpu` is sufficient for 281 vectors and avoids consuming the laptop GPU.

## Build indexes

```powershell
Set-Location backend
uv run python -m app.retrieval build `
  --export-prefix ..\data\retrieval\memematch_281 `
  --index-dir ..\data\retrieval\memematch_281_indexes `
  --text-weight 1.0 `
  --image-weight 1.0
```

This builds 281-vector text (384), image (768), and fused (1,152) indexes. The
index directory is ignored by Git. Rebuild it whenever any export artifact,
encoder identifier, or fusion weight changes.

## Text search

```powershell
Set-Location backend
uv run python -m app.retrieval search-text `
  --index-dir ..\data\retrieval\memematch_281_indexes `
  --text "I'm exhausted after studying all night." `
  --limit 5 `
  --device cpu
```

The text encoder loads lazily at search time and is checked against the model ID
stored in the index manifest. The first query may download its already-configured
MiniLM model into the local Hugging Face cache; no model weights belong in Git.
