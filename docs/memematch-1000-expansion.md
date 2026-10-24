# MemeMatch 1,000-meme expansion preparation

This document prepares a bounded, source-compliant acquisition and embedding run.
It does not authorize downloading images. Confirm the MemeMatch access terms,
license, allowed redistribution/use, and the selected source revision before step 4.

## Versioned retrieval text (v2)

Each embedding is deterministic text assembled in this order when the source field
is non-empty:

1. cleaned OCR (`cleaned_ocr`, `ocr_text`, `ocr`, or `local_ocr`)
2. local meme context
3. caption or title
4. verified template information only
5. global visual context, labelled as potentially generic

Cleaning performs Unicode NFKC normalization, removes control characters, and
collapses whitespace. It does not spell-correct OCR, infer a caption, summarize,
or add facts. The source CSV remains the original field record. Each exported
record additionally stores `text_representation_version`, `text_fields`, and
`text_source_columns`, so every contribution is inspectable.

`source_key` is a stable source identifier selected from `source_key`, `filename`,
`id`, or `meme_id`; row numbers are never identifiers. The exporter rejects an
ambiguous row rather than inventing one.

## Colab workflow (bounded, manual approval required)

1. In the repository checkout, prepare a metadata-only selection manifest. This
   step does not download images:

   ```powershell
   Set-Location backend
   uv run python -m app.memematch_embeddings `
     --metadata-csv C:\approved-source\memematch_metadata.csv `
     --prepare-manifest C:\approved-source\dameme-memematch-1000.manifest.json `
     --target-count 1000 `
     --source-url "<approved-source-url>" `
     --license-status "<reviewed-license-status>" `
     --access-notes "<terms reviewed on YYYY-MM-DD>"
   ```

2. Review the manifest: it contains a source CSV SHA-256, source/license details,
   deterministic `source_key` entries, and the selected count. A count below 1,000
   means the supplied metadata has fewer distinct stable identifiers; do not pad it
   with invented entries.

3. Obtain explicit approval for the bounded acquisition. In Colab, acquire only
   manifest-listed images into `dataset_root/images/`, preserving source filenames
   and a copy of the approved metadata CSV. Do not modify the 281-meme export.

4. Validate source terms again, then export to a new prefix (example shown on
   mounted Drive). The manifest SHA check prevents accidentally embedding a changed
   CSV; resume is safe only for the same CSV, manifest, encoders, and text version.

   ```bash
   cd /content/DaMeme/backend
   uv run python -m app.memematch_embeddings \
     --metadata-csv /content/drive/MyDrive/dameme-1000/memematch_metadata.csv \
     --dataset-root /content/drive/MyDrive/dameme-1000 \
     --manifest /content/drive/MyDrive/dameme-1000/dameme-memematch-1000.manifest.json \
     --output-prefix /content/drive/MyDrive/dameme-1000/exports/memematch_1000_v2 \
     --batch-size 16 --device cpu
   ```

5. Build a separate FAISS index only after export validation succeeds:

   ```bash
   uv run python -m app.retrieval build \
     --export-prefix /content/drive/MyDrive/dameme-1000/exports/memematch_1000_v2 \
     --index-dir /content/drive/MyDrive/dameme-1000/indexes/memematch_1000_v2
   ```

6. Run the existing manual review export against the new index before claiming any
   ranking improvement. Keep the 281 export and index as the baseline.

## Compatibility and deduplication

The v2 export retains the existing `.embeddings.npz`, `.records.json`, and
`.metadata.json` artifact shapes. `text_embeddings`, `image_embeddings`, completed
record status, contiguous `embedding_index`, and unique `source_key` are unchanged,
so the existing FAISS index builder and recommendation engine accept the export.

The manifest removes duplicate stable source identifiers before acquisition. During
ingestion, existing SHA-256 and perceptual-hash checks remain the image-level exact
and near-duplicate safeguards. Filename/template similarity alone is not treated as
proof that two memes are duplicates.
