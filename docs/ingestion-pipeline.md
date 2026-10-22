# Ingestion pipeline

The CLI ingests **existing local files only**. It does not download archives, scrape URLs, unpack ZIPs, or run models.

1. A source/version record and resumable run are created.
2. Files are traversed deterministically and bounded by `--max-images` (default 100).
3. Pillow verifies decode, supported format, dimensions, pixels, and size. Invalid files are retained on disk and recorded as failures.
4. SHA-256 detects byte-identical files; pHash candidate bands find possible near duplicates.
5. Each completed batch persists a checkpoint. Re-running the same source/root skips already inserted stable relative source IDs.
6. Statistics and a JSON manifest distinguish source metadata from validated local records.

Example PowerShell:

```powershell
Set-Location C:\Users\jason\OneDrive\Desktop\DaMeme\backend
uv run python -m app.datasets.cli --database ..\data\metadata.db init-db
uv run python -m app.datasets.cli --database ..\data\metadata.db import ..\data\raw\mood-sample --dataset mood --version sample-v1 --source-url https://example.org/official-source --license-status pending --manifest-reference docs\source-manifest.json --max-images 100
uv run python -m app.datasets.cli --database ..\data\metadata.db stats --json
uv run python -m app.datasets.cli --database ..\data\metadata.db failures --json
uv run python -m app.datasets.cli --database ..\data\metadata.db export-manifest --dataset mood --output ..\data\manifests\mood-sample.json
```

To preserve supplied MOOD fields, pass a JSON object keyed by relative path, e.g. `{"image.png": {"split": "test", "emotion_label": "<source value>", "label_provenance": "MOOD"}}` with `--metadata`. Do not increase the smoke-test cap beyond 100, or proceed to 1K/5K, without confirming source access/terms and user authorization.

For source-specific links, manual access steps, planning storage ranges, and the recorded zero-image acquisition result, see the [Phase 2.5 acquisition report](dataset-acquisition-report.md).

After accepting MemeMatch terms, a Python shell can request only an annotation file (not the full corpus):

```powershell
uv run python -c "from app.datasets.kaggle import download_dataset_file; print(download_dataset_file(relative_path='annotations/annotations/local_context.csv'))"
```

This command is intentionally not run automatically while the Kaggle license remains listed as unknown.
