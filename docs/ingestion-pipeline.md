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

## Phase 3.2 reproducible local MOOD sample

The adapter never extracts an archive wholesale. It reads `final.csv`, selects at most 50 source records across the six supplied emotion columns, and writes only those image members plus a Git-ignored metadata mapping. The source archive and resulting images remain private.

```powershell
Set-Location C:\Users\jason\OneDrive\Desktop\DaMeme\backend
uv run python -m app.datasets.mood --archive C:\Users\jason\Downloads\TAC24_meme_emotion.zip --output ..\data\raw\mood-phase32 --metadata ..\data\raw\mood-phase32-metadata.json --limit 50
uv run python -m app.datasets.cli --database ..\data\processed\phase32-research.db import ..\data\raw\mood-phase32 --dataset mood --version tac24-final-sample-50 --source-url https://github.com/shiv6891/ALFRED_MemeEmotionDetection --license-status unknown --license-notes "Explicit dataset license not verified; private local research only." --access-notes "Do not commit or redistribute source images." --manifest-reference meme_emotion/final.csv --source-reported-count 10005 --metadata ..\data\raw\mood-phase32-metadata.json --max-images 50
uv run python -m app.datasets.cli --database ..\data\processed\phase32-research.db stats --json
```

Run the same import again to verify that it accepts zero additional records. The run checkpoint may mean `skipped` is zero; the invariant is that the database record count does not increase.

For MemeMatch, make only a known-file request, never a root request:

```powershell
uv run python -c "from app.datasets.kaggle import download_dataset_file; print(download_dataset_file(relative_path='annotations/annotations/local_context.csv'))"
```

Do not request image files until an authoritative relative image path is available.

After accepting MemeMatch terms, a Python shell can request only an annotation file (not the full corpus):

```powershell
uv run python -c "from app.datasets.kaggle import download_dataset_file; print(download_dataset_file(relative_path='annotations/annotations/local_context.csv'))"
```

This command is intentionally not run automatically while the Kaggle license remains listed as unknown.
