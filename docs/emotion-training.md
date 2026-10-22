# Emotion training

```powershell
Set-Location C:\Users\jason\OneDrive\Desktop\DaMeme\backend
uv run python -m app.emotion_cli --database ..\data\processed\phase32-research.db prepare --limit 200 --seed 42
uv run python -m app.emotion_cli --database ..\data\processed\phase32-research.db extract-features --limit 200 --feature-version phase4-minilm-blipvision-v1 --device cpu
uv run python -m app.emotion_cli --database ..\data\processed\phase32-research.db --artifacts ..\artifacts\emotion train --model frequency-baseline --feature-version phase4-minilm-blipvision-v1 --seed 42
uv run python -m app.emotion_cli --database ..\data\processed\phase32-research.db --artifacts ..\artifacts\emotion train --model text-baseline --feature-version phase4-minilm-blipvision-v1 --seed 42
uv run python -m app.emotion_cli --database ..\data\processed\phase32-research.db --artifacts ..\artifacts\emotion train --model image-baseline --feature-version phase4-minilm-blipvision-v1 --seed 42
uv run python -m app.emotion_cli --database ..\data\processed\phase32-research.db --artifacts ..\artifacts\emotion train --model simple-fusion --feature-version phase4-minilm-blipvision-v1 --seed 42
uv run python -m app.emotion_cli --database ..\data\processed\phase32-research.db --artifacts ..\artifacts\emotion train --model emotion-fusion --feature-version phase4-minilm-blipvision-v1 --epochs 12 --seed 42
uv run python -m app.emotion_cli --database ..\data\processed\phase32-research.db --artifacts ..\artifacts\emotion evaluate --model emotion-fusion --feature-version phase4-minilm-blipvision-v1 --output-dir ..\artifacts\emotion\reports
```

The command is cache-aware: a repeat only processes missing feature rows. Use one process at a time for a SQLite database. The current development cap is 200 images; do not increase it without revisiting licensing, storage and runtime limits.

```powershell
uv run python -m app.emotion_cli --database ..\data\processed\phase32-research.db extract-features --limit 25 --feature-version phase4-minilm-blipvision-v1 --device cpu
```

For a future resumable expansion, extract a bounded MOOD subset with `app.datasets.mood --limit N`, ingest it with the existing dataset CLI, run `prepare`, then feature extraction in chunks. Do not extract all 10,005 source images until licensing and storage decisions are reconfirmed. CPU throughput must be measured again on the chosen batch size before estimating a full run.

Splits are deterministic from a seed and SHA-256 (or pHash fallback) group identifier, so one group cannot cross train/validation/test. Local artifacts, embeddings, and databases are Git-ignored. Do not treat 50-image output as a scientific evaluation.

## Windows test environment

The project virtual environment can be affected by local file locks. The verified Windows setup uses an external Python 3.11.17 environment at `%LOCALAPPDATA%\DaMeme-venv`:

```powershell
$env:UV_PROJECT_ENVIRONMENT = "$env:LOCALAPPDATA\DaMeme-venv"
Set-Location C:\Users\jason\OneDrive\Desktop\DaMeme\backend
uv sync --group dev --python 3.11
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

The complete backend suite passed in this environment on 2026-10-23: `16 passed in 18.77s` before the Phase 4C reliability tests were added.

## Gated fusion continuation and weighting

`emotion-fusion` checkpoints store model/optimizer state, completed epoch, best validation state/loss, patience, feature dimensions and training configuration. Continue a compatible run without changing the dataset cache:

```powershell
uv run python -m app.emotion_cli --database ..\data\processed\phase32-research.db --artifacts ..\artifacts\emotion train --model emotion-fusion --feature-version phase4-minilm-blipvision-v1 --resume-from ..\artifacts\emotion\emotion-fusion-seed42-n156-v1.pt --epochs 4
```

The default six emotion positive-class weights use inverse prevalence and safely use `1.0` for an absent class. Override them only deliberately, as six comma-separated positive values. The loss terms can also be weighted independently:

```powershell
uv run python -m app.emotion_cli --database ..\data\processed\phase32-research.db --artifacts ..\artifacts\emotion train --model emotion-fusion --feature-version phase4-minilm-blipvision-v1 --emotion-pos-weights 1,2,1,1,1,1 --emotion-loss-weight 1.0 --sentiment-loss-weight 0.5 --epochs 4
```
