# 5,500-catalog emotion training

The Drive expansion is not present in this repository. The training command validates it before use and does not download images or recompute embeddings. The six binary classifier targets come only from rows explicitly marked `MOOD` or `MOOD-TAC24`; MemeMatch emotion scores remain source metadata for retrieval, not ground-truth labels.

In Colab, after cloning this repository and mounting Drive:

```bash
cd /content/DaMeme/backend
pip install -e . scikit-learn torch
BASE=/content/gdrive/MyDrive/DaMeme/expansion_5000
OUT=$BASE/training_outputs
python -m app.expanded_emotion --base "$BASE" --output "$OUT" --model text --seed 42
python -m app.expanded_emotion --base "$BASE" --output "$OUT" --model vision --seed 42
python -m app.expanded_emotion --base "$BASE" --output "$OUT" --model concat --seed 42
python -m app.expanded_emotion --base "$BASE" --output "$OUT" --model gated --seed 42 --epochs 25
```

Baseline commands write versioned pickle artifacts and held-out MOOD metrics. The gated command writes `dameme_5500_gated_seed42.pt`, including model state, optimizer, epoch, loss curves, validation-tuned thresholds, test metrics and feature/label metadata. Resume it with `--resume "$OUT/dameme_5500_gated_seed42.pt"`.

Choose the best model only from validation metrics, then perform a single held-out test evaluation. This initial command emits final test metrics immediately, so for strict model selection run candidates on a development copy with test withheld, then run only the selected configuration against the untouched test set.

For backend retrieval, place the separately built export at `data/retrieval/catalogs/dameme_5500_v1/` with `indexes/`, `exports/dameme_5500_v1.records.json`, `exports/dameme_5500_v1.metadata.json`, and `dameme_5500_metadata.csv`, then set `DAMEME_RECOMMENDATION_CATALOG=dameme_5500_v1`. Preserve source provenance; do not fabricate annotations across the two source systems.

To enable sentence prediction, export the selected **text** baseline artifact and set `DAMEME_EXPANDED_EMOTION_TEXT_MODEL` to its absolute local path. `POST /api/v1/emotion/text` embeds the sentence with the same MiniLM model and returns the six-label probabilities plus validation-tuned thresholds. It is intentionally separate from the image-aware gated meme classifier.
