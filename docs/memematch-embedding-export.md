# MemeMatch embedding export

Phase 5's first iteration exports retrieval features only. It uses the same frozen
MiniLM and BLIP vision encoders as Phase 4: `sentence-transformers/all-MiniLM-L6-v2`
with attention-mask mean pooling, and `Salesforce/blip-image-captioning-base` with
mean-pooled vision hidden states. It does not train a model or download a dataset.

The CSV's absolute `image_path` is never trusted. Set `--dataset-root` to the
directory containing `images/`; the exporter prefers `images/<filename>` and only
accepts fallback relative paths that remain inside that root.

## Google Colab

Mount Drive and clone/upload this source repository as appropriate. From the
repository root, install the pinned backend environment and run the exporter:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%cd /content/DaMeme/backend
!pip install -e . "scikit-learn>=1.6,<2" "torch>=2.6,<3" "transformers>=4.48,<5"
!python -m app.memematch_embeddings \
  --metadata-csv /content/drive/MyDrive/DaMeme/datasets/MemeMatch/memematch_281_metadata.csv \
  --dataset-root /content/drive/MyDrive/DaMeme/datasets/MemeMatch \
  --output-prefix /content/drive/MyDrive/DaMeme/exports/memematch_281 \
  --batch-size 16 \
  --device cuda
```

Use `--device cpu` when no GPU is attached. Re-run the same command after an
interruption: completed and failed source rows are preserved, while previously
completed embeddings are not recomputed. Use `--no-resume` only to deliberately
start an export over.

The output prefix produces three artifacts: `.embeddings.npz` (separate text and
image arrays), `.records.json` (stable source keys and per-row processing status),
and `.metadata.json` (model identifiers, dimensions, source digest, and config).
Keep these artifacts in Drive or another artifact store, not Git.
