# Emotion models

MOOD's `final.csv` stores `anger`, `fear`, `joy`, `sadness`, `surprise`, and `disgust` as independent `label`/`not_label` values. A meme may therefore have more than one positive emotion, so emotion learning uses multi-output binary classifiers and multi-label metrics. `sentiment` is one of positive, negative, or neutral and is treated as multiclass.

Phase 4 baseline features are cached in SQLite as `phase4-hash-text-rgb-v1`: a 128-dimensional normalized `HashingVectorizer` over cached OCR plus BLIP caption text, and a 192-dimensional normalized RGB 8x8 image vector. Ground-truth labels are targets only. These lightweight representations are reproducible CPU fallbacks, not pretrained MiniLM/BLIP-vision embeddings.
