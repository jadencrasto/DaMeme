# Phase 5: emotion-aware retrieval and ranking

Phase 5 builds a portable bundle from the prepared 5,469 candidate vectors; it never downloads data, recomputes MiniLM or BLIP embeddings, or retrains GATED. `phase5_bundle.npz` contains aligned `meme_ids`, normalized MiniLM vectors, six-class GATED probabilities, and portable image basenames. `phase5_metadata.json` records the feature version, class order, thresholds, dimensions, count, and seed.

For a text query, MiniLM produces a normalized vector. Exact cosine retrieval uses FAISS `IndexFlatIP` when available; the identical inner-product calculation falls back to NumPy when FAISS is unavailable. Every retrieval position is resolved through the bundle's ordered `meme_ids`, never through CSV position.

The rank score is:

```text
score = (w_semantic * ((cosine + 1) / 2) + w_emotion * cosine(user_emotions, meme_emotions))
        / (w_semantic + w_emotion)
```

Defaults are `w_semantic=0.8` and `w_emotion=0.2`. These are transparent heuristic weights, not learned ranking parameters. Scores are in `[0, 1]`; ties are deterministic by semantic score then meme ID.

## Colab build

```bash
cd /content/DaMeme/backend
pip install -e . faiss-cpu torch
BASE=/content/gdrive/MyDrive/DaMeme/expansion_5000
OUT=$BASE/phase5_bundle
python -m app.phase5_retrieval \
  --base "$BASE" \
  --checkpoint "$BASE/training_outputs/dameme_5500_gated_seed42.pt" \
  --output "$OUT" \
  --batch-size 128
```

Expected outputs are `$OUT/phase5_bundle.npz` and `$OUT/phase5_metadata.json`. Copy this directory to a serving host and set `DAMEME_PHASE5_BUNDLE_DIR`; set `DAMEME_EXPANDED_EMOTION_TEXT_MODEL` to the selected text-model artifact. `POST /api/v1/recommendations/text` accepts `{ "text": "…", "top_k": 10 }`. It returns IDs, safe basename-only image references, score components, and the text classifier's emotion prediction. Image resolution remains a host-side responsibility; the bundle never exposes absolute Drive paths.

## Evaluation protocol and limitation

Offline evaluation should use validation queries while selecting any weights and reserve test queries for one final report. Candidate pools must exclude the query itself and must be held fixed for semantic-only versus semantic-plus-emotion comparisons. A source-faithful automatic relevance proxy can define a candidate as relevant only when both MOOD records have at least one shared positive six-class label; report Precision@K, Recall@K, HitRate@K, and nDCG@K by query emotion and ranking mode. This measures label agreement, not human satisfaction, humor, or contextual appropriateness. MemeMatch is excluded from supervised MOOD evaluation because its annotations are not equivalent targets.
