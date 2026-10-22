# Dataset metadata schema

`backend/app/datasets/database.py` applies an idempotent SQLite migration. `dataset_sources` retains source/version/access/permission provenance. `memes` keeps a stable source identifier, original URL, local file facts, hashes, split, supplied label and label provenance, annotations JSON, status, and errors. `ingestion_runs` stores a checkpointed directory run; `ingestion_failures` preserves rejected files; `duplicate_relationships` records exact/near relationships without deletion.

`perceptual_hash_bands` indexes four pHash bands. A new image is compared only against records sharing a band, then exact Hamming distance is calculated; this avoids all-pairs scanning at 100K scale. pHash is a heuristic: template variants and text overlays can evade or trigger matches. Near duplicates are review relationships—not automatic merges. Exact byte duplicates receive `duplicate_exact`; both records remain.

`processing_status` is intentionally coarse (`validated`, `duplicate_exact`) in Phase 2. Later OCR, caption, emotion, and embedding stages must use additive versioned status/feature tables rather than overwrite source annotations.
