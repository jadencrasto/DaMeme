# Dataset strategy

Start only after source, terms, and schema are verified. Target a representative 5K–10K controlled sample for initial experiments, then a 50K release corpus and a 100K+ scalable corpus. MemeMatch is a candidate retrieval dataset; MOOD is a candidate emotion benchmark. Neither is automatically ingested by this repository.

Every batch must record source URL, license/terms, collection date, dataset version, checksum, split, and preprocessing configuration. De-duplicate by cryptographic file hash plus perceptual hash review; keep source-level provenance even when binary duplicates collapse. Preserve raw files separately from derived OCR, captions, embeddings, and index snapshots. Failed jobs should resume from a checkpoint and re-run safely with idempotent item keys.

Development uses local asset storage and SQLite metadata; later, object storage and PostgreSQL can scale separately. Storage estimates depend on image dimensions, formats, duplicate rate, and retained derivatives, so Phase 2 must measure them on a licensed sample rather than publish speculative figures. Personal uploads are a later, consent-aware ingestion path and never merge silently with benchmark data.

Phase 2 adds a local-only, checkpointed ingestion boundary. It validates image bytes before indexing metadata, stores hashes and duplicate relationships, and preserves failures for review. The default import cap is 100 images. Archives, raw images, SQLite databases, and generated manifests are ignored by Git.

Phase 2.5 found the official MemeMatch Kaggle distribution and MOOD/ALFRED Drive archive links, but no verifiable dataset-use terms suitable for an unattended smoke download. The source-specific acquisition report records the blocker, available disk space, and planning ranges. Do not begin an official import until a user has reviewed and accepted the applicable terms.
