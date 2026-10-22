# Dataset licensing and provenance

Code availability does not grant image redistribution rights. Store each source URL, access notes, license status/notes, version, supplied manifest reference, and retrieval timestamp in `dataset_sources`; keep original source URLs and label provenance per record.

Use `verified` only after reviewing the dataset's governing terms. Use `pending`, `restricted`, or `unknown` otherwise. Phase 2.5 verified that MemeMatch is distributed through a linked Kaggle DOI and that ALFRED links a Google Drive archive, but did **not** verify redistribution rights, archive sizes, original MemeMatch packaging, MOOD full-data license, or MOOD split definitions. Consequently the repository reports **0 research images downloaded / 0 research images ingested**.

Generated manifests are ignored by Git because they can contain restricted identifiers, URLs, or source annotations. Commit only redacted documentation and tiny synthetic test fixtures.

The user-provided MOOD archive is an official-source artifact but does not, by itself, establish redistribution or research-use permission. Keep it outside the repository; do not commit, redistribute, or extract its image payload until terms are confirmed.
