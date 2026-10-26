# Phase 8B.1 — safe incremental catalog infrastructure

## Scope and status

This phase implements infrastructure only.  It uses synthetic images created by tests; it does not download, stage, ingest, index, or modify any real catalog image.  The existing `memematch_1000_v2` catalog and all current indexes remain untouched and are not registered, selected, or rewritten by this code.

The implementation is in `backend/app/datasets/catalog_generations.py`, with SQLite migration 4 in `database.py`, a read-only `catalog-dry-run` CLI command, and synthetic tests in `backend/tests/test_catalog_generations.py`.

## Permission and provenance gate

Promotion has a hard gate at `CatalogGenerationManager.begin`.  Each selected imported record is joined to its source and its source permission review.  The operation fails unless every source has `review_status == "approved"`.

A review stores all required fields:

- acquisition method;
- terms reference and review record;
- allowed intended use and restrictions;
- attribution;
- review status and date; and
- evidence reference.

An incomplete review, `unknown`, or `rejected` status cannot pass the read-only planner or begin a generation.  Public availability is not considered authorization.  The existing `dataset_sources` source URL remains the verifiable origin field; image-level original IDs/URLs remain on `memes`.

## Generation lifecycle

1. Existing local-only ingestion validates images, computes SHA-256 and pHash, writes deterministic source IDs, checkpoints batches, and retains failures/duplicate relationships.  It neither modifies source images nor deletes rejected records.
2. `begin(generation_id, member_ids, parent_generation_id=...)` accepts only already-imported rows from approved sources.  It creates a `draft` generation with deterministic source keys of the form `<dataset>:<version>:<original-source-id>`, unique membership, and an optional finalized parent's manifest hash.
3. Feature adapters record versioned OCR/caption, MiniLM, OpenCLIP, or FAISS entries keyed by generation source key, image SHA-256, feature version, and model-provenance SHA-256.  `assert_feature_isolation` rejects mixed completed versions for a feature kind.  These are registry/cache adapters only: they do not load models or generate heavyweight features.
4. `finalize` validates membership, writes `manifest.json` and `source_keys.json` to a generation-private staging directory, hashes the manifest, then atomically renames the directory.  A pre-existing final or staging directory blocks overwrites.  A failed rename removes staging and leaves the generation draft.
5. `rollback(generation_id)` verifies the finalized manifest hash and atomically writes a selector in the caller-provided generation root.  It does not alter the production catalog configuration.  Selecting a generation for production remains a separate explicit operation.

Every manifest contains an immutable generation identity hash, parent generation ID, parent manifest hash, ordered members, source keys, source provenance identity, and image SHA-256s.  The code does not claim that a generation is valid for recommendation until its separate feature/index artifacts have been completed and validated.

## Resumable and dry-run behavior

The existing `DatasetIngester` remains the single ingestion implementation.  It orders files deterministically, validates decodes and size/pixel limits, calculates SHA-256/pHash, records exact and near relationships, checkpoints by batch, and reuses the same run/source/root after an interruption.  Its failure ledger is append-only in intent (`INSERT OR REPLACE` only updates the same run/item diagnostic).

`CatalogPlanner.dry_run` is read-only and reuses the same image validation and hashing helpers.  It reports accepted files, corrupt or missing expected files, exact duplicate pairs, and pHash-near pairs.  It requires the same approved permission review before inspecting candidates.

Example permission-review JSON (use real, reviewed evidence—not this placeholder):

```json
{
  "acquisition_method": "approved source package member",
  "terms_reference": "https://example.invalid/terms",
  "review_record": "legal-or-research-review-id",
  "allowed_intended_use": "approved private research use",
  "restrictions": "no redistribution; preserve attribution",
  "attribution": "required source citation",
  "review_status": "approved",
  "reviewed_at": "2026-10-26",
  "evidence_reference": "local reviewed terms record"
}
```

Exact dry-run command (the directory and JSON must be an approved staging fixture/source; the command writes no database or catalog):

```powershell
Set-Location C:\Users\jason\OneDrive\Desktop\DaMeme\backend
rtk uv run python -m app.datasets.cli --database ..\data\phase8b-dry-run.db catalog-dry-run <approved-staging-directory> --permission-review <reviewed-permission.json>
```

Exit status is nonzero when candidate validation finds rejected files.  An unapproved review fails before candidate processing.

## Synthetic validation coverage

The fixture test constructs valid images, an exact duplicate, a pHash-near image, a corrupt file, an expected-but-missing file, and an explicitly rejected provenance record.  It also simulates an interrupted batch by retaining checkpoint index one and resuming the same source/root.

Tests assert:

- permission rejection and approved-review requirements;
- stable, sorted source-key planning and exact/near duplicate reporting;
- corrupt and missing-file diagnostics;
- idempotent checkpoint resume;
- parent hashes and source-key uniqueness in finalized manifests;
- absence of final/staging artifacts after simulated atomic-rename failure;
- rollback selector integrity; and
- MiniLM embedding-version isolation without loading any model.

## Before the first approved 500-item batch

1. Record the exact source package/version, source-file manifest, origin URL(s), governing terms, permitted purpose, attribution, restrictions, reviewer/date, and durable evidence reference.
2. Obtain an affirmative, source-specific approval review.  Unknown terms remain a hard stop.
3. Create a separate approved staging directory; do not point at `memematch_1000_v2` or use source URLs as crawler instructions.
4. Execute the dry run, investigate every corrupt/missing/near-duplicate/policy exception, and preserve decisions in the batch manifest.
5. Ingest only after explicit authorization, into a separate local database/source version.  Build an unselected generation; measure OCR/BLIP/OpenCLIP resources; validate version compatibility and index/source-key alignment.
6. Run the pre-registered development-only coverage evaluation.  A production catalog switch requires a later explicit decision and is outside Phase 8B.1.
