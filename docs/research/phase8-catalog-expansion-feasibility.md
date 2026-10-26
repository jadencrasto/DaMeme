# Phase 8A — catalog expansion feasibility audit

**Status:** complete, read-only audit (26 October 2026).  No images were downloaded, extracted, ingested, or re-encoded; no catalog, index, ratings, held-out queries, or production code was touched.

## Decision

Expanding to 5,000 is technically feasible on the current machine, but it is **not authorized from the locally available material**.  The number of immediately usable additional images is **0**.  There are 200 locally stored MOOD images with no exact-byte overlap with the active catalog, but their documented status is private local research with an unknown license; they are conditionally reviewable, not usable for expansion.  The 281-image legacy MemeMatch directory contributes **0** images: its 280 unique SHA-256 values all occur in the active 1,000-image catalog (one legacy file duplicates another legacy file).

The recommended conditional expansion source is **MemeMatch v1.0**, after an authenticated review records its governing terms, exact package version/layout, permitted research use, attribution, and image/third-party rights.  Its repository links the official Kaggle DOI but does not state a dataset license in the repository README. [MemeMatch repository](https://github.com/TriAnLe171/MemeMatch-v1.0)  Until that review, the bounded Phase 8B plan must stop before acquisition.

Machine-readable results are in [the source inventory](phase8-dataset-source-inventory.json) and [resource estimates](phase8-resource-estimates.json).

## Existing-data audit

| Local material | Files / size | Expansion value | Provenance and permission conclusion |
| --- | ---: | --- | --- |
| `memematch_1000_v2/images` | 1,000; 88,722,626 bytes (84.61 MiB) | Active immutable baseline | License is recorded as unknown. Its manifest maps source keys and metadata rows, but it is not an expansion pool. |
| `memematch_281/images` | 281; 19,591,655 bytes (18.68 MiB) | 0 | SHA-256 audit: 280 unique bytes, all already in active catalog. Do not re-ingest. |
| `raw/mood-phase32` | 200; 5,980,276 bytes (5.70 MiB) | 0 now; 200 conditional candidates | No exact SHA-256 match with active catalog. Existing documentation limits it to private local research because terms are unknown. |
| Workspace archives | 0 archive files present | 0 | Documentation records a user-supplied `TAC24_meme_emotion.zip` outside this workspace (1,281,041,204 bytes) with a 10,005-row `final.csv` and nested image ZIPs. This audit did not access it. |

The active catalog manifest has 1,000 entries and 1,000 image files.  The active catalog has 1,000 unique SHA-256 values.  The provenance documents correctly distinguish the archive's official-source origin from a license grant: source code or a Drive/Kaggle link is not permission to redistribute images.

MOOD availability is therefore partial: the local 200-image material is present, and prior safe archive inspection recorded annotations for 10,005 filenames, but the full archive is not in this workspace and no usable license was verified.  The upstream ALFRED repository exposes a sample CSV/images and links a full dataset, while its paper describes MOOD as six-emotion meme data; neither fact establishes reuse rights. [ALFRED repository](https://github.com/shiv6891/ALFRED_MemeEmotionDetection) [MOOD paper](https://arxiv.org/abs/2403.10279)

## Source assessment

| Candidate | Availability / scale | Coverage and metadata | Rights conclusion | Suitability |
| --- | --- | --- | --- | --- |
| MemeMatch v1.0 | Official repository points to Kaggle; project records version 1, about 32.0 GB, with a claimed nearly-million-item resource | Best prospective modern meme coverage; project records Reddit/ImgFlip (2018–2023), local/global context and related annotations | **Unknown.** Kaggle's previously observed metadata was `Unknown`; repository has no visible dataset license. Terms and underlying image rights must be checked. | Best conditional research source; not a source for current ingestion. |
| MOOD / ALFRED | Existing 200 local images; documentation records 10,005 annotation rows in supplied archive | Emotion labels, sentiment and OCR fields; narrower affect dataset, not demonstrated broad/template-representative | **Unknown / private-research restricted.** No documented redistribution or commercial permission. | Good only for separately governed research analysis if owner terms allow; not the recommended retrieval expansion corpus. |
| Permissioned personal favorites | Future opt-in only | Can fill personal/recency gaps; captions/context must be supplied or derived with provenance | **Verified per item only** when ownership/permission and intended scope are recorded | Valuable as a distinct personal collection, never silently mixed with research corpus. |
| Unattributed scraping, repost dumps, or URL crawling | Not evaluated as an authorized source | Often modern but captions/provenance vary and duplicates are high | **Restricted/unknown; reject by default.** | Not suitable. |

“Availability” means a source location can be identified; it does not mean rights are verified.  Do not infer commercial use or redistribution permission from any of these links.  For MemeMatch, obtain the source terms and an authoritative file manifest before selecting files; if it only distributes URLs, separately review each origin's terms and do not scrape by default.

## Resource estimates

The exact measurements below come from the existing 1,000-image artifacts.  Original images average 88,722.626 bytes.  MiniLM uses 384 float32 values (1,536 raw bytes/item).  OpenCLIP uses 512 float32 values (2,048 raw bytes/item).  The current exact OpenCLIP `IndexFlatIP` index is 2,048,045 bytes for 1,000 items; OpenCLIP encoding on CUDA, batch size 8, took 11.947 s (about 83.7 images/s) with 643,718,144 bytes peak allocated GPU memory.  This is compatible with the 4-GB RTX 3050 for the observed batch.

| Total catalog target | Original images* | MiniLM vectors | OpenCLIP vectors | Exact OpenCLIP FAISS | CSV-like metadata | SQLite planning envelope | OpenCLIP encode† | OCR plan‡ | BLIP plan‡ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5,000 | 423 MiB | 7.3 MiB | 9.8 MiB | 9.8 MiB | 5.2 MiB | 10–30 MB | 1.0 min | 8–28 min | 28–83 min |
| 50,000 | 4.13 GiB | 73.2 MiB | 97.7 MiB | 97.7 MiB | 52.3 MiB | 100–300 MB | 10.0 min | 83–278 min | 4.6–13.9 h |
| 100,000 | 8.26 GiB | 146.5 MiB | 195.3 MiB | 195.3 MiB | 104.6 MiB | 200–600 MB | 19.9 min | 167–556 min | 9.3–27.8 h |

\* Linear extrapolation from current compressed source files, not a source-size promise.  Budget at least 2× original-image storage while staging/validating and retain headroom for rejected data, logs, derivative artifacts, and backups.

† Linear extrapolation from the measured 1,000-image CUDA job.  It excludes model load, disk variance, retries, validation, and any simultaneous pipeline work; re-measure after each source/format change.

‡ No existing end-to-end OCR or BLIP timing was found.  These are deliberately broad planning envelopes: OCR assumes 3–10 images/s CPU-equivalent; BLIP assumes 1–3 images/s on the specified 4-GB GPU.  They are not observed results and must not be used as a scheduling commitment.

At 100K, raw vectors plus two exact in-memory matrices/indexes remain comfortably below 16 GB, but model RAM, decoded-image buffers, SQLite cache, OS, and concurrent workers mean Phase 8B should keep one GPU encoding worker, bounded CPU decoding, and one process per catalog build.  Exact FAISS is still reasonable at 100K; no approximate-index decision is needed before measured latency demands it.

## Incremental ingestion architecture

The current importer already establishes useful foundations: decoded-image validation, SHA-256, pHash bands with exact Hamming confirmation, `dataset_sources`, stable `(dataset_source_id, original_source_id)` identity, duplicate relationships, failures, and checkpointed runs.  Its near-duplicate records are review relationships rather than unsafe automatic merges.  Phase 8B should extend this boundary without mutating `memematch_1000_v2`.

1. Create a new catalog generation, source manifest, and immutable input snapshot.  Record source/version/terms URL, access date, licence status, attribution, source item ID, source URL, filename, annotation mapping, archive/member digest, and collection limitations before a byte enters staging.
2. Stage only the approved batch under a generation-specific raw directory.  Decode/validate one file at a time with image dimensions, pixel, format, and file-size limits.  Persist every rejection with reason and source item ID; do not delete evidence.
3. Compute SHA-256 and pHash.  SHA-256 collisions become exact-duplicate relationships.  Four-band pHash candidates receive a Hamming-distance review queue; do not auto-collapse template variants, crops, text overlays, or cross-source matches.  Preserve all provenance even where one binary is selected as canonical.
4. Derive a deterministic catalog identity from source-manifest digest, ordered accepted-record identities, validation configuration, dedupe decision file, and embedding/model configuration.  Do not reuse a filename as identity.  Reject duplicate `(source, source_item_id)` pairs and namespace personal items as `personal:<owner-or-collection>:<id>`.
5. Produce additive, versioned feature rows for OCR, caption/BLIP, MiniLM and OpenCLIP.  Each row must carry image SHA-256, model/provider/version, preprocessing/config digest, status, elapsed time, and error.  A changed image or model produces a new row, not an overwrite.
6. Write append-only generation artifacts to a new directory: metadata, source keys, vectors, exact indexes, checksums, and a manifest.  Use temporary files plus atomic rename.  An incomplete generation cannot be selected by the application.
7. Resume solely from committed checkpoints keyed by immutable source item and configuration digest.  Record `accepted`, `failed_validation`, `duplicate_exact`, `near_duplicate_review`, `skipped_policy`, `embedding_failed`, and `completed`; reruns must be idempotent.
8. Validate counts, checksums, vector dimensions/normalization, FAISS `ntotal`, source-key ordering, and catalog identity before an explicit catalog-switch decision.  The current 1,000 catalog and its indexes stay immutable and selectable throughout.

Personal favorites use the same validator and feature pipeline but a different source class, consent/ownership declaration, visibility scope, and deletion/takedown path.  They must not alter research source counts, benchmark pools, or experimental ratings.

## Development-only coverage evaluation

Catalog size is not a relevance metric.  Compare an immutable 1K baseline with a new generation only on newly selected, development-only queries; do not access held-out queries or modify historic ratings.  Freeze the query set, policy version, retrieval variants (A0/A3-LH/M1 remain experimental), candidate depth, filters, and adjudication form before generating results.

Measure:

- **Template diversity:** unique normalized template IDs in the full catalog and top-k, effective template count (inverse Simpson), top-template share, and source/template concentration.
- **Situation and humor-category coverage:** a predeclared, manually auditable taxonomy; report per-cell catalog support and percentage of development queries with an eligible top-k item.  Do not treat model-generated labels as ground truth without sample audit.
- **Duplicate/template concentration:** exact/near-duplicate rates, review decisions, same-template rate in top-k, and whether added items merely create text-overlay variants.
- **Retrieval novelty:** proportion of top-k items from the new generation, novel templates versus baseline, and score/rank displacement.  Novelty is descriptive, not automatically better.
- **Cross-query repetition:** unique recommendations over the suite, item/template reuse histogram, and pairwise top-k overlap.  Report with and without diversity controls.
- **Safety-filtered availability:** per safety category and query cell, the fraction with at least one and at least k policy-eligible candidates; log filter reason without exposing disallowed content in reports.
- **Human signal, only if the automatic coverage gates show material diversity/availability change:** a small blinded, randomized side-by-side evaluation of unioned candidates, balanced by taxonomy cell and evaluator, with relevance, sendability, safety appropriateness, and “no suitable recommendation” options.  Keep it development-only, report uncertainty and disagreements, and never overwrite prior ratings.

Pre-register practical gates before the batch: no increase in exact duplicates, no material worsening in top-template or cross-query concentration, increased support in documented sparse taxonomy cells, and no safety-availability regression.  A larger catalog that fails these gates is not a successful expansion.

## Bounded Phase 8B implementation plan

**First batch recommendation:** **500 approved, provenance-complete MemeMatch items**, only after permissions/package facts are recorded.  This is intentionally smaller than the 4,000-item leap to 5K: it exercises source mapping, quality, duplicate rate, feature throughput, disk assumptions, and coverage gates while remaining reviewable.  If terms remain unknown, Phase 8B ends at readiness work and ingests zero items.

1. **Authorization gate (no images):** record the official dataset-card/terms, allowed purpose, redistribution/commercial restrictions, version, package/member manifest, citation, and retention/takedown requirements.  Stop if any item-right or source-license ambiguity remains.
2. **Implementation:** add generation-scoped catalog metadata, source manifest schema, immutable accepted/failed/review ledgers, catalog-identity hashing, versioned feature-status tables, and atomic artifact publishing.  Add a personal-source adapter with consent fields and isolation.
3. **Tests:** unit-test identity collisions, restart/checkpoint idempotency, hash/pHash paths, no mutation of 1K artifacts, failed/skipped ledger semantics, source-field preservation, and FAISS/vector alignment; add a tiny synthetic fixture only.
4. **Authorized pilot intake:** stage at most 500 approved source items, validate and deduplicate them, and manually resolve every near-duplicate/policy exception.  Do not switch the app catalog.
5. **Feature and artifact build:** measure OCR/BLIP/OpenCLIP throughput, GPU/RAM, failure rate, and artifact size; publish a new, unselected catalog-generation manifest with checksums.  Do not tune the archived lexical situation reranker.
6. **Development evaluation:** run the preregistered coverage/novelty/concentration/safety analysis and, only if warranted, the blinded small human study.  Keep held-out queries untouched.
7. **Decision gate:** promote nothing automatically.  Review provenance, quality, resources, and coverage results; then explicitly decide whether to create a 5K generation, repeat a corrected pilot, or stop.

## Validation performed

- Read-only filesystem inventory for image directories and archives.
- SHA-256 overlap audit across active, legacy, and local MOOD image sets.
- Read-only inspection of the active catalog manifest, source/provenance documents, dataset schema, validation/ingestion code, existing OpenCLIP manifest, and local SQLite metadata counts.
- JSON parse validation for the two new machine-readable deliverables.

No image decoder, OCR/BLIP/OpenCLIP processing, database migration, index rebuild, external download, or held-out-query operation was run.
