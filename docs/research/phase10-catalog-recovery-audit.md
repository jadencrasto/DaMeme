# Phase 10A: 1,000-meme catalog recovery audit

**Status:** complete, read-only audit (26 October 2026). No image, catalog, model, index, database, configuration, rating, or evaluation-query artifact was changed.

## Decision

`memematch_1000_v2` is **fully recoverable technically**, but it is **not approved for product integration**. All 1,000 local images, A0 retrieval artifacts, manifests, metadata, and export provenance are present and aligned. However, the project has no MemeMatch `dataset_sources` or permission-review record, and the documented license status is `unknown`. Existing local research use is not evidence of application, redistribution, or public-hosting rights.

The current 281-item A0 catalog remains the only product baseline. This phase did not change that selection.

The accompanying machine-readable inventory is [phase10-catalog-artifact-inventory.json](phase10-catalog-artifact-inventory.json).

## Recovery inventory and integrity

| Artifact | Result | Evidence / implications |
| --- | --- | --- |
| 1K image root | **Fully recoverable** | `data/retrieval/catalogs/memematch_1000_v2/images` contains 1,000 files: 956 JPG and 44 PNG. All decoded successfully. It is Git-ignored, which explains why a tracked-file-only search misses it. |
| Image integrity | **Pass with review queue** | 1,000 readable files, 0 exact-byte duplicate groups, dimensions range from 236×150 to 2,118×2,352 (median 534×499). The established pHash rule (Hamming distance <= 6) identifies 2,768 near-duplicate candidate pairs; these are review relationships, not automatic removals. |
| Acquisition manifest and metadata | **Fully recoverable** | 1,000 manifest entries, 1,000 unique stable source keys, and 1,000 CSV rows. The source-key sequence is identical. Metadata SHA-256 is `f4e24bce…2542710d4`. |
| A0 FAISS retrieval set | **Fully recoverable** | `text.faiss` (384D), `image.faiss` (768D), and `fused.faiss` (1,152D) each contain 1,000 entries. The index manifest, records, and metadata are present. |
| A0 export provenance | **Fully recoverable** | All 1,000 completed record indexes are contiguous and their source-key sequence matches both the A0 manifest and acquisition manifest. Export metadata names `sentence-transformers/all-MiniLM-L6-v2` and `Salesforce/blip-image-captioning-base`. |
| A0 serving compatibility | **Technically compatible** | Constructing the existing `RecommendationEngine` against this catalog succeeded with `ready: true`, 1,000 records, and 1,000 resolvable image paths. The existing resolver correctly uses `images/<filename>` rather than the stale source-location text. |
| M1 OpenCLIP | **Recoverable, experimental only** | `phase7r_v1_openclip_image_index` contains a validated 1,000×512 `ViT-B-32-quickgelu`/OpenAI index plus source mapping. It is dimensionally and architecturally separate from A0. |
| Local/global MiniLM indexes | **Recoverable, research only** | The Phase 7R validator passed 24/24 checks for two 1,000×384 representations. Four local rows are deliberately unavailable/zero. These are ablation artifacts, not the product A0 manifest. |
| SQLite provenance | **Missing for this catalog** | The product DB and Phase 2 smoke DB have no dataset-source rows; the research DB records only a separately restricted MOOD source. No MemeMatch permission record was found. |

The A0 index is a serving-compatible `RetrievalService` manifest. Its separate `embeddings.npz` is not required at runtime because the service loads FAISS indexes and its embedded record mapping. No regeneration is required for technical recovery.

## Catalog quality and coverage comparison

The product catalog has 281 images; 280 filenames occur in the recovered 1K catalog, while 720 images are additional. `Advice_Peeta_14.jpg` is the one 281 filename absent from the 1K catalog. Both CSVs expose the same 28 broad `usage` categories, so the increase is depth within the current taxonomy rather than a documented increase in taxonomy breadth.

The 1K metadata has 475 heuristic template groups. Of those, 404 contain one image; the largest groups contain 10–11 images. This gives the current safety/template-diversity controls a more varied candidate pool, but it also means the heuristic is not a verified template taxonomy. The most frequent labels are broad humor categories (for example parody/spoof, sarcasm/irony, and absurd/random humor); category counts are multi-label and therefore do not establish situation relevance.

Image readability, stable identity, exact-byte duplicates, metadata alignment, and index alignment passed. The established pHash threshold identifies 2,768 candidate near-duplicate pairs, which is expected to include crops, overlays, and template variants; the audit makes no automatic deduplication decision. This audit did not assess semantic relevance, inspect held-out queries, or use preliminary human results as a quality claim.

## Product and experiment boundary

A0 remains the only product retrieval method. The recovered A0 artifacts use the established MiniLM/BLIP fusion layout and retain the existing safety and template-diversity behavior. M1 OpenCLIP remains an offline development candidate; M2 RRF and the archived lexical situation reranker remain excluded. No ordinary frontend option should select any of those artifacts.

If experimentation resumes, it must name a non-product variant, pin the catalog-manifest checksum, use the approved development protocol, and keep all outputs and feedback separate. It must not silently replace the A0 product path.

## Rights and usage boundary

The Phase 8 source inventory and dataset licensing documentation record MemeMatch as `unknown`. They explicitly state that code availability, a Kaggle/Drive link, or local file presence does not grant redistribution or product rights. The current SQLite records do not supply the missing evidence. Therefore:

- Local research use: evidenced by the existing artifacts, but not a permission grant.
- Application use: **not authorized by recorded evidence**.
- Redistribution or public hosting: **not authorized by recorded evidence**.
- Permission gate: must remain unchanged; this audit did not write it.

## Bounded Phase 10B plan

1. **Authorization gate first.** Obtain and record authoritative MemeMatch terms, allowed intended use, restrictions, attribution, retention/takedown requirements, exact source version, and a permission-review result. Stop if product/local-serving rights remain ambiguous.
2. **Create an immutable catalog-generation record.** Reference the current source manifest, CSV digest, image-file digest manifest, and A0 artifact checksums without mutating `memematch_1000_v2`.
3. **Add an explicit product-catalog approval boundary.** The 1K generation must only be selectable after the authorization record is approved. Leave the 281 default untouched.
4. **Run an isolated readiness/integration smoke test.** Use the existing 1K A0 artifacts and image root; verify image routes, safety filtering, template diversity, exclusions, errors, and rollback while no client can silently select it.
5. **Explicit promotion or rollback.** A separately approved, reversible configuration change may select the 1K A0 generation. Rollback is selecting the immutable 281 generation; no index mixing, M1/M2 activation, or retraining is involved.

Estimated technical work after authorization is small (one focused implementation and validation phase), because the runtime artifacts already validate. The authorization/provenance decision is the critical blocker and has no safe engineering shortcut.

## Validation performed

- Decoded all 1,000 images; all succeeded. Calculated SHA-256 and pHash evidence read-only: 0 exact-byte duplicate groups and 2,768 pHash candidate pairs at Hamming distance <= 6.
- Compared 1K manifest, metadata, export records, and serving-index source-key order; all aligned.
- Loaded the existing A0 `RecommendationEngine` against the recovered catalog: ready with 1,000 resolvable images.
- Ran `backend/scripts/validate_openclip_index.py --artifact benchmarks/phase7r_v1_openclip_image_index`: all checks passed.
- Ran `backend/scripts/validate_local_global_indexes.py`: 24/24 checks passed.
- Queried the local SQLite files read-only; no MemeMatch dataset-source or permission-review evidence was found.

No models were downloaded, no vectors or indexes were regenerated, and no held-out queries were accessed.
