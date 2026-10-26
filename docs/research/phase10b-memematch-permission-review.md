# Phase 10B: MemeMatch permission review

**Decision:** **not approved.** The existing `memematch_1000_v2` permission gate must remain unapproved. This is a documentation review only; it does not create or alter a database permission record, catalog generation, or product configuration.

## Source identity and evidence reviewed

| Evidence | Finding |
| --- | --- |
| Recovered catalog | `memematch_1000_v2`, 1,000 local images; acquisition manifest created 24 October 2026; metadata SHA-256 `f4e24bce…2542710d4`. |
| [Phase 10A recovery audit](phase10-catalog-recovery-audit.md) | Technical artifacts are complete, but no MemeMatch dataset-source or permission-review record exists in the local SQLite files. |
| [Phase 8 source inventory](phase8-dataset-source-inventory.json) | Records the catalog's license status as `unknown`; explicitly says images must not be modified. |
| [Dataset acquisition report](../dataset-acquisition-report.md) | Identifies the MemeMatch v1.0 repository, Kaggle DOI, and reported version information. It records that public Kaggle metadata reported an unknown license and that no authenticated terms/package review was completed. |
| [Dataset licensing guidance](../dataset-licensing.md) | States that availability of code, a Kaggle/Drive link, or local files does not grant image redistribution rights; `verified` requires review of governing terms. |
| Permission-gate implementation | `PermissionReview.validate_for_promotion()` requires complete evidence and `review_status == "approved"`; generation membership additionally requires an approved review. |

The documented source is MemeMatch v1.0, whose repository links the Kaggle DOI `10.34740/kaggle/dsv/14510064`. This review did not find a locally retained, authoritative dataset card, license text, accepted terms record, or per-image rights evidence.

## Permission assessment

| Intended use | Decision | Basis |
| --- | --- | --- |
| Existing local research inspection | **Evidence of occurrence, not permission** | Local artifacts show prior research processing, but retained artifacts do not establish a governing grant. |
| Local application serving | **Unknown / not authorized** | No verified terms or approved permission review covers application use. |
| Redistribution or public hosting | **Unknown / not authorized** | No evidence grants redistribution, hosting, or downstream image rights. |
| Attribution obligations | **Unknown** | No authoritative terms record specifies the required citation, notices, or provenance display. |
| Explicit restrictions | **No source term verified** | Project policy itself prohibits treating availability as authorization. |

No automatic approval is appropriate. Public availability does not resolve rights held by the dataset distributor, Reddit/ImgFlip sources, individual image authors, or depicted trademarks/personality rights.

## Permission-gate result

The gate cannot be approved. `memematch_1000_v2` has no corresponding source record and no `source_permission_reviews` row in the inspected databases. Even a new record with `unknown` status would continue to block promotion, as designed.

## Evidence required before reconsideration

1. An authoritative, authenticated copy or stable reference for the governing MemeMatch/Kaggle terms and exact dataset version.
2. A written determination that the intended local application use is allowed, including whether images may be served to local browser clients.
3. Explicit treatment of redistribution/public-hosting, commercial/portfolio use, retention, takedown, and any source-image rights restrictions.
4. Required attribution/citation and a complete source/version/package manifest.
5. A provenance record and a complete `PermissionReview` with source-specific evidence. Only then may an authorized reviewer set `review_status` to `approved`.

Until all five are satisfied, preserve the current 281-item catalog and do not activate, publish, or expose the recovered 1K images through the application.
