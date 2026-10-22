# Phase 2.5 dataset acquisition report

**Verification date:** 22 October 2026  
**Real research images downloaded:** 0  
**Real research images ingested:** 0  
**Reason:** neither official distribution exposed terms that could be verified as permitting a local research smoke import without accepting external access conditions. No access control was bypassed and no large archive was downloaded.

## MemeMatch

- Official repository: <https://github.com/TriAnLe171/MemeMatch-v1.0>
- Dataset link published by that repository: <https://doi.org/10.34740/kaggle/dsv/14510064>
- Paper/proceedings: <https://doi.org/10.1609/icwsm.v20i1.42785>

The proceedings abstract says the resource contains nearly one million image-with-text memes from Reddit `r/Memes` (2018–2023) and ImgFlip, with local/global context plus 14-dimensional sentiment/emotion vectors, topics, and intent annotations. A university research poster describes a curated core of about 301K memes, but it is not a substitute for the dataset card. The repository README links the Kaggle DOI but does not specify the archive format/size, whether binary images are included versus URLs, subsets, broken-link rate, download requirements, or redistribution terms. No root LICENSE file was displayed in the public repository listing inspected.

**Manual next step:** open the Kaggle DOI while authenticated, read and accept the dataset terms if appropriate, and record the exact version, archive contents, size, and license in a source manifest. If only URLs are supplied, do not scrape them until their source terms are reviewed.

## MOOD / ALFRED

- Official repository: <https://github.com/shiv6891/ALFRED_MemeEmotionDetection>
- Full-data link published by the repository: <https://drive.google.com/file/d/1w5on7FdfYi5nGhUnQD82YJND5EcJrKw9/view?usp=sharing>
- Paper: <https://arxiv.org/abs/2403.10279>

The official README says the repository contains `data/train_sample.csv`, `data/sample_memes`, and precomputed sample emotion features; it links the full archive as `TAC24_meme_emotion.zip`. It does not document a redistribution license, the full archive size, authentication state, authoritative split files, or a schema that can be verified without downloading. The paper confirms six basic emotions, but their exact labels/splits must come from the official annotations—not from this project.

**Manual next step:** open the Drive link in a browser while signed in if prompted; review owner-provided access and usage terms before downloading. If terms permit, download only an official, already-small sample; otherwise request written research-use permission from the maintainers. Preserve the CSV and image filename mapping verbatim.

## Phase 2.6 local-archive verification

The user supplied `C:\Users\jason\Downloads\TAC24_meme_emotion.zip` (1,281,041,204 bytes) from the official Drive link. Its outer ZIP central directory contains five entries, no traversal/absolute paths, and a 1.00 aggregate compression ratio:

- `meme_emotion/final.csv` — 10,005 rows; fields: `image`, six binary emotion columns, and `sentiment`.
- `meme_emotion/ocr_final.csv` — the `final.csv` fields plus `ocr_text`.
- `meme_emotion/entity_label.csv` — the OCR fields plus `entity_label`.
- `meme_emotion/final.zip` (550,687,595 bytes) and `meme_emotion/memotion.zip` (729,282,687 bytes), which contain the images and were **not extracted**.

`final.csv` has 10,005 distinct filenames; its six supplied labels are `fear`, `anger`, `joy`, `sadness`, `surprise`, and `disgust`, each represented by positive/`not_` values. Sentiment values are `negative` (6,204), `positive` (2,765), and `neutral` (1,036). There is no train/validation/test split column in this file. The outer archive’s terms remain undocumented, so Phase 2.6 stops at safe inspection: 0 images were extracted, 0 images were ingested, and no annotations were modified.

## KaggleHub selective-download verification

Installed `kagglehub` 0.3.13 as a development acquisition dependency and inspected its public API. `kagglehub.dataset_download(handle, path=...)` documents `path` as an optional file inside a dataset, so the repository now has a tested helper for a safe dataset-relative request. The intended first file is `annotations/annotations/local_context.csv` for handle `trianle171/memematch-v1-0`.

No KaggleHub call was made: Kaggle reports MemeMatch’s license as **Unknown**. Accept the dataset terms in the Kaggle browser and confirm research use before running a request. The helper does not fall back to a 32 GB full-dataset download.

## Disk and scale planning

`C:` reported **146,244,202,496 bytes free** (about 136 GiB) during verification. Actual dataset file sizes remain unverified. The following are planning ranges, not source-reported values, assuming 0.2–1.0 MiB per retained original image before derived features:

| Target | Estimated image storage | Notes |
| --- | ---: | --- |
| 5K | 1–5 GiB | plus SQLite/manifests and working space |
| 10K | 2–10 GiB | leave room for re-downloads and duplicates |
| 50K | 10–50 GiB | likely needs a dedicated storage budget before Phase 8 |

The current pipeline is bounded-memory: it decodes one image at a time and checkpoints batches. SHA-256 is linear in file bytes. pHash candidate lookup uses indexed bands instead of all-pairs comparisons; actual throughput and duplicate overhead remain unmeasured because no permitted real sample was acquired.

## Result

## Phase 3.2 local research integration

**Verification date:** 22 October 2026
**Database:** `data/processed/phase32-research.db` (Git-ignored)
**Image policy:** all source images and extracted annotations remain local and Git-ignored.

### MOOD / ALFRED

The supplied `TAC24_meme_emotion.zip` was used only as a local research artifact. DaMeme read `meme_emotion/final.csv`, selected no more than 50 records by round-robin over its six positive emotion columns, and extracted only the matching members from nested `final.zip`.

- Extracted: **50** images.
- Ingested: **50** images; 0 rejects, 0 exact duplicates, 0 near-duplicate candidates.
- Category distribution: anger 9; fear 9; disgust 8; joy 8; sadness 8; surprise 8.
- Preserved per record: original filename, all six original `final.csv` emotion values, sentiment, derived positive-label field, source URL, and label provenance (`MOOD final.csv`). `final.csv` has no split column, so no split was invented.
- License: **unknown**. The official repository and supplied archive did not expose explicit dataset licensing terms during verification. The sample is for private local research only and must not be committed or redistributed.

### MemeMatch

Kaggle's public metadata endpoint reported version 1, 32,002,485,265 total bytes, and an **Unknown** license. The existing selective KaggleHub helper downloaded only `annotations/annotations/local_context.csv` (36.4 MB); it contains filename/context pairs but has no header and does not reveal the image directory layout. Kaggle's public metadata response listed no files.

A bounded request for the first annotated filename as a root-relative image returned Kaggle HTTP 404. Therefore DaMeme did **not** guess paths, crawl URLs, or issue a root dataset request.

- Image metadata files acquired: **1** local-context CSV.
- MemeMatch images downloaded/ingested: **0 / 0**.
- Blocker: the public source exposes no file manifest, so a verified dataset-relative image path is unavailable. Obtain an authoritative layout/manifest from the maintainers or Kaggle, then use `download_dataset_file(relative_path=...)` for each approved image. Do not use a dataset-root request, which would risk the 32 GB archive.
- License remains **unknown**; preserve any future context/sentiment/topic/usage fields verbatim and retain images locally only.

Phase 2 software is ready for a small local import when an official source’s access terms are verified. DaMeme is **not yet ready for a production or 5K+ corpus acquisition** because the necessary source-specific license/package facts are unresolved. Phase 3 should not begin until an authorized, provenance-complete smoke dataset exists.
