# Dataset sources and access status

Checked 22 October 2026. No research image was downloaded in Phase 2 or the Phase 2.5 acquisition checkpoint. See the [acquisition report](dataset-acquisition-report.md) for exact links and manual access steps.

| Dataset | Intended purpose | Verified access | Image/annotation status | Status |
| --- | --- | --- | --- | --- |
| MemeMatch | Retrieval corpus | [Official repository](https://github.com/TriAnLe171/MemeMatch-v1.0) links [Kaggle DOI](https://doi.org/10.34740/kaggle/dsv/14510064) | Proceedings says nearly one million Reddit/ImgFlip memes and rich annotations; package contents and terms remain unverified | Blocked pending authenticated dataset-card/terms inspection |
| MOOD / ALFRED | Emotion train/evaluation benchmark | [Official repository](https://github.com/shiv6891/ALFRED_MemeEmotionDetection) links [full archive](https://drive.google.com/file/d/1w5on7FdfYi5nGhUnQD82YJND5EcJrKw9/view?usp=sharing) | Repository declares sample CSV/images/features, but does not document reusable full-data terms or authoritative split schema | Blocked pending owner terms/access review |

The ALFRED paper describes MOOD as six basic emotion categories, but this repository does not invent their names or its original train/validation/test splits. The importer accepts supplied split and label fields unchanged; never remap, synthesize, or mix MOOD test items into training.

Before any Stage A download, record the specific dataset version, terms, retrieval date, expected storage, authentication requirement, and source manifest. A Kaggle/Drive link may require account acceptance or authentication; do not bypass it. The official MemeMatch README links a Kaggle DOI but does not establish redistribution permission for the images.

Phase 2.6 verified the local MOOD archive's CSV headers/counts and KaggleHub's individual-file API support; see the [acquisition report](dataset-acquisition-report.md). It did not download MemeMatch files or extract MOOD images because the usage terms remain unresolved.
