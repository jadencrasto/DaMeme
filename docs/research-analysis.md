# Research analysis

## Sources checked

- MemeMatch: [official repository](https://github.com/TriAnLe171/MemeMatch-v1.0), which describes itself as the dataset, paper, and source-code repository and links a dataset DOI. The provided ICWSM DOI was not accessible through the research browser on 22 October 2026; therefore paper-level details that could not be independently inspected are not asserted here.
- Sharma et al., [Emotion-Aware Multimodal Fusion for Meme Emotion Detection](https://arxiv.org/abs/2403.10279), submitted 15 March 2024 and listed there as accepted to *IEEE Transactions on Affective Computing*.
- [ALFRED official implementation](https://github.com/shiv6891/ALFRED_MemeEmotionDetection), which supplies code, sample labels/OCR text, sample memes, precomputed emotion features, and a link to the full dataset.

## MemeMatch

The repository establishes a retrieval-oriented meme resource and identifies the dataset and code as separately available resources. Its public README alone does not provide enough reproducible detail to state dataset size, annotation schema, metrics, architecture, checkpoint availability, license, or hardware requirements. DaMeme therefore treats MemeMatch as a prospective, provenance-tracked retrieval source—not as a guaranteed drop-in dependency. Before adoption, Phase 2 must inspect the official paper/code, dataset terms, splits, and licensing.

The research direction adopted from MemeMatch is the separation of local textual context (OCR) and global visual/template context for meme retrieval. DaMeme's proposed deployment is an adaptation: offline extraction and embeddings, followed by online vector retrieval and transparent metadata-driven reranking.

## ALFRED / MOOD

The paper introduces MOOD, a meme-emotion dataset with six basic emotion categories, and ALFRED, a multimodal framework that explicitly models emotion-enriched visual cues and uses gated cross-modal fusion. The abstract reports a 4.94% F1 advantage over the baselines selected in that study; this is not a DaMeme expected result. It also evaluates transfer/generalization on Memotion, HarMeme, and Dank Memes. The repository confirms an implementation and sample artifacts, but it does not state that reusable pretrained checkpoints are published; Phase 1 makes no checkpoint claim.

DaMeme will borrow the *idea* of separate text/vision representations and learned emotion-aware fusion only after a licensed, reproducible Phase 4 experiment. A lightweight BERT/ViT-compatible interface is planned; it is not implemented now. A query-to-meme retrieval system and emotion-aware reranking are DaMeme's proposed combination and research question, not results from either source.

## Feasibility and constraints

No model, dataset, or dependency with large weights was downloaded. Future SentenceTransformers, EasyOCR, BLIP, RoBERTa, BERT/ViT, and FAISS selections require version, model-card, license, memory, and hardware verification at implementation time. GPU acceleration may improve offline processing, but the Phase 1 API and UI require neither GPU nor model runtime. Do not infer a dataset/model license from its code repository; record it explicitly before ingestion.
