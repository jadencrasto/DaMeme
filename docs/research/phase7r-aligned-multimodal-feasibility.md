# Phase 7R.6A: aligned multimodal retrieval feasibility

## Scope and finding

This is a read-only feasibility audit for `memematch_1000_v2`; no model was downloaded, loaded, or used to create embeddings. The catalog has 1,000 local images (88,722,626 bytes) and a production metadata CSV. Existing A0/A3 retrieval uses 384-D MiniLM text embeddings and the Phase 4 feature pipeline can separately produce BLIP vision features. Those MiniLM and BLIP vectors are **not** a shared text-image space: a MiniLM query must not be searched against BLIP image vectors.

The recommended next experiment is a pinned OpenCLIP `ViT-B-32` using the OpenAI-compatible pretrained checkpoint. It provides one 512-D space for both query text and meme images. OpenCLIP documents the shared `encode_image`/`encode_text` interface and L2-normalized dot-product use, while the OpenAI CLIP model card describes contrastive training of the paired encoders. [OpenCLIP README](https://github.com/mlfoundations/open_clip/blob/main/README.md) [OpenAI CLIP model card](https://huggingface.co/openai/clip-vit-base-patch32)

## Local readiness

The project has FAISS CPU, Transformers 4.57.6, and a cache containing MiniLM and BLIP. It has neither OpenCLIP nor an OpenAI CLIP/SigLIP checkpoint. The host reports an RTX 3050 Laptop GPU with 4,096 MiB VRAM, but the current environment has `torch 2.14.1+cpu` and `torch.cuda.is_available() == false`. Therefore Phase 7R.6B must explicitly provision or select a CUDA-enabled PyTorch environment; a visible GPU driver alone is not sufficient.

## Model comparison

| Candidate | Shared dim | Checkpoint estimate | Catalog vector storage | Hardware assessment | Decision |
| --- | ---: | ---: | ---: | --- | --- |
| OpenAI CLIP ViT-B/32 | 512 | ~605 MB | 2.05 MB float32 | Conservative batches should fit 4 GB VRAM; CPU fallback is feasible but slower | Canonical baseline |
| OpenCLIP ViT-B/32, OpenAI weights | 512 | ~605 MB | 2.05 MB float32 | Same practical envelope, with a maintainable route to future OpenCLIP ablations | **Recommended implementation** |
| SigLIP base patch16/224 | 768 | ~813 MB | 3.07 MB float32 | Tighter 4 GB margin; reasonable later comparator, not the first run | Defer |

The OpenAI checkpoint lists a 605 MB PyTorch file and uses ViT-B/32 contrastive text/image encoders. [Checkpoint inventory](https://huggingface.co/openai/clip-vit-base-patch32/tree/main) SigLIP base patch16/224 lists an 813 MB checkpoint, a 768 hidden size, and Apache-2.0 on its model card. [SigLIP model card](https://huggingface.co/google/siglip-base-patch16-224) [SigLIP configuration](https://huggingface.co/google/siglip-base-patch16-224/blob/main/config.json) OpenCLIP source is MIT-licensed, but each selected pretrained checkpoint's terms and provenance must be recorded separately. [OpenCLIP license](https://github.com/mlfoundations/open_clip/blob/main/LICENSE)

All estimates are planning estimates rather than measured benchmarks. The 1,000-image embedding job is expected to take a few minutes on the target GPU at batch size 8, or tens of minutes on CPU. Actual batch size, peak VRAM/RSS, cold-start time, image-encoding time, query-encoding time, and search/filter time must be measured in 7R.6B.

## Controlled retrieval design

The offline comparison will retain A0 and A3-LH unchanged, then add:

| Variant | Retrieval |
| --- | --- |
| M1 | Normalized CLIP query-text embedding against normalized CLIP image embeddings, searched through a separate exact `IndexFlatIP` index. |
| M2 | A3-LH and M1 each retrieve 100 candidates; reciprocal-rank fusion (`k=60`) combines ranks. This avoids mixing raw MiniLM and CLIP scores. |

Each path uses the same development query text and existing mood preprocessing as its text baseline, retains `SafetyConfig`, retains one-per-template diversity, and uses source-key ascending order to break ties. The aligned model owns its processor: images are decoded as RGB and transformed only by the pinned model's returned preprocessing function. This avoids accidental preprocessing drift.

## Evaluation and stopping rules

Only the 40 development queries may be loaded. Per variant, record final top-10 identities and provenance, top-five/top-ten overlap, novelty, cross-query repetition, safety outcomes, template diversity, candidate counts, query/index/filter latencies, memory, and artifact hashes. Existing human ratings may only be reused for exact query--source identities; unjudged candidates stay unjudged.

Before seeing M1/M2 results, pre-register a fresh category-balanced development subset for human evaluation. Fully judge the union of each variant's top five on that subset. Reject the experiment if CUDA cannot run batch size 1 on the target GPU, if runtime is impractical with no usable GPU environment, or if fresh evaluation provides no descriptive relevance/sendability signal worth the additional complexity. It will not select a production architecture.

## Bounded Phase 7R.6B implementation scope

1. After explicit authorization, add the minimal OpenCLIP dependency and obtain one exact, hash-recorded `ViT-B-32` checkpoint. Do not download or compare SigLIP in the first run.
2. Add a standalone aligned-image exporter that validates all 1,000 source keys and image paths, applies the returned OpenCLIP transform, writes normalized 512-D vectors, an exact FAISS IP index, and a manifest with package/checkpoint/preprocessing/device provenance.
3. Add an offline runner for unchanged A0/A3-LH, M1, and M2 at depth 100 with existing safety/diversity functions and deterministic tie-breaking.
4. Add focused integrity, score/rank-fusion, source-key-alignment, deterministic-ordering, and development-only-isolation tests.
5. Execute only development retrieval, export a fresh preregistered human-evaluation pool, and report measurements without modifying production, ratings, catalog, or held-out queries.
