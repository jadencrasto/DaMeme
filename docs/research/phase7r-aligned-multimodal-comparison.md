# Phase 7R.6B.4: aligned multimodal four-way comparison

## Scope

This is a development-only, offline comparison over 40 benchmark queries. It reuses the validated `memematch_1000_v2` catalog, OpenCLIP image index, saved M1 depth-100 candidates, and the prior saved A0/A3-LH depth-100 comparison. No embedding, retrieval, production, catalog, rating, or held-out-query artifact was changed.

M1 uses OpenCLIP `ViT-B-32-quickgelu` with the `openai` pretrained checkpoint, matching tokenizer, model-supplied preprocessing, normalized 512-D text/image vectors, and exact FAISS inner-product search. M2 is rank-only reciprocal-rank fusion of A3-LH and M1 at `k=60`; raw MiniLM and CLIP scores are never combined.

All variants retrieve 100 candidates then apply the existing safety filter, one-per-template diversity rule, deterministic source-key tie-breaking, and a final top-ten limit. All 40 queries retained ten final candidates.

## Aggregate results

| Variant | Unique final memes | Cross-query repetition | Top-5 overlap vs A0 | Top-10 overlap vs A0 | New final candidates vs A0 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A0 | 172 | 57.0% | 1.000 | 1.000 | 0 |
| A3-LH | 153 | 61.8% | 0.635 | 0.612 | 46 |
| M1 | 165 | 58.8% | 0.155 | 0.155 | 104 |
| M2 | 170 | 57.5% | 0.370 | 0.415 | 74 |

M1 is substantially different from A0, and M2 mediates between text-fusion and aligned-image rankings. Novelty and coverage do not establish relevance, tone fit, or sendability; no architecture winner is selected.

## Diagnostics

Frequently appearing top-five identities are observed repetition, not relevance labels. M1 includes `fffffffuuuuuuuuuuuu.jpg` eight times, `10-guy_102.jpg` and `Advice_Dog_1.jpg` seven times each, and `this-is-fine.jpg` seven times. M2 includes `fffffffuuuuuuuuuuuu.jpg` ten times, `Am_I_The_Only_One_Around_Here_12.jpg` eight times, and `Anakin_Padme_4_Panel_2.jpg` seven times. These identities should be reviewed in a blinded assignment rather than interpreted as appropriate or inappropriate from frequency alone.

Existing ratings may only be reused for exact stable query--meme identities. Most M1/M2 novel identities have no judgment in the earlier text-focused assignments; they must remain unjudged rather than receiving implicit zero relevance.

## Latency and filtering limits

M1 query latency in the saved raw output is measured CUDA query encoding plus exact search. M2 timing is cached candidate fusion/filtering only and is not model-serving latency. Cold model load and the one-time catalog image-export time are separately recorded in earlier OpenCLIP artifacts. Safety and template outcomes are retained per query in the machine-readable results.

## Human-evaluation feasibility

Two earlier comparative assignments consume 16 development query identities, leaving 24 development queries unused. A fresh, preregistered, category-balanced eight-query subset is therefore feasible without held-out access. A four-way pooled top-five assignment has an upper bound of 160 query--meme identities; its exact workload must be calculated only after preregistering the eight query IDs and deduplicating stable identities. Existing ratings can be reused only where the same query and source key match exactly; reuse does not replace the need to rate M1/M2-introduced identities.

## Next step

Pre-register the eight unused development queries before inspecting per-query outcomes for selection, create a blinded deduplicated union-of-top-five pool, and judge situation relevance, tone fit, and sendability. Do not choose an architecture from retrieval overlap, novelty, or these descriptive results alone.
