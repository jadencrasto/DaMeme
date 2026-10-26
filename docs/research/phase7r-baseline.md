# Phase 7R.0 research baseline and architecture audit

**Scope.** This is a code-and-artifact audit performed without running inference, downloading models, training, or acquiring data. “Current” below means the checked-in implementation and local artifacts as inspected on 25 October 2026. Claims labelled *measured* are backed by a checked-in result artifact; all other observations are implementation facts or explicitly marked hypotheses/uncertainties.

## 1. Current online recommendation path

`GET /api/v1/recommendations/text` accepts free text, optional six-label emotion and three-label sentiment, usage filters, weights, a selectable mood, and retrieval controls ([`backend/app/api/v1/recommendations.py`](../../backend/app/api/v1/recommendations.py)). The default path is:

1. The engine lazily loads a named local catalog, its CSV metadata, and three FAISS indexes. It validates record count, source-key alignment, dimensions, and export provenance before serving ([`recommendation_engine.py:59`](../../backend/app/recommendation_engine.py:59), [`retrieval.py:181`](../../backend/app/retrieval.py:181)). The default catalog is `memematch_281`; `memematch_1000_v2` is an explicit alternative ([`core/config.py:34`](../../backend/app/core/config.py:34)).
2. If a mood is selected, the query becomes `Situation: <text>\nSelected mood context: <static description>.`; it is not classified from the text ([`moods.py:71`](../../backend/app/moods.py:71)).
3. The effective query is mean-pooled and L2-normalized with `sentence-transformers/all-MiniLM-L6-v2`, then searched against the **text** FAISS index ([`emotion.py:67`](../../backend/app/emotion.py:67), [`retrieval.py:222`](../../backend/app/retrieval.py:222)).
4. `semantic` mode returns text cosine candidates. `hybrid` additionally BM25-searches the exported text documents and uses RRF to select/order candidates. `hybrid_cross_encoder` optionally reranks the post-metadata shortlist using a locally cached cross-encoder and RRF-fuses its order with the metadata rank ([`hybrid_retrieval.py`](../../backend/app/hybrid_retrieval.py)). Semantic is the API default; default candidate, BM25, and cross-encoder limits are 50, 50, and 20 respectively ([`api/v1/recommendations.py:61`](../../backend/app/api/v1/recommendations.py:61)).
5. The candidate metadata scorer applies raw-cosine thresholding (if requested), a narrow source-text safety filter, exact usage-category filtering, and the weighted score below. Template diversity is applied last ([`recommendations.py:313`](../../backend/app/recommendations.py:313)).

For text search, with active normalized weights \(w_s,w_e,w_p\), the final score is:

\[
  w_s\frac{\mathrm{cosine}+1}{2} + w_e E_{requested} + w_p P_{requested}.
\]

`E` and `P` are the means of available source local/global probabilities. Default raw weights are `(0.90, 0.08, 0.02)`; they are normalized over request-intended components, so missing metadata contributes zero rather than renormalizing an individual item ([`recommendations.py:22`](../../backend/app/recommendations.py:22), [`recommendations.py:337`](../../backend/app/recommendations.py:337)). An emotion-only endpoint bypasses semantic retrieval and ranks the entire metadata catalog with semantic weight zero.

## 2. Models, vectors, indexes, and ranking

| Component | Current implementation | Dimensions / method | Used online for text recommendations? |
| --- | --- | --- | --- |
| Query/document text encoder | Frozen `all-MiniLM-L6-v2`, attention-mask mean pooling, L2 normalization | 384 | Yes |
| Image feature encoder | BLIP image-captioning base vision encoder, mean pool over vision states, L2 normalization | 768 | Exported, but no text endpoint queries it |
| FAISS | `IndexFlatIP` over normalized vectors | text 384; image 768; fused 1,152 | Text index only for current user text flow |
| Fused vector | Concatenated weighted text and image vectors, normalized; weights default 1/1 | 1,152 | Built and validated but not selected by an API recommendation path |
| Keyword retrieval | In-memory token BM25, `k1=1.2`, `b=0.75` | Exact lexical terms over exported document text | Only `hybrid` variants |
| Rank fusion | Reciprocal Rank Fusion \(1/(60+r)\) | Rank-only; no score blending | Hybrid and successful cross-encoder order fusion |
| Cross encoder | `cross-encoder/ms-marco-MiniLM-L6-v2` | Pairwise query/document logits; capped shortlist | Optional, cached-only |
| Emotion baselines | HashingVectorizer-128 + RGB 8x8 (192), or frozen MiniLM + BLIP vision; logistic baselines and gated fusion | 320 for lightweight concatenation; 1,152 pretrained concatenation | No—stored prediction outputs are not read by the recommendation ranker |

The artifact metadata confirms 281 completed records for the legacy catalog and 1,000 for v2, both with 384/768 source embeddings ([`data/retrieval/memematch_281/memematch_281.metadata.json`](../../data/retrieval/memematch_281/memematch_281.metadata.json), [`data/retrieval/catalogs/memematch_1000_v2/exports/memematch_1000_v2.metadata.json`](../../data/retrieval/catalogs/memematch_1000_v2/exports/memematch_1000_v2.metadata.json)). `IndexFlatIP` is exact, not ANN; its current implementation even retrieves all rows before deterministic sort ([`retrieval.py:156`](../../backend/app/retrieval.py:156)).

## 3. Context, OCR/caption, labels, and categories

### Retrieval text and MemeMatch contexts

The v2 export constructs one text field, in order, from cleaned OCR, `local_context`, caption/title, **verified** template data, and `global_context`. Each field is labelled in the composed string; global context is explicitly labelled “may be generic” ([`memematch_embeddings.py:104`](../../backend/app/memematch_embeddings.py:104)). This is a single MiniLM representation: there is no separate local-versus-global index, query routing, or per-context score. Legacy/v2 catalogs therefore differ in text provenance, and text representation version must be controlled in experiments.

The SQLite multimodal workflow is separate from the serving catalog. EasyOCR extracts text and boxes; BLIP produces a short visual caption; `combined_context` concatenates both ([`multimodal.py:31`](../../backend/app/multimodal.py:31), [`multimodal.py:85`](../../backend/app/multimodal.py:85)). Those outputs feed offline emotion-feature extraction as OCR + caption text ([`emotion.py:201`](../../backend/app/emotion.py:201)), but the recommendation engine reads CSV/export artifacts—not `multimodal_analyses`. Thus OCR/BLIP analysis is not currently an online enrichment step for the two shipped MemeMatch catalogs.

### Emotion, sentiment, and usage

MemeMatch local/global annotation probabilities are parsed for the requested six training labels (`anger`, `fear`, `joy`, `sadness`, `surprise`, `disgust`) and the three sentiments, then averaged across scopes. They are source metadata, not DaMeme classifier predictions or calibrated suitability labels ([`recommendations.py:45`](../../backend/app/recommendations.py:45), [`docs/emotion-aware-ranking.md`](../emotion-aware-ranking.md)). Usage strings split on `|`, comma, or semicolon and become a hard filter only when the caller supplies preferences ([`recommendations.py:70`](../../backend/app/recommendations.py:70)). They neither add a ranking feature nor train a model.

Source emotion/sentiment values do affect text ranking only when their corresponding request parameter is set. The evaluation runner passes `sentiment=None`, so its reported items correctly show `sentiment_relevance: null` ([`docs/human-relevance-evaluation.md`](../human-relevance-evaluation.md)).

### Six trained labels versus 32 selectable moods

The six labels are multi-label targets used by the offline MOOD emotion pipeline; sentiment is a separate three-class target ([`emotion.py:25`](../../backend/app/emotion.py:25)). The UI/API taxonomy has 32 static moods, each with a description, synonyms, an uneven mapping to one or more of those six labels, and a sentiment attribute ([`moods.py:23`](../../backend/app/moods.py:23)). There is no 32-class mood model.

At request time, a selected mood has two effects: its description is appended to the MiniLM/BM25/cross-encoder query, and—only if `emotion` was omitted—the highest-weight underlying label becomes the metadata emotion request ([`recommendation_engine.py:143`](../../backend/app/recommendation_engine.py:143), [`moods.py:76`](../../backend/app/moods.py:76)). The mood's `sentiment` field is not propagated to ranking. Ties resolve by dictionary insertion order, an implementation detail rather than a psychological calibration. This mapping is a product taxonomy assumption, not a validated mapping from user language to affect.

## 4. Existing but unused or only partially used metadata

The 281 CSV includes local/global probabilities for `anticipation`, `optimism`, `pessimism`, `trust`, `love`, and `neutral`, in addition to the six rankable emotions and sentiment. These non-six emotion labels are not retrieval features, rank signals, or selectable API emotion values. `reddit_score`, `reddit_created_utc`, local/global topic text and topic probabilities are also not used for retrieval or ranking ([`memematch_281_metadata.csv`](../../data/retrieval/memematch_281/memematch_281_metadata.csv)).

The v2 CSV adds/retains `source_key`, `template_group_heuristic`, acquisition/path-review fields, `local_Extracted Text + Title`, and `global_Caption` ([`memematch_1000_metadata_v2.csv`](../../data/retrieval/catalogs/memematch_1000_v2/memematch_1000_metadata_v2.csv)). `source_key` aligns records and `template_group_heuristic` supports diversity; path fields support catalog construction. `local_Extracted Text + Title` and `global_Caption` are not recognized by the current v2 text-field selector, so they are not embedded unless their content is duplicated in one of the accepted fields. This is important but should be confirmed against each exported record's `text_fields` before calling it a corpus-wide omission.

The database holds provenance, hashes, dimensions, dataset split, labels, OCR boxes/confidence, caption/provider/configuration, feature vectors, model metrics, and model prediction probabilities ([`datasets/database.py`](../../backend/app/datasets/database.py)). None of these SQLite tables is joined by the live `RecommendationEngine`; they are available only to offline/data workflows. Exact/near-duplicate relationships also do not currently constrain retrieval.

## 5. Semantic, hybrid, and cross-encoder behavior

Semantic mode is MiniLM cosine over a single document representation. Hybrid mode computes semantic candidates plus BM25 candidates and RRF-selects a union; Phase 6 metadata scoring still receives the original cosine only. In hybrid mode the code searches all semantic rows so that lexical-only IDs without semantic vectors cannot enter; on the present aligned catalog that distinction has no visible effect ([`hybrid_retrieval.py:89`](../../backend/app/hybrid_retrieval.py:89)).

Cross-encoder mode runs after safety and metadata scoring on up to the requested 20 candidates. It ranks query/document pairs by logit but does not numerically mix logits with cosine; it RRF-fuses the pairwise order with metadata order ([`recommendation_engine.py:156`](../../backend/app/recommendation_engine.py:156)). The document is the same composed text used by BM25, not an explicitly local/global-aware prompt. If weights are absent from the local Hugging Face cache, `local_files_only=True` produces a non-fatal fallback to hybrid ranking ([`hybrid_retrieval.py:107`](../../backend/app/hybrid_retrieval.py:107)).

## 6. Evaluation evidence and its limits

### Measured, checked-in evidence

- A completed single-rater human review covers 38 deduplicated query–meme pairs from five fixed queries and top-five output per mode. Mean Precision@5 / nDCG@5: semantic **0.44 / 0.7890**, hybrid **0.44 / 0.7568**, hybrid-cross-encoder **0.40 / 0.8007** ([`backend/evaluation_phase7a.metrics.json`](../../backend/evaluation_phase7a.metrics.json)). The cross encoder improved this small set's graded nDCG but lowered binary Precision@5; neither difference establishes general superiority.
- The Phase 7A export logged nonzero cross-encoder latency for its five calls, about **422–637 ms** per query; that is an observed local run, not a hardware-independent SLO ([`backend/evaluation_phase7a.json`](../../backend/evaluation_phase7a.json)).
- The mood diagnostic tested five paired scenarios with semantic mode on the 1,000-item catalog. All five pairs had different top-five IDs when only the selected static mood changed ([`backend/evaluation_phase7b.json`](../../backend/evaluation_phase7b.json)). This proves sensitivity to appended mood wording, not mood appropriateness or user benefit.
- The offline emotion document reports a 200-image functional sample split into 156 train / 26 validation / 18 test and explicitly warns that label supports can be one ([`docs/emotion-evaluation.md`](../emotion-evaluation.md)). No checked-in numeric emotion F1/accuracy report was found in this audit, so no model-quality number is asserted.

### Do not mistake these for measured facts

No controlled comparison establishes a best embedding model, separate local/global retrieval, query expansion quality, category quality, feedback effectiveness, safety recall, or scalable latency. The human comparison is five prompts, one completed rating set, overlapping candidate pools, and an unknown rater protocol beyond the stored 0–3 scale. The system documents this limitation itself ([`docs/human-relevance-evaluation.md`](../human-relevance-evaluation.md)).

## 7. Bottlenecks and likely failure modes

| Evidence-backed risk | Why it matters | Evidence |
| --- | --- | --- |
| One vector conflates OCR, local situation, and potentially generic global image context | A strong generic visual caption can dilute situational text; query intent cannot prefer a scope | Single-field construction, [`memematch_embeddings.py:104`](../../backend/app/memematch_embeddings.py:104) |
| Emotion is a small metadata adjustment and mood is template text | Fine-grained moods collapse to one six-label fallback and static wording; sentiment mapping is unused | 0.90/0.08/0.02 default and `primary_classifier_emotion`, [`recommendations.py:22`](../../backend/app/recommendations.py:22), [`moods.py:76`](../../backend/app/moods.py:76) |
| Source annotation quality/calibration is unknown | Averaging local/global probability assumes comparability; annotations may not reflect recommendation appropriateness | Code/doc explicitly distinguish source metadata from classifier outputs, [`recommendations.py:5`](../../backend/app/recommendations.py:5) |
| Exhaustive FlatIP and hybrid full semantic search | Exact all-row search is simple at 1,000, but hybrid explicitly scans all semantic rows; it will be a scaling concern | [`retrieval.py:156`](../../backend/app/retrieval.py:156), [`hybrid_retrieval.py:97`](../../backend/app/hybrid_retrieval.py:97) |
| Cross encoder is CPU-expensive and cache-dependent | Observed 422–637 ms in the small evaluation; absent cache silently changes the mode's behavior | Phase 7A artifacts and cached fallback implementation |
| Weak/noisy source text and unindexed fields | OCR errors are preserved; useful title/caption-like v2 columns may be excluded by the selector | CSV headers and source-faithful/no-correction policy, [`memematch_embeddings.py:61`](../../backend/app/memematch_embeddings.py:61) |
| Corpus and evaluation coverage | 281 default items / 1,000 optional items, no released relevance benchmark, and limited human sample make overfitting easy | catalog configuration and Phase 7A metrics |
| Safety and diversity are narrow heuristics | Keyword filtering misses image-only/contextual risk; filename/template grouping may over- or under-collapse variants | [`recommendations.py:232`](../../backend/app/recommendations.py:232), [`recommendations.py:355`](../../backend/app/recommendations.py:355) |

## 8. Dataset fields and provenance for experiments

**MemeMatch retrieval artifacts.** Catalog records preserve stable source keys, source digest, exporter configuration, text-field provenance, models, dimensions, and completion status. Available CSV content includes local/global context, scope-specific 14-dimensional annotation families (11 emotion-like labels plus positive/negative/neutral), usage categories, and—on the 281 export—Reddit score/timestamp/topics. v2 additionally carries source-key/template/path-review fields and text/caption-like columns noted above. The artifacts report the source CSV locations and SHA-256 values; preserve these in every experiment run.

**MOOD / ALFRED emotion data.** Local archive inspection found 10,005 rows with original image name, six binary label/not-label columns, and three-class sentiment; no authoritative split column. The bounded private research sample extracted/ingested 50 images; later functional evaluation used 200 images. License remains unknown and the project says source images/derived artifacts are Git-ignored ([`docs/dataset-acquisition-report.md`](../dataset-acquisition-report.md)). Do not treat it as a licensed production training set without fresh review.

**Uncertainties.** MemeMatch's public license is still recorded as unknown in project documentation; original corpus scale and dataset-card details should be verified from the approved source before any expansion. Topic semantics, usage-label quality, local/global annotation provenance, and whether all v2 caption/title fields were embedded are not established by current evaluation.

## 9. Hardware, caching, and inference limitations

The ordinary recommendation path lazy-loads MiniLM via `from_pretrained` without `local_files_only`; a cold environment may try to obtain it from the normal Hugging Face cache rather than remaining offline. BLIP and EasyOCR are likewise lazy and normally download/cache weights on first invocation ([`multimodal.py:31`](../../backend/app/multimodal.py:31), [`docs/model-setup.md`](../model-setup.md)). The cross encoder is the exception: it is cache-only and safely falls back.

CPU is the documented default. GPU/CUDA needs a compatible user-installed PyTorch build; project guidance says not to load EasyOCR and BLIP together on the 4 GB GPU ([`docs/model-setup.md`](../model-setup.md)). Existing exports were produced on CUDA with batches 16 (281) and 8 (1,000), but that is artifact provenance rather than a current host requirement. Serving requires local FAISS indexes, matching metadata/export files, a compatible MiniLM cache for text queries, and optional local images only to serve image files. There is no measured memory, throughput, or concurrency budget.

## Research baseline conclusion

The current system is a reproducible, local, semantic-first catalog ranker with modest source-metadata affect adjustment. It has enough provenance and a small completed human comparison to support careful ablations, but not enough labeled relevance data or calibration evidence to justify a new recommendation model claim yet. Phase 7R experiments should preserve catalog/version/field manifests, start with the existing five-query human protocol as a smoke check, and add a held-out, multi-rater evaluation plan before optimizing to the observed Phase 7A numbers.
