# Research-to-product mapping

| Component | Research source | Original method | DaMeme adaptation | Implementation phase |
| --- | --- | --- | --- | --- |
| OCR/local text context | MemeMatch direction | Local textual meme context | Offline, provenance-tracked OCR field | 3 |
| Global visual context | MemeMatch direction | Global image/template context | Caption/template features stored separately from OCR | 3 |
| Multimodal retrieval | MemeMatch direction | Multimodal retrieval system | Semantic vector retrieval followed by metadata ranking | 5–6 |
| Emotion labels | MOOD / ALFRED | Six basic-emotion dataset | Evaluate compatible labels only after dataset review | 4 |
| Emotion fusion | ALFRED | Emotion-enriched visual cues and gated cross-modal fusion | Reproducible baseline before any simplified fusion | 4 |
| Recommendation reranking | DaMeme | Not claimed as a paper result | Compare semantic-only retrieval with emotion-aware reranking | 6, 8 |
| User uploads | DaMeme | Not covered in this Phase 1 research mapping | Separate consent, validation, provenance pipeline | Later than 7 |
