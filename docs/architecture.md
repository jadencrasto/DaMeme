# Architecture

The browser and API are independently deployable. The API owns request validation, metadata orchestration, and recommendation contracts; it does not load models in Phase 1. Offline processing will own expensive OCR, captions, embeddings, quality checks, and index construction. This prevents query latency from being tied to ingestion and enables restartable batch jobs.

```mermaid
flowchart LR
  UI[React + Vite] --> API[FastAPI /api/v1]
  API --> META[(SQLite now / PostgreSQL later)]
  API --> INDEX[FAISS index]
  OFFLINE[Offline ingestion jobs] --> OCR[OCR + captions]
  OCR --> EMBED[Embeddings + emotion features]
  EMBED --> META
  EMBED --> INDEX
  ASSETS[Local assets now / object storage later] --> OFFLINE
```

```mermaid
sequenceDiagram
  participant U as User
  participant A as API
  participant V as Vector index
  participant M as Metadata store
  U->>A: future text or emotion query
  A->>V: semantic candidate retrieval
  V-->>A: candidate identifiers
  A->>M: fetch metadata/features
  A-->>A: future emotion-aware rerank + explanation
  A-->>U: ranked, attributed results
```

Metadata, binary assets, vectors, model artifacts, and processing state have distinct storage boundaries. SQLite is adequate for the academic MVP; migrate metadata to PostgreSQL when concurrent writers, richer querying, or service deployment demand it. FAISS remains a replaceable vector-store boundary. For 100K+ memes, use versioned shards/index snapshots, incremental ingestion, idempotency keys, content hashes, checkpoints, and failure records. No background worker is implemented in Phase 1.
