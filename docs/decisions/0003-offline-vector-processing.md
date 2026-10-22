# ADR 0003: Offline processing and FAISS boundary

**Decision:** create embeddings/indexes offline and query through a replaceable FAISS boundary. **Why:** predictable online latency and resumable ingestion. **Consequence:** data/index versions and input provenance are mandatory in later phases.
