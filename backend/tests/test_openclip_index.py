import json
from pathlib import Path

import faiss
import numpy as np

ARTIFACT = Path(__file__).resolve().parents[1] / "benchmarks" / "phase7r_v1_openclip_image_index"


def test_openclip_index_mapping_and_vectors_are_aligned() -> None:
    manifest = json.loads((ARTIFACT / "manifest.json").read_text(encoding="utf-8"))
    mapping = json.loads((ARTIFACT / "source_keys.json").read_text(encoding="utf-8"))
    with np.load(ARTIFACT / "embeddings.npz") as payload:
        vectors = payload["embeddings"]
    index = faiss.read_index(str(ARTIFACT / "openclip.faiss"))
    assert manifest["model"]["dimension"] == 512
    assert manifest["model"]["pretrained"] == "openai"
    assert len(mapping) == len({item["source_key"] for item in mapping}) == 1000
    assert [item["position"] for item in mapping] == list(range(1000))
    assert vectors.shape == (1000, 512)
    assert vectors.dtype == np.float32
    assert np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
    assert index.ntotal == 1000 and index.d == 512
    assert np.allclose(index.reconstruct_n(0, 1000), vectors, atol=1e-6)
