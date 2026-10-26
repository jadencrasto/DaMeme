"""Read-only validator for a completed OpenCLIP image index."""
import argparse
import hashlib
import json
from pathlib import Path

import faiss
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    mapping = json.loads((root / "source_keys.json").read_text(encoding="utf-8"))
    with np.load(root / "embeddings.npz") as payload:
        vectors = payload["embeddings"]
    index = faiss.read_index(str(root / "openclip.faiss"))
    checks = {"count": len(mapping) == 1000, "unique_keys": len({item["source_key"] for item in mapping}) == 1000, "mapping_order": [item["position"] for item in mapping] == list(range(1000)), "image_hashes": all(digest(PROJECT_ROOT / item["image_path"]) == item["image_sha256"] for item in mapping), "shape_dtype": vectors.shape == (1000, 512) and vectors.dtype == np.float32, "finite": bool(np.isfinite(vectors).all()), "unit_norm": bool(np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4)), "faiss": index.ntotal == 1000 and index.d == 512 and bool(np.allclose(index.reconstruct_n(0, 1000), vectors, atol=1e-6)), "provenance": manifest["model"]["architecture"] == "ViT-B-32-quickgelu" and manifest["model"]["pretrained"] == "openai", "hashes": all(digest(root / name) == value for name, value in manifest["artifacts"].items())}
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} | {name}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
