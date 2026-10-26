"""Validate the checked-in Phase 7R local/global index artifacts.

This script is deliberately read-only: it verifies the existing files and does
not create, replace, or otherwise mutate any artifact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss
except ImportError as error:  # pragma: no cover - depends on the local environment
    print(f"FAIL | CPU FAISS import: {error}")
    print("Run from backend with: rtk uv run python scripts/validate_local_global_indexes.py")
    raise SystemExit(1) from error


ARTIFACT_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "phase7r_v1_local_global_indexes"
)
EXPECTED_SHAPE = (1000, 384)
NORM_ATOL = 1e-4
VECTOR_ATOL = 1e-6
VECTOR_RTOL = 1e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append((name, passed, detail))

    manifest_path = ARTIFACT_DIRECTORY / "manifest.json"
    with manifest_path.open(encoding="utf-8") as manifest_file:
        manifest: dict[str, Any] = json.load(manifest_file)

    required_fields = {"dimensions", "counts", "availability", "artifact_sha256"}
    check(
        "manifest fields",
        required_fields.issubset(manifest),
        f"found: {', '.join(sorted(manifest))}",
    )
    dimensions = manifest["dimensions"]
    counts = manifest["counts"]
    availability = manifest["availability"]
    expected_hashes = manifest["artifact_sha256"]

    embeddings: dict[str, np.ndarray] = {}
    indexes: dict[str, Any] = {}
    for representation in ("local", "global"):
        npz_name = f"{representation}.embeddings.npz"
        index_name = f"{representation}.faiss"
        npz_path = ARTIFACT_DIRECTORY / npz_name
        index_path = ARTIFACT_DIRECTORY / index_name

        with np.load(npz_path) as archive:
            check(
                f"{representation} NPZ member",
                archive.files == ["embeddings"],
                f"members: {archive.files}",
            )
            embeddings[representation] = np.asarray(archive["embeddings"])
        indexes[representation] = faiss.read_index(str(index_path))

        actual_hash = sha256(npz_path)
        check(
            f"{npz_name} SHA-256",
            actual_hash == expected_hashes.get(npz_name),
            actual_hash,
        )
        actual_hash = sha256(index_path)
        check(
            f"{index_name} SHA-256",
            actual_hash == expected_hashes.get(index_name),
            actual_hash,
        )

    for representation in ("local", "global"):
        vectors = embeddings[representation]
        index = indexes[representation]
        available = np.asarray(availability[representation], dtype=bool)

        check(
            f"{representation} embedding shape",
            vectors.shape == EXPECTED_SHAPE,
            f"actual: {vectors.shape}",
        )
        check(
            f"{representation} manifest count/dimension",
            counts.get(representation) == EXPECTED_SHAPE[0] and dimensions == EXPECTED_SHAPE[1],
            f"count={counts.get(representation)}, dimensions={dimensions}",
        )
        check(
            f"{representation} FAISS size/dimension",
            index.ntotal == EXPECTED_SHAPE[0] and index.d == EXPECTED_SHAPE[1],
            f"ntotal={index.ntotal}, d={index.d}",
        )
        check(
            f"{representation} finite vectors",
            bool(np.isfinite(vectors).all()),
            f"non-finite values={np.size(vectors) - np.isfinite(vectors).sum()}",
        )
        check(
            f"{representation} availability length",
            available.shape == (EXPECTED_SHAPE[0],),
            f"actual: {available.shape}",
        )

        norms = np.linalg.norm(vectors[available], axis=1)
        check(
            f"{representation} available unit L2 norms",
            bool(np.allclose(norms, 1.0, atol=NORM_ATOL, rtol=0.0)),
            f"min={norms.min():.8f}, max={norms.max():.8f}, atol={NORM_ATOL}",
        )

        stored_vectors = np.asarray(index.reconstruct_n(0, index.ntotal))
        max_difference = float(np.max(np.abs(stored_vectors - vectors)))
        check(
            f"{representation} FAISS vectors match NPZ",
            bool(np.allclose(stored_vectors, vectors, atol=VECTOR_ATOL, rtol=VECTOR_RTOL)),
            f"max_abs_diff={max_difference:.8g}, atol={VECTOR_ATOL}, rtol={VECTOR_RTOL}",
        )

    local_available = np.asarray(availability["local"], dtype=bool)
    local_unavailable = ~local_available
    check(
        "local unavailable row count",
        int(local_unavailable.sum()) == 4,
        f"unavailable={int(local_unavailable.sum())}",
    )
    check(
        "local unavailable rows are zero",
        bool(np.all(embeddings["local"][local_unavailable] == 0.0)),
        f"max_abs={float(np.abs(embeddings['local'][local_unavailable]).max()):.8g}",
    )
    global_available = np.asarray(availability["global"], dtype=bool)
    check(
        "global rows all available",
        bool(global_available.all()),
        f"available={int(global_available.sum())}/{global_available.size}",
    )

    for name, passed, detail in checks:
        print(f"{'PASS' if passed else 'FAIL'} | {name}: {detail}")
    failures = sum(not passed for _, passed, _ in checks)
    print(f"{'PASS' if not failures else 'FAIL'} | {len(checks) - failures}/{len(checks)} checks passed")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
