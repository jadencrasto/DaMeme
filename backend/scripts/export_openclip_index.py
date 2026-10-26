"""Resumable, offline OpenCLIP image exporter for the fixed MemeMatch-1000 catalog."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np
import open_clip
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CATALOG = ROOT / "data" / "retrieval" / "catalogs" / "memematch_1000_v2"
MODEL, PRETRAINED, DIMENSION = "ViT-B-32-quickgelu", "openai", 512


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save_checkpoint(path: Path, vectors: np.ndarray, keys: list[str], image_hashes: list[str]) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, vectors=vectors, source_keys=np.array(keys), image_hashes=np.array(image_hashes))
    os.replace(temporary, path)


def load_catalog() -> tuple[list[str], list[Path], list[str], str]:
    csv_path = CATALOG / "memematch_1000_metadata_v2.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys = [row["filename"] for row in rows]
    paths = [CATALOG / "images" / key for key in keys]
    if len(keys) != 1000 or len(set(keys)) != 1000 or any(not path.is_file() for path in paths):
        raise ValueError("catalog must contain 1,000 unique, readable image paths")
    return keys, paths, [digest(path) for path in paths], digest(csv_path)


def export(output: Path, batch_size: int = 8) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    keys, paths, hashes, csv_hash = load_catalog()
    checkpoint = output / "progress.npz"
    vectors = np.full((len(keys), DIMENSION), np.nan, dtype=np.float32)
    if checkpoint.exists():
        with np.load(checkpoint) as saved:
            if list(saved["source_keys"]) != keys or list(saved["image_hashes"]) != hashes:
                raise ValueError("stale or mismatched progress checkpoint")
            vectors = np.asarray(saved["vectors"], dtype=np.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED, device=device)
    model.eval()
    failures: list[dict[str, str]] = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    pending = [index for index in range(len(keys)) if not np.isfinite(vectors[index]).all()]
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        images = []
        good = []
        for index in batch:
            try:
                with Image.open(paths[index]) as source:
                    images.append(preprocess(source.convert("RGB")))
                good.append(index)
            except (OSError, ValueError) as error:
                failures.append({"source_key": keys[index], "error": str(error)})
        if images:
            try:
                with torch.inference_mode():
                    encoded = model.encode_image(torch.stack(images).to(device))
                    encoded = encoded / encoded.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                vectors[good] = encoded.cpu().numpy().astype(np.float32)
            except torch.cuda.OutOfMemoryError as error:
                torch.cuda.empty_cache()
                raise RuntimeError(f"CUDA OOM at batch size {len(images)}; restart with a smaller batch") from error
        if offset % (batch_size * 10) == 0:
            save_checkpoint(checkpoint, vectors, keys, hashes)
    save_checkpoint(checkpoint, vectors, keys, hashes)
    if failures or not np.isfinite(vectors).all():
        (output / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        raise RuntimeError(f"embedding failed for {len(failures)} images; no completed index written")
    np.savez_compressed(output / "embeddings.tmp.npz", embeddings=vectors)
    os.replace(output / "embeddings.tmp.npz", output / "embeddings.npz")
    index = faiss.IndexFlatIP(DIMENSION)
    index.add(vectors)
    faiss.write_index(index, str(output / "openclip.tmp.faiss"))
    os.replace(output / "openclip.tmp.faiss", output / "openclip.faiss")
    mapping = [{"position": index, "source_key": key, "image_path": str(path.relative_to(ROOT)), "image_sha256": image_hash} for index, (key, path, image_hash) in enumerate(zip(keys, paths, hashes, strict=True))]
    (output / "source_keys.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    elapsed = time.perf_counter() - started
    manifest = {"format_version": 1, "variant": "phase7r-openclip-image-v1", "catalog": {"metadata_csv": str(CATALOG / "memematch_1000_metadata_v2.csv"), "metadata_sha256": csv_hash, "image_count": len(keys)}, "model": {"architecture": MODEL, "pretrained": PRETRAINED, "dimension": DIMENSION, "normalization": "l2", "preprocessing": "open_clip.create_model_and_transforms returned transform"}, "environment": {"torch": torch.__version__, "open_clip": getattr(open_clip, "__version__", "2.32.0"), "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None, "batch_size": batch_size}, "performance": {"elapsed_seconds": elapsed, "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()) if device == "cuda" else None}, "artifacts": {name: digest(output / name) for name in ("embeddings.npz", "openclip.faiss", "source_keys.json")}, "created_at": datetime.now(UTC).isoformat()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    checkpoint.unlink(missing_ok=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(export(args.output, args.batch_size), indent=2))
