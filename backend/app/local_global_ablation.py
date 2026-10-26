"""Offline Phase 7R.4 local/global text representation audit utilities."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from app.emotion import PretrainedFeatureExtractor
from app.memematch_embeddings import _field, _verified_template, clean_source_text, stable_meme_id


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ScopedText:
    source_key: str
    local: str
    global_: str
    combined: str
    fields: dict[str, str]


def _join(parts: list[tuple[str, tuple[str, str] | None]]) -> tuple[str, dict[str, str]]:
    used = [(label, value) for label, value in parts if value]
    values: list[str] = []
    fields: dict[str, str] = {}
    seen: set[str] = set()
    for label, value in used:
        assert value
        normalized = value[1]
        if normalized.casefold() in seen:
            continue
        seen.add(normalized.casefold())
        values.append(f"{label}: {normalized}")
        fields[label] = value[0]
    return "\n\n".join(values), fields


def scoped_text(row: dict[str, str]) -> ScopedText:
    """A0–A3 source-faithful text scopes; never infers a caption or template."""
    key = stable_meme_id(row)
    local, local_fields = _join([
        ("Local OCR", _field(row, "cleaned_ocr", "ocr_text", "ocr", "local_ocr")),
        ("Local context", _field(row, "local_context")),
        ("Local title", _field(row, "title", "meme_title")),
    ])
    global_, global_fields = _join([
        ("Global context", _field(row, "global_context")),
        ("Verified template", _verified_template(row)),
    ])
    combined, _ = _join([
        ("Local OCR", _field(row, "cleaned_ocr", "ocr_text", "ocr", "local_ocr")),
        ("Local context", _field(row, "local_context")),
        ("Caption or title", _field(row, "caption", "title", "meme_title")),
        ("Verified template", _verified_template(row)),
        ("Global context", _field(row, "global_context")),
    ])
    return ScopedText(key, local, global_, combined, {**{f"local:{k}": v for k, v in local_fields.items()}, **{f"global:{k}": v for k, v in global_fields.items()}})


def overlap(left: str, right: str) -> str:
    left, right = clean_source_text(left), clean_source_text(right)
    if not left or not right:
        return "missing"
    if left.casefold() == right.casefold():
        return "exact"
    if left.casefold() in right.casefold() or right.casefold() in left.casefold():
        return "containment"
    tokens = set(left.casefold().split()) & set(right.casefold().split())
    return "partial" if tokens else "distinct"


def field_audit(metadata_csv: Path, output: Path) -> dict[str, object]:
    with metadata_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    comparisons = (("local_context", "local_Extracted Text + Title"), ("global_context", "global_Caption"))
    report: dict[str, object] = {"format_version": 1, "source_sha256": _digest(metadata_csv), "rows": len(rows), "comparisons": {}}
    for left, right in comparisons:
        counts = {name: 0 for name in ("exact", "containment", "partial", "distinct", "missing")}
        examples: dict[str, list[dict[str, str]]] = {name: [] for name in counts}
        for row in rows:
            result = overlap(row.get(left, ""), row.get(right, ""))
            counts[result] += 1
            if len(examples[result]) < 3:
                examples[result].append({"source_key": stable_meme_id(row), left: row.get(left, ""), right: row.get(right, "")})
        report["comparisons"][f"{left}__{right}"] = {"counts": counts, "examples": examples}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def build_indexes(metadata_csv: Path, output: Path, *, batch_size: int = 32) -> dict[str, object]:
    """Build A1/A2 vectors and FlatIP indexes without touching production artifacts."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    with metadata_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    scopes = [scoped_text(row) for row in rows]
    keys = [item.source_key for item in scopes]
    if len(keys) != 1000 or len(set(keys)) != len(keys):
        raise ValueError("ablation requires 1,000 unique source keys")
    extractor = PretrainedFeatureExtractor("cpu")

    def encode(values: list[str]) -> tuple[np.ndarray, list[bool]]:
        available = [bool(value.strip()) for value in values]
        vectors = np.zeros((len(values), 384), dtype=np.float32)
        positions = [index for index, value in enumerate(available) if value]
        for start in range(0, len(positions), batch_size):
            batch = positions[start : start + batch_size]
            vectors[batch] = extractor.embed_texts([values[index] for index in batch])
        return vectors, available

    local, local_available = encode([item.local for item in scopes])
    global_, global_available = encode([item.global_ for item in scopes])
    for name, vectors, available in (("local", local, local_available), ("global", global_, global_available)):
        if vectors.shape != (1000, 384) or not np.isfinite(vectors).all():
            raise ValueError(f"{name} embedding shape or values are invalid")
        norms = np.linalg.norm(vectors, axis=1)
        if any(not np.isclose(norms[index], 1.0, atol=1e-4) for index, value in enumerate(available) if value):
            raise ValueError(f"{name} nonempty vectors must be normalized")
        if any(norms[index] != 0 for index, value in enumerate(available) if not value):
            raise ValueError(f"{name} empty fields must be zero vectors")
    output.mkdir(parents=True, exist_ok=True)
    for name, vectors in (("local", local), ("global", global_)):
        temporary = output / f"{name}.tmp.npz"
        np.savez_compressed(temporary, embeddings=vectors)
        os.replace(temporary, output / f"{name}.embeddings.npz")
        index = faiss.IndexFlatIP(384)
        index.add(vectors)
        temporary_index = output / f"{name}.tmp.faiss"
        faiss.write_index(index, str(temporary_index))
        os.replace(temporary_index, output / f"{name}.faiss")
    manifest = {
        "format_version": 1, "variant": "phase7r-local-global-a1-a3", "source_csv": str(metadata_csv),
        "source_sha256": _digest(metadata_csv), "model_id": extractor.text_model_id,
        "encoder": {"pooling": "attention-mask mean", "max_length": 256, "normalization": "l2", "device": "cpu"},
        "source_keys": keys, "dimensions": 384, "counts": {"local": 1000, "global": 1000},
        "availability": {"local": local_available, "global": global_available},
        "representations": {"A0": "existing v2 production combined index (referenced, not regenerated)",
                              "A1": "local", "A2": "global", "A3": "reuse A1/A2 vectors; fuse at retrieval time"},
        "artifact_sha256": {name: _digest(output / name) for name in ("local.embeddings.npz", "global.embeddings.npz", "local.faiss", "global.faiss")},
    }
    temporary_manifest = output / "manifest.tmp.json"
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, output / "manifest.json")
    return {"output": str(output), "local_missing": local_available.count(False), "global_missing": global_available.count(False)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-indexes", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    result = build_indexes(args.metadata_csv, args.output, batch_size=args.batch_size) if args.build_indexes else field_audit(args.metadata_csv, args.output)
    print(json.dumps(result, indent=2))
