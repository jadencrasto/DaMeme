"""Bounded, local-only extraction for a MOOD/ALFRED research sample."""

import argparse
import csv
import io
import json
import shutil
import tempfile
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

MOOD_SOURCE_URL = "https://github.com/shiv6891/ALFRED_MemeEmotionDetection"
EMOTIONS = ("fear", "anger", "joy", "sadness", "surprise", "disgust")


def _safe_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe archive member: {name}")
    return path


def _positive_emotions(row: dict[str, str]) -> list[str]:
    return [emotion for emotion in EMOTIONS if row.get(emotion) == emotion]


def _select_rows(rows: Iterable[dict[str, str]], limit: int) -> list[dict[str, str]]:
    """Round-robin positive labels to retain a small, category-spread sample."""
    if limit < 1 or limit > 200:
        raise ValueError("MOOD development sample limit must be between 1 and 200")
    buckets = {emotion: [] for emotion in EMOTIONS}
    for row in rows:
        for emotion in _positive_emotions(row):
            buckets[emotion].append(row)
    selected, seen = [], set()
    while len(selected) < limit:
        added = False
        for emotion in EMOTIONS:
            while buckets[emotion]:
                row = buckets[emotion].pop(0)
                filename = row["image"]
                if filename not in seen:
                    selected.append(row)
                    seen.add(filename)
                    added = True
                    break
            if len(selected) == limit:
                break
        if not added:
            break
    return selected


def _metadata(rows: Iterable[dict[str, str]]) -> dict[str, dict[str, object]]:
    return {
        row["image"]: {
            "source_url": MOOD_SOURCE_URL,
            "emotion_label": ",".join(_positive_emotions(row)),
            "label_provenance": "MOOD final.csv",
            "annotations": {
                "filename": row["image"],
                "emotions": {emotion: row[emotion] for emotion in EMOTIONS},
                "sentiment": row["sentiment"],
            },
        }
        for row in rows
    }


def extract_mood_sample(
    archive: Path, output_directory: Path, metadata_path: Path, limit: int = 50
) -> int:
    """Extract only selected final.zip members, preserving final.csv annotations."""
    output_directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as outer:
        rows = list(
            csv.DictReader(
                io.TextIOWrapper(outer.open("meme_emotion/final.csv"), encoding="utf-8-sig")
            )
        )
        selected = _select_rows(rows, limit)
        with outer.open("meme_emotion/final.zip") as nested_stream:
            with tempfile.SpooledTemporaryFile(max_size=16 * 1024 * 1024) as temporary:
                shutil.copyfileobj(nested_stream, temporary)
                temporary.seek(0)
                with zipfile.ZipFile(temporary) as images:
                    available = set(images.namelist())
                    for row in selected:
                        filename = row["image"]
                        member = _safe_member(f"final/{filename}")
                        if member.as_posix() not in available:
                            raise ValueError(f"final.zip does not contain {filename}")
                        destination = output_directory / filename
                        if not destination.exists():
                            with (
                                images.open(member.as_posix()) as source,
                                destination.open("wb") as target,
                            ):
                                shutil.copyfileobj(source, target)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(_metadata(selected), indent=2), encoding="utf-8")
    return len(selected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract a bounded MOOD research sample")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args(argv)
    print(extract_mood_sample(args.archive, args.output, args.metadata, args.limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
