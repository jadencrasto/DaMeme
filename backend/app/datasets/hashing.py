import hashlib
from pathlib import Path

import imagehash
from PIL import Image, ImageOps


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(path: Path) -> str:
    with Image.open(path) as image:
        return str(imagehash.phash(ImageOps.exif_transpose(image)))


def hash_bands(value: str, *, band_count: int = 4) -> list[str]:
    """Return indexable bands; callers compare only candidates sharing a band."""
    width = len(value) // band_count
    return [f"{index}:{value[index * width : (index + 1) * width]}" for index in range(band_count)]


def hamming_distance(first: str, second: str) -> int:
    return (int(first, 16) ^ int(second, 16)).bit_count()
