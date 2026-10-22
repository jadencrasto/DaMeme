from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP", "GIF"}


class ImageValidationError(ValueError):
    """A candidate file cannot safely be accepted as an image."""


@dataclass(frozen=True)
class ImageInfo:
    image_format: str
    width: int
    height: int
    file_size: int


def validate_image(path: Path, *, max_bytes: int, max_pixels: int) -> ImageInfo:
    """Decode and inspect a local image without retaining decoded pixel buffers."""
    if not path.is_file():
        raise ImageValidationError("file does not exist or is not a regular file")
    if path.stat().st_size > max_bytes:
        raise ImageValidationError(f"file exceeds configured maximum of {max_bytes} bytes")
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            normalized = ImageOps.exif_transpose(image)
            image_format = image.format or "UNKNOWN"
            width, height = normalized.size
    except (OSError, UnidentifiedImageError, SyntaxError) as exc:
        raise ImageValidationError(f"image decoding failed: {exc}") from exc
    if image_format not in SUPPORTED_FORMATS:
        raise ImageValidationError(f"unsupported image format: {image_format}")
    if width < 1 or height < 1:
        raise ImageValidationError("image has invalid dimensions")
    if width * height > max_pixels:
        raise ImageValidationError(f"image exceeds configured maximum of {max_pixels} pixels")
    return ImageInfo(
        image_format=image_format, width=width, height=height, file_size=path.stat().st_size
    )
