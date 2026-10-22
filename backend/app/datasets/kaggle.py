"""Explicit, bounded access helpers for the official MemeMatch Kaggle dataset."""

from collections.abc import Callable
from pathlib import Path

MEMEMATCH_HANDLE = "trianle171/memematch-v1-0"


def download_dataset_file(
    *, relative_path: str, downloader: Callable[..., str] | None = None
) -> Path:
    """Download one dataset-relative file without a full-dataset fallback.

    The caller must check source terms before invoking this function. KaggleHub
    documents ``path`` as an optional file path inside the requested dataset.
    """
    if not relative_path or Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
        raise ValueError("relative_path must be a safe dataset-relative path")
    if downloader is None:
        import kagglehub

        downloader = kagglehub.dataset_download
    return Path(downloader(MEMEMATCH_HANDLE, path=relative_path))
