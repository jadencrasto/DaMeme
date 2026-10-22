import pytest

from app.datasets.kaggle import MEMEMATCH_HANDLE, download_dataset_file


def test_kaggle_helper_requests_one_relative_dataset_file() -> None:
    requested: dict[str, str] = {}

    def fake_download(handle: str, *, path: str) -> str:
        requested.update(handle=handle, path=path)
        return "C:/cache/local_context.csv"

    result = download_dataset_file(
        relative_path="annotations/annotations/local_context.csv", downloader=fake_download
    )

    assert result.name == "local_context.csv"
    assert requested == {
        "handle": MEMEMATCH_HANDLE,
        "path": "annotations/annotations/local_context.csv",
    }


def test_kaggle_helper_rejects_unsafe_relative_path(tmp_path) -> None:
    with pytest.raises(ValueError, match="safe"):
        download_dataset_file(relative_path="../dataset.zip")
