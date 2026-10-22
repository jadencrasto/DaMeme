from pathlib import Path

from app.datasets.ingestion import DatasetIngester
from app.multimodal import AnalysisService, CaptionResult, OcrResult, OcrToken, combine_context


class FakeOcr:
    def analyze(self, _: Path) -> OcrResult:
        return OcrResult([OcrToken("HELLO", [[0, 0]], 0.9)], "fake-ocr")


class FakeCaption:
    def analyze(self, _: Path) -> CaptionResult:
        return CaptionResult("a synthetic scene", "fake-caption")


def test_context_preserves_ocr_and_caption() -> None:
    assert (
        combine_context(FakeOcr().analyze(Path("x")), FakeCaption().analyze(Path("x")))
        == "Text: HELLO. Visual: a synthetic scene."
    )


def test_analysis_is_cached(tmp_path: Path) -> None:
    image = tmp_path / "meme.png"
    image.write_bytes(b"not used by fake providers")
    ingester = DatasetIngester(tmp_path / "data.db")
    # Insert a minimal registered meme directly: providers are mocked and never decode it.
    ingester.connection.execute(
        "INSERT INTO dataset_sources(name,version,source_url,license_status,retrieval_date) VALUES ('synthetic','v1','https://example.test','verified','now')"
    )
    ingester.connection.execute(
        "INSERT INTO memes(dataset_source_id,original_source_id,local_image_path,sha256,annotation_json,license_status,ingestion_timestamp,processing_status,dataset_version) VALUES (1,'one',?,?, '{}','verified','now','validated','v1')",
        (str(image), "hash"),
    )
    ingester.connection.commit()
    service = AnalysisService(tmp_path / "data.db", FakeOcr(), FakeCaption())
    first, second = service.analyze_meme(1), service.analyze_meme(1)
    assert first["combined_context"] == "Text: HELLO. Visual: a synthetic scene."
    assert second["cached"] is True
    service.close()
    ingester.close()
