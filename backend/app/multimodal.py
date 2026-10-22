"""Lazy, replaceable OCR/caption providers and cached Phase 3 analysis."""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from app.datasets.database import connect, migrate

PIPELINE_VERSION = "phase3-v1"


@dataclass(frozen=True)
class OcrToken:
    text: str
    box: list[list[float]]
    confidence: float


@dataclass(frozen=True)
class OcrResult:
    tokens: list[OcrToken]
    provider: str

    @property
    def text(self) -> str:
        return " ".join(token.text for token in self.tokens).strip()


@dataclass(frozen=True)
class CaptionResult:
    text: str
    provider: str


class OcrProvider(Protocol):
    def analyze(self, image: Path) -> OcrResult: ...


class CaptionProvider(Protocol):
    def analyze(self, image: Path) -> CaptionResult: ...


class EasyOcrProvider:
    def __init__(self, device: str = "cpu"):
        self.device, self._reader = device, None

    def analyze(self, image: Path) -> OcrResult:
        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError("EasyOCR is not installed; install the multimodal extra") from exc
        if self._reader is None:
            self._reader = easyocr.Reader(["en"], gpu=self.device == "cuda", verbose=False)
        rows = self._reader.readtext(str(image), detail=1)
        tokens = [
            OcrToken(
                text=row[1],
                box=[[float(coordinate) for coordinate in point] for point in row[0]],
                confidence=float(row[2]),
            )
            for row in rows
        ]
        tokens.sort(
            key=lambda token: (
                min(point[1] for point in token.box),
                min(point[0] for point in token.box),
            )
        )
        return OcrResult(tokens, "easyocr")


class BlipCaptionProvider:
    model_id = "Salesforce/blip-image-captioning-base"

    def __init__(self, device: str = "cpu"):
        self.device, self._processor, self._model = device, None, None

    def analyze(self, image: Path) -> CaptionResult:
        try:
            from PIL import Image
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError as exc:
            raise RuntimeError(
                "BLIP dependencies are not installed; install the multimodal extra"
            ) from exc
        if self._model is None:
            self._processor = BlipProcessor.from_pretrained(self.model_id)
            self._model = BlipForConditionalGeneration.from_pretrained(self.model_id).to(
                self.device
            )
        with Image.open(image) as source:
            inputs = self._processor(images=source.convert("RGB"), return_tensors="pt").to(
                self.device
            )
        import torch

        with torch.inference_mode():
            output = self._model.generate(**inputs, max_new_tokens=20, do_sample=False)
        return CaptionResult(
            self._processor.decode(output[0], skip_special_tokens=True).strip(), self.model_id
        )


def combine_context(ocr: OcrResult, caption: CaptionResult | None) -> str:
    parts = []
    if ocr.text:
        parts.append(f"Text: {ocr.text}.")
    if caption and caption.text:
        parts.append(f"Visual: {caption.text}.")
    return " ".join(parts)


class AnalysisService:
    def __init__(self, database: Path, ocr: OcrProvider, caption: CaptionProvider):
        self.connection, self.ocr, self.caption = connect(database), ocr, caption
        migrate(self.connection)

    def analyze_meme(self, meme_id: int, force: bool = False) -> dict[str, object]:
        meme = self.connection.execute("SELECT * FROM memes WHERE id = ?", (meme_id,)).fetchone()
        if not meme:
            raise KeyError("meme not found")
        config = json.dumps(
            {"ocr": type(self.ocr).__name__, "caption": type(self.caption).__name__}, sort_keys=True
        )
        digest = hashlib.sha256(config.encode()).hexdigest()
        cached = self.connection.execute(
            "SELECT * FROM multimodal_analyses WHERE meme_id=? AND image_sha256=? AND pipeline_version=? AND configuration_digest=?",
            (meme_id, meme["sha256"], PIPELINE_VERSION, digest),
        ).fetchone()
        if cached and not force:
            return {**dict(cached), "cached": True}
        ocr, caption, warnings = self.ocr.analyze(Path(meme["local_image_path"])), None, []
        try:
            caption = self.caption.analyze(Path(meme["local_image_path"]))
        except Exception as exc:
            warnings.append(str(exc))
        status = "completed" if caption else "partial"
        now = datetime.now(UTC).isoformat()
        values = (
            meme_id,
            meme["sha256"],
            PIPELINE_VERSION,
            digest,
            status,
            ocr.text,
            json.dumps([asdict(t) for t in ocr.tokens]),
            (sum(t.confidence for t in ocr.tokens) / len(ocr.tokens)) if ocr.tokens else None,
            caption.text if caption else "",
            combine_context(ocr, caption),
            ocr.provider,
            caption.provider if caption else type(self.caption).__name__,
            config,
            "; ".join(warnings) or None,
            now,
        )
        with self.connection:
            self.connection.execute(
                "INSERT INTO multimodal_analyses(meme_id,image_sha256,pipeline_version,configuration_digest,status,ocr_text,ocr_boxes_json,ocr_confidence,visual_caption,combined_context,ocr_provider,caption_provider,provenance_json,warning_details,processed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(meme_id,image_sha256,pipeline_version,configuration_digest) DO UPDATE SET status=excluded.status,ocr_text=excluded.ocr_text,ocr_boxes_json=excluded.ocr_boxes_json,ocr_confidence=excluded.ocr_confidence,visual_caption=excluded.visual_caption,combined_context=excluded.combined_context,warning_details=excluded.warning_details,processed_at=excluded.processed_at",
                values,
            )
        return {
            "meme_id": meme_id,
            "status": status,
            "ocr_text": ocr.text,
            "visual_caption": caption.text if caption else "",
            "combined_context": combine_context(ocr, caption),
            "cached": False,
        }

    def analyze_batch(self, limit: int, force: bool = False) -> dict[str, int]:
        rows = self.connection.execute(
            "SELECT id FROM memes ORDER BY id LIMIT ?", (limit,)
        ).fetchall()
        counts = {"completed": 0, "partial": 0, "skipped": 0, "failed": 0}
        for row in rows:
            try:
                result = self.analyze_meme(row["id"], force)
                counts["skipped" if result.get("cached") else result["status"]] += 1
            except Exception:
                counts["failed"] += 1
        return counts

    def close(self) -> None:
        self.connection.close()


def analyze_image(image: Path, ocr: OcrProvider, caption: CaptionProvider) -> dict[str, object]:
    if not image.is_file():
        raise FileNotFoundError(image)
    ocr_result, caption_result = ocr.analyze(image), caption.analyze(image)
    return {
        "ocr_text": ocr_result.text,
        "ocr_tokens": [asdict(token) for token in ocr_result.tokens],
        "visual_caption": caption_result.text,
        "combined_context": combine_context(ocr_result, caption_result),
        "ocr_provider": ocr_result.provider,
        "caption_provider": caption_result.provider,
    }
