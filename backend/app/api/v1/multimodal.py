from pathlib import Path

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.multimodal import AnalysisService, BlipCaptionProvider, EasyOcrProvider
from app.schemas.multimodal import AnalysisResponse, ModelInfoResponse

router = APIRouter(tags=["multimodal"])


def _database_path() -> Path:
    value = get_settings().database_url
    if not value.startswith("sqlite:///"):
        raise HTTPException(503, "Only SQLite analysis storage is configured")
    return Path(value.removeprefix("sqlite:///"))


@router.get("/models/multimodal", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        ocr_provider="EasyOCR (optional)", caption_provider="Salesforce BLIP base (optional)"
    )


@router.get("/memes/{meme_id}/analysis", response_model=AnalysisResponse)
def get_analysis(meme_id: int) -> AnalysisResponse:
    service = AnalysisService(_database_path(), EasyOcrProvider(), BlipCaptionProvider())
    try:
        row = service.connection.execute(
            "SELECT * FROM multimodal_analyses WHERE meme_id=? ORDER BY id DESC LIMIT 1", (meme_id,)
        ).fetchone()
        if not row:
            raise HTTPException(404, "Analysis not found")
        return AnalysisResponse(**dict(row), cached=True)
    finally:
        service.close()


@router.post("/memes/{meme_id}/analyze", response_model=AnalysisResponse)
async def analyze_meme(meme_id: int) -> AnalysisResponse:
    def analyze() -> dict[str, object]:
        service = AnalysisService(_database_path(), EasyOcrProvider(), BlipCaptionProvider())
        try:
            return service.analyze_meme(meme_id)
        finally:
            service.close()

    try:
        return AnalysisResponse(**await run_in_threadpool(analyze))
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
