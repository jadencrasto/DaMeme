from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request

from app.core.config import get_settings
from app.product_feedback import (
    DuplicateProductFeedback,
    ProductFeedback,
    feedback_analytics,
    record_product_feedback,
)
from app.schemas.recommendations import FeedbackRequest, FeedbackResponse

router = APIRouter(tags=["product feedback"])


@router.post("/feedback", response_model=FeedbackResponse, status_code=201)
def submit_feedback(payload: FeedbackRequest) -> FeedbackResponse:
    database_url = get_settings().database_url
    if not database_url.startswith("sqlite:///"):
        raise HTTPException(503, "Product feedback requires a local SQLite database.")
    try:
        record_product_feedback(Path(database_url.removeprefix("sqlite:///")), ProductFeedback(payload.request_id, payload.meme_id, payload.selected_mood, payload.retrieval_variant, payload.value))
    except DuplicateProductFeedback as exc:
        raise HTTPException(409, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(503, f"Product feedback storage is unavailable: {exc}") from exc
    return FeedbackResponse(accepted=True)


@router.get("/feedback/analytics")
def product_feedback_analytics(request: Request, limit: int = Query(ge=1, le=50, default=20), offset: int = Query(ge=0, default=0)) -> dict[str, object]:
    """Local diagnostic only; never expose shared feedback from a non-local host."""
    if request.client is None or request.client.host not in {"127.0.0.1", "::1", "testclient"}:
        raise HTTPException(403, "Product feedback analytics are available only on the local development host.")
    database_url = get_settings().database_url
    if not database_url.startswith("sqlite:///"):
        raise HTTPException(503, "Product feedback storage is unavailable.")
    try:
        return feedback_analytics(Path(database_url.removeprefix("sqlite:///")), limit=limit, offset=offset)
    except OSError as exc:
        raise HTTPException(503, "Product feedback storage is unavailable.") from exc
