from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.emotion import EmotionService

router = APIRouter(tags=["emotion"])


class TextEmotionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)


def database() -> Path:
    value = get_settings().database_url
    if not value.startswith("sqlite:///"):
        raise HTTPException(503, "Only SQLite emotion storage is configured")
    return Path(value.removeprefix("sqlite:///"))


@router.get("/models/emotion")
def models() -> list[dict[str, object]]:
    service = EmotionService(database())
    try:
        return [
            dict(row)
            for row in service.connection.execute(
                "SELECT id,model_name,model_version,feature_version,trained_at FROM emotion_models ORDER BY id DESC"
            )
        ]
    finally:
        service.close()


@router.get("/memes/{meme_id}/emotion")
def get_prediction(meme_id: int) -> dict[str, object]:
    service = EmotionService(database())
    try:
        row = service.connection.execute(
            "SELECT * FROM emotion_predictions WHERE meme_id=? ORDER BY id DESC LIMIT 1", (meme_id,)
        ).fetchone()
        if not row:
            raise HTTPException(404, "Emotion prediction not found")
        return {
            "meme_id": meme_id,
            "status": row["status"],
            "emotion_probabilities": __import__("json").loads(row["emotion_probabilities_json"]),
            "sentiment": __import__("json").loads(row["sentiment_probabilities_json"]),
            "predicted_at": row["predicted_at"],
            "cached": True,
        }
    finally:
        service.close()


@router.post("/memes/{meme_id}/predict-emotion")
def predict(meme_id: int) -> dict[str, object]:
    service = EmotionService(database())
    try:
        return {"meme_id": meme_id, **service.predict(meme_id)}
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    finally:
        service.close()


@router.post("/emotion/text")
def predict_text(request: TextEmotionRequest) -> dict[str, object]:
    """Infer from an exported text-only model; image fusion is not applicable to a sentence."""
    settings = get_settings()
    if settings.expanded_emotion_text_model is None:
        raise HTTPException(503, "No expanded text emotion model is configured")
    try:
        from app.emotion import PretrainedFeatureExtractor
        from app.expanded_emotion import predict_text_embedding

        embedding = PretrainedFeatureExtractor().text(request.text)
        return predict_text_embedding(settings.expanded_emotion_text_model, embedding)
    except (FileNotFoundError, ValueError, OSError, ImportError) as exc:
        raise HTTPException(503, str(exc)) from exc
