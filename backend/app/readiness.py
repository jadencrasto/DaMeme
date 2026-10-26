"""Local-only readiness checks for the fixed A0 catalog."""
from __future__ import annotations

from pathlib import Path

from app.core.config import Settings
from app.datasets.database import connect, migrate
from app.recommendation_engine import RecommendationEngine


def runtime_readiness(settings: Settings, engine: RecommendationEngine) -> dict[str, object]:
    errors: list[str] = []
    database_ready = False
    if not settings.database_url.startswith("sqlite:///"):
        errors.append("DAMEME_DATABASE_URL must use a local sqlite:/// path.")
    else:
        try:
            connection = connect(Path(settings.database_url.removeprefix("sqlite:///")))
            migrate(connection)
            connection.close()
            database_ready = True
        except OSError as exc:
            errors.append(f"Database unavailable: {exc}")
    try:
        from transformers import AutoConfig
        AutoConfig.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
        model_ready = True
    except Exception:
        model_ready = False
        errors.append("MiniLM is not in the local cache; provision it before starting offline.")
    catalog = settings.active_catalog()
    status = engine.status()
    if not status["ready"]:
        errors.append(str(status["detail"]))
    image_count = 0
    if catalog.image_root is None:
        errors.append("DAMEME_RECOMMENDATION_IMAGE_ROOT is not configured for the 281-item catalog.")
    elif not catalog.image_root.is_dir():
        errors.append(f"Configured recommendation image root does not exist: {catalog.image_root}")
    elif status["ready"] and hasattr(engine, "available_image_count"):
        image_count = engine.available_image_count()
        if image_count == 0:
            errors.append("No catalog images resolve below DAMEME_RECOMMENDATION_IMAGE_ROOT.")
    return {"ready": not errors, "detail": " ".join(errors) if errors else None, "catalog_id": catalog.catalog_id, "meme_count": status["meme_count"], "image_count": image_count, "database_ready": database_ready, "model_ready": model_ready}
