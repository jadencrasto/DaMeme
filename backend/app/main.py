import asyncio
import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import router as api_router
from app.core.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="DaMeme API", version="0.1.0", docs_url="/docs", redoc_url=None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.middleware("http")
    async def recommendation_timeout(request: Request, call_next):
        if request.url.path in {
            f"{settings.api_v1_prefix}/recommendations/text",
            f"{settings.api_v1_prefix}/recommendations/emotion",
        }:
            try:
                return await asyncio.wait_for(call_next(request), timeout=settings.recommendation_request_timeout_seconds)
            except TimeoutError:
                return JSONResponse(status_code=504, content={"detail": "Recommendation request timed out. Please try again."})
        return await call_next(request)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled API error", exc_info=exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    return app


app = create_app()
