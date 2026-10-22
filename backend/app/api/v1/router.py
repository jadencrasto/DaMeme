from fastapi import APIRouter

from app.api.v1.health import router as health_router
from app.api.v1.multimodal import router as multimodal_router

router = APIRouter()
router.include_router(health_router)
router.include_router(multimodal_router)
