from fastapi import APIRouter

from app.moods import MOODS
from app.schemas.recommendations import MoodResponse, MoodTaxonomyResponse

router = APIRouter(tags=["moods"])


@router.get("/emotions", response_model=MoodTaxonomyResponse)
def emotions() -> MoodTaxonomyResponse:
    return MoodTaxonomyResponse(moods=[MoodResponse(id=mood.id, label=mood.label, group=mood.group, description=mood.description, synonyms=list(mood.synonyms), classifier_weights=mood.classifier_weights, sentiment=mood.sentiment) for mood in MOODS])
