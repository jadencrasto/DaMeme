from pydantic import BaseModel


class ModelInfoResponse(BaseModel):
    ocr_provider: str
    caption_provider: str
    lazy_loading: bool = True


class AnalysisResponse(BaseModel):
    meme_id: int
    status: str
    ocr_text: str = ""
    visual_caption: str = ""
    combined_context: str = ""
    cached: bool = False
