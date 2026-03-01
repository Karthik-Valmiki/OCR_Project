from pydantic import BaseModel


class OCRResponse(BaseModel):
    status: str
    model_used: str
    text: str


# feat(core): define OCRResponse schema for structured API output
