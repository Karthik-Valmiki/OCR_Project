from pydantic import BaseModel


class OCRResponse(BaseModel):
    status: str
    model_used: str
    text: str
