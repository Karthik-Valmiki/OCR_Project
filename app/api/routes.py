from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image
from app.services.ocr_service import OCRService
from app.core.schemas import OCRResponse
import io

router = APIRouter()


@router.post("/ocr", response_model=OCRResponse)
async def run_ocr(file: UploadFile = File(...), model_type: str = Form("printed")):

    # Validate Model type
    if model_type not in ["printed", "handwritten"]:
        raise HTTPException(
            status_code=422, detail="model_type must be 'printed' or 'handwritten'"
        )

    # Validate File type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded image must be an image")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        result = OCRService.extract_text(image, model_type)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
