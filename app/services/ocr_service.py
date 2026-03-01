from PIL import Image
from app.models.trocr_model import trocr_model
from app.utils.image_utils import preprocess_image
import time

"""
feat(service): implement OCRService orchestration with preprocessing and inference timing
"""


class OCRService:

    @staticmethod
    def extract_text(image: Image.Image, model_type: str = "printed"):

        if not isinstance(image, Image.Image):
            raise ValueError("Invalid image input")

        try:
            processed_image = preprocess_image(image)

            start = time.time()
            text = trocr_model.predict(processed_image, model_type=model_type)
            end = time.time()

            print(f"[INFO] OCR inference took {end - start:.2f} seconds")

            return {"status": "success", "model_used": model_type, "text": text.strip()}

        except Exception as e:
            raise RuntimeError(f"OCR processing failed: {str(e)}")
