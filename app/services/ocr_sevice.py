from PIL import Image
from app.models.trocr_model import trocr_model
from app.utils.image_utils import preprocess_image


class OCRService:

    @staticmethod
    def extract_text(image: Image.Image, model_type: str = "printed"):

        if image is None:
            raise ValueError("Image cannot be None")

        # Preprocess
        processed_image = preprocess_image(image)

        # Predict
        text = trocr_model.predict(processed_image, model_type=model_type)

        # Minimal cleanup
        text = text.strip()

        return {"status": "success", "model_used": model_type, "text": text}
