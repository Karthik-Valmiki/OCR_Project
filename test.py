from pathlib import Path
from PIL import Image
from app.services.ocr_service import OCRService

BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "app" / "sample_1.png"

image = Image.open(IMAGE_PATH)

result = OCRService.extract_text(image, model_type="printed")

print(result)
