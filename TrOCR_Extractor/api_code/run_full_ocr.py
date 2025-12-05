import torch
import pytesseract
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import io
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pydantic import BaseModel, Field  # <-- Added for professional schema definition
import uvicorn
import os
from typing import Literal

# --- 1. Global Configuration ---
# NOTE: Update this path if Tesseract isn't in this location
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
try:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    print(f"Tesseract command set to: {TESSERACT_PATH}")
except Exception:
    print(
        "WARNING: Could not set Tesseract path. Ensure Tesseract is installed or the path is correct."
    )

# --- NEW: Pydantic Models for Schema Definition ---

# 1. Define the possible model choices as a string union (used for both types and validation)
ModelType = Literal["printed", "handwritten"]


# 2. Define the formal response structure for professional API documentation
class OCRResult(BaseModel):
    """Schema for the successful OCR extraction result."""

    extracted_text: str = Field(
        ...,
        description="The final extracted text content from the image.",
        examples=["Flat name: Abcde", "Date: 27-09-2000"],
    )
    model_used: ModelType = Field(
        ...,
        description="The TrOCR model that was used for the extraction.",
        examples=["printed"],
    )


# --- 2. Global Model Initialization (Load both once) ---
try:
    # 1. Printed Model (Best for clean, structured text)
    PRINTED_MODEL_NAME = "microsoft/trocr-base-printed"
    printed_processor = TrOCRProcessor.from_pretrained(PRINTED_MODEL_NAME)
    printed_model = VisionEncoderDecoderModel.from_pretrained(PRINTED_MODEL_NAME)

    # 2. Handwritten Model (Best for natural, free-form script)
    HANDWRITTEN_MODEL_NAME = "microsoft/trocr-base-handwritten"
    handwritten_processor = TrOCRProcessor.from_pretrained(HANDWRITTEN_MODEL_NAME)
    handwritten_model = VisionEncoderDecoderModel.from_pretrained(
        HANDWRITTEN_MODEL_NAME
    )

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    printed_model.to(DEVICE)
    handwritten_model.to(DEVICE)

    # Store both models and processors in a dictionary for easy access
    MODEL_MAP = {
        "printed": (printed_processor, printed_model),
        "handwritten": (handwritten_processor, handwritten_model),
    }

    print(f"Dual TrOCR Models loaded successfully on device: {DEVICE}")
except Exception as e:
    print(f"ERROR: Failed to load one or both TrOCR models: {e}")
    MODEL_MAP = {}


# --- 3. Core Extraction Function ---
def extract_text_from_image_bytes(image_bytes: bytes, model_type: ModelType) -> str:
    """Uses adaptive logic: Tesseract segmentation tailored for printed or handwritten text."""

    if model_type not in MODEL_MAP:
        raise RuntimeError(f"Model type '{model_type}' is not loaded.")

    processor, model = MODEL_MAP[model_type]
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # --- ADAPTIVE SEGMENTATION LOGIC ---
    if model_type == "printed":
        # Printed: Use high-precision line segmentation (PSM 7, Level 4) - Best for structure.
        config_str = "--psm 7 --oem 1"
        detection_levels = [4]
        print(
            "--- DEBUG: Using high-precision PSM 7 (Line) segmentation for PRINTED. ---"
        )
    else:  # model_type == "handwritten"
        # Handwritten: Use PSM 6 (Single Uniform Block) and look for Textlines (Level 3).
        # This is a robust mode for messy script, reducing fragmentation errors.
        config_str = "--psm 6 --oem 1"
        detection_levels = [3]  # Level 3: Textline bounding box
        print(
            "--- DEBUG: Using balanced PSM 6 (Textline) segmentation for HANDWRITTEN. ---"
        )

    # Tesseract pre-processing
    line_data = pytesseract.image_to_data(
        pil_image,
        output_type=pytesseract.Output.DICT,
        config=config_str,
        lang="eng",
    )

    raw_boxes = []

    for i in range(len(line_data["level"])):
        # Filter for the detected line/block level
        if (
            line_data["level"][i] in detection_levels
            and line_data["text"][i].strip() != ""
        ):
            (x, y, w, h) = (
                line_data["left"][i],
                line_data["top"][i],
                line_data["width"][i],
                line_data["height"][i],
            )
            raw_boxes.append({"top": y, "box": (x, y, w, h)})

    print(
        f"--- TESSERACT DEBUG ({model_type}): Found {len(raw_boxes)} boxes for cropping. ---"
    )

    # FALLBACK: If Tesseract finds nothing, return its full raw text (for debugging)
    if not raw_boxes:
        print("--- DEBUG: Falling back to full Tesseract string (PSM 3). ---")
        return pytesseract.image_to_string(
            pil_image, config="--psm 3 --oem 1", lang="eng"
        ).strip()

    # Sort boxes and TrOCR Recognition Loop
    raw_boxes.sort(key=lambda b: b["top"])
    final_boxes = raw_boxes

    texts = []
    Y_THRESHOLD = 15
    last_y_coord = -Y_THRESHOLD

    for box_info in final_boxes:
        (x, y, w, h) = box_info["box"]

        # Insert newline only if there is a significant vertical jump
        if abs(y - last_y_coord) > Y_THRESHOLD:
            texts.append("\n")

        # Crop the line image for TrOCR (with padding)
        padding = 5
        cropped_line = pil_image.crop(
            (
                max(0, x - padding),
                max(0, y - padding),
                min(pil_image.width, x + w + padding),
                min(pil_image.height, y + h + padding),
            )
        )

        # TrOCR Recognition
        pixel_values = processor(
            images=cropped_line, return_tensors="pt"
        ).pixel_values.to(DEVICE)

        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_length=150, num_beams=4)

        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        texts.append(text + " ")
        last_y_coord = y

    # Final cleanup
    final_text = "".join(texts).strip()
    final_text = (
        final_text.replace(" \n ", "\n").replace("\n\n", "\n").replace("\n\n", "\n")
    )

    return final_text


# =========================================================
# === FastAPI Application Layer ===
# =========================================================

app = FastAPI(
    title="Adaptive TrOCR Extractor API",
    description="A robust OCR service that uses two specialized TrOCR models (printed and handwritten) "
    "with adaptive Tesseract segmentation for high accuracy.",
    version="1.0.0",  # Professional touch
)


@app.post(
    "/ocr/extract",
    summary="Extract Text from Image using Selected Model",
    description="Upload a high-quality image and choose 'printed' for structured text "
    "or 'handwritten' for free-form script.",
    response_model=OCRResult,  # Use the defined schema for validation and docs
    status_code=200,
)
async def extract_text_api(
    # --- ENHANCED INPUT DOCUMENTATION ---
    image_file: UploadFile = File(
        ...,
        description="The image file (PNG, JPG) containing the text to be recognized.",
        media_type=["image/png", "image/jpeg"],
    ),
    model_choice: ModelType = Form(
        default="printed",
        description="Select the model type to optimize for: 'printed' or 'handwritten'. "
        "Use 'handwritten' for forms or messy documents.",
    ),
):
    """Handles the file upload and adaptive TrOCR processing."""

    try:
        if not os.path.exists(TESSERACT_PATH):
            raise pytesseract.pytesseract.TesseractNotFoundError

        if not MODEL_MAP:
            raise RuntimeError(
                "Models failed to load at startup. Check console for errors."
            )

        image_bytes = await image_file.read()

        extracted_text = extract_text_from_image_bytes(image_bytes, model_choice)

        # Return dictionary matches the OCRResult model structure
        return {"extracted_text": extracted_text, "model_used": model_choice}

    except pytesseract.pytesseract.TesseractNotFoundError:
        raise HTTPException(
            status_code=500,
            detail=f"Tesseract OCR Engine not found. Path: {TESSERACT_PATH}",
        )
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        print(f"Error during OCR processing: {e}")
        raise HTTPException(
            status_code=500, detail=f"An internal processing error occurred: {e}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
