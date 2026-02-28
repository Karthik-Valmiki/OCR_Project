import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class TrOCRModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[INFO] Loading models on device: {self.device}")

        # Printed model
        self.printed_processor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-base-printed"
        )
        self.printed_model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-printed"
        ).to(self.device)

        # Handwritten model
        self.handwritten_processor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )
        self.handwritten_model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-handwritten"
        ).to(self.device)

        print("[INFO] Models loaded successfully")

    def predict(self, image, model_type: str = "printed"):

        if model_type == "printed":
            processor = self.printed_processor
            model = self.printed_model
        elif model_type == "handwritten":
            processor = self.handwritten_processor
            model = self.handwritten_model
        else:
            raise ValueError("Invalid model_type. Use 'printed' or 'handwritten'.")

        model.eval()

        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(
            self.device
        )

        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_length=64,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )

        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        generated_text = generated_text.strip()

        return generated_text


# Singleton instance
trocr_model = TrOCRModel()
