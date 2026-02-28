# OCR Project — Production-Oriented Development Plan

##  Project Plan

Build a modular, production-ready Optical Character Recognition (OCR) system structured in two phases:

- **Phase 1 → Stable, usable MVP**
- **Phase 2 → Optimized, scalable, production-grade system**

This project evolves from a functional prototype to a high-performance document intelligence service.

---

#  Phase 1 — MVP (Functional, Stable, Demonstrable)


Deliver a working OCR API that:

- Accepts image input (printed or handwritten)
- Extracts text reliably
- Returns structured JSON output
- Optionally exports extracted text into a document

> No over-optimization. No premature scaling.

---

##  Phase 1 Architecture

### Core Pipeline

Image Upload
↓
Basic Preprocessing (RGB conversion)
↓
TrOCR Model (Printed / Handwritten)
↓
Extracted Text
↓
Return JSON Response


---

##  Design Principles - (Phase 1)

To keep the MVP clean and stable:

- ❌ No Tesseract segmentation
- ❌ No multi-stage detection logic
- ❌ No fallback heuristics
- ❌ No complex layout handling
- ✅ Single responsibility: text recognition

The goal is reliability, not sophistication.

---

##  API Specification

### Endpoint: `/ocr/extract`

**Method:** `POST`  
**Input:**  
- Image file (`.png`, `.jpg`, `.jpeg`)
- Model selection (`printed` or `handwritten`)

**Response Format:**

```json
{
  "extracted_text": "Recognized text from image",
  "model_used": "printed"
}
```

# Phase 2 — Optimization & Production Architecture
Phase 2 introduces:

Image preprocessing (deskew, contrast normalization)

Deep-learning text detection (CRAFT / DBNet)

Batch inference optimization

Model acceleration (FP16 / ONNX)

Confidence scoring

Layout awareness (tables, forms)

Scalable deployment (Docker + queue workers)

Tech Stack (Phase 1)

    Python
    FastAPI
    PyTorch
    HuggingFace Transformers
    TrOCR (Printed + Handwritten)
    Uvicorn

