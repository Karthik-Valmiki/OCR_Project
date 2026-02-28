from fastapi import FastAPI
from api.routes import router

app = FastAPI(title="TR-OCR MVP")

app.include_router(router)
