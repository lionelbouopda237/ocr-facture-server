from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
from PIL import Image
import io

app = FastAPI()

# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- OCR Endpoint ----
@app.post("/ocr")
async def ocr_invoice(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        text = pytesseract.image_to_string(image)

        return {
            "success": True,
            "extracted_text": text
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
