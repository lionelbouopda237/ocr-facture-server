from fastapi import FastAPI, UploadFile, File
import pytesseract
import cv2
import numpy as np
from PIL import Image
import io

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ocr")
async def ocr_invoice(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Convertir en OpenCV
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Prétraitement
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Augmenter contraste
        gray = cv2.equalizeHist(gray)

        # Threshold adaptatif
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 2
        )

        # OCR config optimisée
        custom_config = r'--oem 3 --psm 6 -l fra'

        text = pytesseract.image_to_string(thresh, config=custom_config)

        return {
            "success": True,
            "extracted_length": len(text),
            "raw_text": text
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

from fastapi.middleware.cors import CORSMiddleware




