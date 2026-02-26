from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
import cv2
import numpy as np
import io

app = FastAPI()

# ---------------- CORS ----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Prétraitement image ----------------
def preprocess_image(contents):
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Augmenter la résolution (important pour OCR)
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Améliorer contraste
    gray = cv2.equalizeHist(gray)

    # Adaptive threshold (meilleur que OTSU pour factures)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2
    )

    return thresh


# ---------------- Endpoint OCR ----------------
@app.post("/ocr")
async def ocr_invoice(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        processed_image = preprocess_image(contents)

        text = pytesseract.image_to_string(
            processed_image,
            lang="fra",                 # ← LANGUE FRANÇAISE
            config="--oem 3 --psm 6"
        )

        return {
            "success": True,
            "raw_text": text
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
