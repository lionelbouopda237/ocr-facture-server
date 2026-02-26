from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
import cv2
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_image(contents):
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # 🔥 REDUIRE TAILLE IMAGE (clé performance)
    height, width = image.shape[:2]
    max_width = 1200

    if width > max_width:
        scale = max_width / width
        image = cv2.resize(image, (int(width * scale), int(height * scale)))

    # Grayscale simple
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold simple rapide
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    return thresh


@app.post("/ocr")
async def ocr_invoice(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        processed = preprocess_image(contents)

        text = pytesseract.image_to_string(
            processed,
            lang="fra",
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
