from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
import cv2
import numpy as np
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# PREPROCESSING AVANCÉ
# ==========================

def preprocess_image(contents):
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    height, width = image.shape[:2]
    max_width = 1600

    if width > max_width:
        scale = max_width / width
        image = cv2.resize(image, (int(width * scale), int(height * scale)))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Débruitage plus avancé
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Augmentation contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Threshold adaptatif
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2
    )

    return thresh


# ==========================
# EXTRACTION INTELLIGENTE
# ==========================

def extract_data(text):

    # Nettoyage espaces multiples
    clean_text = re.sub(r'\s+', ' ', text)

    # Numéro facture
    invoice_number = None
    match_invoice = re.search(r'facture\s*(n°|no|numero)?\s*[:\-]?\s*(\w+)', clean_text, re.IGNORECASE)
    if match_invoice:
        invoice_number = match_invoice.group(2)

    # Date
    date = None
    match_date = re.search(r'\d{2}/\d{2}/\d{4}', clean_text)
    if match_date:
        date = match_date.group()

    # Montants (tous les nombres format argent)
    amounts = re.findall(r'\d{1,3}(?:[\s.,]\d{3})*(?:[.,]\d{2})?', clean_text)

    # Nettoyage montants
    numeric_amounts = []
    for amt in amounts:
        cleaned = amt.replace(" ", "").replace(",", ".")
        try:
            numeric_amounts.append(float(cleaned))
        except:
            pass

    # Détection net à payer (priorité)
    net_a_payer = None
    match_net = re.search(r'net\s*a\s*payer.*?(\d[\d\s.,]+)', clean_text, re.IGNORECASE)
    if match_net:
        raw_net = match_net.group(1)
        raw_net = raw_net.replace(" ", "").replace(",", ".")
        try:
            net_a_payer = float(raw_net)
        except:
            pass

    return {
        "numero_facture": invoice_number,
        "date": date,
        "net_a_payer": net_a_payer,
        "montants_detectes": numeric_amounts
    }


# ==========================
# ENDPOINT PRINCIPAL
# ==========================

@app.post("/ocr")
async def ocr_invoice(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        processed = preprocess_image(contents)

        text = pytesseract.image_to_string(
            processed,
            lang="fra",
            config="--oem 3 --psm 6 -c preserve_interword_spaces=1"
        )

        extracted = extract_data(text)

        return {
            "success": True,
            "raw_text": text,
            "extracted_data": extracted
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
