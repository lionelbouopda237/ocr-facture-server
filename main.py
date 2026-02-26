from fastapi import FastAPI, File, UploadFile
import pytesseract
import cv2
import numpy as np
from PIL import Image
import io
import re

app = FastAPI()

@app.post("/ocr")
async def ocr_invoice(file: UploadFile = File(...)):

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Convertir en OpenCV
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Amélioration contraste
    gray = cv2.equalizeHist(gray)

    # Binarisation
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # OCR
    text = pytesseract.image_to_string(thresh, lang='fra')

    clean_text = text.replace("\n", " ")

    data = {}

    # Numéro facture
    num = re.search(r'FACTURE\s*N[°o]?\s*(\d+)', clean_text)
    if num:
        data["numero"] = num.group(1)

    # Date
    date = re.search(r'(\d{2}/\d{2}/\d{4})', clean_text)
    if date:
        data["date"] = date.group(1)

    # Heure
    heure = re.search(r'(\d{2}:\d{2}:\d{2})', clean_text)
    if heure:
        data["heure"] = heure.group(1)

    # Net à payer
    net = re.search(r'NET.*?(\d[\d\s]+)', clean_text)
    if net:
        data["net_a_payer"] = int(net.group(1).replace(" ", ""))

    # Montant TTC
    ttc = re.search(r'TTC.*?(\d[\d\s]+)', clean_text)
    if ttc:
        data["montant_ttc"] = int(ttc.group(1).replace(" ", ""))

    # Colis
    colis = re.search(r'Colis\s*:\s*(\d+)', clean_text)
    if colis:
        data["colis"] = int(colis.group(1))

    # Casier
    casier = re.search(r'Casier\s*:\s*(\d+)', clean_text)
    if casier:
        data["casier"] = int(casier.group(1))

    # EMB plein
    embp = re.search(r'EMB\s*Plein\s*:\s*(\d+)', clean_text)
    if embp:
        data["emb_plein"] = int(embp.group(1))

    # EMB vide
    embv = re.search(r'EMB\s*Vide\s*:\s*(\d+)', clean_text)
    if embv:
        data["emb_vide"] = int(embv.group(1))

    return {
        "success": True,
        "extracted": data,
        "raw_text": text
    }