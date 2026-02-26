from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import pytesseract
import cv2
import numpy as np
import io
import re
from datetime import datetime

app = FastAPI()

# ---- CORS ----
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

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return thresh

def extract_structured_data(text):

    data = {
        "date": None,
        "heure": None,
        "numero_facture": None,
        "client": None,
        "produits": [],
        "totaux": {},
        "emballages": None,
        "emballages_vides": None,
        "colis": None,
        "remise": None,
        "ristourne": None,
        "ristourne_en_cours": None,
        "raw_text": text
    }

    lines = text.split("\n")

    # --- Date & Heure ---
    date_match = re.search(r"\d{2}/\d{2}/\d{4}", text)
    if date_match:
        data["date"] = date_match.group()

    heure_match = re.search(r"\d{2}:\d{2}", text)
    if heure_match:
        data["heure"] = heure_match.group()

    # --- Numéro facture ---
    facture_match = re.search(r"(FACTURE|Facture|N°)\s*[:\-]?\s*(\S+)", text)
    if facture_match:
        data["numero_facture"] = facture_match.group(2)

    # --- Extraction Produits ---
    for line in lines:
        # Format supposé : NomProduit  Qté  PU  Total
        product_match = re.search(
            r"(.+?)\s+(\d+)\s+([\d.,]+)\s+([\d.,]+)",
            line
        )
        if product_match:
            produit = {
                "nom": product_match.group(1).strip(),
                "quantite": int(product_match.group(2)),
                "prix_unitaire": product_match.group(3),
                "total_ligne": product_match.group(4)
            }
            data["produits"].append(produit)

    # --- Totaux ---
    total_match = re.search(r"TOTAL\s*[:\-]?\s*([\d.,]+)", text, re.IGNORECASE)
    if total_match:
        data["totaux"]["total"] = total_match.group(1)

    remise_match = re.search(r"REMISE\s*[:\-]?\s*([\d.,]+)", text, re.IGNORECASE)
    if remise_match:
        data["remise"] = remise_match.group(1)

    ristourne_match = re.search(r"RISTOURNE\s*[:\-]?\s*([\d.,]+)", text, re.IGNORECASE)
    if ristourne_match:
        data["ristourne"] = ristourne_match.group(1)

    # --- Emballages / Colis ---
    colis_match = re.search(r"COLIS\s*[:\-]?\s*(\d+)", text, re.IGNORECASE)
    if colis_match:
        data["colis"] = int(colis_match.group(1))

    emballage_match = re.search(r"EMBALLAGES?\s*[:\-]?\s*(\d+)", text, re.IGNORECASE)
    if emballage_match:
        data["emballages"] = int(emballage_match.group(1))

    emballage_vide_match = re.search(r"VIDES?\s*[:\-]?\s*(\d+)", text, re.IGNORECASE)
    if emballage_vide_match:
        data["emballages_vides"] = int(emballage_vide_match.group(1))

    return data


@app.post("/ocr")
async def ocr_invoice(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        processed_image = preprocess_image(contents)

        text = pytesseract.image_to_string(
            processed_image,
            lang="eng",
            config="--oem 3 --psm 6"
        )

        structured = extract_structured_data(text)

        return {
            "success": True,
            "data": structured
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
