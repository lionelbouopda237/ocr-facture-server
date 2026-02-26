from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import pytesseract
import cv2
import numpy as np
import re
import sqlite3

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# DATABASE
# ==========================

def init_db():
    conn = sqlite3.connect("invoices.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS invoices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type_document TEXT,
            numero_facture TEXT,
            numero_commande TEXT,
            date_facture TEXT,
            colis INTEGER,
            casier INTEGER,
            emb_plein INTEGER,
            emb_vide INTEGER,
            montant_ttc REAL,
            net_a_payer REAL,
            devise TEXT,
            ocr_confidence REAL,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ==========================
# IMAGE PREPROCESSING
# ==========================

def preprocess_image(contents):
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Resize si trop large
    h, w = image.shape[:2]
    if w > 1200:
        scale = 1200 / w
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Amélioration contraste
    gray = cv2.equalizeHist(gray)

    # Suppression bruit
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive threshold (robuste fond variable)
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
# OCR CONFIDENCE
# ==========================

def get_ocr_confidence(image):
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    confidences = [int(conf) for conf in data["conf"] if conf != "-1"]
    if confidences:
        return round(sum(confidences) / len(confidences), 2)
    return 0.0

# ==========================
# CLEAN HELPERS
# ==========================

def clean_amount(text):
    try:
        text = text.replace(" ", "")
        return float(text)
    except:
        return None

def clean_int(text):
    try:
        return int(text)
    except:
        return None

# ==========================
# EXTRACTION INTELLIGENTE
# ==========================

def extract_data(text):

    clean_text = re.sub(r'\s+', ' ', text)
    lower = clean_text.lower()

    # TYPE
    type_doc = "FACTURE" if "facture" in lower else None

    # NUMERO FACTURE
    match_num = re.search(r'facture\s*n?[°o]?\s*(\d+)', lower)
    numero_facture = match_num.group(1) if match_num else None

    # DATE
    match_date = re.search(r'\d{2}/\d{2}/\d{4}', clean_text)
    date = match_date.group() if match_date else None

    # EXTRACTION MONTANTS
    numbers = re.findall(r'\d[\d\s]{2,}', clean_text)
    amounts = []

    for n in numbers:
        val = clean_amount(n)
        if val and val > 1000:
            amounts.append(val)

    amounts = sorted(amounts)

    net_a_payer = amounts[-1] if amounts else None
    montant_ttc = amounts[-2] if len(amounts) > 1 else None

    # COLIS
    match_colis = re.search(r'colis\s*[:\-]?\s*(\d+)', lower)
    colis = clean_int(match_colis.group(1)) if match_colis else None

    # CASIER
    match_casier = re.search(r'casier\s*[:\-]?\s*(\d+)', lower)
    casier = clean_int(match_casier.group(1)) if match_casier else None

    # EMB
    match_emb_plein = re.search(r'emb\s*plein\s*[:\-]?\s*(\d+)', lower)
    emb_plein = clean_int(match_emb_plein.group(1)) if match_emb_plein else None

    match_emb_vide = re.search(r'emb\s*vide\s*[:\-]?\s*(\d+)', lower)
    emb_vide = clean_int(match_emb_vide.group(1)) if match_emb_vide else None

    # NUMERO COMMANDE
    match_commande = re.search(r'[A-Z]{1,3}\d{5,}', clean_text)
    numero_commande = match_commande.group() if match_commande else None

    devise = "FCFA" if "fcfa" in lower else None

    return {
        "type_document": type_doc,
        "numero_facture": numero_facture,
        "numero_commande": numero_commande,
        "date_facture": date,
        "colis": colis,
        "casier": casier,
        "emb_plein": emb_plein,
        "emb_vide": emb_vide,
        "montant_ttc": montant_ttc,
        "net_a_payer": net_a_payer,
        "devise": devise
    }

# ==========================
# ROUTES
# ==========================

@app.post("/ocr")
async def ocr_invoice(file: UploadFile = File(...)):

    contents = await file.read()
    processed = preprocess_image(contents)

    text = pytesseract.image_to_string(
        processed,
        lang="fra",
        config="--oem 3 --psm 6"
    )

    confidence = get_ocr_confidence(processed)

    extracted = extract_data(text)

    # Save DB
    conn = sqlite3.connect("invoices.db")
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO invoices (
            type_document, numero_facture, numero_commande,
            date_facture, colis, casier,
            emb_plein, emb_vide,
            montant_ttc, net_a_payer,
            devise, ocr_confidence, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        extracted["type_document"],
        extracted["numero_facture"],
        extracted["numero_commande"],
        extracted["date_facture"],
        extracted["colis"],
        extracted["casier"],
        extracted["emb_plein"],
        extracted["emb_vide"],
        extracted["montant_ttc"],
        extracted["net_a_payer"],
        extracted["devise"],
        confidence,
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()

    return {
        "success": True,
        "ocr_confidence": confidence,
        "extracted_data": extracted
    }

@app.get("/factures")
def get_invoices():
    conn = sqlite3.connect("invoices.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM invoices ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows
