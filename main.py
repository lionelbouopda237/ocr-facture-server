from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import cv2
import numpy as np
import re
import sqlite3
import pytesseract
import easyocr

app = FastAPI()

# ==========================
# Configuration CORS
# ==========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permet toutes les origines (change selon ton besoin)
    allow_credentials=True,
    allow_methods=["*"],  # Permet toutes les méthodes HTTP
    allow_headers=["*"],  # Permet tous les headers
)

# ==========================
# Initialisation OCR
# ==========================
reader = easyocr.Reader(['fr'], gpu=False)  # Mode CPU activé pour éviter les problèmes de mémoire

# ==========================
# BASE DE DONNÉES
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
# Préprocessing de l'image (réduction de la taille)
# ==========================
def preprocess_image(contents):
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Limiter la largeur à 800px pour économiser la mémoire
    h, w = image.shape[:2]
    max_width = 800
    if w > max_width:
        scale = max_width / w
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 2
    )

    return thresh

# ==========================
# Fonction OCR (Tesseract + EasyOCR si nécessaire)
# ==========================
def run_ocr(image):
    # Utilisation de EasyOCR pour extraire le texte
    result = reader.readtext(image)
    easy_text = " ".join([r[1] for r in result])
    easy_conf = np.mean([r[2] for r in result]) * 100 if result else 0

    # Si la confiance d'EasyOCR est faible, utiliser Tesseract
    if easy_conf < 40:
        tess_text = pytesseract.image_to_string(
            image,
            lang="fra",
            config="--oem 3 --psm 6"
        )
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        confs = [int(c) for c in data["conf"] if c != "-1"]
        tess_conf = sum(confs) / len(confs) if confs else 0

        if tess_conf > easy_conf:
            return tess_text, tess_conf

    return easy_text, easy_conf

# ==========================
# Extraction des données
# ==========================
def clean_amount(text):
    try:
        return float(text.replace(" ", ""))
    except:
        return None

def clean_int(text):
    try:
        return int(text)
    except:
        return None

def extract_data(text):
    clean_text = re.sub(r'\s+', ' ', text)
    lower = clean_text.lower()

    # Type document
    type_doc = "FACTURE" if "facture" in lower else None

    # Numéro de facture
    match_num = re.search(r'facture\s*n?[°o]?\s*(\d+)', lower)
    numero_facture = match_num.group(1) if match_num else None

    # Date
    match_date = re.search(r'\d{2}/\d{2}/\d{4}', clean_text)
    date = match_date.group() if match_date else None

    # Extraction des montants
    numbers = re.findall(r'\d[\d\s]{2,}', clean_text)
    amounts = []
    for n in numbers:
        val = clean_amount(n)
        if val and val > 1000:
            amounts.append(val)

    amounts = sorted(amounts)
    net_a_payer = amounts[-1] if amounts else None
    montant_ttc = amounts[-2] if len(amounts) > 1 else None

    # Colis
    match_colis = re.search(r'colis\s*[:\-]?\s*(\d+)', lower)
    colis = clean_int(match_colis.group(1)) if match_colis else None

    # Casier
    match_casier = re.search(r'casier\s*[:\-]?\s*(\d+)', lower)
    casier = clean_int(match_casier.group(1)) if match_casier else None

    # Emballage Plein / Vide
    match_emb_plein = re.search(r'emb\s*plein\s*[:\-]?\s*(\d+)', lower)
    emb_plein = clean_int(match_emb_plein.group(1)) if match_emb_plein else None

    match_emb_vide = re.search(r'emb\s*vide\s*[:\-]?\s*(\d+)', lower)
    emb_vide = clean_int(match_emb_vide.group(1)) if match_emb_vide else None

    # Numéro de commande (code alphanumérique)
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
# Routes
# ==========================
@app.post("/ocr")
async def ocr_invoice(file: UploadFile = File(...)):
    contents = await file.read()
    processed = preprocess_image(contents)

    text, confidence = run_ocr(processed)
    extracted = extract_data(text)

    # Insertion des données dans la base SQLite
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
        "ocr_confidence": round(confidence, 2),
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
