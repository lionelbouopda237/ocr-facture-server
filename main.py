from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import declarative_base, sessionmaker
import pytesseract
import cv2
import numpy as np
import re

app = FastAPI()

# ==========================
# BASE DE DONNÉES SQLITE
# ==========================

DATABASE_URL = "sqlite:///./factures.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Facture(Base):
    __tablename__ = "factures"

    id = Column(Integer, primary_key=True, index=True)
    fournisseur = Column(String)
    numero_facture = Column(String)
    date = Column(String)
    net_a_payer = Column(Float)
    ocr_confidence = Column(Float)

Base.metadata.create_all(bind=engine)

# ==========================
# CORS
# ==========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# PREPROCESSING
# ==========================

def preprocess_image(contents):
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

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
# SCORE OCR
# ==========================

def compute_ocr_confidence(image):
    data = pytesseract.image_to_data(image, lang="fra", output_type=pytesseract.Output.DICT)

    confidences = [
        int(conf)
        for conf in data["conf"]
        if conf != "-1"
    ]

    if len(confidences) == 0:
        return 0

    return round(sum(confidences) / len(confidences), 2)

# ==========================
# EXTRACTION
# ==========================

def clean_amount(value):
    value = value.replace(" ", "").replace(",", ".")
    try:
        return float(value)
    except:
        return None

def extract_data(text):

    clean_text = re.sub(r'\s+', ' ', text)

    lines = text.split("\n")
    fournisseur = None
    for line in lines:
        if len(line.strip()) > 5 and line.isupper():
            fournisseur = line.strip()
            break

    invoice_number = None
    match_invoice = re.search(r'(facture|invoice).*?(\w+)', clean_text, re.IGNORECASE)
    if match_invoice:
        invoice_number = match_invoice.group(2)

    date = None
    match_date = re.search(r'\d{2}/\d{2}/\d{4}', clean_text)
    if match_date:
        date = match_date.group()

    net_a_payer = None
    match_net = re.search(r'net\s*a\s*payer.*?(\d[\d\s.,]+)', clean_text, re.IGNORECASE)
    if match_net:
        net_a_payer = clean_amount(match_net.group(1))

    return fournisseur, invoice_number, date, net_a_payer

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
            config="--oem 3 --psm 6"
        )

        confidence = compute_ocr_confidence(processed)

        fournisseur, numero_facture, date, net_a_payer = extract_data(text)

        # Sauvegarde en base
        db = SessionLocal()
        nouvelle_facture = Facture(
            fournisseur=fournisseur,
            numero_facture=numero_facture,
            date=date,
            net_a_payer=net_a_payer,
            ocr_confidence=confidence
        )
        db.add(nouvelle_facture)
        db.commit()
        db.close()

        return {
            "success": True,
            "ocr_confidence": confidence,
            "extracted_data": {
                "fournisseur": fournisseur,
                "numero_facture": numero_facture,
                "date": date,
                "net_a_payer": net_a_payer
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# ==========================
# ENDPOINT LISTE FACTURES
# ==========================

@app.get("/factures")
def get_factures():
    db = SessionLocal()
    factures = db.query(Facture).all()
    db.close()

    return factures
