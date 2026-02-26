from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.orm import declarative_base, sessionmaker
import pytesseract
import cv2
import numpy as np
import re
import datetime

app = FastAPI()

# =====================================================
# DATABASE
# =====================================================

DATABASE_URL = "sqlite:///./factures.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Facture(Base):
    __tablename__ = "factures"

    id = Column(Integer, primary_key=True, index=True)
    fournisseur = Column(String)
    numero_facture = Column(String)
    date_facture = Column(String)
    type_document = Column(String)
    net_a_payer = Column(Float)
    montant_ht = Column(Float)
    montant_tva = Column(Float)
    montant_ttc = Column(Float)
    ocr_confidence = Column(Float)
    raw_text = Column(Text)
    created_at = Column(String)

Base.metadata.create_all(bind=engine)

# =====================================================
# CORS
# =====================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# MULTI PREPROCESSING
# =====================================================

def preprocess_variants(contents):
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    variants = []

    # Version 1 : CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    v1 = clahe.apply(gray)
    variants.append(v1)

    # Version 2 : Bilateral
    v2 = cv2.bilateralFilter(gray, 9, 75, 75)
    variants.append(v2)

    # Version 3 : Simple grayscale
    variants.append(gray)

    return variants

# =====================================================
# OCR MULTI CONFIG
# =====================================================

def run_best_ocr(images):

    configs = [
        "--oem 3 --psm 6",
        "--oem 3 --psm 4",
        "--oem 3 --psm 11"
    ]

    best_text = ""
    best_conf = 0

    for img in images:
        for config in configs:

            data = pytesseract.image_to_data(
                img,
                lang="fra",
                config=config,
                output_type=pytesseract.Output.DICT
            )

            confidences = [
                int(conf)
                for conf in data["conf"]
                if conf != "-1"
            ]

            if not confidences:
                continue

            avg_conf = sum(confidences) / len(confidences)

            if avg_conf > best_conf:
                best_conf = avg_conf
                best_text = pytesseract.image_to_string(
                    img,
                    lang="fra",
                    config=config
                )

    return best_text, round(best_conf, 2)

# =====================================================
# EXTRACTION MAXIMALE
# =====================================================

def clean_amount(value):
    value = value.replace(" ", "").replace(",", ".")
    try:
        return float(value)
    except:
        return None

def detect_type(text):
    t = text.lower()
    if "devis" in t:
        return "DEVIS"
    if "bon de livraison" in t:
        return "BON_LIVRAISON"
    if "avoir" in t:
        return "AVOIR"
    if "facture" in t:
        return "FACTURE"
    return "INCONNU"

def extract_all(text):

    clean_text = re.sub(r'\s+', ' ', text)

    # Dates
    dates = re.findall(r'\d{2}/\d{2}/\d{4}', clean_text)

    # Numéros potentiels
    numeros = re.findall(r'(facture|invoice).*?(\w+)', clean_text, re.IGNORECASE)

    # Montants
    montants_bruts = re.findall(r'\d[\d\s.,]+', clean_text)
    montants = [clean_amount(m) for m in montants_bruts]
    montants = [m for m in montants if m and m > 100]

    # Champs spécifiques
    montant_ht = None
    montant_tva = None
    montant_ttc = None
    net = None

    match_ht = re.search(r'(ht|hors taxe).*?(\d[\d\s.,]+)', clean_text, re.IGNORECASE)
    if match_ht:
        montant_ht = clean_amount(match_ht.group(2))

    match_tva = re.search(r'(tva).*?(\d[\d\s.,]+)', clean_text, re.IGNORECASE)
    if match_tva:
        montant_tva = clean_amount(match_tva.group(2))

    match_ttc = re.search(r'(ttc).*?(\d[\d\s.,]+)', clean_text, re.IGNORECASE)
    if match_ttc:
        montant_ttc = clean_amount(match_ttc.group(2))

    match_net = re.search(r'net\s*a\s*payer.*?(\d[\d\s.,]+)', clean_text, re.IGNORECASE)
    if match_net:
        net = clean_amount(match_net.group(1))

    return {
        "type_document": detect_type(text),
        "dates_detectees": dates,
        "numeros_detectes": numeros,
        "tous_montants": montants,
        "montant_ht": montant_ht,
        "montant_tva": montant_tva,
        "montant_ttc": montant_ttc,
        "net_a_payer": net
    }

# =====================================================
# ENDPOINT
# =====================================================

@app.post("/ocr")
async def ocr_invoice(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        variants = preprocess_variants(contents)

        text, confidence = run_best_ocr(variants)

        extracted = extract_all(text)

        db = SessionLocal()
        facture = Facture(
            fournisseur=None,
            numero_facture=None,
            date_facture=extracted["dates_detectees"][0] if extracted["dates_detectees"] else None,
            type_document=extracted["type_document"],
            montant_ht=extracted["montant_ht"],
            montant_tva=extracted["montant_tva"],
            montant_ttc=extracted["montant_ttc"],
            net_a_payer=extracted["net_a_payer"],
            ocr_confidence=confidence,
            raw_text=text,
            created_at=str(datetime.datetime.now())
        )
        db.add(facture)
        db.commit()
        db.close()

        return {
            "success": True,
            "ocr_confidence": confidence,
            "extracted_data": extracted
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/factures")
def get_factures():
    db = SessionLocal()
    factures = db.query(Facture).all()
    db.close()
    return factures
