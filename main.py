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

    id = Column(Integer, primary_key=True)
    type_document = Column(String)
    numero_facture = Column(String)
    numero_commande = Column(String)
    date_facture = Column(String)
    colis = Column(Integer)
    casier = Column(Integer)
    emb_plein = Column(Integer)
    emb_vide = Column(Integer)
    montant_ht = Column(Float)
    tva = Column(Float)
    montant_ttc = Column(Float)
    net_a_payer = Column(Float)
    devise = Column(String)
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
# PREPROCESSING OPTIMISÉ
# =====================================================

def preprocess(contents):
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # 🔥 Upscale important pour petite police
    image = cv2.resize(image, None, fx=1.8, fy=1.8, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # CLAHE contraste local
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Sharpen
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    gray = cv2.filter2D(gray, -1, kernel)

    return gray

# =====================================================
# OCR
# =====================================================

def run_ocr(image):

    data = pytesseract.image_to_data(
        image,
        lang="fra",
        config="--oem 3 --psm 4",
        output_type=pytesseract.Output.DICT
    )

    confidences = [int(conf) for conf in data["conf"] if conf != "-1"]
    confidence = round(sum(confidences)/len(confidences),2) if confidences else 0

    text = pytesseract.image_to_string(
        image,
        lang="fra",
        config="--oem 3 --psm 4"
    )

    return text, confidence

# =====================================================
# OUTILS
# =====================================================

def clean_amount(value):
    value = value.replace(" ", "").replace(",", ".")
    try:
        return float(value)
    except:
        return None

def clean_int(value):
    try:
        return int(re.sub(r"\D", "", value))
    except:
        return None

# =====================================================
# EXTRACTION INTELLIGENTE
# =====================================================

def extract_data(text):

    clean_text = re.sub(r'\s+', ' ', text)
    lower = clean_text.lower()

    # Type document
    type_doc = "FACTURE" if "facture" in lower else None

    # Numéro facture
    match_num = re.search(r'facture\s*n?[°o]?\s*(\d+)', lower)
    numero_facture = match_num.group(1) if match_num else None

    # Date
    match_date = re.search(r'\d{2}/\d{2}/\d{4}', clean_text)
    date = match_date.group() if match_date else None

    # Numéro commande
    match_commande = re.search(r'commande\s*[:\-]?\s*([a-zA-Z0-9]+)', lower)
    numero_commande = match_commande.group(1) if match_commande else None

    # Colis
    match_colis = re.search(r'colis\s*[:\-]?\s*(\d+)', lower)
    colis = clean_int(match_colis.group(1)) if match_colis else None

    # Casier
    match_casier = re.search(r'casier\s*[:\-]?\s*(\d+)', lower)
    casier = clean_int(match_casier.group(1)) if match_casier else None

    # EMB plein
    match_emb_plein = re.search(r'emb\s*plein\s*[:\-]?\s*(\d+)', lower)
    emb_plein = clean_int(match_emb_plein.group(1)) if match_emb_plein else None

    # EMB vide
    match_emb_vide = re.search(r'emb\s*vide\s*[:\-]?\s*(\d+)', lower)
    emb_vide = clean_int(match_emb_vide.group(1)) if match_emb_vide else None

    # Devise
    devise = "FCFA" if "fcfa" in lower else None

    # Montants
    match_ht = re.search(r'(ht).*?(\d[\d\s.,]+)', lower)
    montant_ht = clean_amount(match_ht.group(2)) if match_ht else None

    match_tva = re.search(r'(tva).*?(\d[\d\s.,]+)', lower)
    tva = clean_amount(match_tva.group(2)) if match_tva else None

    match_ttc = re.search(r'(ttc).*?(\d[\d\s.,]+)', lower)
    montant_ttc = clean_amount(match_ttc.group(2)) if match_ttc else None

    match_net = re.search(r'net\s*a\s*payer.*?(\d[\d\s.,]+)', lower)
    net_a_payer = clean_amount(match_net.group(1)) if match_net else None

    return {
        "type_document": type_doc,
        "numero_facture": numero_facture,
        "numero_commande": numero_commande,
        "date_facture": date,
        "colis": colis,
        "casier": casier,
        "emb_plein": emb_plein,
        "emb_vide": emb_vide,
        "montant_ht": montant_ht,
        "tva": tva,
        "montant_ttc": montant_ttc,
        "net_a_payer": net_a_payer,
        "devise": devise
    }

# =====================================================
# ENDPOINT OCR
# =====================================================

@app.post("/ocr")
async def ocr_invoice(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        processed = preprocess(contents)
        text, confidence = run_ocr(processed)

        data = extract_data(text)

        db = SessionLocal()
        facture = Facture(
            **data,
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
            "extracted_data": data
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

# =====================================================
# GET FACTURES
# =====================================================

@app.get("/factures")
def get_factures():
    db = SessionLocal()
    factures = db.query(Facture).all()
    db.close()
    return factures
