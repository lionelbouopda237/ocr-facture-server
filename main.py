from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.orm import declarative_base, sessionmaker
import easyocr
import cv2
import numpy as np
import re
import datetime

app = FastAPI()

# ==========================
# DATABASE
# ==========================

DATABASE_URL = "sqlite:///./factures.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Facture(Base):
    __tablename__ = "factures"
    id = Column(Integer, primary_key=True)
    type_document = Column(String)
    date_facture = Column(String)
    montant_ht = Column(Float)
    montant_tva = Column(Float)
    montant_ttc = Column(Float)
    net_a_payer = Column(Float)
    ocr_confidence = Column(Float)
    raw_text = Column(Text)
    created_at = Column(String)

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
# EASY OCR (chargé une seule fois)
# ==========================

reader = easyocr.Reader(['fr'], gpu=False)

# ==========================
# PREPROCESSING
# ==========================

def preprocess(contents):
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Agrandir image (important pour petites factures)
    image = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return gray

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

    match_date = re.search(r'\d{2}/\d{2}/\d{4}', clean_text)
    date = match_date.group() if match_date else None

    match_net = re.search(r'net\s*a\s*payer.*?(\d[\d\s.,]+)', clean_text, re.IGNORECASE)
    net = clean_amount(match_net.group(1)) if match_net else None

    return date, net

# ==========================
# ENDPOINT OCR
# ==========================

@app.post("/ocr")
async def ocr_invoice(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = preprocess(contents)

        results = reader.readtext(image)

        extracted_text = ""
        confidences = []

        for (bbox, text, confidence) in results:
            extracted_text += text + " "
            confidences.append(confidence)

        avg_confidence = round(sum(confidences)/len(confidences)*100, 2) if confidences else 0

        date, net = extract_data(extracted_text)

        db = SessionLocal()
        facture = Facture(
            type_document="FACTURE",
            date_facture=date,
            net_a_payer=net,
            ocr_confidence=avg_confidence,
            raw_text=extracted_text,
            created_at=str(datetime.datetime.now())
        )
        db.add(facture)
        db.commit()
        db.close()

        return {
            "success": True,
            "ocr_confidence": avg_confidence,
            "extracted_data": {
                "date": date,
                "net_a_payer": net
            }
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

# ==========================
# GET FACTURES
# ==========================

@app.get("/factures")
def get_factures():
    db = SessionLocal()
    factures = db.query(Facture).all()
    db.close()
    return factures
