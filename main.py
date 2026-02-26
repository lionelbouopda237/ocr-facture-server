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
# SCANNER INTELLIGENT (détection + perspective)
# =====================================================

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def scan_document(image):

    orig = image.copy()
    ratio = image.shape[0] / 500.0
    image = cv2.resize(image, (int(image.shape[1] / ratio), 500))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break
    else:
        return orig  # si pas détecté, retourne image originale

    pts = screenCnt.reshape(4, 2) * ratio
    rect = order_points(pts)

    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

    return warped

# =====================================================
# PREPROCESSING FINAL
# =====================================================

def preprocess(contents):
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    scanned = scan_document(image)

    gray = cv2.cvtColor(scanned, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    return gray

# =====================================================
# OCR + SCORE
# =====================================================

def run_ocr(image):

    data = pytesseract.image_to_data(
        image,
        lang="fra",
        config="--oem 3 --psm 6",
        output_type=pytesseract.Output.DICT
    )

    confidences = [int(conf) for conf in data["conf"] if conf != "-1"]

    confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0

    text = pytesseract.image_to_string(
        image,
        lang="fra",
        config="--oem 3 --psm 6"
    )

    return text, confidence

# =====================================================
# EXTRACTION LARGE
# =====================================================

def clean_amount(value):
    value = value.replace(" ", "").replace(",", ".")
    try:
        return float(value)
    except:
        return None

def extract_data(text):

    clean_text = re.sub(r'\s+', ' ', text)

    type_doc = "FACTURE" if "facture" in clean_text.lower() else "INCONNU"

    match_date = re.search(r'\d{2}/\d{2}/\d{4}', clean_text)
    date = match_date.group() if match_date else None

    match_ht = re.search(r'(ht|hors taxe).*?(\d[\d\s.,]+)', clean_text, re.IGNORECASE)
    ht = clean_amount(match_ht.group(2)) if match_ht else None

    match_tva = re.search(r'(tva).*?(\d[\d\s.,]+)', clean_text, re.IGNORECASE)
    tva = clean_amount(match_tva.group(2)) if match_tva else None

    match_ttc = re.search(r'(ttc).*?(\d[\d\s.,]+)', clean_text, re.IGNORECASE)
    ttc = clean_amount(match_ttc.group(2)) if match_ttc else None

    match_net = re.search(r'net\s*a\s*payer.*?(\d[\d\s.,]+)', clean_text, re.IGNORECASE)
    net = clean_amount(match_net.group(1)) if match_net else None

    return type_doc, date, ht, tva, ttc, net

# =====================================================
# ENDPOINT
# =====================================================

@app.post("/ocr")
async def ocr_invoice(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        processed = preprocess(contents)
        text, confidence = run_ocr(processed)
        type_doc, date, ht, tva, ttc, net = extract_data(text)

        db = SessionLocal()
        facture = Facture(
            type_document=type_doc,
            date_facture=date,
            montant_ht=ht,
            montant_tva=tva,
            montant_ttc=ttc,
            net_a_payer=net,
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
            "extracted_data": {
                "type_document": type_doc,
                "date": date,
                "montant_ht": ht,
                "tva": tva,
                "montant_ttc": ttc,
                "net_a_payer": net
            }
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/factures")
def get_factures():
    db = SessionLocal()
    factures = db.query(Facture).all()
    db.close()
    return factures
