import pytesseract
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import cv2
import numpy as np
from PIL import Image

app = FastAPI()

# Modèle de données pour la réponse
class InvoiceData(BaseModel):
    total: str
    products: list

# Fonction de prétraitement de l'image (OpenCV)
def preprocess_image(image_path: str) -> np.ndarray:
    # Chargement de l'image
    img = cv2.imread(image_path)

    # Conversion en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Augmentation du contraste avec un seuillage adaptatif
    img_contrast = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Enlever le bruit avec un flou gaussien
    img_denoised = cv2.GaussianBlur(img_contrast, (5, 5), 0)

    # Sauvegarder l'image traitée
    cv2.imwrite('processed_image.jpg', img_denoised)

    return img_denoised

# Fonction d'extraction de texte avec Tesseract
def extract_text_from_image(image_path: str) -> str:
    # Charger l'image prétraitée
    img = Image.open(image_path)

    # Effectuer l'OCR avec Tesseract
    extracted_text = pytesseract.image_to_string(img, lang='fra')  # Utilisez 'eng' pour l'anglais si nécessaire

    return extracted_text

# Fonction de nettoyage du texte extrait
def clean_extracted_text(extracted_text: str) -> str:
    # Enlever les caractères indésirables (espaces supplémentaires, caractères spéciaux)
    cleaned_text = re.sub(r"[^A-Za-z0-9.,€\n]", " ", extracted_text)  # Garde les caractères alphanumériques et les symboles pertinents
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()  # Remplacer les espaces multiples par un seul
    return cleaned_text

# Fonction pour extraire les données de la facture
def extract_invoice_data(cleaned_text: str):
    # Exemple d'extraction via des expressions régulières
    total_match = re.search(r"TOTAL\s*[:;]?\s*(\d+[\.,]?\d*)", cleaned_text)
    products_match = re.findall(r"(Produit|Article)\s*[:;]?\s*(\d+)\s*([A-Za-z ]+)\s*([\d,\.]+)", cleaned_text)
    
    # Extraction des montants (ici le total)
    total = total_match.group(1) if total_match else "Non trouvé"

    # Extraction des produits
    products = []
    for match in products_match:
        products.append({
            'product_code': match[1],
            'product_name': match[2],
            'price': match[3]
        })
    
    # Renvoyer un dictionnaire avec les informations
    invoice_data = {
        'total': total,
        'products': products
    }

    return invoice_data

# Fonction de l'API FastAPI pour traiter les images de factures
@app.post("/ocr")
async def extract_invoice(file: UploadFile = File(...)):
    # Sauvegarder l'image temporairement
    with open("temp_image.jpg", "wb") as f:
        f.write(await file.read())
    
    # Prétraiter l'image
    processed_image = preprocess_image("temp_image.jpg")
    
    # Extraire le texte de l'image
    extracted_text = extract_text_from_image("temp_image.jpg")
    
    # Nettoyer le texte extrait
    cleaned_text = clean_extracted_text(extracted_text)
    
    # Extraire les informations de la facture
    invoice_data = extract_invoice_data(cleaned_text)
    
    return InvoiceData(total=invoice_data['total'], products=invoice_data['products'])
