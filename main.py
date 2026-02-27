import pytesseract
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

app = FastAPI()

# Fonction pour extraire le texte de l'image
def extract_text_from_image(image: Image.Image) -> str:
    # Utilisation de pytesseract pour extraire le texte
    text = pytesseract.image_to_string(image)
    return text

# Endpoint pour télécharger l'image et récupérer le texte extrait
@app.post("/ocr")
async def ocr_invoice(file: UploadFile = File(...)):
    # Lire l'image téléchargée
    image_data = await file.read()
    
    # Convertir les données binaires en image
    image = Image.open(io.BytesIO(image_data))
    
    # Extraire le texte avec OCR
    extracted_text = extract_text_from_image(image)
    
    return {"extracted_text": extracted_text}

# Pour démarrer le serveur avec 'uvicorn main:app --reload' si ce fichier est exécuté directement
