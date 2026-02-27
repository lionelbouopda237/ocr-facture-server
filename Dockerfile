# Étape 1: Utiliser une image Python officielle comme base
FROM python:3.9-slim

# Étape 2: Définir le répertoire de travail
WORKDIR /app

# Étape 3: Installer les dépendances système nécessaires pour OpenCV et Tesseract
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*  # Nettoyer les caches d'apt pour réduire l'image

# Étape 4: Mettre à jour pip et installer virtualenv (optionnel, mais recommandé)
RUN pip install --upgrade pip

# Étape 5: Créer un environnement virtuel pour éviter des conflits de dépendances (optionnel mais recommandé)
RUN python -m venv /env
ENV PATH="/env/bin:$PATH"

# Étape 6: Copier le fichier requirements.txt dans le conteneur
COPY requirements.txt .

# Étape 7: Installer toutes les dépendances Python listées dans requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Étape 8: Copier tout le code source de l'application dans le conteneur
COPY . .

# Étape 9: Exposer le port sur lequel l'application FastAPI va tourner
EXPOSE 8000

# Étape 10: Définir la commande d'exécution (ici avec Uvicorn pour démarrer FastAPI)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
