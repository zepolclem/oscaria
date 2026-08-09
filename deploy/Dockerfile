# Image de l'application web OscarIA (FastAPI + Gradio).
# Contexte de build = racine du repo (le lock uv couvre tout le workspace) :
#   docker build -f deploy/Dockerfile .

# --- Étape 1 : installation des dépendances avec uv ---
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS build

WORKDIR /code
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Manifestes seuls d'abord : la couche de dépendances n'est reconstruite
# que si un pyproject/lock change, pas à chaque modification de code.
COPY pyproject.toml uv.lock ./
COPY ml/pyproject.toml ml/
COPY dl/pyproject.toml dl/
COPY collecte/scraper/pyproject.toml collecte/scraper/
COPY app/pyproject.toml app/

# Venv jetable propre au conteneur : n'installe QUE les dépendances du membre
# app (le piège d'élagage du venv partagé ne s'applique pas ici).
RUN uv sync --frozen --no-dev --package oscaria-app

# --- Étape 2 : image finale légère ---
FROM python:3.12-slim-bookworm

WORKDIR /code
ENV PATH="/code/.venv/bin:$PATH" \
    GRADIO_TEMP_DIR=/tmp/gradio

COPY --from=build /code/.venv /code/.venv
COPY app/src app/src
# Code du pilier plaques : l'app importe flouter/train_plaques (inférence seule).
COPY dl/src dl/src
# Checkpoint versionné (git add -f, décision utilisateur 2026-08-07) : embarqué
# dans l'image — pas de volume ni de téléchargement au démarrage.
COPY dl/models/plaques_baseline.pt dl/models/plaques_baseline.pt
# Modèle du pilier prix (~170 Ko), produit par `ml/src/entrainement.py` et versionné.
# Même principe que le checkpoint plaques : embarqué, pas téléchargé au démarrage.
COPY app/models app/models
ENV PYTHONPATH=/code/dl/src \
    CHEMIN_MODELE=/code/dl/models/plaques_baseline.pt

# /data : point de montage du volume modèle — propriété d'oscaria pour que le
# téléchargement au premier démarrage puisse y écrire.
RUN useradd --create-home oscaria && mkdir -p /data && chown oscaria /data
USER oscaria

EXPOSE 8000
# `--app-dir app/src` plutôt que le module `app.src.main` : les fichiers de app/src
# s'importent à plat entre eux (`main` -> `page_prix` -> `prix` -> `preparation`), ce qui
# suppose ce dossier sur sys.path. `PYTHONPATH=/code/dl/src` reste valable en parallèle
# pour les imports du pilier plaques (`flouter`, `train_plaques`).
CMD ["uvicorn", "--app-dir", "app/src", "main:app", "--host", "0.0.0.0", "--port", "8000"]
