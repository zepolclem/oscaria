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

RUN useradd --create-home oscaria
USER oscaria

EXPOSE 8000
CMD ["uvicorn", "app.src.main:app", "--host", "0.0.0.0", "--port", "8000"]
