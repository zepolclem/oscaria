"""Application web OscarIA — API FastAPI + interface Gradio.

Endpoints :
- `GET /sante` : healthcheck Docker/Coolify.
- `POST /plaques` : upload d'une photo → photo avec plaques floutées (démo).

Le modèle de détection se charge au démarrage depuis `CHEMIN_MODELE`
(défaut `/data/plaques_baseline.pt`) ; si le fichier est absent et `URL_MODELE`
définie, il est téléchargé une fois (volume persistant → pas de re-téléchargement).
Sans modèle, l'app démarre quand même : l'onglet démo affiche « modèle non chargé »
et `/plaques` répond 503 — le healthcheck reste vert.

AVERTISSEMENT DÉMO : rappel 0,876 mesuré sur la validation interne d'un dataset
international (ADR 0001) ; le transfert sur photos réelles françaises n'est pas
encore mesuré (Phase 3, ADR 0002). Ne pas considérer le floutage comme garanti.
"""

from __future__ import annotations

import io
import os
import sys
import urllib.request
from pathlib import Path

import gradio as gr
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

# Code du pilier plaques (dl/src) : PYTHONPATH dans l'image Docker, repli chemin
# relatif pour le lancement local depuis la racine du repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dl" / "src"))

from flouter import SEUIL_DEMO, detecter_plaques, flouter_image  # noqa: E402

BANDEAU = (
    "⚠️ Démo non validée — rappel 0,88 mesuré sur le dataset d'entraînement "
    "(international) ; la fiabilité sur photos réelles françaises n'est pas encore "
    "mesurée (ADR 0002 à venir). Le floutage n'est pas garanti."
)

app = FastAPI(title="OscarIA")

_MODELE = None
_DEVICE = None


def _charger_modele():
    """Charge (télécharge au besoin) le checkpoint ; renvoie True si le modèle est prêt."""
    global _MODELE, _DEVICE
    if _MODELE is not None:
        return True
    chemin = Path(os.environ.get("CHEMIN_MODELE", "/data/plaques_baseline.pt"))
    url = os.environ.get("URL_MODELE")
    if not chemin.exists() and url:
        chemin.parent.mkdir(parents=True, exist_ok=True)
        tmp = chemin.with_suffix(".tmp")
        try:
            requete = urllib.request.Request(url)
            jeton = os.environ.get("GITHUB_TOKEN")
            if jeton:
                requete.add_header("Authorization", f"Bearer {jeton}")
                requete.add_header("Accept", "application/octet-stream")
            with urllib.request.urlopen(requete, timeout=120) as reponse, open(tmp, "wb") as f:
                f.write(reponse.read())
            tmp.rename(chemin)  # écriture atomique : jamais de checkpoint tronqué
        except Exception as e:  # noqa: BLE001 — démarrage dégradé assumé
            print(f"téléchargement du modèle impossible : {e}", file=sys.stderr)
            tmp.unlink(missing_ok=True)
            return False
    if not chemin.exists():
        return False
    import torch

    from train_plaques import charger_checkpoint

    _DEVICE = torch.device("cpu")  # conteneur sans GPU
    _MODELE, _ = charger_checkpoint(chemin, _DEVICE)
    return True


@app.on_event("startup")
def demarrage() -> None:
    if _charger_modele():
        print("modèle plaques chargé")
    else:
        print("modèle plaques ABSENT — démarrage en mode dégradé", file=sys.stderr)


@app.get("/sante")
def sante() -> dict[str, str]:
    """Healthcheck pour Docker/Coolify (vert même sans modèle : l'app répond)."""
    return {"statut": "ok", "modele": "charge" if _MODELE is not None else "absent"}


@app.post("/plaques")
async def plaques(photo: UploadFile) -> Response:
    """Photo → photo JPEG avec plaques détectées floutées."""
    if not _charger_modele():
        raise HTTPException(status_code=503, detail="modèle non chargé")
    try:
        img = Image.open(io.BytesIO(await photo.read()))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"image illisible : {e}") from e
    boites = detecter_plaques(_MODELE, _DEVICE, img, SEUIL_DEMO)
    floutee = flouter_image(img, boites)
    tampon = io.BytesIO()
    floutee.save(tampon, format="JPEG", quality=90)
    return Response(content=tampon.getvalue(), media_type="image/jpeg",
                    headers={"X-Plaques-Detectees": str(len(boites))})


def flouter_demo(img: Image.Image | None):
    if img is None:
        return None, "Charge une photo d'abord."
    if not _charger_modele():
        return None, "Modèle non chargé sur ce serveur (checkpoint absent)."
    boites = detecter_plaques(_MODELE, _DEVICE, img, SEUIL_DEMO)
    return flouter_image(img, boites), f"{len(boites)} plaque(s) détectée(s) et floutée(s)."


with gr.Blocks(title="OscarIA — démo floutage de plaques") as demo:
    gr.Markdown(f"# OscarIA — floutage de plaques (démo)\n\n{BANDEAU}")
    with gr.Row():
        entree = gr.Image(type="pil", label="Photo d'annonce")
        sortie = gr.Image(label="Photo floutée")
    message = gr.Markdown()
    gr.Button("Flouter les plaques", variant="primary").click(
        flouter_demo, inputs=entree, outputs=[sortie, message]
    )

app = gr.mount_gradio_app(app, demo, path="/")
