"""Page « Plaques » — API et démo de floutage des plaques d'immatriculation.

Besoin RGPD : une photo d'annonce laisse lire l'immatriculation, qui rattache le véhicule à
une personne. On détecte la plaque et on la floute — **pas de lecture, pas d'OCR** :
minimisation des données, on ne produit jamais la valeur qu'on cherche à protéger.

Le modèle se charge au démarrage depuis `CHEMIN_MODELE` (défaut
`/code/dl/models/plaques_baseline.pt`), avec téléchargement de secours si `URL_MODELE` est
définie. Son absence ne fait pas tomber l'application : l'onglet affiche l'indisponibilité et
`/plaques` répond 503, `main.py` sert les autres piliers normalement.

Contenu déplacé depuis `app/src/main.py` lors de la fusion des piliers Prix et Plaques : un
fichier par pilier, `main.py` réduit à un assembleur. Comportement inchangé.
"""

from __future__ import annotations

import io
import os
import sys
import urllib.request
from pathlib import Path

import gradio as gr
from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

# Code du pilier plaques (`dl/src`) : `PYTHONPATH` dans l'image Docker, repli sur le chemin
# relatif pour un lancement local depuis la racine du dépôt.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dl" / "src"))

from flouter import SEUIL_DEMO, detecter_plaques, flouter_image  # noqa: E402

router = APIRouter(tags=["plaques"])

BANDEAU = (
    "⚠️ **Démo non validée** — rappel 0,88 mesuré sur le dataset d'entraînement "
    "(international) ; la fiabilité sur photos réelles françaises n'est pas encore mesurée "
    "(ADR 0008 à venir). Le floutage n'est pas garanti."
)

_MODELE = None
_DEVICE = None


def charger_modele() -> bool:
    """Charge (télécharge au besoin) le checkpoint. Renvoie True si le modèle est prêt.

    Idempotente et appelée aussi à chaque requête : un premier chargement raté ne condamne
    pas le service, la tentative suivante peut réussir (fichier monté entre-temps, réseau
    revenu).
    """
    global _MODELE, _DEVICE
    if _MODELE is not None:
        return True
    chemin = Path(os.environ.get("CHEMIN_MODELE", "/code/dl/models/plaques_baseline.pt"))
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
            print(f"téléchargement du modèle plaques impossible : {e}", file=sys.stderr)
            tmp.unlink(missing_ok=True)
            return False
    if not chemin.exists():
        return False
    import torch

    from train_plaques import charger_checkpoint

    _DEVICE = torch.device("cpu")  # conteneur sans GPU
    _MODELE, _ = charger_checkpoint(chemin, _DEVICE)
    return True


def etat() -> str:
    """« charge » / « absent » — remonté par le healthcheck de `main.py`."""
    return "charge" if _MODELE is not None or charger_modele() else "absent"


@router.post("/plaques")
async def plaques(photo: UploadFile) -> Response:
    """Photo → photo JPEG avec les plaques détectées floutées."""
    if not charger_modele():
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


def _flouter_demo(img: Image.Image | None):
    if img is None:
        return None, "Charge une photo d'abord."
    if not charger_modele():
        return None, "Modèle non chargé sur ce serveur (checkpoint absent)."
    boites = detecter_plaques(_MODELE, _DEVICE, img, SEUIL_DEMO)
    return flouter_image(img, boites), f"{len(boites)} plaque(s) détectée(s) et floutée(s)."


def construire() -> None:
    """Monte les composants de l'onglet dans le contexte Gradio courant."""
    gr.Markdown(f"### Flouter les plaques d'une photo d'annonce\n\n{BANDEAU}")
    with gr.Row():
        entree = gr.Image(type="pil", label="Photo d'annonce")
        sortie = gr.Image(label="Photo floutée")
    message = gr.Markdown()
    gr.Button("Flouter les plaques", variant="primary").click(
        _flouter_demo, inputs=entree, outputs=[sortie, message]
    )
