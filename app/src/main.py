"""Application web OscarIA — assemblage des piliers.

Ce fichier ne contient **aucune logique métier** : il branche les routes et les onglets que
chaque pilier définit dans son propre module (`page_prix.py`, `page_plaques.py`), et expose
le healthcheck.

C'est délibéré. Les piliers avancent sur des branches séparées ; tant que chacun livre son
fichier de page, `main.py` n'a pas à changer et les fusions restent propres — la version
précédente, où chaque pilier ajoutait ici ses routes et son formulaire, produisait un conflit
à chaque itération. Ajouter un pilier = déposer son module et une ligne dans `PAGES`.

Chaque module de page expose :
    router      (facultatif) un APIRouter monté sur l'application
    construire()             monte les composants Gradio dans l'onglet courant
    etat()      (facultatif) « charge » / « absent », remonté par le healthcheck

Un pilier dont le module est absent, ou dont le modèle ne charge pas, **ne fait pas tomber
l'application** : son onglet affiche l'indisponibilité et les autres continuent de servir.
Le healthcheck reste vert — un healthcheck qui rougit parce qu'un pilier sur deux manque
fait redémarrer le conteneur en boucle chez Coolify, et on perd aussi celui qui marchait.
"""

from __future__ import annotations

import importlib
import sys

import gradio as gr
from fastapi import FastAPI

# (module, libellé de l'onglet). L'ordre est celui d'affichage.
PAGES = [
    ("page_prix", "Estimer le prix"),
    ("page_plaques", "Flouter les plaques"),
]

app = FastAPI(title="OscarIA")

_charges: dict[str, object] = {}
_erreurs: dict[str, str] = {}

for _nom, _ in PAGES:
    try:
        _module = importlib.import_module(_nom)
    except Exception as e:  # noqa: BLE001 — démarrage dégradé assumé, cf. docstring
        _erreurs[_nom] = f"{type(e).__name__}: {e}"
        print(f"pilier {_nom} indisponible — {e}", file=sys.stderr)
        continue
    _charges[_nom] = _module
    if hasattr(_module, "router"):
        app.include_router(_module.router)


@app.get("/sante")
def sante() -> dict:
    """Healthcheck Docker/Coolify.

    Vert dès que l'application répond ; l'état pilier par pilier est renvoyé à côté, pour
    diagnostiquer un modèle manquant sans faire échouer le déploiement. Un module chargé
    n'implique pas un modèle chargé : les piliers qui savent faire la différence l'exposent
    par `etat()`.
    """
    piliers = {}
    for nom, _ in PAGES:
        module = _charges.get(nom)
        if module is None:
            piliers[nom] = "absent"
        elif hasattr(module, "etat"):
            piliers[nom] = module.etat()
        else:
            piliers[nom] = "charge"
    return {"statut": "ok", "piliers": piliers}


with gr.Blocks(title="OscarIA") as demo:
    gr.Markdown(
        "# OscarIA\n"
        "Assistant d'aide à la décision pour la vente d'un véhicule d'occasion entre "
        "particuliers."
    )
    with gr.Tabs():
        for nom, libelle in PAGES:
            with gr.Tab(libelle):
                module = _charges.get(nom)
                if module is not None and hasattr(module, "construire"):
                    module.construire()
                else:
                    gr.Markdown(
                        f"**Pilier indisponible sur ce serveur.**\n\n"
                        f"`{_erreurs.get(nom, 'module absent')}`"
                    )

app = gr.mount_gradio_app(app, demo, path="/")
