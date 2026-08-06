"""Application web OscarIA — squelette de validation de la chaîne de déploiement.

FastAPI porte l'API (healthcheck, futurs endpoints /prix et /plaques) ;
Gradio fournit l'interface web montée sur /.
"""

import gradio as gr
from fastapi import FastAPI

app = FastAPI(title="OscarIA")


@app.get("/sante")
def sante() -> dict[str, str]:
    """Healthcheck pour Docker/Coolify."""
    return {"statut": "ok"}


def echo(texte: str) -> str:
    return f"OscarIA a bien reçu : {texte}"


demo = gr.Interface(
    fn=echo,
    inputs=gr.Textbox(label="Message de test"),
    outputs=gr.Textbox(label="Réponse"),
    title="OscarIA — squelette de déploiement",
    description="Interface provisoire : sera remplacée par le formulaire annonce (prix) et l'upload photos (plaques).",
    flagging_mode="never",
)

app = gr.mount_gradio_app(app, demo, path="/")
