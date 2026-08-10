"""Page « Prix » — API et formulaire du pilier estimation de prix.

Tout ce qui est propre à ce pilier vit ici : le contrat d'entrée, la route `/prix` et
l'onglet Gradio. `main.py` ne fait que les assembler avec ceux des autres piliers.

Ce découpage n'est pas cosmétique : les piliers Prix et Plaques avancent sur des branches
séparées. Un `main.py` où chacun ajoute son formulaire garantit un conflit de fusion à chaque
itération ; un fichier par pilier réduit `main.py` à quelques lignes que personne n'a besoin
de modifier pour livrer sa page.
"""

from __future__ import annotations

from typing import Annotated, Literal

import gradio as gr
from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

import prix as service

router = APIRouter(tags=["prix"])

MENTION = (
    "Estimation produite automatiquement par un modèle statistique entraîné sur "
    "20 915 annonces de **particuliers** publiées sur leboncoin. **Aide à la décision, pas "
    "une expertise.** Trois limites à connaître : le modèle apprend des **prix demandés**, "
    "pas des prix de vente conclus — le prix réellement obtenu est généralement plus bas "
    "après négociation ; l'état du véhicule est **déclaré par le vendeur**, jamais vérifié ; "
    "et la couverture annoncée est une moyenne, meilleure sur l'entrée de gamme que sur les "
    "véhicules à plus de 20 000 €."
)


class Vehicule(BaseModel):
    """Contrat d'entrée de l'API. Miroir exact du formulaire — contrat v2, tout obligatoire.

    Plus aucun champ facultatif. Le carnet 05 a montré que les champs laissés vides tiraient
    l'estimation vers le bas : le modèle avait appris des annonces que « champ manquant =
    annonce bâclée = pas cher », signal sans aucun sens pour un vendeur qui ignore une
    valeur. Plutôt que de corriger ce biais après coup, le contrat v2 supprime sa cause :
    huit champs qu'un vendeur connaît toujours, tous exigés (ADR ML 0007).

    Les bornes reprennent le domaine réellement observé dans le jeu d'entraînement. Elles ne
    servent pas à protéger le modèle (un arbre ne plante pas sur une valeur extrême, il la
    rabat sur sa dernière coupe) mais à **refuser une saisie absurde plutôt que de rendre un
    prix qui aurait l'air normal** : un kilométrage négatif produisait sinon une estimation
    parfaitement plausible.
    """

    # `extra="forbid"` : un champ inconnu vaut 422, pas un silence. Le contrat a déjà perdu
    # `region`, `mois_mec`, puis `couleur`, `puissance_fisc`, `portes`, `places`, `critair`
    # et `ct_valide_jusqu_a` (v2) ; sans ça, un client resté sur une ancienne version les
    # enverrait encore et recevrait un prix calculé SANS eux — juste, mais pas celui qu'il
    # croit avoir demandé. Un refus explicite vaut mieux qu'une réponse trompeuse.
    model_config = ConfigDict(extra="forbid")

    marque: str
    modele: str
    annee_mec: Annotated[int, Field(ge=1980, le=2030, description="Année de mise en circulation")]
    kilometrage: Annotated[float, Field(ge=0, le=1_000_000)]
    energie: str
    boite: str
    # Contraint aux 4 crans en vigueur. Sans cette contrainte, une valeur d'une ancienne
    # échelle (8 puis 5 crans) passerait en 200 et retomberait sur « inconnu » : le client
    # recevrait un prix calculé comme si l'état n'était pas renseigné, sans avertissement.
    etat: Literal[tuple(service.ETATS.values())]  # type: ignore[valid-type]
    puissance_din: Annotated[float, Field(ge=1, le=2_000)]


@router.post("/prix")
def estimer_prix(v: Vehicule) -> dict:
    """Un véhicule → une fourchette de prix."""
    return service.estimer(v.model_dump())


@router.get("/prix/contrat")
def contrat() -> dict:
    """Ce que le modèle attend et ce qu'il vaut — transparence exigée par le bloc BC04."""
    return service.contrat()


def _euros(n: float) -> str:
    return f"{n:,.0f}".replace(",", " ") + " €"


def _modeles(marque):
    """Recharge la liste des modèles quand la marque change (liste en cascade).

    804 modèles au total, 6 par marque en médiane : proposer la liste complète serait
    inutilisable, et laisser un modèle incohérent avec sa marque n'aurait aucun sens.
    """
    return gr.Dropdown(choices=service.modeles_de(marque), value=None)


def _estimer(marque, modele, annee, km, energie, boite, etat_libelle, din):
    """Adapte la saisie Gradio au contrat du service et rend le résultat en Markdown.

    Gradio ne sait pas rendre un champ obligatoire : un composant vide envoie `None` sans
    broncher. Le garde-fou vit donc ici — on nomme ce qui manque au lieu d'estimer, parce
    qu'estimer sans ces champs reproduirait exactement le biais que le contrat v2 supprime.
    """
    saisie = {"marque": marque, "modele": modele, "annee_mec": annee, "kilometrage": km,
              "energie": energie, "boite": boite,
              # Le formulaire affiche des libellés français ; le modèle ne connaît que ses codes.
              "etat": service.ETATS.get(etat_libelle),
              "puissance_din": din}
    libelles = {"marque": "la marque", "modele": "le modèle", "annee_mec": "l'année",
                "kilometrage": "le kilométrage", "energie": "l'énergie", "boite": "la boîte",
                "etat": "l'état", "puissance_din": "la puissance DIN"}
    manquants = [libelles[c] for c, v in saisie.items() if v is None]
    if manquants:
        return ("**Tous les champs sont nécessaires à l'estimation.** "
                f"Il manque : {', '.join(manquants)}.")
    r = service.estimer(saisie)
    part = round(r["couverture"] * 10)
    return (
        f"## {_euros(r['bas'])} — {_euros(r['haut'])}\n\n"
        f"Estimation centrale : **{_euros(r['central'])}**\n\n"
        f"Cette fourchette contient le **prix affiché** d'annonces comparables "
        f"**{part} fois sur 10** ({r['couverture']:.1%} mesurés sur 3 977 annonces jamais "
        f"vues par le modèle).\n\n"
        f"*{MENTION}*"
    )


def construire() -> None:
    """Monte les composants de l'onglet dans le contexte Gradio courant."""
    gr.Markdown(
        "### Estimer le prix de votre véhicule\n"
        "Huit champs, **tous nécessaires** — les informations de base de votre véhicule. "
        "Un formulaire incomplet fausserait l'estimation, il n'est donc pas accepté."
    )

    with gr.Row():
        marque = gr.Dropdown(service.MARQUES, label="Marque", value="RENAULT")
        modele = gr.Dropdown(service.modeles_de("RENAULT"), label="Modèle", value=None)
        annee = gr.Number(label="Année de mise en circulation", value=2015, precision=0)
    with gr.Row():
        km = gr.Number(label="Kilométrage", value=120_000, minimum=0, maximum=1_000_000)
        energie = gr.Dropdown(service.ENERGIES, label="Énergie", value="Diesel")
        boite = gr.Dropdown(service.BOITES, label="Boîte de vitesses", value="Manuelle")
    with gr.Row():
        etat = gr.Dropdown(list(service.ETATS), label="État déclaré", value="Usure normale")
        # Sans valeur par défaut : une puissance pré-remplie « plausible » serait une saisie
        # déguisée en donnée — exactement le biais que le contrat v2 supprime.
        din = gr.Number(label="Puissance DIN (ch)", value=None, minimum=1, maximum=2_000)

    marque.change(_modeles, inputs=marque, outputs=modele)

    sortie = gr.Markdown()
    gr.Button("Estimer", variant="primary").click(
        _estimer,
        inputs=[marque, modele, annee, km, energie, boite, etat, din],
        outputs=sortie,
    )
