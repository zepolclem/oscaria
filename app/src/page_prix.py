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
    "Estimation produite automatiquement par un modèle statistique entraîné sur des annonces "
    "de particuliers. **Aide à la décision, pas une expertise** : le prix réellement obtenu "
    "dépend de la négociation, du marché local et de l'état constaté du véhicule. "
    "La couverture annoncée est une moyenne : elle est meilleure sur les véhicules "
    "d'entrée de gamme que sur ceux à plus de 20 000 €."
)


class Vehicule(BaseModel):
    """Contrat d'entrée de l'API. Miroir exact du formulaire.

    Les champs facultatifs valent `None` par défaut : `HistGradientBoosting` traite
    nativement les valeurs manquantes — il apprend de quel côté d'une coupe envoyer les
    trous — donc un champ vide n'est pas une erreur, seulement une information de moins.

    Les bornes reprennent le domaine réellement observé dans le jeu d'entraînement. Elles ne
    servent pas à protéger le modèle (un arbre ne plante pas sur une valeur extrême, il la
    rabat sur sa dernière coupe) mais à **refuser une saisie absurde plutôt que de rendre un
    prix qui aurait l'air normal** : un kilométrage négatif produisait sinon une estimation
    parfaitement plausible.
    """

    # `extra="forbid"` : un champ inconnu vaut 422, pas un silence. Le contrat a déjà perdu
    # `region` et `mois_mec` ; sans ça, un client resté sur l'ancienne version les enverrait
    # encore et recevrait un prix calculé SANS eux — juste, mais pas celui qu'il croit avoir
    # demandé. Un refus explicite vaut mieux qu'une réponse trompeuse.
    model_config = ConfigDict(extra="forbid")

    marque: str
    modele: str | None = None
    annee_mec: Annotated[int, Field(ge=1980, le=2030, description="Année de mise en circulation")]
    kilometrage: Annotated[float, Field(ge=0, le=1_000_000)] | None = None
    energie: str | None = None
    boite: str | None = None
    # Contraint aux 5 crans en vigueur. Sans cette contrainte, une valeur de l'ancienne
    # échelle à 8 crans passait en 200 et retombait sur « inconnu » : le client recevait un
    # prix calculé comme si l'état n'avait pas été renseigné, sans aucun avertissement.
    etat: Literal[tuple(service.ETATS.values())] | None = None  # type: ignore[valid-type]
    couleur: str | None = None
    puissance_din: Annotated[float, Field(ge=1, le=2_000)] | None = None
    puissance_fisc: Annotated[float, Field(ge=1, le=100)] | None = None
    portes: Annotated[float, Field(ge=1, le=9)] | None = None
    places: Annotated[float, Field(ge=1, le=9)] | None = None
    critair: Annotated[float, Field(ge=0, le=5)] | None = None
    ct_valide_jusqu_a: Annotated[float, Field(ge=2000, le=2100)] | None = None


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


def _estimer(marque, modele, annee, km, energie, boite, etat_libelle, couleur,
             din, fisc, portes, places, critair, ct):
    """Adapte la saisie Gradio au contrat du service et rend le résultat en Markdown."""
    r = service.estimer({
        "marque": marque, "modele": modele, "annee_mec": annee, "kilometrage": km,
        "energie": energie, "boite": boite,
        # Le formulaire affiche des libellés français ; le modèle ne connaît que ses codes.
        "etat": service.ETATS.get(etat_libelle),
        "couleur": couleur,
        "puissance_din": din, "puissance_fisc": fisc, "portes": portes, "places": places,
        "critair": critair, "ct_valide_jusqu_a": ct,
    })
    part = round(r["couverture"] * 10)
    return (
        f"## {_euros(r['bas'])} — {_euros(r['haut'])}\n\n"
        f"Estimation centrale : **{_euros(r['central'])}**\n\n"
        f"Cette fourchette contient le prix de vente **{part} fois sur 10** "
        f"({r['couverture']:.1%} mesurés sur 3 977 annonces jamais vues par le modèle).\n\n"
        f"*{MENTION}*"
    )


def construire() -> None:
    """Monte les composants de l'onglet dans le contexte Gradio courant."""
    gr.Markdown(
        "### Estimer le prix de votre véhicule\n"
        "Renseignez ce que vous savez. Les champs de la seconde section sont "
        "**facultatifs** : les laisser vides réduit la précision sans bloquer l'estimation."
    )

    with gr.Row():
        marque = gr.Dropdown(service.MARQUES, label="Marque", value="RENAULT")
        modele = gr.Dropdown(service.modeles_de("RENAULT"), label="Modèle", value=None)
        annee = gr.Number(label="Année de mise en circulation", value=2015, precision=0)
    with gr.Row():
        km = gr.Number(label="Kilométrage", value=120_000, minimum=0, maximum=1_000_000)
        energie = gr.Dropdown(service.ENERGIES, label="Énergie", value="Diesel")
        boite = gr.Dropdown(service.BOITES, label="Boîte de vitesses", value="Manuelle")
    etat = gr.Dropdown(list(service.ETATS), label="État déclaré", value="Usure normale")

    marque.change(_modeles, inputs=marque, outputs=modele)

    with gr.Accordion("Informations complémentaires (facultatif)", open=False):
        with gr.Row():
            din = gr.Number(label="Puissance DIN (ch)", value=None, minimum=1, maximum=2_000)
            fisc = gr.Number(label="Puissance fiscale (CV)", value=None, minimum=1, maximum=100)
            critair = gr.Number(label="Vignette Crit'Air (0 à 5)", value=None,
                                minimum=0, maximum=5)
        with gr.Row():
            portes = gr.Number(label="Nombre de portes", value=None, minimum=1, maximum=9)
            places = gr.Number(label="Nombre de places", value=None, minimum=1, maximum=9)
            ct = gr.Number(label="Contrôle technique valide jusqu'à (année)", value=None,
                           minimum=2000, maximum=2100)
        couleur = gr.Dropdown(service.COULEURS, label="Couleur", value=None)

    sortie = gr.Markdown()
    gr.Button("Estimer", variant="primary").click(
        _estimer,
        inputs=[marque, modele, annee, km, energie, boite, etat, couleur,
                din, fisc, portes, places, critair, ct],
        outputs=sortie,
    )
