"""Service d'estimation du prix — charge l'artefact et sert une fourchette.

Reçoit la saisie d'un formulaire (un véhicule, pas un jeu de données), la met dans l'état
exact où le modèle l'attend, et renvoie une **fourchette**, jamais un prix ponctuel
(`ml/AGENTS.md`).

Les artefacts viennent de `ml/src/entrainement.py` :

    app/models/prix.joblib   le modèle sérialisé
    app/models/prix.json     le vocabulaire des catégories, les tables de dérivation,
                             la date de référence, la MAE par tranche

Aucun import de `ml/` : ce module tourne dans l'image de l'application, qui n'embarque que
`app/`. Tout ce que le service doit savoir voyage dans le JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from preparation import (
    CAT,
    ETATS_LIBELLES,
    FREQUENCE_INCONNUE,
    NUM,
    construire_jeu,
)

MODELES = Path(__file__).resolve().parent.parent / "models"

# Chargés une fois à l'import, pas à chaque requête : désérialiser un modèle par appel
# ajouterait des centaines de millisecondes de latence pour un résultat identique.
#
# SÉCURITÉ — `joblib.load` déroule du pickle, donc exécute du code arbitraire. Acceptable
# ici, et seulement ici : le fichier est produit par `ml/src/entrainement.py` dans ce dépôt,
# versionné avec le code, et copié dans l'image au build. Il n'est jamais téléversé, ni lu
# depuis une entrée utilisateur, ni récupéré sur le réseau. Le jour où l'artefact viendrait
# d'ailleurs (registre de modèles, dépôt tiers), ce chargement devient une faille et il faut
# passer à un format sans exécution (ONNX, ou export des paramètres en JSON).
_modeles = joblib.load(MODELES / "prix.joblib")  # {"q10": …, "q50": …, "q90": …}
_meta: dict[str, Any] = json.loads((MODELES / "prix.json").read_text(encoding="utf-8"))

# Constante de conformalisation, mesurée sur un jeu de calibration jamais vu à l'ajustement.
# C'est elle qui transforme trois quantiles appris — systématiquement optimistes — en un
# intervalle dont la couverture est mesurée. Positive : les bornes brutes étaient trop
# étroites et sont élargies d'autant.
CONFORMAL_Q: float = _meta["conformal_q"]
COUVERTURE: float = _meta["couverture_test"]

CATEGORIES: dict[str, list[str]] = _meta["categories"]
# Date figée à l'entraînement. C'est elle, et non la date du jour, qui sert à calculer l'âge :
# sinon les âges d'entrée glissent hors de la plage apprise à mesure que le modèle vieillit,
# et le service dérive sans que rien ne le signale.
DATE_REFERENCE = pd.Timestamp(_meta["date_reference"])

# Libellés du formulaire (5 crans, regroupement calé sur les prix observés — cf.
# `preparation.ETATS_GROUPES`). Les valeurs techniques ne sont jamais traduites côté encodage.
ETATS = ETATS_LIBELLES

MARQUES = [m for m in CATEGORIES["marque"] if m != _meta["categorie_inconnue"]]
COULEURS = [c for c in CATEGORIES["couleur"] if c != _meta["categorie_inconnue"]]
ENERGIES = _meta["energies_saisissables"]
BOITES = ["Manuelle", "Automatique"]
MODELES_PAR_MARQUE: dict[str, list[str]] = _meta["modeles_par_marque"]
FREQUENCE_MODELE: dict[str, int] = _meta["frequence_modele"]


def modeles_de(marque: str | None) -> list[str]:
    """Modèles proposables pour une marque — alimente la liste en cascade du formulaire."""
    return MODELES_PAR_MARQUE.get(marque or "", [])


def _age(annee_mec: int) -> float:
    """Âge en années, calculé depuis l'ANNÉE de mise en circulation.

    Le formulaire ne demande plus le mois : `entrainement.py` recalcule l'âge de la même
    façon avant d'apprendre, pour que le service ne parle jamais une langue plus précise que
    celle du modèle. Coût mesuré de la perte du mois : −4 € de MAE, c'est-à-dire rien.
    """
    return (DATE_REFERENCE - pd.Timestamp(year=int(annee_mec), month=1, day=1)).days / 365.25


def estimer(saisie: dict[str, Any]) -> dict[str, Any]:
    """Une saisie de formulaire -> une fourchette de prix.

    Les dérivations reproduisent celles de `clean_leboncoin` : le modèle n'a jamais vu
    « boîte automatique » ni « mise en circulation », il a vu `boite_auto` et `age`.
    """
    marque = saisie.get("marque")
    energie = saisie.get("energie")

    ligne = {
        "age": _age(saisie["annee_mec"]),
        "kilometrage": saisie.get("kilometrage"),
        "puissance_din": saisie.get("puissance_din"),
        "puissance_fisc": saisie.get("puissance_fisc"),
        "portes": saisie.get("portes"),
        "places": saisie.get("places"),
        "critair": saisie.get("critair"),
        "ct_valide_jusqu_a": saisie.get("ct_valide_jusqu_a"),
        "boite_auto": 1 if saisie.get("boite") == "Automatique" else 0,
        # Marque hors table premium -> NaN, et non 0 : « pas de palier connu » n'est pas
        # « généraliste ». HistGradientBoosting apprend où envoyer les valeurs manquantes.
        "niveau": _meta["niveau_par_marque"].get(marque),
        # Modèle exact, encodé par sa fréquence au train — la même table que celle apprise.
        # Un modèle absent vaut 0 : plus rare que le plus rare vu, ce qui prolonge l'échelle
        # au lieu d'ouvrir une branche que le modèle n'a jamais apprise.
        "modele_freq": float(
            FREQUENCE_MODELE.get(saisie.get("modele") or "", FREQUENCE_INCONNUE)
        ),
        # Énergies trop rares pour être apprises séparément (GNV : 3 annonces) : regroupées,
        # comme à l'entraînement.
        "energie_grp": "Autre" if energie in _meta["energies_rares"] else energie,
        "marque": marque,
        "couleur": saisie.get("couleur"),
        "etat": saisie.get("etat"),
    }

    X = construire_jeu(pd.DataFrame([ligne]), categories=CATEGORIES)

    # Les bornes viennent de deux modèles distincts, pas d'un écart appliqué à une prédiction
    # centrale : c'est ce qui rend la fourchette ADAPTATIVE. Mesuré sur le test, elle va de
    # ~1 900 € de large en bas de gamme à ~13 700 € en haut — un écart qu'une largeur unique
    # ne pourrait pas produire.
    bas = float(_modeles["q10"].predict(X)[0]) - CONFORMAL_Q
    central = float(_modeles["q50"].predict(X)[0])
    haut = float(_modeles["q90"].predict(X)[0]) + CONFORMAL_Q

    # CROISEMENT DE QUANTILES. Les trois modèles sont ajustés indépendamment : rien ne les
    # contraint à rester ordonnés. Mesuré sur les 19 882 annonces : les bornes ne s'inversent
    # jamais, mais l'estimation centrale sort de la fourchette dans **1,20 % des cas**
    # (dépassement médian 275 €, maximum 2 647 €). Une fourchette qui ne contient pas son
    # propre prix central est incompréhensible pour un vendeur.
    #
    # On rabat le central dans l'intervalle plutôt que d'élargir l'intervalle : c'est lui qui
    # porte la garantie de couverture mesurée, le déformer invaliderait la promesse. Le
    # central n'est qu'une estimation ponctuelle, la rabattre ne casse rien.
    if bas > haut:
        bas, haut = haut, bas
    central = min(max(central, bas), haut)

    # Un prix négatif affiché à un vendeur détruirait toute crédibilité : plancher dur.
    bas, central = max(bas, 0.0), max(central, 0.0)

    return {
        "bas": round(bas),
        "central": round(central),
        "haut": round(haut),
        "largeur": round(haut - bas),
        # La couverture RÉELLEMENT mesurée sur le jeu de test, pas la cible visée. Les deux
        # ne coïncident pas exactement (79,5 % pour une cible de 80 %) et c'est le chiffre
        # mesuré qui doit être affiché — annoncer la cible serait promettre plus que ce
        # qu'on a constaté.
        "couverture": COUVERTURE,
    }


def contrat() -> dict[str, Any]:
    """Ce que le service attend et ce qu'il vaut — pour l'interface et pour l'API."""
    return {
        "features_num": NUM,
        "features_cat": CAT,
        "metriques": _meta["metriques"],
        # La promesse et sa vérification, exposées ensemble pour qu'elle soit contrôlable de
        # l'extérieur : la cible visée, la couverture constatée, et le détail par tranche de
        # prix — c'est ce dernier qui révèle que la garantie n'est pas uniforme.
        "quantiles": _meta["quantiles"],
        "alpha": _meta["alpha"],
        "conformal_q": _meta["conformal_q"],
        "couverture_test": _meta["couverture_test"],
        "largeur_mediane_test": _meta["largeur_mediane_test"],
        "couverture_par_tranche": _meta["couverture_par_tranche"],
        "date_reference": _meta["date_reference"],
        "version_sklearn": _meta["version_sklearn"],
    }
