"""Mise en forme des données pour le modèle Prix — **partagée entraînement / service**.

Ce module n'a qu'une raison d'être : garantir qu'un véhicule est encodé de la même façon
quand il sert à entraîner et quand il arrive d'un formulaire. C'est la classe de bug la plus
coûteuse d'un service de machine learning — le *décalage entraînement/service* — parce qu'elle
ne lève aucune erreur : elle rend juste des prédictions fausses.

D'où la duplication interdite : `ml/src/entrainement.py` **importe** ce fichier, il n'en
recopie pas le contenu. Il vit dans `app/` parce que c'est l'image de l'application qui doit
l'embarquer ; `ml/` n'est jamais copié dans le conteneur.

Aucune dépendance vers `ml/`, aucun accès disque, aucun artefact chargé : rien d'autre que la
transformation. C'est ce qui le rend importable des deux côtés sans ordre d'initialisation.
"""

from __future__ import annotations

import pandas as pd

# Jeu de variables. Dérivé du carnet 04, puis remanié pour alléger le formulaire — chaque
# écart est chiffré en validation croisée 5 plis, pas décidé au jugé :
#
#   - `region` RETIRÉE            −14 € de MAE. Importance par permutation 0,0014 au carnet
#                                 04 : elle n'apportait pas d'information, seulement du bruit
#                                 dans lequel l'arbre creusait des coupes non généralisables.
#   - `age` depuis l'ANNÉE seule   −4 € de MAE. Le formulaire ne demande plus le mois de mise
#                                 en circulation ; la perte de finesse est dans le bruit.
#   - `modele_freq` AJOUTÉE       −35 € de MAE. Le modèle exact du véhicule, encodé par sa
#                                 fréquence dans le train (cf. plus bas).
#
# Ces listes sont le **contrat d'entrée** du service : les modifier impose de ré-entraîner ET
# de reprendre le formulaire, jamais l'un sans l'autre.
NUM = ["age", "kilometrage", "puissance_din", "puissance_fisc", "portes", "places",
       "critair", "ct_valide_jusqu_a", "boite_auto", "niveau", "modele_freq"]
CAT = ["energie_grp", "marque", "couleur", "etat"]

# Regroupement de l'échelle d'état déclaré, de 8 crans à 5. Calé sur les **prix médians
# observés**, pas sur le sens commun : `not_drivable` (1 300 €), `damaged` et
# `major_repairs_needed` (1 500 € chacun) sont indiscernables en prix, les fusionner ne perd
# rien ; `undamaged` (9 000 €) et `excellent_condition` (16 000 €) sont séparés par 7 000 €,
# les fusionner en perdrait. Mesuré : 1 467 € de MAE contre 1 471 € à 8 crans — le
# regroupement est même très légèrement gagnant.
ETATS_GROUPES = {
    "not_drivable": "1_hors_service",
    "damaged": "1_hors_service",
    "major_repairs_needed": "1_hors_service",
    "minor_repairs_needed": "2_a_reparer",
    "normal_wear_and_tear": "3_usure",
    "good_overall_condition": "4_bon",
    "undamaged": "4_bon",
    "excellent_condition": "5_excellent",
}

# Libellés du formulaire -> cran du modèle. Ordre d'affichage, du pire au meilleur.
ETATS_LIBELLES = {
    "Ne roule pas ou grosses réparations": "1_hors_service",
    "Petites réparations à prévoir": "2_a_reparer",
    "Usure normale": "3_usure",
    "Bon état": "4_bon",
    "Excellent état": "5_excellent",
}

# Fréquence attribuée à un modèle jamais vu à l'entraînement. Zéro plutôt que « valeur
# manquante » : la fréquence est une échelle continue où le minimum appris vaut 1, donc 0
# prolonge naturellement le « encore plus rare que le plus rare ». Un NaN enverrait la ligne
# dans une branche que le modèle n'a jamais eu l'occasion d'apprendre.
FREQUENCE_INCONNUE = 0.0

# Bornes de tranche du carnet 04 §3 — elles fixent la largeur de la fourchette servie.
TRANCHES = [500, 2000, 5000, 10000, 20000, 50000]

# Modalité de repli des catégories absentes. Une valeur explicite plutôt qu'un trou :
# « non renseigné » est une information, et l'arbre peut l'isoler par une coupe.
INCONNU = "(inconnu)"


def construire_jeu(
    df: pd.DataFrame, categories: dict[str, list[str]] | None = None
) -> pd.DataFrame:
    """Met un DataFrame au format exact attendu par le modèle.

    `categories` fige le vocabulaire de chaque colonne catégorielle et **doit** être fourni à
    l'inférence. `HistGradientBoostingRegressor` indexe les catégories **par position** dans
    le dtype pandas : laisser pandas déduire les modalités d'une seule ligne de formulaire
    renumérote tout, et une `RENAULT` se retrouve lue comme une `ABARTH`. Silencieusement.

    À l'entraînement (`categories=None`), le vocabulaire est déduit du jeu complet puis
    enregistré dans les métadonnées. À l'inférence, toute valeur hors vocabulaire retombe sur
    `INCONNU` — une marque jamais vue n'a pas à faire tomber le service.
    """
    X = df.reindex(columns=NUM + CAT).copy()
    X[NUM] = X[NUM].astype("float64")
    for c in CAT:
        col = X[c].fillna(INCONNU).astype("string")
        if categories is not None:
            col = col.where(col.isin(categories[c]), INCONNU)
            X[c] = pd.Categorical(col, categories=categories[c])
        else:
            X[c] = pd.Categorical(col)
    return X
