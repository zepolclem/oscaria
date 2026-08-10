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

# Jeu de variables — contrat v2, **tous champs obligatoires**. Le carnet 05 a montré que les
# champs facultatifs vides tiraient l'estimation vers le bas : le modèle avait appris des
# annonces que « champ manquant = annonce bâclée = pas cher », signal sans aucun sens pour un
# vendeur qui ne connaît pas la valeur. Le v2 supprime la cause : plus de champ facultatif,
# et un contrat réduit aux variables qui portent le prix. Chaque écart est chiffré en
# validation croisée 5 plis (carnet 06), pas décidé au jugé :
#
#   - `region` RETIRÉE (v1)       −14 € de MAE. Importance par permutation 0,0014.
#   - `age` depuis l'ANNÉE seule   −4 € de MAE (v1). Le formulaire ne demande plus le mois.
#   - `modele_freq` AJOUTÉE (v1)  −35 € de MAE. Le modèle exact, encodé par sa fréquence.
#   - `critair`, `ct` RETIRÉS      −4 € de MAE : du bruit (importances 0,0002 et −0,0005).
#   - `couleur` RETIRÉE           −15 € de MAE : du bruit aussi (0,0001).
#   - `puissance_fisc` RETIRÉE    +11 € de MAE : redondante avec la DIN (0,045 vs 0,208).
#   - `portes`, `places` RETIRÉES +14 € de MAE : le seul retrait non trivial, assumé.
#   - états 5 -> 4 crans          +17 € de MAE (cf. ETATS_GROUPES ci-dessous).
#
# Cumul v2 : +43 € de MAE, le prix d'un formulaire de 8 champs qu'un vendeur remplit en
# entier — contre un biais de −506 € (médian) quand il laissait les champs vides.
#
# Ces listes sont le **contrat d'entrée** du service : les modifier impose de ré-entraîner ET
# de reprendre le formulaire, jamais l'un sans l'autre.
NUM = ["age", "kilometrage", "puissance_din", "boite_auto", "niveau", "modele_freq"]
CAT = ["energie_grp", "marque", "etat"]

# Regroupement de l'échelle d'état déclaré, de 8 crans à 4. Calé sur les **prix médians
# observés**, pas sur le sens commun : `not_drivable` (1 300 €), `damaged`,
# `major_repairs_needed` (1 500 € chacun) et `minor_repairs_needed` (~2 000 €) restent
# proches, les fusionner coûte peu (+17 € de MAE mesurés au carnet 06, fusion comprise dans
# le passage 5 -> 4 crans) ; `undamaged` (9 000 €) et `excellent_condition` (16 000 €) sont
# séparés par 7 000 €, les fusionner perdrait vraiment — ils restent distincts.
ETATS_GROUPES = {
    "not_drivable": "1_a_reparer",
    "damaged": "1_a_reparer",
    "major_repairs_needed": "1_a_reparer",
    "minor_repairs_needed": "1_a_reparer",
    "normal_wear_and_tear": "2_usure",
    "good_overall_condition": "3_bon",
    "undamaged": "3_bon",
    "excellent_condition": "4_excellent",
}

# Libellés du formulaire -> cran du modèle. Ordre d'affichage, du pire au meilleur.
ETATS_LIBELLES = {
    "Ne roule pas ou réparations à prévoir": "1_a_reparer",
    "Usure normale": "2_usure",
    "Bon état": "3_bon",
    "Excellent état": "4_excellent",
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
