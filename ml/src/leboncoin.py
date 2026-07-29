"""Nettoyage et features — dataset `leboncoin-private`, pilier Prix OscarIA.

Entrée : `ml/data/leboncoin-private/raw/annonces.parquet` (20 915 annonces de particuliers,
produit par `collecte/scraper/dataset.py`). Sortie : un DataFrame prêt à modéliser, plus un
**journal** qui chiffre chaque ligne perdue — aucune suppression silencieuse (attendu BC04).

Module distinct de `features.py`, qui reste dédié au dataset `french-second-hand-cars` (colonnes
codées en dur). Les utilitaires génériques, eux, sont réutilisés depuis `features.py`.

Décisions appliquées ici : cf. plan de nettoyage et le Gate EDA
(`ml/notebooks/leboncoin-private/01_eda_inspection.ipynb`).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import _normalize, load_premium_table, to_num

# Échelle d'état déclaré (`vehicle_damage`), du pire au meilleur. Ordre figé : il porte
# l'information ordinale, le réordonner casserait l'encodage.
ETATS_ORDONNES = [
    "not_drivable",
    "damaged",
    "major_repairs_needed",
    "minor_repairs_needed",
    "normal_wear_and_tear",
    "good_overall_condition",
    "undamaged",
    "excellent_condition",
]

# Modalités d'énergie trop rares pour être apprises séparément (GNV = 3 lignes) : regroupées.
# Électriques et hybrides sont CONSERVÉS — le motif de l'ADR 0001 (batterie en location rendant
# le prix non comparable) reposait sur une colonne du dataset 2023 qui n'existe pas ici.
ENERGIES_RARES = ["GPL", "Autre", "Gaz Naturel (GNV)"]

# Kilométrage sentinelle : 999 999 n'est pas un kilométrage, c'est un champ rempli au maximum.
# On neutralise la valeur sans jeter la ligne (le reste de l'annonce est valide).
KM_SENTINELLE = 999_999

# Quintuplet identifiant un véhicule : deux annonces qui coïncident dessus sont la même voiture
# republiée. À dédupliquer AVANT tout split, sinon fuite entre entraînement et test.
CLES_VEHICULE = ["marque", "modele", "annee", "kilometrage", "prix_eur"]

# Colonnes volontairement écartées des features. Le motif compte autant que l'exclusion.
COLONNES_ECARTEES = {
    "car_price_min": "estimation de prix de leboncoin (45 %) — le modèle recopierait un estimateur "
                     "tiers au lieu du marché, et s'effondrerait sur les 55 % qui ne l'ont pas. "
                     "Réservée comme comparateur externe à l'évaluation.",
    "car_price_max": "idem car_price_min.",
    "old_price": "prix précédent de la même annonce (9 %) — ancre quasi circulaire pour la cible. "
                 "Réservé au pilier Date (baisse de prix ↔ délai de vente).",
    "is_import": "'false' sur 20 915 / 20 915 — variance nulle, aucune information.",
    "body": "vide sur 20 915 / 20 915 — la description n'est pas rendue sur le listing leboncoin.",
}


def _attributs(df: pd.DataFrame) -> pd.DataFrame:
    """Déplie le bloc `attributes` du JSON `raw` en colonnes.

    Les colonnes chaudes du Parquet ne reprennent qu'une partie des attributs leboncoin ; le
    reste (puissance, portes, Crit'Air, contrôle technique…) n'existe que là.
    """
    return pd.DataFrame(
        [json.loads(s).get("attributes", {}) for s in df["raw"]], index=df.index
    )


def _age_annees(
    attrs: pd.DataFrame, annee: pd.Series, ref: pd.Timestamp, min_year: int
) -> tuple[pd.Series, int]:
    """Âge en années fractionnaires depuis la mise en circulation (`issuance_date`, MM/YYYY).

    Plus fin que l'année-millésime : janvier vs décembre, c'est presque un an d'écart réel.
    Repli sur l'année-millésime dans deux cas :

    - la date ne se parse pas (`issuance_date` absent, ~1 % des annonces) ;
    - **elle se parse mais est aberrante** — quelques vendeurs saisissent `12/0994` pour 1994 ou
      `01/0165` pour 1965. Le format est valide, l'année ne l'est pas : sans ce garde-fou l'âge
      part à plus de 1 000 ans. La colonne `annee`, elle, est correcte sur ces annonces.

    Renvoie `(age, n_dates_aberrantes)`.
    """
    mec = pd.to_datetime(
        attrs.get("issuance_date", pd.Series(index=attrs.index, dtype="object")),
        format="%m/%Y", errors="coerce",
    )
    aberrante = mec.notna() & ~mec.dt.year.between(min_year, ref.year + 1)
    mec = mec.mask(aberrante)
    age = (ref - mec).dt.days / 365.25
    return age.fillna(ref.year - annee).astype("float64"), int(aberrante.sum())


def clean_leboncoin(
    parquet_path: str | Path,
    premium_path: str | Path,
    min_price: float = 500,
    max_price: float | None = 50_000,
    min_year: int = 1980,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nettoie le dataset leboncoin et dérive les features du pilier Prix.

    Renvoie `(df_propre, journal)` où `journal` chiffre chaque règle appliquée : c'est la pièce
    qui rend le nettoyage défendable, pas un effet de bord.

    - `min_price` / `max_price` : périmètre de prix. `max_price` vaut 50 000 € par défaut, la
      décision **ADR 0002** en vigueur ; passer 100_000 ou None pour mesurer d'autres périmètres.
      Le fichier `raw/` n'est jamais modifié.
    - `min_year` : les véhicules d'avant 1980 relèvent de la collection, pas du marché d'occasion
      grand public — leur cote n'est pas comparable (miroir de `features.clean_cars`).
    """
    df = pd.read_parquet(parquet_path)
    journal: list[dict] = []
    n = len(df)

    def note(regle: str, avant: int) -> None:
        journal.append({"regle": regle, "perdues": avant - len(df), "restantes": len(df)})

    # --- garde-fous ---------------------------------------------------------------
    avant = len(df)
    df = df[df["prix_eur"] >= min_price]
    note(f"prix_eur >= {min_price:,.0f} EUR", avant)

    if max_price is not None:
        avant = len(df)
        df = df[df["prix_eur"] <= max_price]
        note(f"prix_eur <= {max_price:,.0f} EUR (ADR 0002)", avant)

    avant = len(df)
    df = df[df["annee"] >= min_year]
    note(f"annee >= {min_year}", avant)

    avant = len(df)
    df = df.drop_duplicates(subset=CLES_VEHICULE, keep="first")
    note("quasi-doublons vehicule (marque+modele+annee+km+prix)", avant)

    df = df.copy()

    # Sentinelle km : on neutralise la VALEUR, on garde la LIGNE.
    n_sentinelle = int((df["kilometrage"] >= KM_SENTINELLE).sum())
    df.loc[df["kilometrage"] >= KM_SENTINELLE, "kilometrage"] = pd.NA
    journal.append({"regle": f"kilometrage >= {KM_SENTINELLE} -> NaN ({n_sentinelle} valeurs, "
                             f"aucune ligne jetee)", "perdues": 0, "restantes": len(df)})

    # --- features -----------------------------------------------------------------
    attrs = _attributs(df)
    ref_date = pd.Timestamp(df["scraped_at"].max()).tz_localize(None)

    df["age"], n_mec_aberrantes = _age_annees(attrs, df["annee"], ref_date, min_year)
    journal.append({"regle": f"mise en circulation aberrante -> repli sur annee "
                             f"({n_mec_aberrantes} valeurs)", "perdues": 0, "restantes": len(df)})
    df["boite_auto"] = (df["boite"] == "Automatique").astype("int8")

    df["energie_grp"] = df["energie"].where(~df["energie"].isin(ENERGIES_RARES), "Autre")

    df["etat_ord"] = pd.Categorical(df["etat"], categories=ETATS_ORDONNES, ordered=True)
    df["etat_niveau"] = df["etat_ord"].cat.codes.astype("int8")

    df["puissance_din"] = to_num(attrs.get("horse_power_din", pd.Series(dtype="object")))
    df["puissance_fisc"] = to_num(attrs.get("horsepower", pd.Series(dtype="object")))
    df["portes"] = to_num(attrs.get("doors", pd.Series(dtype="object"))).astype("Int64")
    df["places"] = to_num(attrs.get("seats", pd.Series(dtype="object"))).astype("Int64")
    df["critair"] = to_num(attrs.get("critair", pd.Series(dtype="object"))).astype("Int64")
    df["ct_valide_jusqu_a"] = to_num(
        attrs.get("vehicle_technical_inspection_a", pd.Series(dtype="object"))
    ).astype("Int64")
    df["couleur"] = attrs.get("vehicule_color")

    # Palier premium : jointure sur la marque NORMALISÉE (tirets -> espaces, majuscules), sinon
    # MERCEDES-BENZ et LAND-ROVER manquent la table de référence.
    ref = load_premium_table(premium_path)
    ref["_cle"] = ref["marque"].map(_normalize)
    df["_cle"] = df["marque"].map(_normalize)
    df = df.merge(ref[["_cle", "palier", "niveau"]], on="_cle", how="left").drop(columns="_cle")

    # `raw` et `images` ont été consommés. Les garder ferait voyager les colonnes écartées
    # (`car_price_*`, `old_price`, `body`) dans le jeu nettoyé, où elles n'ont rien à faire.
    # Elles restent récupérables à tout moment depuis le Parquet `raw/`, jointure sur `ad_id`.
    df = df.drop(columns=[c for c in ("raw", "images") if c in df.columns])

    return df, pd.DataFrame(journal)


def colonnes_modele(df: pd.DataFrame) -> list[str]:
    """Colonnes destinées au modèle — tout le reste est identifiant, brut ou écarté."""
    features = [
        "age", "annee", "kilometrage", "puissance_din", "puissance_fisc", "portes", "places",
        "critair", "ct_valide_jusqu_a", "boite_auto", "niveau", "etat_niveau",
        "energie_grp", "marque", "modele", "couleur", "region", "palier", "etat",
    ]
    return [c for c in features if c in df.columns]


def resume_journal(journal: pd.DataFrame, n_depart: int) -> str:
    """Rend le journal lisible : chaque règle, son coût, et le total qui doit boucler."""
    lignes = [f"{'regle':<58}{'perdues':>9}{'restantes':>11}", "-" * 78]
    for _, r in journal.iterrows():
        lignes.append(f"{r['regle']:<58}{r['perdues']:>9}{r['restantes']:>11}")
    total = int(journal["perdues"].sum())
    lignes += ["-" * 78,
               f"{'TOTAL':<58}{total:>9}{n_depart - total:>11}",
               f"depart {n_depart} - {total} perdues = {n_depart - total}"]
    return "\n".join(lignes)
