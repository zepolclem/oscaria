"""Feature engineering partagé — pilier Prix OscarIA.

Nettoyage du dataset brut, extraction de la marque depuis `carmodel` et
jointure de la classification premium (table `ml/references/premium_brand.csv`).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def to_num(serie: pd.Series) -> pd.Series:
    """Extrait un nombre d'une colonne texte sale ('27 297 Km', '11 080 €', '12 mois')."""
    return pd.to_numeric(
        serie.astype(str)
        .str.replace(r"[^\d,.-]", "", regex=True)  # garde chiffres . , -
        .str.replace(",", ".", regex=False)
        .replace("", np.nan),
        errors="coerce",
    )


def clean_cars(raw_path: str | Path) -> pd.DataFrame:
    """Nettoyage minimal du dataset brut -> DataFrame exploitable.

    - retire les électriques purs (garde les hybrides) ;
    - parse la cible et les colonnes numériques sales ;
    - dérive les features catégorielles : boîte auto, garantie constructeur, durée garantie ;
    - garde-fous : prix valide, année plausible.
    Le fichier brut n'est jamais modifié.
    """
    df = pd.read_csv(raw_path)

    # retirer les électriques purs
    df["énergie"] = df["énergie"].astype(str).str.strip()
    df = df[df["énergie"] != "Electrique"].copy()

    # cible + numériques
    df["price"] = to_num(df["price"])
    df["kilometrage"] = to_num(df["kilométragecompteur"])
    df["puissance_din"] = to_num(df["puissancedin"])
    df["puissance_fisc"] = to_num(df["puissancefiscale"])
    df["annee"] = pd.to_numeric(df["année"], errors="coerce")

    # features dérivées
    df["boite_auto"] = (df["boîtedevitesse"].astype(str).str.strip() == "automatique").astype(int)
    df["garantie_constructeur"] = (
        df["garantieconstructeur"].astype(str).str.strip() == "en cours"
    ).astype(int)
    df["garantie_mois"] = to_num(df["garantie"])

    # garde-fous
    df = df[df["price"].notna()]
    df = df[df["annee"].between(1980, 2026)]
    return df


def load_premium_table(path: str | Path) -> pd.DataFrame:
    """Charge la table de référence marque -> palier premium.

    Colonnes attendues : marque, palier (generaliste/premium/luxe), niveau (0/1/2).
    """
    ref = pd.read_csv(path)
    ref["marque"] = ref["marque"].str.strip().str.upper()
    return ref


def _normalize(carmodel: str) -> str:
    """Majuscules, tirets -> espaces, espaces multiples compactés."""
    s = str(carmodel).upper().replace("-", " ")
    return re.sub(r"\s+", " ", s).strip()


def extract_brand(carmodel: str, brands: list[str]) -> str | None:
    """Renvoie la marque connue en préfixe de `carmodel`, sinon None.

    `brands` est testé du plus long au plus court pour que les marques
    multi-mots (ALFA ROMEO, LAND ROVER) l'emportent sur un préfixe partiel.
    Le préfixe doit s'arrêter sur une frontière de mot (évite qu'une marque
    soit un bout d'un mot plus long).
    """
    text = _normalize(carmodel)
    for brand in sorted(brands, key=lambda b: -len(b)):
        b = _normalize(brand)
        if text == b or text.startswith(b + " "):
            return brand
    return None


def add_brand_features(
    df: pd.DataFrame, ref: pd.DataFrame, carmodel_col: str = "carmodel"
) -> pd.DataFrame:
    """Ajoute les colonnes `marque`, `palier`, `niveau` à partir de `carmodel`.

    Ne modifie pas `df` en place : renvoie une copie enrichie.
    """
    brands = ref["marque"].tolist()
    out = df.copy()
    out["marque"] = out[carmodel_col].map(lambda x: extract_brand(x, brands))
    out = out.merge(ref[["marque", "palier", "niveau"]], on="marque", how="left")
    return out
