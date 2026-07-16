"""Feature engineering partagé — pilier Prix OscarIA.

Extraction de la marque depuis `carmodel` et jointure de la classification
premium (table de référence `ml/references/premium_brand.csv`).
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


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
