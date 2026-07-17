"""Feature engineering partagé — pilier Prix OscarIA.

Nettoyage du dataset brut, extraction de la marque depuis `carmodel` et
jointure de la classification premium (table `ml/references/premium_brand.csv`).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# Référence pour l'âge : données collectées ~février 2023
# (miseencirculation max = 01/02/2023). REF_DATE juste après -> âges tous positifs.
REF_YEAR = 2023
REF_DATE = pd.Timestamp("2023-03-01")

# Catégories de carburant figées -> code ordinal stable entre les sous-échantillons
# (les électriques purs sont retirés en amont).
CARBURANTS = [
    "Essence",
    "Diesel",
    "Hybride essence électrique",
    "Hybride diesel électrique",
]


def to_num(serie: pd.Series) -> pd.Series:
    """Extrait le PREMIER nombre d'une colonne texte sale.

    Gère les espaces de milliers ('27 297 Km', '11 080\\xa0€') et la décimale
    ',' ou '.' ('5,2 l/100km'). On prend le premier nombre et on s'arrête :
    concaténer tous les chiffres corromprait les unités qui en contiennent
    ('4 l/100km' doit donner 4, pas 4100).
    """
    ext = serie.astype(str).str.extract(
        r"(-?\d[\d\s  ]*(?:[.,]\d+)?)", expand=False
    )
    return pd.to_numeric(
        ext.str.replace(r"[\s  ]", "", regex=True)  # colle les milliers
        .str.replace(",", ".", regex=False),
        errors="coerce",
    )


def clean_cars(raw_path: str | Path, max_price: float | None = 50_000) -> pd.DataFrame:
    """Nettoyage minimal du dataset brut -> DataFrame exploitable.

    - retire les électriques purs (garde les hybrides) ;
    - parse la cible et les colonnes numériques sales ;
    - dérive les features : boîte auto, garanties, première main, motorisation, âge fin ;
    - garde-fous : prix valide, année plausible ;
    - applique le périmètre produit `max_price` (défaut 50 000 €, décision ADR 0002) —
      passer `max_price=None` pour les données complètes.
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

    # âge en années FRACTIONNAIRES depuis la date de mise en circulation
    # (plus fin que l'année-millésime : janv. vs déc. = ~1 an d'écart réel ;
    # 193 annonces ont d'ailleurs année ≠ année de mise en circulation).
    # Repli sur l'année-millésime si la date ne se parse pas.
    mec = pd.to_datetime(
        df["miseencirculation"].astype(str).str.strip(),
        format="%d/%m/%Y", errors="coerce",
    )
    df["age"] = ((REF_DATE - mec).dt.days / 365.25).fillna(REF_YEAR - df["annee"])

    # features dérivées
    df["boite_auto"] = (df["boîtedevitesse"].astype(str).str.strip() == "automatique").astype(int)
    df["garantie_constructeur"] = (
        df["garantieconstructeur"].astype(str).str.strip() == "en cours"
    ).astype(int)
    df["garantie_mois"] = to_num(df["garantie"])
    df["premiere_main"] = (
        df["premièremain(déclaratif)"].astype(str).str.strip() == "oui"
    ).astype(int)

    # motorisation / pollution
    # énergie -> code ordinal stable (catégories figées, inconnu = -1)
    df["energie_code"] = pd.Categorical(
        df["énergie"], categories=CARBURANTS
    ).codes
    df["critair"] = to_num(df["crit'air"])            # vignette pollution 1..4
    df["conso"] = to_num(df["consommationmixte"])     # "5.2 l/100km" -> 5.2

    # modèle exact nettoyé (retours-ligne, espaces multiples) — encodage
    # (ex. fréquence) à calculer sur le train uniquement, dans les notebooks
    df["modele"] = (
        df["carmodel"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    )

    # garde-fous
    df = df[df["price"].notna()]
    df = df[df["annee"].between(1980, 2026)]

    # périmètre produit (ADR 0002)
    if max_price is not None:
        df = df[df["price"] <= max_price]
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
