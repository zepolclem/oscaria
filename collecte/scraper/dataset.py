#!/usr/bin/env python3
"""Rapatriement du dataset serveur (Postgres + MinIO) vers le disque local.

Le scraper (`scraper.py`) pousse vers le serveur ; ce script fait le trajet inverse, pour rendre
les données consommables en local. Chaque sortie atterrit dans le pilier qu'elle alimente :

    ml/data/leboncoin-private/raw/annonces.parquet            -> pilier Prix (tabulaire)
    dl/data/leboncoin-private/raw/images/<etat>/<ad_id>/NN.jpg -> pilier État (ImageFolder)

Commandes :
    python dataset.py annonces
    python dataset.py images [--workers 8]
    python dataset.py all
"""
from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from typing import Any

import pandas as pd
from psycopg.rows import dict_row

# Plomberie partagée avec le collecteur : connexion Postgres (verify-ca via la CA Coolify),
# client S3/MinIO et chargement du `.env`. Pas de duplication de config.
from scraper import BUCKET, pg_connect, s3_client

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))

# Chaque sortie va dans le pilier qu'elle alimente (cf. AGENTS.md §Structure cible) : le tabulaire
# sert le modèle Prix, les photos servent le modèle État. Un seul dataset, deux destinations.
PARQUET_PATH = os.path.join(ROOT, "ml", "data", "leboncoin-private", "raw", "annonces.parquet")
IMAGES_DIR = os.path.join(ROOT, "dl", "data", "leboncoin-private", "raw", "images")

# Préfixe des clés MinIO, retiré pour obtenir le chemin local : la racine locale est déjà
# `<dataset>/raw/images`, ré-empiler `scraping/leboncoin/` n'apporterait rien.
KEY_PREFIX = "scraping/leboncoin/"

# Colonnes entières nullables : Postgres renvoie None -> pandas passerait en float64 (2008.0).
INT_COLS = ("annee", "kilometrage", "nb_images")


# --------------------------------------------------------------------------- annonces
def export_annonces() -> int:
    """Table `annonces` -> Parquet. Écriture atomique (tmp + replace)."""
    with pg_connect() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            rows: list[dict[str, Any]] = cur.execute(
                "SELECT * FROM annonces ORDER BY pk").fetchall()
    if not rows:
        print("aucune annonce en base")
        return 0

    df = pd.DataFrame(rows)

    # `raw` est du JSONB -> psycopg le rend en dict. Parquet n'a pas de schéma stable à y appliquer
    # (attributs leboncoin hétérogènes) : on le garde en texte JSON, décodable à la demande.
    if "raw" in df:
        df["raw"] = df["raw"].map(lambda v: json.dumps(v, ensure_ascii=False)
                                  if v is not None else None)
    # NUMERIC -> Decimal, que pyarrow refuse dans une colonne `object`.
    if "prix_eur" in df:
        df["prix_eur"] = df["prix_eur"].map(
            lambda v: float(v) if isinstance(v, Decimal) else v).astype("float64")
    for col in INT_COLS:
        if col in df:
            df[col] = df[col].astype("Int64")

    os.makedirs(os.path.dirname(PARQUET_PATH), exist_ok=True)
    tmp = PARQUET_PATH + ".tmp"
    df.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, PARQUET_PATH)

    size_mb = os.path.getsize(PARQUET_PATH) / 1e6
    print(f"{len(df)} annonces -> {PARQUET_PATH} ({size_mb:.1f} Mo)")
    print(df["etat"].fillna("(null)").value_counts().to_string())
    return len(df)


# --------------------------------------------------------------------------- images
_local = threading.local()


def _s3():
    """Un client boto3 par thread : un client partagé n'est pas thread-safe."""
    if not hasattr(_local, "client"):
        _local.client = s3_client()
    return _local.client


def remote_sizes() -> dict[str, int]:
    """Taille de chaque objet du bucket, en ~23 requêtes paginées.

    Sert au test d'idempotence. Un `head_object` par image coûterait 22 000 allers-retours pour
    la même information.
    """
    sizes: dict[str, int] = {}
    paginator = s3_client().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=KEY_PREFIX):
        for obj in page.get("Contents", []):
            sizes[obj["Key"]] = obj["Size"]
    return sizes


def wanted_keys() -> list[str]:
    """Clés à rapatrier, lues dans `image_keys`.

    C'est la colonne Postgres qui fait foi, pas le contenu du bucket : elle seule relie une image
    à son annonce et à son état. Lister le bucket ramènerait aussi d'éventuels orphelins.
    """
    with pg_connect() as conn:
        rows = conn.execute(
            "SELECT image_keys FROM annonces WHERE image_keys IS NOT NULL ORDER BY pk").fetchall()
    return [key for (keys,) in rows for key in keys]


def local_path(key: str) -> str:
    return os.path.join(IMAGES_DIR, key[len(KEY_PREFIX):] if key.startswith(KEY_PREFIX) else key)


def fetch_one(key: str) -> int:
    """Télécharge une image. Renvoie ses octets écrits, 0 si déjà à jour."""
    path = local_path(key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    _s3().download_file(BUCKET, key, tmp)
    os.replace(tmp, path)
    return os.path.getsize(path)


def export_images(workers: int = 8) -> tuple[int, int]:
    """Miroir MinIO -> disque. Idempotent : une reprise ne retélécharge que ce qui manque."""
    keys = wanted_keys()
    sizes = remote_sizes()
    todo = [
        k for k in keys
        # Déjà présent ET de la bonne taille -> rien à faire. Une taille distante inconnue
        # (clé listée en base mais absente du bucket) tombe dans le `todo` et échouera
        # bruyamment, ce qui est le comportement voulu : on veut le savoir.
        if not (os.path.exists(local_path(k)) and os.path.getsize(local_path(k)) == sizes.get(k))
    ]
    print(f"{len(keys)} images référencées, {len(keys) - len(todo)} déjà à jour, "
          f"{len(todo)} à télécharger")
    if not todo:
        return 0, 0

    done = failed = total_bytes = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_safe_fetch, todo):
            if result is None:
                failed += 1
            else:
                done += 1
                total_bytes += result
            if (done + failed) % 500 == 0:
                print(f"  {done + failed}/{len(todo)} ({total_bytes / 1e6:.0f} Mo)", flush=True)
    print(f"{done} images écrites ({total_bytes / 1e6:.0f} Mo), {failed} en échec")
    return done, failed


def _safe_fetch(key: str) -> int | None:
    """Une image morte ne doit pas tuer le run — on la signale et on continue."""
    try:
        return fetch_one(key)
    except Exception as exc:
        print(f"    KO {key}: {exc}", flush=True)
        return None


# --------------------------------------------------------------------------- cli
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("annonces", help="Postgres -> annonces.parquet")
    img = sub.add_parser("images", help="MinIO -> images/<etat>/<ad_id>/NN.jpg")
    img.add_argument("--workers", type=int, default=8)
    a = sub.add_parser("all", help="les deux")
    a.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if args.cmd in ("annonces", "all"):
        export_annonces()
    if args.cmd in ("images", "all"):
        export_images(args.workers)


if __name__ == "__main__":
    main()
