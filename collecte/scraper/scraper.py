#!/usr/bin/env python3
"""Collecteur leboncoin -> Postgres + MinIO.

Pas de navigateur : le listing leboncoin est rendu côté serveur dans `__NEXT_DATA__`, et le gate
DataDome porte sur le fingerprint TLS (JA3). `curl_cffi` rejoue le handshake d'un vrai Chrome ->
HTTP 200 sans challenge JS, ~0,5 s/page au lieu de ~60 s en navigateur headless.

Commandes :
    python scraper.py collect --condition damaged --pages 1-22
    python scraper.py images                 # draine les annonces sans image_keys
    python scraper.py all                    # collect toutes les catégories + images
    python scraper.py loop --every 3600      # boucle `all` (mode conteneur)
    python scraper.py stats
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from typing import Any, Iterator

import boto3
import psycopg
from botocore.client import Config
from curl_cffi import requests

HERE = os.path.dirname(os.path.abspath(__file__))


def load_env() -> None:
    """Charge `.env` (gitignored) sans dépendance externe. Les vars déjà définies gagnent."""
    path = os.path.join(HERE, ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


load_env()

SEARCH_BASE = "https://www.leboncoin.fr/recherche?category=2&owner_type=private"
NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.S
)

# États véhicule leboncoin (attribut vehicle_damage) -> nb de pages à collecter.
# Classes rares : prises EN ENTIER (leboncoin n'en a pas plus, elles plafonnent le dataset vision).
# Classes fréquentes : au plafond leboncoin (max_pages=100, soit ~3 500 annonces) -> volume pour le
# modèle Prix.
ETATS: dict[str, int] = {
    "damaged": 22,               # ~736 dispo -> épuisé
    "not_drivable": 35,          # ~1 217 dispo -> épuisé
    "major_repairs_needed": 100,
    "minor_repairs_needed": 100,
    "normal_wear_and_tear": 100,
    "undamaged": 100,
    "good_overall_condition": 100,
    "excellent_condition": 100,
}

# Quotas de photos par état. Les images ne servent qu'au pilier État (vision) : inutile de stocker les
# photos de dizaines de milliers de voitures saines qui n'alimentent que le modèle Prix (où l'image ne
# sert pas). On prend TOUT des classes abîmées (rares) et un échantillon des autres -> dataset vision
# équilibré et stockage maîtrisé. None = pas de limite.
IMG_QUOTAS: dict[str, int | None] = {
    "damaged": None,
    "not_drivable": None,
    "major_repairs_needed": 1000,
    "minor_repairs_needed": 1000,
    "normal_wear_and_tear": 700,
    "undamaged": 700,
    "good_overall_condition": 700,
    "excellent_condition": 700,
}

# Bruit vendeur/affichage : hors périmètre véhicule, on ne le garde pas en colonne.
NOISE_ATTRS = {"rating_score", "rating_count", "profile_picture_url"}

BUCKET = os.environ.get("MINIO_BUCKET", "oscaria-images")
RATE_SLEEP = float(os.environ.get("RATE_SLEEP", "1.5"))  # ~40 req/min = stable
IMG_BATCH = int(os.environ.get("IMG_BATCH", "200"))

DDL = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS annonces (
  pk BIGSERIAL PRIMARY KEY, source TEXT NOT NULL, ad_id TEXT NOT NULL, url TEXT,
  etat TEXT, etat_label TEXT, owner_type TEXT, prix_eur NUMERIC, marque TEXT, modele TEXT,
  annee INT, kilometrage INT, energie TEXT, boite TEXT, ville TEXT, code_postal TEXT,
  departement TEXT, region TEXT, nb_images INT, images TEXT[], image_keys TEXT[],
  raw JSONB NOT NULL, scraped_at TIMESTAMPTZ, inserted_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE(source, ad_id));
CREATE INDEX IF NOT EXISTS idx_annonces_etat ON annonces(etat);
CREATE INDEX IF NOT EXISTS idx_annonces_owner ON annonces(owner_type);
CREATE INDEX IF NOT EXISTS idx_annonces_source ON annonces(source);
CREATE INDEX IF NOT EXISTS idx_annonces_prix ON annonces(prix_eur);
CREATE INDEX IF NOT EXISTS idx_annonces_raw_gin ON annonces USING GIN(raw);
"""

INSERT_SQL = """
INSERT INTO annonces (source, ad_id, url, etat, etat_label, owner_type, prix_eur, marque, modele,
  annee, kilometrage, energie, boite, ville, code_postal, departement, region, nb_images, images,
  raw, scraped_at)
VALUES (%(source)s, %(ad_id)s, %(url)s, %(etat)s, %(etat_label)s, %(owner_type)s, %(prix_eur)s,
  %(marque)s, %(modele)s, %(annee)s, %(kilometrage)s, %(energie)s, %(boite)s, %(ville)s,
  %(code_postal)s, %(departement)s, %(region)s, %(nb_images)s, %(images)s, %(raw)s, now())
ON CONFLICT (source, ad_id) DO NOTHING
"""


# --------------------------------------------------------------------------- infra
def pg_connect() -> psycopg.Connection:
    """Connexion Postgres. sslmode=verify-ca si la CA Coolify est montée (anti-MITM)."""
    dsn = os.environ["PG_URL"]
    ca = os.environ.get("PG_CA_CERT", "/etc/ssl/certs/coolify-ca.crt")
    if not os.path.isabs(ca):  # chemin relatif au dossier du script (ex. ../infra/coolify-ca.crt)
        ca = os.path.normpath(os.path.join(HERE, ca))
    kwargs: dict[str, Any] = {}
    if os.path.exists(ca):
        # verify-ca : la chaîne est vérifiée, pas le hostname (le CN est le nom du conteneur).
        kwargs = {"sslmode": "verify-ca", "sslrootcert": ca}
    return psycopg.connect(dsn, **kwargs)


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["MINIO_ENDPOINT"],
        aws_access_key_id=os.environ["MINIO_ACCESS_KEY"],
        aws_secret_access_key=os.environ["MINIO_SECRET_KEY"],
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
        region_name="us-east-1",
    )


def http_session() -> requests.Session:
    """Session curl_cffi imitant Chrome (TLS/JA3) — la clé du contournement DataDome."""
    proxies = None
    if os.environ.get("PROXY_URL"):  # secours seulement : inutile sous ~40 req/min
        proxies = {"http": os.environ["PROXY_URL"], "https": os.environ["PROXY_URL"]}
    return requests.Session(impersonate="chrome", proxies=proxies, timeout=25)


# --------------------------------------------------------------------------- scraping
def fetch_page(session: requests.Session, page: int, condition: str | None) -> dict[str, Any]:
    """Récupère une page de recherche et renvoie le bloc searchData. {} si bloqué."""
    url = SEARCH_BASE + (f"&vehicle_damage={condition}" if condition else "") + f"&page={page}"
    resp = session.get(url)
    if resp.status_code != 200:
        return {}
    m = NEXT_DATA_RE.search(resp.text)
    if not m:  # challenge DataDome : pas de SSR
        return {}
    try:
        return json.loads(m.group(1))["props"]["pageProps"]["searchData"]
    except (KeyError, json.JSONDecodeError):
        return {}


def to_row(ad: dict[str, Any]) -> dict[str, Any]:
    """Annonce brute -> ligne SQL (colonnes chaudes + JSON complet dans `raw`)."""
    attrs = {a["key"]: a.get("value_label") or a.get("value") for a in ad.get("attributes", [])
             if a.get("key") not in NOISE_ATTRS}
    dmg = next((a for a in ad.get("attributes", []) if a.get("key") == "vehicle_damage"), None)
    loc = ad.get("location") or {}
    owner = ad.get("owner") or {}
    imgs = (ad.get("images") or {}).get("urls_large") or (ad.get("images") or {}).get("urls") or []
    price = ad.get("price")

    def as_int(v: Any) -> int | None:
        if v is None:
            return None
        digits = re.sub(r"[^0-9]", "", str(v))
        return int(digits) if digits else None

    payload = {
        "source": "leboncoin", "id": ad.get("list_id"), "url": ad.get("url"),
        "subject": ad.get("subject"), "body": ad.get("body"),
        "etat": dmg.get("value") if dmg else None,
        "etat_label": (dmg.get("value_label") or dmg.get("value")) if dmg else None,
        "owner": {"type": owner.get("type"), "no_salesmen": owner.get("no_salesmen")},
        "location": {k: loc.get(k) for k in
                     ("city", "zipcode", "department_name", "region_name", "lat", "lng")},
        "attributes": attrs, "nb_images": (ad.get("images") or {}).get("nb_images"),
        "images": imgs, "first_publication_date": ad.get("first_publication_date"),
    }
    return {
        "source": "leboncoin", "ad_id": str(ad.get("list_id")), "url": ad.get("url"),
        "etat": payload["etat"], "etat_label": payload["etat_label"],
        "owner_type": owner.get("type"),
        "prix_eur": (price[0] if isinstance(price, list) and price else price),
        "marque": attrs.get("u_car_brand"), "modele": attrs.get("u_car_model"),
        "annee": as_int(attrs.get("regdate")), "kilometrage": as_int(attrs.get("mileage")),
        "energie": attrs.get("fuel"), "boite": attrs.get("gearbox"),
        "ville": loc.get("city"), "code_postal": loc.get("zipcode"),
        "departement": loc.get("department_name"), "region": loc.get("region_name"),
        "nb_images": payload["nb_images"], "images": imgs,
        "raw": json.dumps(payload, ensure_ascii=False),
    }


def collect(condition: str | None, first: int, last: int) -> int:
    session, added = http_session(), 0
    with pg_connect() as conn:
        conn.execute(DDL)
        conn.commit()
        for page in range(first, last + 1):
            data = fetch_page(session, page, condition)
            ads = data.get("ads") or []
            if not ads:  # 403 de rate-limit : transitoire, on souffle et on réessaie une fois
                print(f"  p{page}: bloqué, pause 30 s", flush=True)
                time.sleep(30)
                ads = (fetch_page(session, page, condition) or {}).get("ads") or []
                if not ads:
                    print(f"  p{page}: toujours bloqué, skip", flush=True)
                    continue
            with conn.cursor() as cur:
                for ad in ads:
                    cur.execute(INSERT_SQL, to_row(ad))
                    added += cur.rowcount
            conn.commit()
            print(f"  p{page}: {len(ads)} annonces (total ajouté {added})", flush=True)
            time.sleep(RATE_SLEEP + random.uniform(0, 0.5))
    return added


# --------------------------------------------------------------------------- images
def pending_images(conn: psycopg.Connection, limit: int) -> list[tuple]:
    """Annonces à photographier, en respectant IMG_QUOTAS (quota = nb d'annonces avec photos/état)."""
    already = dict(conn.execute(
        "SELECT COALESCE(etat,'unknown'), count(*) FROM annonces "
        "WHERE image_keys IS NOT NULL GROUP BY 1").fetchall())
    rows: list[tuple] = []
    for etat, quota in IMG_QUOTAS.items():
        if len(rows) >= limit:
            break
        room = limit - len(rows)
        if quota is not None:
            room = min(room, max(0, quota - already.get(etat, 0)))
        if room <= 0:
            continue
        rows += conn.execute(
            "SELECT ad_id, source, COALESCE(etat,'unknown'), images FROM annonces "
            "WHERE image_keys IS NULL AND etat = %s AND images IS NOT NULL "
            "AND array_length(images,1) > 0 ORDER BY inserted_at LIMIT %s", (etat, room)
        ).fetchall()
    return rows


def sync_images(limit: int = IMG_BATCH) -> tuple[int, int]:
    """Télécharge les photos manquantes -> MinIO, puis renseigne image_keys. Idempotent."""
    s3, session = s3_client(), http_session()
    done_ads = done_imgs = 0
    with pg_connect() as conn:
        rows = pending_images(conn, limit)
        for ad_id, source, etat, urls in rows:
            keys: list[str] = []
            for i, url in enumerate(urls):
                ext = (re.search(r"\.(jpg|jpeg|png|webp)(?:\?|$)", url, re.I) or [None, "jpg"])[1]
                key = f"scraping/{source}/{etat}/{ad_id}/{i:02d}.{ext.lower()}"
                try:
                    r = session.get(url)
                    if r.status_code != 200:
                        continue
                    s3.put_object(Bucket=BUCKET, Key=key, Body=r.content,
                                  ContentType=f"image/{'jpeg' if ext.lower() in ('jpg','jpeg') else ext.lower()}")
                    keys.append(key)
                except Exception as exc:  # image morte : on continue, l'annonce reste traitée
                    print(f"    img KO {url[:60]}: {exc}", flush=True)
            if keys:
                conn.execute("UPDATE annonces SET image_keys = %s WHERE source=%s AND ad_id=%s",
                             (keys, source, ad_id))
                conn.commit()
                done_ads += 1
                done_imgs += len(keys)
    return done_ads, done_imgs


# --------------------------------------------------------------------------- cli
def stats() -> None:
    with pg_connect() as conn:
        rows = conn.execute(
            "SELECT COALESCE(etat,'(null)'), count(*), count(image_keys) "
            "FROM annonces GROUP BY 1 ORDER BY 2 DESC").fetchall()
        total = conn.execute("SELECT count(*), count(image_keys) FROM annonces").fetchone()
    print(f"{'état':<26}{'annonces':>10}{'avec images':>13}")
    for etat, n, img in rows:
        print(f"{etat:<26}{n:>10}{img:>13}")
    print(f"{'TOTAL':<26}{total[0]:>10}{total[1]:>13}")


def run_all() -> None:
    for condition, pages in ETATS.items():
        print(f"=== {condition} (1-{pages}) ===", flush=True)
        added = collect(condition, 1, pages)
        print(f"=== {condition}: +{added} ===", flush=True)
    while True:  # draine tout le backlog images
        ads, imgs = sync_images()
        print(f"images: +{ads} annonces, {imgs} fichiers", flush=True)
        if ads == 0:
            break


def parse_pages(spec: str) -> tuple[int, int]:
    first, _, last = spec.partition("-")
    return int(first), int(last or first)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--condition", default=None, help=f"un de : {', '.join(ETATS)}")
    c.add_argument("--pages", default="1-1", help="ex. 1-22")
    i = sub.add_parser("images")
    i.add_argument("--limit", type=int, default=IMG_BATCH)
    sub.add_parser("all")
    sub.add_parser("stats")
    lp = sub.add_parser("loop")
    lp.add_argument("--every", type=int, default=3600, help="secondes entre deux passes")
    args = ap.parse_args()

    if args.cmd == "collect":
        first, last = parse_pages(args.pages)
        print(f"+{collect(args.condition, first, last)} annonces")
    elif args.cmd == "images":
        ads, imgs = sync_images(args.limit)
        print(f"+{ads} annonces, {imgs} images")
    elif args.cmd == "all":
        run_all()
    elif args.cmd == "stats":
        stats()
    elif args.cmd == "loop":
        while True:
            try:
                run_all()
            except Exception as exc:  # une passe qui casse ne doit pas tuer le conteneur
                print(f"passe en erreur: {exc}", file=sys.stderr, flush=True)
            print(f"--- pause {args.every}s ---", flush=True)
            time.sleep(args.every)


if __name__ == "__main__":
    main()
