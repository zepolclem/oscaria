# Collecteur leboncoin → Postgres + MinIO

Collecte d'annonces auto **de particuliers** (`owner_type=private`), **stratifiées par état véhicule**.
Apporte la cible produit OscarIA — le **vendeur particulier** — et un **label d'état déclaré**
exploitable pour le pilier État. Seule source du projet à ne pas être 100 % vendeurs professionnels :
c'est le levier pour confronter le biais mesuré par l'ADR 0005.

> Dataset candidat provisoire (slug `leboncoin-private`), non encore validé au Gate.

## Pourquoi il n'y a pas de navigateur ici

Le listing leboncoin (Next.js) est **rendu côté serveur** dans `__NEXT_DATA__`, et le gate DataDome
porte sur le **fingerprint TLS (JA3)**, pas sur le JavaScript. Mesuré :

| Client | Résultat |
|---|---|
| `urllib` (User-Agent Chrome) | **403** |
| `curl_cffi` (`impersonate="chrome"`) | **200**, 35 annonces/page |
| Playwright headless + proxy mobile | **403** (hard-block) |

Exécuter du JS n'apporterait **aucune donnée supplémentaire** — juste ~700 Mo de RAM et une surface de
détection (CDP, canvas, WebGL) à masquer. `curl_cffi` rejoue le handshake d'un vrai Chrome :
**~0,5 s/page** contre ~60 s en navigateur, et 100 % de succès là où Playwright était bloqué.

**Débit** : ~40 req/min (`RATE_SLEEP=1.5`) est stable. Les 403 sont du **rate-limit transitoire**, pas
un ban : le script fait une pause de 30 s et réessaie. **Le proxy mobile n'est pas nécessaire** à ce
rythme — la variable `PROXY_URL` reste disponible en secours.

## Architecture

```
scraper.py ──curl_cffi──► leboncoin (__NEXT_DATA__)
     │
     ├──► Postgres (Earth, PGVector 18) : table `annonces`
     │      colonnes chaudes + raw JSONB + images TEXT[] + image_keys TEXT[]
     │      UNIQUE(source, ad_id) -> dédup native, relance sans doublon
     │
     └──► MinIO (bucket `oscaria-images`)
            clé : scraping/leboncoin/<etat>/<ad_id>/NN.jpg
            le label est dans le chemin -> dataset de classification prêt à l'emploi

dataset.py  ◄── trajet inverse : serveur -> disque local, chacun dans son pilier
     ├──► ml/data/leboncoin-private/raw/annonces.parquet             (pilier Prix)
     └──► dl/data/leboncoin-private/raw/images/<etat>/<ad_id>/NN.jpg (pilier État, ImageFolder)
```
**Le script tourne en local** (membre du workspace `uv`) : il est assez rapide pour ne pas justifier
d'infrastructure dédiée — la collecte complète prend ~15 min. Ce qui compte, c'est que **les données
sont persistées côté serveur** (Postgres + MinIO sur Coolify) : le PC ne fait que pousser, rien n'est
stocké localement, et une interruption ne perd rien (dédup `ON CONFLICT`, reprise idempotente).

## Configuration

Copier `.env.example` → `.env` (gitignored) et renseigner `PG_URL`, `MINIO_*`. La CA Coolify
(`../infra/coolify-ca.crt`) sert à la connexion Postgres en **`verify-ca`** — chaîne vérifiée
(anti-MITM), hostname non vérifié car le CN du certificat est le nom du conteneur.

## Usage

```bash
cd collecte/scraper
uv run python scraper.py stats                                # volumétrie par état
uv run python scraper.py collect --condition damaged --pages 1-22
uv run python scraper.py images --limit 200                   # draine les photos manquantes
uv run python scraper.py all                                  # les 8 états + images
uv run python scraper.py loop --every 21600                   # boucle (si besoin d'un run continu)
```
`uv run` installe les dépendances et charge `.env` automatiquement.

> ⚠️ Ne **jamais** lancer `uv sync` depuis ce dossier : le venv est partagé par tout le workspace,
> et un `sync` depuis un membre **élague** les dépendances des autres (torch, scikit-learn,
> jupyterlab disparaissent). Pour rafraîchir l'environnement : `uv sync --all-packages` à la racine.

### Rapatrier le dataset en local (`dataset.py`)

Trajet inverse de la collecte. Chaque sortie va dans le pilier qu'elle alimente, comme le veut la
structure du repo (`AGENTS.md`) : le tabulaire sert le modèle Prix, les photos le modèle État.

```bash
uv run python dataset.py annonces              # -> ml/data/leboncoin-private/raw/annonces.parquet
uv run python dataset.py images --workers 12   # -> dl/data/leboncoin-private/raw/images/
uv run python dataset.py all
```

- **Parquet** : colonnes chaudes typées (`prix_eur` float, `annee`/`kilometrage` `Int64` nullable),
  `images`/`image_keys` en `list<string>` natif, `raw` JSONB conservé en **colonne texte JSON**
  (attributs leboncoin trop hétérogènes pour un schéma typé). Écriture atomique.
- **Images** : la colonne `image_keys` fait foi, pas un listing du bucket — elle seule relie une
  image à son annonce et à son état. Le préfixe `scraping/leboncoin/` est retiré, donc le **label
  reste le premier segment du chemin** → `torchvision.datasets.ImageFolder` directement utilisable.
- **Idempotent** : un fichier déjà présent à la bonne taille est sauté. Les tailles distantes sont
  lues en un listing paginé (~23 requêtes) plutôt qu'un `head_object` par image (~22 000).
  Une interruption se reprend sans tout retélécharger.

## États véhicule (`vehicle_damage`)

| Code | Label | Volume ~ (FR, particuliers) |
|---|---|---|
| `excellent_condition` | Excellent état (proche du neuf) | ~51 700 |
| `undamaged` | Non endommagé | ~9 500 |
| `good_overall_condition` | Bon état général | ~97 200 |
| `normal_wear_and_tear` | Traces d'usure normales | ~26 000 |
| `minor_repairs_needed` | Réparations mineures à prévoir | ~9 900 |
| `major_repairs_needed` | Réparations majeures à prévoir | ~4 600 |
| `damaged` | Endommagé | ~740 |
| `not_drivable` | Non roulant | ~1 230 |

Les classes rares sont collectées **en entier**, les fréquentes **plafonnées** (~29 pages) → dataset
plus équilibré pour la classification.

> ⚠️ **`vehicle_damage` = état mécanique/global déclaré par le vendeur, pas un dégât carrosserie
> visible.** Une voiture « non roulant » peut avoir une caisse intacte (moteur HS). C'est une feature
> de décote pertinente pour le **pilier Prix**, mais **pas un label de vérité terrain pour le modèle
> vision** — voir ADR 0007 (domain shift CarDD). Souvent non renseigné (`etat` NULL).

## Requêtes utiles

```sql
SELECT etat, count(*), count(image_keys) FROM annonces GROUP BY etat ORDER BY 2 DESC;
SELECT count(*) FILTER (WHERE image_keys IS NULL) AS backlog_images FROM annonces;
SELECT unnest(image_keys) FROM annonces WHERE etat = 'damaged';
```

## Limites / garde-fous

- **Plafond ~3 500 résultats par requête** (100 pages × 35). Pour aller au-delà, découper l'espace de
  recherche (marque × modèle, prix × année, région) — bonus : échantillon équilibré plutôt que les
  3 500 annonces les plus récentes.
- **Légal** : constituer et stocker un dataset par scraping est le schéma sanctionné par la
  jurisprudence (Cass. La Centrale 15/10/2025, Entreparticuliers 2022) ; contraire aux CGU leboncoin.
  **ADR à écrire** — finalité recherche/non-commerciale, non-republication, rétention.
- **Données personnelles** : les annonces contiennent nom de vendeur, localisation, `has_phone`. On ne
  conserve que les champs véhicule + `owner.type` ; le reste n'est pas extrait en colonnes.
- **`body`** (description) est vide sur le listing : la description complète n'est que sur la page
  annonce (non collectée).
