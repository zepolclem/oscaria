# Collecte `occasion.largus.fr` → dataset candidat

Brique de collecte d'annonces auto d'occasion. Le scraping tourne sur le **n8n-playwright du VPS**
(`n8n-nodes-playwright`, opération *Run Custom Script*) ; ce dossier ne fait que **piloter** ce n8n
via son API REST et **accumuler le dataset en local**.

> Dataset candidat provisoire (slug de travail `largus-occasion`), non encore validé au Gate.
> Voir `docs/decisions/` (ADR) et la vision produit dans
> `docs/plans/2026-07-20-dl-classifieur-etat-carrosserie.md`.

## Architecture

```
Webhook (POST {page:N})  ->  Playwright: goto listing ?currentpage=N -> URLs annonces
                                         -> pour chaque annonce: goto + pageExtractor  ->  Respond (JSON)
        n8n (VPS) ▲                                                                              │
                  │  node n8n.js deploy   (déploie/active le workflow via l'API REST)            │
   local ─────────┴──  node n8n.js collect (appelle le webhook page/page, écrit NDJSON+images) ◄─┘
```

- **`extract.js`** — `pageExtractor()`, **source de vérité** de l'extraction (contexte navigateur).
  Combine le JSON-LD `schema.org/Car` + `BreadcrumbList` (prix, marque, modèle, localisation) et le DOM
  « Informations générales » (specs bruts largus : boîte, portes, puissance, mise en circulation…).
  Il est **inliné** dans le `scriptCode` du node Playwright au `deploy` (pas de copie divergente).
- **`n8n.js`** — CLI de pilotage (deploy / status / collect).
- **`scrape-one.js`** — debugger local **optionnel** (`npm i playwright` puis `node scrape-one.js <url>`),
  utile pour tester l'extraction hors n8n. Non requis pour la collecte.

## Configuration

Copier `.env.example` → `.env` (gitignored) et renseigner :

```
N8N_BASE_URL=https://n8n-playwright.govroumvroum.fr
N8N_API_KEY=<clef API REST publique n8n>   # header X-N8N-API-KEY
N8N_WEBHOOK_PATH=largus-collect
```

> ⚠️ La clef API est un secret : jamais commitée (`.env` est ignoré), à régénérer si exposée.

## Usage

```bash
node n8n.js deploy               # (ré)installe + active le workflow "largus-collect" sur le VPS
node n8n.js status               # id, état, URL du webhook
node n8n.js collect --pages 1-5  # collecte les pages 1 à 5 (24 annonces/page)
```

Sortie (gitignored, convention repo) :
```
ml/data/largus-occasion/raw/
├── annonces.ndjson       # 1 annonce brute / ligne (append, dédup par uuid, resume-friendly)
└── images/<uuid>/NN.webp # photos pleine résolution (large-)
```

## Schéma d'une annonce (brut largus)

```json
{
  "source": "occasion.largus.fr", "url": "...", "id": "<uuid>", "scraped_at": "<ISO8601>",
  "titre": "...", "prix_eur": 12990, "marque": "BMW", "modele": "Série 3",
  "informations_generales": { "Marque":"BMW", "Modèle":"Série 3", "Mise en circulation":"01/03/2008",
    "Année":"2008", "Kilométrage":"246 000 km", "Énergie":"Diesel", "Boîte de vitesses":"Automatique",
    "Couleur":"Noir", "Portes":"3", "Places":"4", "Puissance":"286 ch", "Puissance fiscale":"18 cv",
    "État":"Occasion" },
  "description": "...", "vendeur": { "type":"pro", "siret":"..." },
  "localisation": { "region":"...", "departement":"...", "ville":"...", "code_postal":"..." },
  "images": ["https://assets2.largus.fr/.../large-...-0.webp", "..."],
  "jsonld_car": { ... }
}
```

### Mapping différé vers `ml/src/features.py` (colonnes FR)

| largus (`informations_generales`) | features.py (dataset `french-second-hand-cars`) |
|---|---|
| `prix_eur` | `price` |
| `Kilométrage` | `kilométragecompteur` |
| `Année` | `année` |
| `Mise en circulation` (dd/mm/yyyy) | `miseencirculation` |
| `Énergie` | `énergie` |
| `Boîte de vitesses` | `boîtedevitesse` |
| `marque` + `modele` (+ titre) | `carmodel` |
| `Puissance` / `Puissance fiscale` | `puissancedin` / `puissancefiscale` |

Le mapping/nettoyage réel se fera dans un notebook EDA `ml/notebooks/largus-occasion/` (hors périmètre ici).

## Garde-fous / limites (contexte BC04)

- **Politesse** : 1 navigateur, 2–5 s + jitter entre annonces, ~3 s entre pages ; UA réaliste.
- **Légal (dette assumée)** : constituer + stocker un dataset par scraping = schéma sanctionné
  (Cass. La Centrale 15/10/2025, Entreparticuliers 2022). **ADR à écrire** ; finalité recherche /
  non-commerciale, non-republication, rétention à cadrer.
- **Biais source** : premier test = **24/24 vendeurs pros** (SIRET). largus ne corrigera **pas** le biais
  « 100 % pros » d'ADR 0005 s'il ne remonte pas de particuliers → `vendeur.type` loggé pour le mesurer.
- **Volume** : ~49 Mo/page (~29 images × 24 annonces). 100 pages ≈ 5 Go. Le listing par défaut = ~416 pages.
```
