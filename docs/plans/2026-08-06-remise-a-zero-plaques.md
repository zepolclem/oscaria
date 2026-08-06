# Remise à zéro n°2 — pilier DL « détection + floutage des plaques »

## Contexte

Le pilier « reconnaissance de dégâts » (arc ADR 0001→0007) est abandonné : trop fastidieux, promesse produit ramenée trop bas. Le nouveau pilier deep learning répond au besoin RGPD déjà consigné en fiche 0007 : **détecter les plaques d'immatriculation et les flouter** avant tout livrable. Périmètre acté : détection de boîte + flou gaussien, **pas de lecture/OCR** (minimisation RGPD) ; visages plus tard. Structure du projet conservée, datasets dégâts supprimés.

Décisions utilisateur déjà actées : **commit tombeau** avant suppression (l'arc actuel n'a jamais été commité — 0 commit sur toutes branches, vérifié) ; **leboncoin-private : garder `raw/` seulement** (22 204 photos, 2,4 Go, domaine cible non reconstructible) ; **téléchargement du dataset Kaggle `andrewmvd/car-plate-detection` validé** (433 images, boîtes plaques PASCAL VOC, licence CC0, 213 Mo).

Protocole répliqué de la remise à zéro de juillet (`docs/plans/2026-07-29-cardd-remise-a-zero.md`) : plan déposé dans git avant nettoyage, branche dédiée, commit unique 4 blocs, README ADR vidé avec pointeur historique, numérotation ADR repartant à 0001.

---

## Phase 0 — Remise à zéro

### 0a. Commit tombeau (sur `reset/cardd-baseline`, AVANT tout nettoyage)

Supprimer d'abord les 2 doublons exacts à la racine : `rm cases_saines.csv verdicts.csv` (copies identiques sous `dl/data/`).

Puis ajouter explicitement :

- **Code untracked (8 fichiers, ~1 619 lignes)** : `dl/src/annonce.py annoter.py binaire.py experiences_annonce.py experiences_binaire.py negatifs.py train_cardd.py tri_vues.py`
- **Carnets untracked (6)** : `dl/notebooks/cardd/ leboncoin-private/ real_test/`
- **Les 7 fiches ADR** : `docs/decisions/0001-*.md … 0007-*.md`
- **Tracked modifiés** : `dl/src/cardd.py localize.py detect.py`, `docs/decisions/README.md`
- **Annotations humaines gitignorées — `git add -f` OBLIGATOIRE** (5 CSV) :
  `dl/data/leboncoin-private/annot/verdicts.csv`, `boites_471.csv`, `manifeste.csv`,
  `dl/data/cardd/negatifs_grille/cases_saines.csv`, `manifeste_grille.csv`.
  Les manifestes sont indispensables : `verdicts.csv`/`cases_saines.csv` portent des noms neutres (`0001.jpg`) indéchiffrables sans le mapping `chemin_origine` du manifeste.
- **NON sauvés, perte assumée** : `dl/models/*.pt` (3 × 43 Mo — binaires lourds dans un dépôt déjà poussé, métriques préservées dans les ADR, pilier abandonné) ; CSV de sorties machine (`probas_*`, `leviers_*`, résultats — les chiffres qui font foi sont dans les ADR).

Message : `chore(dl): tombeau de l'arc dégâts avant remise à zéro plaques` — 4 blocs (contexte / Sauvé / Non sauvé + justification / Suite dans `docs/plans/2026-08-06-remise-a-zero-plaques.md`).

### 0b. Nouvelle branche

`git switch -c reset/plaques` — le tombeau reste le tip de `reset/cardd-baseline` (conclusion de l'arc juillet), le nettoyage ouvre le nouvel arc. Pas de push sans demande explicite.

### 0c. Nettoyage (sur `reset/plaques`)

**Tracked → `git rm`** (récupérables dans l'historique) :
- `dl/src/cardd.py localize.py infer.py` (spécifiques dégâts / couplés format CarDD)
- les 8 fichiers src + 3 dossiers de carnets + 7 ADR entrés au tombeau
- les 5 CSV du tombeau (restent dans le commit tombeau)
- `dl/data/cardd/raw/.gitkeep`, `dl/data/real_test/raw/.gitkeep`

**Untracked/gitignorés → `rm` (perte définitive, actée)** :
- `dl/data/cardd/` (5,8 Go), `dl/data/car_damages/` (1,0 Go), `dl/data/real_test/` (8,1 Mo)
- `dl/data/leboncoin-private/annot/` (164 Mo) et `annonces/` (466 Mo)
- `dl/models/*.pt`, `dl/src/__pycache__/`

**Conservés** : `dl/src/device.py` (générique MPS/CUDA/CPU), `dl/src/detect.py` (détourage véhicule Faster R-CNN inférence pure — resservira ; porte le fait mesuré « backward détection cassé sur MPS »), `dl/AGENTS.md`, `dl/pyproject.toml`, `dl/notebooks/00_smoke_device.ipynb`, `docs/plans/2026-07-29-*.md`, `dl/data/leboncoin-private/raw/` (2,4 Go), tout `ml/` et `collecte/`.

### 0d. Réécritures (même commit que 0c)

1. `docs/decisions/README.md` : convention ADR light conservée, index vidé, numérotation repart à 0001, mention « fiches 0001–0007 de l'arc dégâts supprimées le 2026-08-06, consultables dans l'historique git (tip de `reset/cardd-baseline`, commit tombeau) ».
2. `dl/AGENTS.md` : titre + « Objectif du pilier » → plaques (détection + floutage, pas d'OCR) ; process (EDA → STOP → verdict ADR), conventions `data/<slug>/raw/` et règles inchangés.
3. `docs/plans/2026-08-06-remise-a-zero-plaques.md` : copie de ce plan, déposée avant le commit.
4. `mkdir -p dl/data/plaques/raw && touch dl/data/plaques/raw/.gitkeep && git add` — `.gitignore` déjà compatible (vérifié : `dl/data/**` + `!dl/data/**/` + `!**/.gitkeep`).
5. Docstrings de `device.py`/`detect.py` : références au « pilier État » et à l'ADR 0004 → une ligne « voir historique git ».
6. **Nouvelle convention à inscrire dans le plan déposé** : toute annotation humaine créée sous `dl/data/` est commitée `git add -f` dès sa création, avec son manifeste.

Commit unique : `chore: remise à zéro du pilier État — place au pilier plaques` (4 blocs).

### Vérification Phase 0

`git log --oneline -2` = tombeau + nettoyage ; `git show --stat <tombeau>` liste les 5 CSV ; `git status` propre ; `dl/src/` = `device.py` + `detect.py` seulement ; `docs/decisions/` = README seul ; `dl/models/` vide ; `du -sh dl/data` ≈ 2,4 Go ; `import device, detect` passe. Aucun push.

---

## Phase 1 — Dataset plaques

### 1a. Téléchargement
```
curl -L -o dl/data/plaques/raw/car-plate-detection.zip \
  https://www.kaggle.com/api/v1/datasets/download/andrewmvd/car-plate-detection
unzip … -d dl/data/plaques/raw/
```
Si 401/403 ou si `file` révèle du HTML : demander le token `kaggle.json` à l'utilisateur (aucune installation sans accord). Zip conservé (gitignoré, évite un re-téléchargement).

### 1b. Intégrité
433 images + 433 XML appariés (`CarsN.png`/`CarsN.xml`), chaque XML parse (`xml.etree` stdlib), chaque boîte satisfait `0 ≤ xmin < xmax ≤ largeur`.

### 1c. EDA — `dl/notebooks/plaques/01_eda.ipynb`
Process AGENTS.md : volumes, résolutions, **nombre de plaques par image** (impose le format multi-boîtes), distribution des tailles de plaques (aire relative + largeur px — conditionne résolution d'entrée et ancres), ratios d'aspect, échantillon visuel ~20 images (part de plaques « format UE long » vs autres — c'est l'écart que la Phase 3 mesurera), limite consignée : pas d'identité voiture documentée → split par image.

**STOP — validation utilisateur sur le résumé chiffré avant toute modélisation.**

Commit proposé en fin de phase (sur feu vert) : `feat(dl): dataset plaques — EDA et verdict`.

---

## Phase 2 — Baseline détection

### Fichiers
- `dl/src/plaques.py` (neuf) : parseur VOC → dataset PyTorch format détection torchvision (`{"boxes": [N,4], "labels": …}`, 1 classe), split train/val reproductible (~346/87), **éval IoU maison** (~50 lignes : appariement glouton, rappel/précision à IoU et score donnés, courbe précision/rappel) — lisible par un novice, zéro dépendance nouvelle.
- `dl/src/train_plaques.py` (neuf) : boucle d'entraînement, checkpoint autoporteur `dl/models/plaques_baseline.pt` (state_dict + config + seuil).
- `dl/notebooks/plaques/02_baseline.ipynb`.

### Choix du modèle : décidé par la fiche ADR 0001, pas avant
Contrainte mesurée (docstring `detect.py`) : backward des détecteurs torchvision cassé sur MPS. Ordre de mesure : (1) smoke test d'un pas d'entraînement sur MPS pour confirmer/infirmer en 2026 ; (2) si cassé, chronométrer une époque CPU d'un candidat léger (ex. `fasterrcnn_mobilenet_v3_large_fpn`, 346 images) ; (3) si > ~15 min/époque, bascule PC RTX 5070 Ti (outillée par `recommend_install()` + `00_smoke_device.ipynb`).

ADR 0001 consigne : dataset retenu (verdict EDA), cible (boîte mono-classe, pas d'OCR), modèle + device + hyperparamètres, et la **hiérarchie des métriques : rappel prioritaire** (plaque ratée = fuite RGPD ; faux positif = un flou en trop). Pas de jeu de test interne : le vrai test est le lot leboncoin de la Phase 3 (leçon de l'arc mort : ROC 0,509 sur le domaine cible).

### Vérification
Overfit volontaire sur 10 images (le loss s'écrase), rappel@IoU 0,5 + courbe précision/rappel sur val, checkpoint rechargeable. Commit proposé : `feat(dl): baseline détection plaques + ADR 0001`.

---

## Phase 3 — Transfert sur photos réelles françaises (la phase-leçon)

### 3a. Lot d'annotation aveugle
Ressusciter le protocole depuis le tombeau (`git show <sha>:dl/src/annoter.py`) → `dl/src/annoter_plaques.py` : même mécanique (échantillonnage 1 photo/annonce, noms neutres, manifeste séparé, page HTML autonome avec export CSV), mais page de **tracé de boîtes** (rectangle à la souris, bouton « aucune plaque visible » → `-1,-1,-1,-1`). Export `plaques_verite.csv` (`fichier,x1,y1,x2,y2`).

Lot : **200 photos** de `leboncoin-private/raw/images/` (tous dossiers d'état confondus), filtre véhicule via `detect.py` consigné comme périmètre. **Dès l'export : `git add -f` du CSV vérité + manifeste** (nouvelle convention).

### 3b. Mesure — `dl/notebooks/plaques/03_transfert_reel.ipynb`
Inférence MPS (inférence seule = OK) puis : rappel/précision à IoU ≥ 0,5 et ≥ 0,3 ; **couverture après marge** (fraction de chaque plaque vérité couverte par l'union des boîtes prédites élargies — la métrique RGPD réelle) ; courbe rappel/seuil de confiance → fixe le seuil bas de la Phase 4 ; variante avec/sans recadrage véhicule préalable (`recadrer()` de `detect.py`) — mesurée, pas présumée.

### 3c. ADR 0002 — transfert
Chiffres + verdict. Gate : si rappel < ~0,80 au seuil bas → options à trancher en ADR (fine-tuning sur la moitié du lot français, l'autre moitié en test ; ou dataset complémentaire). Rien décidé avant les chiffres.

### Vérification
200/200 jugées, CSV commités `-f`, tableau rappel/précision/couverture remonté (STOP), ADR 0002 écrite. Commit proposé : `feat(dl): transfert plaques mesuré sur lot leboncoin — ADR 0002`.

---

## Phase 4 — Floutage

### `dl/src/flouter.py` (neuf)
- `flouter_image(img, boites, marge, rayon_min)` : boîtes élargies d'une marge relative (esprit `recadrer()`), flou gaussien PIL à **rayon proportionnel à la hauteur de boîte** avec plancher.
- `flouter_dossier(...)` : détection (checkpoint Phase 2, seuil bas fixé par la courbe Phase 3) → écriture dans un dossier de sortie, **jamais dans `raw/`**.
- CLI : `uv run python dl/src/flouter.py --entree <dossier> --sortie <dossier> [--seuil … --marge …]`.
- Réglages par défaut consignés en **ADR 0003 — politique de floutage**, avec la couverture mesurée à ces réglages.

### Démo — `dl/notebooks/plaques/04_demo_floutage.ipynb`
Grille avant/après ~12 photos. Règle livrable : seules les images **floutées** sortent du poste.

### Vérification
CLI de bout en bout sans erreur ; % de plaques vérité entièrement couvertes par une zone floutée aux réglages ADR 0003 (métrique finale du pilier) ; contrôle visuel : plaques illisibles. Commit proposé : `feat(dl): floutage des plaques — CLI, démo, ADR 0003`.

---

## Récapitulatif

| Phase | Créés | Modifiés | Supprimés |
|---|---|---|---|
| 0 | plan dans `docs/plans/`, `dl/data/plaques/raw/.gitkeep` | `docs/decisions/README.md`, `dl/AGENTS.md`, docstrings `device.py`/`detect.py` | listes 0c |
| 1 | `dl/notebooks/plaques/01_eda.ipynb`, données `raw/` | — | — |
| 2 | `dl/src/plaques.py`, `dl/src/train_plaques.py`, carnet 02, ADR 0001, `plaques_baseline.pt` (gitignoré) | — | — |
| 3 | `dl/src/annoter_plaques.py`, carnet 03, ADR 0002, CSV vérité (`add -f`) | — | — |
| 4 | `dl/src/flouter.py`, carnet 04, ADR 0003 | — | — |

Réutilisation : `device.py`, `detect.py` (filtre véhicule, `recadrer()`, fait MPS), protocole `annoter.py` (depuis le tombeau). Dépendances : `dl/pyproject.toml` couvre déjà tout — **aucune installation prévue**.

## Contraintes transverses

- Commits : **exactement ceux listés** — les 2 de la Phase 0 sont demandés par ce plan ; ceux de fin de phases 1–4 sont proposés et attendent un feu vert au moment venu. **Aucun push sans demande explicite.**
- STOP de validation utilisateur : fin d'EDA (Phase 1) et tableau de transfert (Phase 3), conformément au process `dl/AGENTS.md`.
- Jamais modifier `raw/` ; vigilance RGPD (le pilier existe pour ça).
