# dl/ — Pilier Plaques (détection + floutage) : process et conventions

> Ce fichier cadre le **process**. Aucun choix de modèle n'y figure : ces décisions se
> prennent après le choix du dataset, et se consignent dans `docs/decisions/`.

## Objectif du pilier

**Détecter les plaques d'immatriculation** sur les photos d'annonce et les **flouter** avant
tout livrable (besoin RGPD). Détection de boîte + flou uniquement — **pas de lecture/OCR** de
la plaque : lire serait un traitement de donnée personnelle supplémentaire, contraire à la
minimisation. Le floutage des **visages** viendra dans un second temps. Hiérarchie des
métriques : **rappel prioritaire** (une plaque ratée = fuite RGPD ; un faux positif ne coûte
qu'un flou en trop).

## Itération datasets

On teste **plusieurs datasets candidats** avant de choisir celui à présenter. Conventions par
candidat (miroir de `ml/`) :

- `data/<dataset-slug>/raw/` : données brutes, **jamais modifiées**. Contenu gitignored,
  déposé à part ; seule l'arborescence est versionnée (`.gitkeep`).
- `notebooks/<dataset-slug>/` : notebooks d'exploration propres à ce candidat.
- Verdict **garder / écarter** + **objectif d'entraînement** retenu, documenté dans
  `docs/decisions/` (ADR light).

Le code partagé vit dans `src/` : il rend les EDA comparables d'un candidat à l'autre.

## Process par candidat

1. **EDA d'inspection standardisée** : volumes & splits, classes & équilibre (par image et par
   instance), type et qualité des labels, résolutions/formats, doublons, licence & provenance,
   biais de source, échantillons visuels.
2. **STOP — validation utilisateur.** Résumé chiffré remonté, aucune étape suivante sans accord.
3. **Verdict** garder / écarter + objectif, consigné.

Nettoyage, split et modélisation ne se décident qu'une fois un dataset **retenu**.

## Environnement

- **uv** : membre du workspace racine (`dl/pyproject.toml`), `uv.lock` partagé.
- Dépendances **minimales au besoin** (YAGNI) : on n'ajoute une lib que quand l'étape l'exige.
- **Device** : MPS sur le Mac (M1 Pro), CUDA si bascule sur le PC (RTX 5070 Ti). Helper
  `get_device()` dans `src/` (repli CPU). Import notebooks : `sys.path.insert(0, "../../src")`
  (ou `"../src"` pour un notebook directement sous `notebooks/`, ex. `00_smoke_device`).
- **Install selon la machine** : `recommend_install()` (`src/device.py`) tourne **sans torch**
  et dit quel env poser — Mac Apple Silicon → wheels PyPI par défaut (MPS) ; PC NVIDIA
  Blackwell → index CUDA 12.8. Le notebook `00_smoke_device.ipynb` confirme que le device
  attendu calcule vraiment. **On reste sur le Mac (MPS) pour l'instant.**
- Format **hybride** : notebooks pour l'exploration, code stabilisé extrait vers `src/`.

## Règles

- Jamais installer sans accord explicite.
- Jamais modifier un fichier `raw/`.
- Split **train/val/test sans data leakage** (jamais la même voiture des deux côtés).
- Métriques **honnêtes** (precision/recall/F1 + matrice de confusion), pas l'accuracy seule.
- Rigueur > performance apparente : une limite documentée vaut mieux qu'un score gonflé.
- Photos = données perso possibles (plaques, visages, EXIF) : vigilance RGPD.
