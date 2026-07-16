# ml/ — Pilier Prix : process et conventions

> Ce fichier cadre le **process**. Aucun choix de modélisation n'y figure : ces décisions se prennent après le choix du dataset, et se consignent dans `docs/decisions/`.

## Objectif du pilier

Aide à la décision **Prix** pour le vendeur particulier : restituer une **fourchette de prix avec incertitude**, jamais un prix ponctuel. Exigence produit, indépendante du dataset retenu.

## Itération datasets

On teste **plusieurs datasets candidats** avant de choisir celui à présenter. Aucun dataset n'est nommé ici : le choix des candidats est une décision à venir.

Conventions par candidat :

- `data/<dataset-slug>/raw/` : données brutes, **jamais modifiées**. Toute transformation produit un nouveau fichier ailleurs.
- `notebooks/<dataset-slug>/` : notebooks d'exploration propres à ce candidat.
- Verdict **garder / écarter** documenté dans `docs/decisions/`, avec les raisons.

Le code partagé vit dans `src/` : il rend les EDA comparables d'un candidat à l'autre.

## Process par candidat

1. **EDA d'inspection standardisée** : dimensions, colonnes, types, taux de valeurs manquantes, qualité générale, aberrations.
2. **STOP — validation utilisateur.** Résumé chiffré remonté, aucune étape suivante sans accord.
3. **Verdict** garder / écarter, consigné.

Nettoyage, feature engineering et modélisation ne se décident qu'une fois un dataset **retenu**.

## Environnement

- **uv** : `pyproject.toml` + `uv.lock`, environnement reproductible.
- Dépendances **minimales au besoin** (YAGNI) : on n'ajoute une lib que quand l'étape en cours l'exige.
- Format **hybride** : notebooks pour l'exploration, code stabilisé extrait vers `src/`.

## Règles

- Jamais installer sans accord explicite.
- Jamais modifier un fichier `raw/`.
- Rigueur > performance apparente : une limite documentée vaut mieux qu'un score gonflé.
