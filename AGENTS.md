# OscarIA — Cadrage projet

## Le projet

**OscarIA** est un assistant IA d'aide à la décision pour le **vendeur particulier** de voiture d'occasion. Il optimise trois décisions — **État / Prix / Date** — autour de la satisfaction du vendeur : maximiser le cash net, minimiser le délai de vente, minimiser la charge de gestion. Positionnement : tiers **neutre**, grand public.

**Posture** : l'équipe incarne la société OscarIA — on se présente comme la startup qui conçoit et vend cette solution. Le client final du produit reste le vendeur particulier.

Certification visée : **bloc BC04** (RNCP38616) — « mener un projet IA en équipe en intégrant les contraintes légales et des considérations éthiques ». Ce n'est **pas** le bloc modèle : on est jugé sur le pilotage, la rigueur, la transparence sur les limites des données et la gestion des biais — pas sur la performance brute.

## Périmètre

1. **ML — pilier Prix (priorité actuelle)** : restituer une **fourchette de prix avec incertitude**, pas un prix ponctuel (exigence produit).
2. **DL — classifieur d'état (plus tard)** : reconnaissance binaire **intact / abîmé** sur photos.

Aucun choix de méthode ou de modèle n'est figé à ce stade : ces décisions se prennent **après** le choix des datasets.

## Contraintes dures

- Les répertoires `ai-dev-teaching-hub`, `transfer-learning-alyra` et `alyra-brain` sont des supports de cours : **lecture seule absolue**. Aucune écriture, aucune commande mutative dedans. Les trois ont le même niveau de vérité.
- `alyra-brain` (`/Users/zepolclem/Developer/alyra/alyra-brain`) contient les résumés Markdown de tous les cours (Parties 3-9) extraits des PDF : entrer par `Index.md` (synthèses dans `Parties/`, ~90 fiches dans `Concepts/`, plus `Glossaire.md`).
- Tout le travail (code, données, docs, commits) se fait dans **ce repo**.
- Ne jamais installer de dépendance ou d'outil sans accord explicite.
- `.local/` (gitignored) contient les **notes perso locales** : à lire pour contexte, mais **jamais** commit ni référencé dans un livrable. Les données brutes vont dans `ml/data/` (arborescence versionnée, contenu gitignored — déposer les fichiers à part).

## Process de travail

1. **Brainstorm → validation → code.** Pas d'implémentation sans cadrage validé.
2. **Gate EDA** : toute exploration d'un dataset s'arrête après l'inspection et attend la validation utilisateur avant nettoyage ou modélisation.
3. **Décisions consignées** dans `docs/decisions/` (format ADR light : contexte, décision, alternatives écartées). C'est une pièce maîtresse du dossier BC04.

## Attendus transverses (BC04)

- **Transparence** sur les limites de la donnée : la rigueur prime sur la performance apparente.
- **Explicabilité** : fourchette + incertitude, pas une boîte noire.
- **Vigilance biais** : côté données comme côté modèle.
- **Note légale/éthique** : RGPD (le VIN devient une donnée personnelle si rattachable à une personne), AI Act (afficher l'usage de l'IA et l'incertitude, rester une aide à la décision), accès aux données (privilégier partenariats/API ; scraping concurrentiel risqué en France — jurisprudence leboncoin / La Centrale).

## Structure cible du repo

Les dossiers sont créés au fil de l'eau, quand ils deviennent nécessaires.

```
oscaria/
├── AGENTS.md                      # ce fichier
├── docs/
│   └── decisions/                 # journal de décisions (ADR light)
├── ml/                            # pilier Prix
│   ├── AGENTS.md                  # process ML et itération datasets
│   ├── pyproject.toml             # environnement uv
│   ├── data/
│   │   └── <dataset-slug>/raw/    # un sous-dossier par dataset candidat, raw jamais modifié
│   ├── notebooks/
│   │   └── <dataset-slug>/        # notebooks d'exploration par dataset
│   ├── src/                       # code partagé extrait des notebooks
│   └── models/                    # artefacts entraînés
└── dl/                            # brique état carrosserie (plus tard)
```
