# OscarIA

Assistant IA d'aide à la décision pour le **vendeur particulier** de voiture d'occasion. Le projet explore trois dimensions : **État / Prix / Date**, avec une position de tiers neutre.

## Accès jury

- [Démo OscarIA](https://oscaria.zepolclem.dev)
- [Dépôt GitHub](https://github.com/zepolclem/oscaria)

La démo en ligne permet de parcourir les fonctionnalités disponibles. Le présent dépôt contient le code, les notebooks, les décisions de conception et les artefacts de modèle nécessaires pour comprendre le projet.

## Ce que démontre le projet

- **Prix** : une estimation sous forme de fourchette accompagnée d'une incertitude, et non un prix ponctuel.
- **Plaques** : détection puis floutage de plaques d'immatriculation, sans lecture du contenu de la plaque.
- **Pilotage responsable** : décisions documentées, limites explicitées, vigilance sur les biais, les données personnelles et la reproductibilité.

Certification visée : **RNCP38616 - BC04**, mener un projet IA en équipe en intégrant contraintes légales et considérations éthiques. La rigueur et la transparence priment sur la performance brute.

## Parcourir le dépôt

- [`docs/decisions/`](docs/decisions/) : journal des décisions (ADR light) et alternatives écartées.
- [`ml/`](ml/) : pilier Prix et notebooks d'exploration.
- [`dl/`](dl/) : détection et floutage des plaques.
- [`app/`](app/) : application servie dans la démo.

## Données et confidentialité

Les annonces brutes collectées, les images sources et les secrets d'environnement ne sont pas publiés. Les données sont conservées localement dans les répertoires prévus, ignorés par Git. Le dépôt ne doit contenir ni annonce brute, ni photo personnelle non floutée, ni identifiant d'accès.

## Environnement

Python est géré par **uv** (`pyproject.toml` + `uv.lock`). Les consignes de travail figurent dans [AGENTS.md](AGENTS.md) et [ml/AGENTS.md](ml/AGENTS.md).
