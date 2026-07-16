# OscarIA

Assistant IA d'aide à la décision pour le **vendeur particulier** de voiture d'occasion : il optimise **État / Prix / Date** en tiers neutre.

Priorité actuelle — **pilier Prix** : restituer une **fourchette de prix avec incertitude**, jamais un prix ponctuel.

Certification visée : **bloc BC04** (RNCP38616) — piloter un projet IA en équipe avec contraintes légales et éthiques. On est jugé sur le pilotage, la rigueur et la transparence, pas sur la performance brute.

## Structure

```
oscaria/
├── AGENTS.md        # cadrage projet (périmètre, contraintes, process)
├── ml/              # pilier Prix
│   ├── AGENTS.md    # process ML et itération datasets
│   └── data/        # données brutes — non versionnées (voir ci-dessous)
└── docs/decisions/  # journal de décisions (ADR light) — au fil de l'eau
```

## Données

Les données brutes ne sont **pas versionnées** : déposer les fichiers dans `ml/data/` (gitignored), ils se connectent automatiquement. Seule l'arborescence est suivie.

## Environnement

Python géré par **uv** (`pyproject.toml` + `uv.lock`). Voir [ml/AGENTS.md](ml/AGENTS.md).
