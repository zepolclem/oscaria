# Journal de décisions (ADR)

Ce dossier consigne les décisions techniques structurantes du projet OscarIA, au format
**ADR light** (*Architecture Decision Record*, fiche de décision) : contexte, décision,
alternatives écartées, conséquences.

C'est une pièce du dossier de certification **bloc BC04** (piloter un projet IA avec rigueur,
transparence sur les limites et gestion des biais). Chaque fiche est datée et immuable :
une décision qui change donne une **nouvelle** fiche qui remplace l'ancienne (statut `Remplacé`).

## Trois espaces, trois numérotations

Les piliers avancent en parallèle sur des branches séparées. Une numérotation unique produisait
des collisions — deux fiches `0002` ont été écrites le même jour sur deux branches. Chaque
espace numérote donc **indépendamment**, depuis `0001` :

| Espace | Périmètre | Index |
|---|---|---|
| **ML** | pilier Prix : dataset, cible, modèle, incertitude | [`ml/`](ml/README.md) |
| **DL** | pilier Plaques : détection, floutage, transfert | [`dl/`](dl/README.md) |
| **Socle** | transverse : déploiement, packaging, infrastructure | [`socle/`](socle/README.md) |

Une fiche se cite par son espace **et** son numéro — « ADR ML 0004 », « ADR DL 0002 » — jamais
par le numéro seul, qui est ambigu.

## Historique

Les fiches 0001–0007 de l'arc « reconnaissance de dégâts » ont été supprimées lors de la
remise à zéro du 2026-08-06 ; elles restent consultables dans l'historique git (commit
tombeau au tip de la branche `reset/cardd-baseline`). Certains renvois du dépôt pointent
encore vers elles — notamment `collecte/scraper/README.md` — et sont donc **morts** : ils
n'ont volontairement pas été renumérotés, pour ne pas les rendre faussement valides.

La branche locale `reset/plaques` est un second tombeau (2026-08-09) : elle porte l'arc
d'évaluation UC3M-LP, abandonné sans verdict, et un premier entraînement de la baseline
plaques (AP 0,932) distinct du modèle livré. Clôture consignée dans
[ADR DL 0003](dl/0003-cloture-arc-leboncoin-uc3m.md).

Voir les plans en cours dans [`docs/plans/`](../plans/).
