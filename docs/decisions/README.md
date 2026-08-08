# Journal de décisions (ADR)

Ce dossier consigne les décisions techniques structurantes du projet OscarIA, au format
**ADR light** (*Architecture Decision Record*, fiche de décision) : contexte, décision,
alternatives écartées, conséquences.

C'est une pièce du dossier de certification **bloc BC04** (piloter un projet IA avec rigueur,
transparence sur les limites et gestion des biais). Chaque fiche est datée et immuable :
une décision qui change donne une **nouvelle** fiche qui remplace l'ancienne (statut `Remplacé`).

## Index

| # | fiche | date | en une ligne |
|---|---|---|---|
| 0001 | [Baseline détection plaques](0001-baseline-detection-plaques.md) | 2026-08-06 | dataset car-plate-detection gardé ; Faster R-CNN MobileNetV3 fine-tuné sur CPU (backward MPS corrompu — mesuré) ; rappel prioritaire, pas de test interne |
| 0002 | [Transfert photos réelles](0002-transfert-plaques-photos-reelles.md) | 2026-08-08 | 87 photos perso annotées en aveugle : rappel réel 0,895, couverture 95,3 %, 4/95 plaques sans flou ; transfert validé, seuil 0,3 confirmé, pas de fine-tuning |

Les fiches 0001–0007 de l'arc « reconnaissance de dégâts » ont été supprimées lors de la
remise à zéro du 2026-08-06 ; elles restent consultables dans l'historique git (commit
tombeau au tip de la branche `reset/cardd-baseline`). La numérotation repart de **0001**.

Voir le plan en cours dans [`docs/plans/`](../plans/).
