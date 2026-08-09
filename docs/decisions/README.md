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
| 0002 | [Dataset leboncoin-private](0002-dataset-leboncoin-private.md) | 2026-08-08 | gardé — 20 915 annonces, 100 % particuliers ; six limites déclarées (pas de pros, pas d'historique, pas de prix de vente, état déclaratif) |
| 0003 | [Cible et périmètre](0003-cible-prix-demande-et-perimetre.md) | 2026-08-08 | on prédit un **prix demandé**, pas un prix de vente ; périmètre [500 €, 50 000 €], 1 033 annonces écartées et journalisées |
| 0004 | [Colonnes écartées](0004-colonnes-ecartees.md) | 2026-08-08 | l'estimation de leboncoin (`car_price_*`) refusée par positionnement, pas par technique ; `old_price` circulaire |
| 0005 | [Arbres vs linéaire](0005-modele-arbres-vs-lineaire.md) | 2026-08-08 | HistGradientBoosting retenu (1 475 € contre 2 301 €, 0 prix négatif) ; le linéaire n'était pas lisible non plus — marque et modèle emboîtés |
| 0006 | [Contrat d'entrée du formulaire](0006-contrat-entree-formulaire.md) | 2026-08-08 | 7 champs obligatoires + 7 facultatifs en valeurs manquantes natives ; chaque changement chiffré, états regroupés à 5 crans |
| 0007 | [Fourchette conformalisée](0007-fourchette-conformalisee.md) | 2026-08-08 | quantiles + CQR, couverture **79,5 % mesurée** pour une cible de 80 % ; garantie marginale et non conditionnelle (72 % au-delà de 20 000 €) |

Les fiches 0001–0007 de l'arc « reconnaissance de dégâts » ont été supprimées lors de la
remise à zéro du 2026-08-06 ; elles restent consultables dans l'historique git (commit
tombeau au tip de la branche `reset/cardd-baseline`). La numérotation repart de **0001**.

Voir le plan en cours dans [`docs/plans/`](../plans/).
