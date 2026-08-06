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
| 0001 | [Cible multi-étiquette sur CarDD](0001-cible-multi-etiquette-cardd.md) | 2026-07-30 | 6 types de dégât, macro-F1 arbitre, baseline à trois lignes → **0,815** sur CarDD |
| 0002 | [Non-transfert de CarDD sur photos d'annonce](0002-non-transfert-cardd-photos-annonce.md) | 2026-07-31 | aire ROC **0,509** sur 471 photos réelles : classement aléatoire, aucun seuil ne le répare |
| 0003 | [L'état déclaré par le vendeur](0003-etiquette-declaree-asymetrique.md) | 2026-07-31 | annotation aveugle de 648 photos : étiquette fiable à **98 % côté intact**, **40 % côté abîmé** |
| 0004 | [Binaire sur le domaine annonce](0004-binaire-domaine-annonce.md) | 2026-08-01 | apprenable — précision moyenne **0,689** contre un plancher de 0,223, en 384 px sur détourage |
| 0005 | [Greffe de négatifs dans CarDD](0005-greffe-negatifs-cardd.md) | 2026-08-01 | le modèle apprend à dire « rien » : fausse alerte **80,2 % → 2,9 %**, typage préservé |
| 0006 | [Tri des vues](0006-tri-des-vues.md) | 2026-08-03 | 4 photos d'annonce sur 10 sont inexploitables ; les filtrer se fait à **0,972** d'aire ROC, sans annotation nouvelle |
| 0007 | [Pilier État recadré en cohérence](0007-pilier-etat-coherence-annonce.md) | 2026-08-04 | promesse ramenée au niveau annonce → **0,807** ; signale 1 annonce sur 5, juste 9 fois sur 10 |

Fil conducteur : 0001 entraîne un modèle qui marche sur son jeu ; 0002 mesure qu'il ne marche pas
sur le domaine visé ; 0003 établit quelles étiquettes sont exploitables ; 0004 reconstruit sur le
bon domaine ; 0005 corrige l'angle mort du modèle d'origine ; 0006 traite le désordre des photos
réelles ; 0007 ramène la promesse produit au niveau de ce qui est réellement mesuré.

Sept pistes ont été écartées **sur mesure** et non par intuition : réglage de seuils sur un
classement aléatoire, découpage en tuiles avec le modèle d'origine, entraînement sur l'état déclaré,
annotation supplémentaire (courbe d'apprentissage plate), septième classe « intact », capture guidée,
agrégation par maximum.

Les dix fiches précédentes ont été supprimées lors de la remise à zéro du 2026-07-29 : la moitié
décrivaient un dataset de prix abandonné. Elles restent consultables dans l'historique `git`
(branche `main`). La numérotation repart de **0001**.

Voir le plan en cours dans [`docs/plans/`](../plans/).
