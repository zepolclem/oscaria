# Décisions — pilier ML (Prix)

Fiches de décision du pilier **Prix** : estimation du prix demandé d'un véhicule d'occasion à
partir des caractéristiques saisies par le vendeur.

Format et règles communes : voir le [journal général](../README.md). Numérotation propre à cet
espace ; une fiche se cite « ADR ML 0004 ».

## Index

| # | fiche | date | en une ligne |
|---|---|---|---|
| 0001 | [Dataset leboncoin-private](0001-dataset-leboncoin-private.md) | 2026-08-08 | gardé — 20 915 annonces, 100 % particuliers ; six limites déclarées (pas de pros, pas d'historique, pas de prix de vente, état déclaratif) |
| 0002 | [Cible et périmètre](0002-cible-prix-demande-et-perimetre.md) | 2026-08-08 | on prédit un **prix demandé**, pas un prix de vente ; périmètre [500 €, 50 000 €], 1 033 annonces écartées et journalisées |
| 0003 | [Colonnes écartées](0003-colonnes-ecartees.md) | 2026-08-08 | l'estimation de leboncoin (`car_price_*`) refusée par positionnement, pas par technique ; `old_price` circulaire |
| 0004 | [Arbres vs linéaire](0004-modele-arbres-vs-lineaire.md) | 2026-08-08 | HistGradientBoosting retenu (1 475 € contre 2 301 €, 0 prix négatif) ; le linéaire n'était pas lisible non plus — marque et modèle emboîtés |
| 0005 | [Contrat d'entrée du formulaire](0005-contrat-entree-formulaire.md) | 2026-08-08 | 7 champs obligatoires + 7 facultatifs en valeurs manquantes natives ; chaque changement chiffré, états regroupés à 5 crans |
| 0006 | [Fourchette conformalisée](0006-fourchette-conformalisee.md) | 2026-08-08 | quantiles + CQR, couverture **79,5 % mesurée** pour une cible de 80 % ; garantie marginale et non conditionnelle (72 % au-delà de 20 000 €) |
