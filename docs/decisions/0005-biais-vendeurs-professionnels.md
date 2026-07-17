# ADR 0005 — Biais de représentativité : 100 % d'annonces de vendeurs professionnels

- **Statut** : Accepté (limite documentée)
- **Date** : 2026-07-17

## Contexte

En inventoriant les colonnes inexploitées du dataset, on a découvert que la colonne `vendeur`
vaut **« Professionnel » pour 100 % des 2 441 annonces**. Le jeu de données ne contient
**aucune vente entre particuliers** : uniquement des annonces de garages, concessions et
mandataires.

Or la cible produit d'OscarIA est le **vendeur particulier**. Les prix affichés par des
professionnels sont **systématiquement différents** des prix de vente entre particuliers :
marge du professionnel, garantie commerciale incluse, frais de remise en état, TVA sur la
marge. L'écart usuel constaté sur le marché est de l'ordre de 10 à 20 % au-dessus du prix
entre particuliers.

C'est un **biais de représentativité** (l'échantillon ne représente pas la population visée),
distinct d'un problème de qualité de données : les prix sont justes, mais ce sont des prix
« pros ».

## Décision

1. **Assumer et documenter la limite** : le modèle actuel estime un **prix professionnel
   affiché**, pas un prix de transaction entre particuliers. Toute restitution produit doit le
   préciser (transparence, attendu du bloc BC04).
2. **Ne pas corriger arbitrairement** (ex. appliquer −15 % forfaitaire) : sans données de
   ventes entre particuliers, un tel coefficient serait invérifiable — pire que la limite
   documentée.
3. **Orienter la recherche du prochain dataset candidat** vers des données contenant des
   ventes entre particuliers (ou permettant de distinguer pro / particulier), conformément à
   l'itération multi-datasets prévue (`ml/AGENTS.md`).

## Alternatives écartées

- **Correction forfaitaire du prix** (−10 à −20 %) : coefficient invérifiable avec les données
  actuelles, fausse précision.
- **Ignorer le biais** : contraire à l'exigence de transparence sur les limites de la donnée
  (cœur du bloc BC04).

## Conséquences

- La fourchette restituée s'interprète comme « prix affiché en vente professionnelle » — utile
  au particulier comme **référence haute** de négociation, à défaut d'un prix particulier.
- Ce biais devient un **critère de sélection** pour le dataset candidat #2.
- Pour le dossier : la découverte tardive (colonne inspectée après plusieurs itérations)
  montre l'intérêt d'inventorier **toutes** les colonnes dès l'analyse exploratoire, y compris
  celles qu'on ne compte pas utiliser.
