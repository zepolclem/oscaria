# Clôture de l'arc leboncoin/UC3M-LP — reconstruction d'une historique ADR logique

- **Date** : 2026-08-09
- **Statut** : validé (décisions utilisateur du 2026-08-09)

## Contexte

Un audit de cohérence des fiches de décision (ADR) et des modèles entraînés a décelé que le
pilier Plaques s'était développé sur **deux lignes parallèles** : la branche locale
`reset/plaques` (jamais poussée) et `main` (via `zepolclem/no-plan-question`). Chacune a
écrit ses propres fiches 0001/0002 et entraîné son propre modèle. Trois incohérences en
résultent sur `main` :

1. **Photos leboncoin** : trois versions contradictoires de leur sort — « `raw/` intact »
   (fiche 0002 de la branche locale, 2026-08-06), « dossier vide » (ADR DL 0002 de `main`,
   2026-08-08), « alimentent le pilier plaques » (ADR ML 0001, même jour).
2. **AP 0,932 orphelin** : cette précision moyenne (*Average Precision*) appartient à
   l'entraînement de la branche locale (taux d'apprentissage 1e-3), pas au modèle déployé
   de `main` (taux 0,005, checkpoint différent — vérifié par empreinte git), dont l'ADR DL
   0001 ne consigne aucune mesure.
3. **Plan UC3M-LP jamais soldé** : la branche locale prévoyait de mesurer le transfert sur
   le dataset UC3M-LP (plaques espagnoles, même gabarit que les françaises) ; l'inférence
   n'a jamais été lancée, aucun verdict n'existe. La « gate 0,80 » citée par l'ADR DL 0002
   de `main` provient de ce plan absent de `main`.

## Décisions utilisateur

- **Photos leboncoin : oubliées pour le deep learning.** Elles devaient entraîner le pilier
  Plaques, mais leboncoin floute les plaques à la source dans la grande majorité des
  annonces — elles ne servent à rien. À consigner là où ça a sa place.
- **Arc UC3M-LP : clos sans verdict.** Aucune mesure n'a jamais été produite ; la question
  du transfert a reçu une meilleure réponse sur 87 photos réelles françaises (ADR DL 0002).
  Ce qui ne sert à rien, on l'oublie — mais on le consigne (BC04 : résultats négatifs et
  abandons inclus).
- **Branche `reset/plaques` : tombeau local** (pattern existant `reset/cardd-baseline`),
  ni supprimée ni poussée.
- **Disque : suppression du seul `dl/data/uc3m-lp/raw/UC3M-LP.zip`** (4,2 Go, doublon de
  l'archive extraite, re-téléchargeable sur Zenodo) ; le dataset extrait est conservé.

## Marches

1. **Fiche `docs/decisions/dl/0003-cloture-arc-leboncoin-uc3m.md`** : sortie des photos
   leboncoin du pilier, clôture UC3M-LP sans verdict, généalogie des deux entraînements
   (à qui appartient l'AP 0,932). Index `docs/decisions/dl/README.md` mis à jour.
2. **Errata datés** (les fiches restent immuables, on ajoute une note, on ne réécrit pas) :
   ADR ML 0001 (ligne « photos alimentent le pilier plaques »), ADR DL 0002 (renvoi vers la
   branche locale rendu autoportant), `docs/decisions/README.md` (tombeau `reset/plaques`
   ajouté à l'historique).
3. **Ménage disque** : suppression du zip UC3M-LP.

## Vérification

- Tout lecteur de `main` seul reconstitue l'histoire complète du pilier Plaques, sans
  renvoi pendant vers une branche invisible.
- Les renvois internes des fiches touchées résolvent vers des fichiers existants.
- `reset/plaques` toujours listée par `git branch` ; dataset UC3M extrait intact.
