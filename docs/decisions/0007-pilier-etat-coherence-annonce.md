# 0007 — Le pilier État devient « cohérence déclaration ↔ photos », au niveau annonce

- **Date** : 2026-08-04
- **Statut** : Acceptée
- **Restreint** : [0004](0004-binaire-domaine-annonce.md) — même modèle, promesse différente
- **S'appuie sur** : [0003](0003-etiquette-declaree-asymetrique.md) (granularité de l'étiquette) et
  [0006](0006-tri-des-vues.md) (filtrage amont)

## Contexte

Deux constats ont imposé de changer la **promesse** plutôt que le modèle.

**La performance ne soutient pas la promesse initiale.** Le binaire de la fiche 0004 atteint 0,689
de précision moyenne ; à son meilleur seuil, précision 0,61 et rappel 0,65 — **quatre alertes
fausses sur dix**. Annoncer « cette voiture est abîmée » sur cette base serait indéfendable devant
un jury lisant les mêmes chiffres.

**Le désordre des photos est structurel** ([0006](0006-tri-des-vues.md)) : environ quatre photos sur
dix ne montrent pas de carrosserie exploitable.

**Et la granularité était fausse depuis le début** : la fiche 0003 a mesuré que l'état déclaré est
« vrai au niveau de l'annonce et faux au niveau de trois photos sur quatre ». Évaluer photo par
photo revenait à se battre contre une étiquette qui n'a jamais prétendu décrire une photo.

Une piste a été explorée puis écartée : la **capture guidée** (l'application demande au vendeur de
photographier telle zone). Elle présuppose que le vendeur sait déjà où est le dégât — la
localisation étant alors faite par l'humain, la valeur du modèle se réduit à nommer le type. S'y
ajoute l'absence totale de données de capture guidée : ce serait une démonstration sans mesure.

## Décision

**OscarIA ne dit pas « cette voiture est abîmée » mais « les photos de cette annonce ne soutiennent
pas l'état déclaré ».** Verdict au niveau **annonce**, pas photo.

Chaîne retenue :

```
photos de l'annonce
  → tri des vues (0006)        écarte intérieurs, documents, captures, photos ratées
  → détourage véhicule          detect.py, Faster R-CNN COCO en inférence pure
  → binaire d'état              384 px sur détourage, configuration gagnante de la fiche 0004
  → agrégation par MOYENNE      score d'annonce
  → comparaison à l'état déclaré
```

Code : `dl/src/annonce.py`, `dl/src/experiences_annonce.py`. Carnet :
`dl/notebooks/leboncoin-private/03_chaine_coherence.ipynb`.

Ce recadrage est cohérent avec le positionnement de **tiers neutre** du projet : OscarIA vérifie une
déclaration, il ne juge pas un véhicule.

## Mesures

600 annonces (300 `damaged`, 300 `undamaged`/`excellent_condition`), 1 813 photos, plancher 0,500.
`not_drivable` est exclu : 73 % de ses photos montrent une carrosserie intacte.

| niveau | filtre tri | règle | aire ROC | précision moyenne |
|---|---|---|---|---|
| photo, sans agrégation | non | — | 0,729 | 0,722 |
| annonce | non | max | 0,770 | 0,775 |
| annonce | non | moyenne | 0,793 | 0,799 |
| annonce | non | moyenne des 2 meilleures | 0,781 | 0,791 |
| annonce | oui | max | 0,799 | 0,808 |
| annonce | **oui** | **moyenne** | **0,807** | 0,807 |
| annonce | oui | moyenne des 2 meilleures | 0,806 | 0,810 |

Décomposition des deux effets, mesurés séparément : **agrégation +0,064**, **filtrage +0,014**.

Point de fonctionnement, règle « moyenne » avec filtre (593 annonces) :

| seuil | précision | rappel | part des annonces signalées |
|---|---|---|---|
| 0,344 | **0,903** | 0,345 | 19,1 % |
| 0,202 | 0,851 | 0,483 | 28,3 % |
| 0,085 | 0,801 | 0,625 | 39,0 % |

Score médian par état déclaré : **0,013** (intact) contre **0,178** (abîmé).

## Ce que ça donne côté produit

Au seuil le plus prudent : **OscarIA signale une annonce sur cinq, et neuf fois sur dix la
déclaration est effectivement contredite par les photos.** Il rate deux tiers des cas — comportement
attendu d'un tiers neutre, qui préfère se taire qu'accuser à tort.

## Alternatives écartées

| alternative | raison du rejet |
|---|---|
| capture guidée par l'application | présuppose que le vendeur localise lui-même le dégât ; la valeur du modèle se réduit à nommer le type. Et aucune donnée de capture guidée n'existe → démonstration sans mesure |
| garder la promesse « URL → état du véhicule » | 0,61 de précision au meilleur seuil : quatre alertes fausses sur dix, indéfendable |
| **agrégation par maximum** | mesuré inférieur (0,799 contre 0,807). Contre-intuitif : le maximum amplifie le bruit — une seule fausse alerte condamne l'annonce. À 0,61 de précision par photo, réduire le bruit rapporte plus que capter le signal rare |
| fermer le pilier État | le modèle est trop faible pour *détecter*, pas pour *contredire une déclaration*. La barre est plus basse et elle est atteinte |
| inclure `not_drivable` | 73 % de carrosseries intactes : ce serait demander de deviner une panne mécanique depuis une photo de tôle |

## Limites

- **L'évaluation compare le modèle à l'état DÉCLARÉ, pas à une vérité terrain.** Une annonce
  `damaged` dont le dégât est mécanique compte comme un échec alors que le modèle a raison.
  Cohérent avec la promesse, mais interdit de parler de détection de dégât.
- **Lot équilibré 50/50 par construction** ; la prévalence réelle est inconnue, donc la précision
  au même seuil différerait en population réelle.
- **Non comparable au 0,855 de la fiche 0004** : là-bas, le verdict d'un humain sur une photo avec
  22 % de prévalence ; ici, la déclaration d'un vendeur sur une annonce entière avec 50 %. Deux
  questions, deux planchers, deux populations.
- **Probabilités non calibrées** : la perte est pondérée pour compenser le déséquilibre, ce qui
  décale mécaniquement les sorties. Afficher « 0,34 » comme un risque de 34 % serait une allégation
  trompeuse au sens de l'obligation de transparence de l'AI Act. **La calibration est un prérequis
  à tout affichage** — non faite à ce jour.
- **7 annonces sur 600** perdent toutes leurs photos après filtrage : le produit doit répondre
  « photos insuffisantes », jamais un score par défaut.
- Binaire entraîné sur 471 photos annotées par **un seul annotateur non expert** ; un seul segment
  (particuliers, une plateforme, juillet 2026).
- **RGPD** : plaques et visages présents dans les photos, masquage requis avant tout livrable.
