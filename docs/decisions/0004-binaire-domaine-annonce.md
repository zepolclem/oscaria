# 0004 — Binaire « abîmé / intact » entraîné sur le domaine annonce, en 384 px sur détourage

- **Date** : 2026-08-01
- **Statut** : Acceptée
- **Suite de** : [0002](0002-non-transfert-cardd-photos-annonce.md) (non-transfert) et
  [0003](0003-etiquette-declaree-asymetrique.md) (étiquetage)

## Contexte

CarDD ne transfère pas ([0002](0002-non-transfert-cardd-photos-annonce.md)). Il faut donc un modèle
entraîné sur des photos du bon domaine. On dispose de **471 photos annotées à l'aveugle**
(105 abîmées, 366 intactes), une photo par annonce — donc 471 voitures différentes, ce qui rend la
fuite par véhicule structurellement impossible à l'intérieur du lot.

## Décision

**Classifieur binaire ResNet18 à une sortie**, entraîné sur les verdicts humains du domaine
annonce, avec :

- **initialisation ImageNet** (pas CarDD) ;
- **entrée 384 px** appliquée sur le **détourage du véhicule** (`detect.py`, marge 5 %, repli sur
  l'image entière si aucun véhicule détecté) ;
- perte d'entropie croisée binaire pondérée par `pos_weight` (déséquilibre 3,5 pour 1) ;
- augmentation légère à l'entraînement seulement : retournement horizontal, variation
  colorimétrique ;
- **évaluation en validation croisée à 5 blocs** : chaque photo est prédite par un modèle qui ne l'a
  jamais vue.

**Métrique arbitre : la précision moyenne** (aire sous la courbe précision/rappel), pas l'aire ROC.
En déséquilibre, l'aire ROC flatte — les 366 négatifs dominent le calcul. La précision moyenne ne
regarde que la classe rare, et son plancher n'est pas 0,5 mais la prévalence : **0,223**.

Code : `dl/src/binaire.py`, expériences rejouables par `dl/src/experiences_binaire.py`,
carnet `dl/notebooks/leboncoin-private/01_annotation_et_pilote.ipynb`.

## Mesures

**La tâche s'apprend**, et ce ne sont pas les données qui manquent :

| essai (224 px, photo entière) | aire ROC | précision moyenne |
|---|---|---|
| plancher (hasard) | 0,500 | **0,223** |
| init ImageNet, 100 % des données | 0,818 | 0,615 |
| init CarDD, 100 % des données | 0,782 | 0,578 |
| ImageNet, 25 % des données | 0,802 | 0,584 |
| ImageNet, 50 % des données | 0,816 | 0,617 |
| ImageNet, 75 % des données | 0,810 | 0,627 |

**Leviers d'échelle**, chacun isolé, mêmes photos et même découpage :

| configuration | aire ROC | précision moyenne | gain | durée |
|---|---|---|---|---|
| 224 px, photo entière | 0,818 | 0,615 | référence | 94 s |
| 224 px, détourée | 0,831 | 0,657 | +0,042 | 90 s |
| 384 px, photo entière | 0,844 | 0,672 | +0,057 | 420 s |
| **384 px, détourée** | **0,855** | **0,689** | **+0,074** | 309 s |

Au meilleur seuil : précision 0,61, rappel 0,65.

## Lecture, avec ses réserves

- **La tâche est apprenable** : 0,689 contre un plancher de 0,223, soit 3,1 fois le hasard — là où
  CarDD plafonnait au tirage au sort sur les mêmes photos.
- **La courbe d'apprentissage est plate** (0,584 à 25 %, 0,615 à 100 %, incertitude ±0,05) : à 25 %,
  le modèle voit ~94 photos dont ~21 abîmées et fait déjà aussi bien qu'avec 377. Annoter davantage
  n'aurait rien apporté **à cette résolution** — c'est ce qui a orienté l'effort vers l'échelle
  plutôt que vers l'annotation.
- **Pris isolément, chaque gain d'échelle est à la limite du bruit** (±0,05). Ce qui fonde la
  conclusion, c'est la cohérence du motif : deux manipulations indépendantes vont dans le même sens
  et se cumulent approximativement. Le bruit ne produit pas cet ordre.
- **Le tronc CarDD n'apporte rien** (0,578 contre 0,615). L'écart reste dans le bruit : on ne dira
  pas qu'il nuit, seulement qu'il ne sert pas. Ses filtres, spécialisés sur des textures en gros
  plan, ne valent pas un point de départ généraliste.
- **Économie** : 384 px sur détourage (309 s) est **plus rapide** que 384 px sur photo entière
  (420 s), les images recadrées étant plus petites à décoder. Le levier le plus efficace est aussi
  l'un des moins chers.

## Alternatives écartées

| alternative | raison du rejet |
|---|---|
| initialiser depuis `cardd_baseline.pt` | mesuré inférieur (0,578 contre 0,615) — le tronc spécialisé gros plan ne transfère pas mieux qu'ImageNet |
| annoter 1 000 photos de plus avant d'entraîner | la courbe d'apprentissage plate montre que le gain aurait été nul à cette résolution ; la mesure a coûté 12 minutes de machine et économisé 15 minutes d'annotation inutile |
| entraîner sur les 5 355 négatifs déclarés contre 105 positifs annotés | déséquilibre 50 pour 1 ; écarté au profit d'une mesure préalable de faisabilité — reste ouvert si la précision doit monter |
| aire ROC comme arbitre | flatte en déséquilibre ; la précision moyenne est la mesure honnête sur la classe rare |
| sélectionner l'époque sur le bloc de validation | le transformerait en jeu de réglage et gonflerait le résultat : budget d'époques fixe et identique pour toutes les configurations |

## Conséquences

- **État produit** : sur 10 alertes « voiture abîmée », ~4 sont fausses ; sur 10 dégâts réels,
  ~6,5 sont détectés. C'est un signal d'aide à la décision défendable — cohérent avec le
  positionnement d'OscarIA (aide, pas verdict) et avec l'obligation d'affichage d'incertitude de
  l'AI Act — mais pas un verdict automatisable.
- **Le détourage devient un composant de la chaîne**, plus une option : il apporte un gain à coût
  nul et supprime au passage le fond (allée de gravier, hangar) comme raccourci potentiel.
- **Pistes ouvertes, non tranchées** : refaire la courbe d'apprentissage à 384 px détouré (la
  platitude constatée à 224 px pourrait être un effet de la résolution) ; inspecter les erreurs pour
  vérifier que le modèle ne répond pas à « voiture vieille ou sale » ; monter à 512 px.

## Limites

- **471 photos, dont 105 abîmées** : incertitude d'environ ±0,05 sur la précision moyenne. Tout
  écart inférieur n'est pas interprétable.
- **Un seul annotateur, non expert**, aucun accord inter-annotateurs mesuré
  ([0003](0003-etiquette-declaree-asymetrique.md)).
- **Les 177 photos « à jeter » sont exclues** de toutes les mesures : en production, un quart des
  photos ne sont pas notables, ce qui suppose un tri des vues en amont.
- **Un seul segment** : annonces de particuliers, une plateforme, collecte de juillet 2026. Rien
  n'est établi pour les annonces professionnelles ni pour d'autres sources.
