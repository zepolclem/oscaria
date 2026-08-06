# 0001 — Cible multi-étiquette sur CarDD, et protocole de la baseline

- **Date** : 2026-07-30
- **Statut** : Acceptée
- **Pilier** : État (carrosserie)
- **Portée** : dataset CarDD uniquement — voir [0002](0002-non-transfert-cardd-photos-annonce.md)
  pour ce qu'il advient sur photos d'annonce.

## Contexte

Le pilier État repart de zéro après la remise à plat du 2026-07-29. Objectif de la phase :
entraîner correctement un ResNet sur CarDD, avec des métriques honnêtes. Cinq décisions étaient à
prendre avant d'écrire du code, chacune contrainte par une mesure faite sur les données.

Mesures d'entrée (annotations COCO de CarDD) :

- 2 816 / 810 / 374 images en `train2017` / `val2017` / `test2017` ;
- **0 image sans annotation de dégât** — le jeu ne contient aucune voiture intacte ;
- **38 % des images portent au moins deux types de dégât** ;
- déséquilibre marqué : `scratch` 1 507 contre `tire flat` 219, soit ~7 pour 1 ;
- aucune fuite détectable entre découpages (0 nom de fichier commun, 0 doublon par empreinte md5).

## Décision

1. **Cible multi-étiquette à 6 sorties indépendantes** (`dent`, `scratch`, `crack`,
   `glass shatter`, `lamp broken`, `tire flat`), perte d'entropie croisée binaire par classe
   pondérée par `pos_weight` = négatifs / positifs. Le binaire « abîmé / pas abîmé » est **dérivé
   à l'inférence** (au moins une classe au-dessus du seuil), jamais entraîné.
2. **Découpages natifs de CarDD**, utilisés tels quels. `test2017` reste fermé jusqu'à
   l'évaluation finale.
3. **macro-F1 comme métrique arbitre**, tableau precision / rappel / F1 par classe toujours
   produit, micro-F1 en secondaire. L'exactitude est refusée.
4. **Baseline en trois lignes** : planchers triviaux, backbone gelé, affinage complet.
5. **Budget fixe de 10 époques**, poids de la meilleure époque sur validation — jamais la dernière.

## Alternatives écartées

| alternative | raison du rejet |
|---|---|
| binaire « intact / abîmé » entraîné sur CarDD | **impossible** : 0 image intacte dans le jeu, le modèle répondrait « abîmé » à 100 % et obtiendrait un score parfait sans rien apprendre |
| cible mono-étiquette (une classe par image, softmax) | obligerait à inventer une règle de « dégât dominant » pour 38 % des images, et à compter comme erreur une prédiction correcte du second dégât |
| détection ou segmentation de zones | plus lourd à entraîner et à évaluer ; les masques CarDD restent disponibles si la localisation redevient nécessaire |
| exactitude comme métrique | ne jamais prédire `tire flat` (59 des 810 images de validation) donne déjà 92,7 % — la métrique récompenserait l'abandon des classes rares |
| micro-F1 comme arbitre | un échec total sur une classe rare y est presque invisible ; macro-F1 le rend impossible à masquer |
| arrêt précoce | chaque variante s'arrêterait à une époque différente, la comparaison ne serait plus à budget de calcul égal |

## Conséquences

Résultats obtenus (`dl/notebooks/cardd/01_baseline.ipynb`, validation `val2017`) :

| modèle | macro-F1 | micro-F1 |
|---|---|---|
| plancher « toujours oui » | 0,379 | 0,406 |
| plancher « fréquences » | 0,259 | 0,377 |
| ResNet18 backbone gelé (3 078 paramètres entraînés sur 11,2 M) | 0,717 | 0,717 |
| **ResNet18 affinage complet** | **0,815** | 0,802 |

- Le transfert d'apprentissage fait l'essentiel : +0,338 sur le plancher avec 0,03 % des
  paramètres entraînés. L'affinage complet ajoute 0,098.
- Aucune classe sacrifiée : écart macro/micro de −0,013 (contre +0,117 pour le plancher aléatoire),
  et `tire flat` atteint F1 = 0,931 malgré 219 exemples.
- Classe faible : `crack` à F1 = 0,629, avec des confusions systématiques vers `dent` et `scratch`.
- Coût dominé par le décodage JPEG (29 s/époque) et non par le calcul (15 s) : à 224 px, un GPU
  plus rapide n'apporterait presque rien.

**Limites assumées.**
- L'absence de fuite est vérifiée par nom de fichier et empreinte md5 : la même voiture
  photographiée sous un autre angle dans deux découpages ne serait pas détectée.
- La ligne « backbone gelé » n'a pas atteint son plateau au budget de 10 époques : elle est
  sous-évaluée, la comparaison reste équitable mais l'écart réel est plus faible.
- 43 % des images de validation portent au moins une erreur au seuil 0,5, alors que le macro-F1
  affiche 0,815 : la métrique par classe est bien plus indulgente qu'une exigence « toutes les
  étiquettes justes ».
- **Le point 1 porte une conséquence lourde** : le modèle n'a aucune sortie pour dire « rien ». Sur
  une voiture intacte, il choisit le type le moins improbable. Mesuré en [0002](0002-non-transfert-cardd-photos-annonce.md).
