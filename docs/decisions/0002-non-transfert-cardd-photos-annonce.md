# 0002 — Le modèle CarDD ne transfère pas sur les photos d'annonce

- **Date** : 2026-07-31
- **Statut** : Acceptée
- **Remplace** : rien. **Restreint** la portée de [0001](0001-cible-multi-etiquette-cardd.md).

## Contexte

Le modèle de [0001](0001-cible-multi-etiquette-cardd.md) atteint macro-F1 = 0,815 sur CarDD, jeu
composé de **gros plans** où le dégât occupe 30 à 80 % du cadre. La cible produit d'OscarIA est
tout autre : une photo d'annonce cadre la voiture entière à plusieurs mètres, où une bosse d'aile
occupe ~2 % de l'image et, après redimensionnement à 224 px, une dizaine de pixels.

Deux mesures ont été faites, dans cet ordre.

**Sondage sur 13 photos** (`dl/notebooks/real_test/01_sondage_recadrage.ipynb`), trois variantes :
photo entière, détourage du véhicule, tuiles glissantes sur le détourage. Résultat : le modèle
prédit des dégâts à 0,61–0,95 sur les **trois voitures intactes** du lot, et la variante « tuiles »
sature tout le monde à 0,96–0,99. Médiane du maximum : 0,946 (entière), 0,879 (détourée),
0,996 (tuiles).

**Mesure sur 471 photos annotées** (voir [0003](0003-etiquette-declaree-asymetrique.md) pour le
protocole d'annotation), via l'aire sous la courbe ROC — qui ne dépend d'aucun seuil et mesure
seulement la capacité à *ordonner* :

| signal | aire ROC |
|---|---|
| maximum des 6 probabilités | **0,509** |
| `scratch` | 0,544 |
| `crack` / `tire flat` | 0,510 |
| `glass shatter` | 0,504 |
| `dent` | 0,460 |
| `lamp broken` | 0,447 |

La probabilité moyenne de `dent` est même **plus élevée sur les voitures intactes** (0,813) que sur
les abîmées (0,767).

## Décision

**Le modèle CarDD n'est pas utilisable sur le domaine « photo d'annonce », et aucun réglage ne peut
l'y rendre utilisable.** Le pilier État bascule vers un modèle entraîné sur des photos du bon
domaine — voir [0004](0004-binaire-domaine-annonce.md).

CarDD reste valable dans son périmètre : typer un dégât sur un gros plan de dégât. Il est conservé
comme brique éventuelle en aval d'un recadrage serré, pas comme détecteur de premier niveau.

## Alternatives écartées

| alternative | raison du rejet |
|---|---|
| régler les seuils par classe sur le domaine annonce | **impossible** : une aire ROC de 0,509 signifie un classement aléatoire. Un seuil déplace le compromis précision/rappel le long d'un classement ; il ne peut pas créer d'information dans un classement qui n'en contient pas. Les distributions se recouvrent complètement (intactes 0,61–0,95, abîmées 0,33–1,00) |
| découper la photo en tuiles pour retrouver des gros plans | mesuré, contre-productif : une tuile de 224 px prise dans un détourage de 1 000 px ne contient souvent que de la peinture, une jante ou du bitume — hors distribution d'entraînement, le réseau y sort des valeurs élevées arbitraires. S'y ajoute la multiplicité des tirages (quelques dizaines de tuiles, on garde le maximum) : la Porsche intacte passe de 0,612 à 0,991 |
| détourer le véhicule puis appliquer CarDD | mesuré, sans effet sur ce lot : la médiane du maximum **baisse** (0,879 contre 0,946), les photos testées étant déjà des plans rapprochés |
| utiliser CarDD pour pré-trier les photos à annoter | à 0,509, ce tri vaut un tirage au sort |
| conclure dès le sondage sur 13 photos | 13 photos jugées à l'œil suggèrent, elles ne tranchent pas. Le sondage laissait espérer qu'un seuil suffirait ; la mesure sur 471 photos l'a réfuté |

## Conséquences

- **La cause est identifiée et documentée** : CarDD ne contient aucune voiture intacte
  ([0001](0001-cible-multi-etiquette-cardd.md)), donc la probabilité a priori d'un dégât valait 1
  pendant tout l'apprentissage. Le réseau ne répond pas à « y a-t-il un dégât ? » mais à « lequel
  des six, sachant qu'il y en a un ». Il n'a aucune sortie pour dire « rien ».
- Les leviers d'optimisation prévus sur CarDD (résolution 384, seuils par classe, ResNet50,
  augmentation) sont **suspendus** : ils amélioreraient le typage sur gros plan, pas la capacité à
  détecter un dégât sur une photo d'annonce.
- Le checkpoint `dl/models/cardd_baseline.pt` est conservé, ainsi que son carnet et ses métriques.
- **Portée de la mesure** : 471 photos d'annonces de particuliers, un seul annotateur, plateforme
  unique. La conclusion « pas de transfert » est robuste (0,509 est très loin de tout seuil
  d'utilité), mais son ampleur exacte sur d'autres segments n'est pas établie.

## Leçon de méthode

Une petite mesure qualitative (13 photos) a suggéré une conclusion — « il faut recadrer » — que la
mesure quantitative a réfutée. Ce n'est pas le recadrage qui manquait, c'est le domaine
d'entraînement. La bonne réponse a coûté 471 annotations et vingt minutes.
