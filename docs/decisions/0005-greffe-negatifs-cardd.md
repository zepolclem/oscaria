# 0005 — Greffer des négatifs « intact » dans CarDD : adopté

- **Date** : 2026-08-01
- **Statut** : Acceptée
- **Suite de** : [0001](0001-cible-multi-etiquette-cardd.md) (cible multi-étiquette) et
  [0002](0002-non-transfert-cardd-photos-annonce.md) (non-transfert)

## Contexte

CarDD ne contient **aucune voiture intacte** ([0001](0001-cible-multi-etiquette-cardd.md)) : la
probabilité a priori d'un dégât valait 1 pendant tout l'apprentissage. Le modèle ne répond donc pas
à « y a-t-il un dégât ? » mais à « lequel des six, sachant qu'il y en a un ». Sur une voiture
intacte, il choisit le type le moins improbable — `dent` à 0,946 sur une Passat impeccable.

Constat qui rend la correction possible : **l'architecture sait déjà dire « rien »**. En
multi-étiquette, « intact » est le vecteur `[0,0,0,0,0,0]`, que la perte d'entropie croisée binaire
gère nativement. Il ne manquait pas une septième classe, il manquait des **exemples**.

Source de négatifs retenue : photos leboncoin **déclarées intactes** (fiables à ~98 %,
[0003](0003-etiquette-declaree-asymetrique.md)), découpées en tuiles après détourage pour
ressembler au cadrage de CarDD. Un jeu externe (Stanford Cars, CompCars…) aurait introduit un biais
de source : le réseau aurait appris à reconnaître la provenance de l'image, pas le dégât.

## Décision

Ré-entraîner le modèle 6 classes sur **CarDD + 1 671 tuiles de carrosserie intacte** à cible
`[0,0,0,0,0,0]`, protocole de la fiche 0001 inchangé (ResNet18, 224 px, Adam 1e-4, lots de 32,
10 époques à budget fixe, meilleure époque sur validation, `test2017` fermé). `pos_weight` est
recalculé sur l'ensemble combiné.

Anti-fuite : tuiles issues d'annonces **hors** du lot d'annotation, découpage train/val par annonce.

## Mesures

| volet | baseline | greffé |
|---|---|---|
| typage — macro-F1 sur `val2017` seul | 0,815 | **0,818** |
| fausse alerte sur tuiles intactes jamais vues | **80,2 %** | **2,9 %** |
| probabilité maximale moyenne sur ces tuiles | 0,743 | 0,067 |
| aire ROC sur les 471 photos annotées, pleine photo | 0,509 | 0,640 |
| aire ROC sur les 471, tuiles sur détourage | 0,496 | **0,706** |

Checkpoint : `dl/models/cardd_greffe.pt`. Carnet : `dl/notebooks/cardd/02_greffe_negatifs.ipynb`.

## Alternatives écartées

| alternative | raison du rejet |
|---|---|
| ajouter une 7e classe « intact » | inutile : en multi-étiquette, « intact » est déjà représentable par le vecteur nul. Une classe supplémentaire aurait introduit une concurrence artificielle entre « intact » et les six types |
| greffer un dataset externe de voitures intactes | biais de source rédhibitoire — positifs et négatifs de styles différents, le réseau apprend la provenance. Écueil déjà mesuré dans ce projet |
| régler les seuils du modèle d'origine | tué par l'aire ROC de 0,509 ([0002](0002-non-transfert-cardd-photos-annonce.md)) : aucun seuil ne crée d'information dans un classement aléatoire |

## Conséquences

- **Le modèle sait se taire.** C'est ce qui rend la localisation par tuiles à nouveau envisageable :
  avec 80 % de fausses alertes, une carte de tuiles était inexploitable ; à 2,9 %, elle redevient
  une carte de dégâts. C'était l'échec de la variante « tuiles » du sondage initial.
- **Il reste sous le binaire dédié** (0,706 contre 0,855, [0004](0004-binaire-domaine-annonce.md)).
  Attendu : le binaire est entraîné exactement sur le domaine et la tâche d'évaluation, tandis que
  la greffe apprend une tâche plus riche avec des positifs d'un autre style.
- **Rôle dans le produit** : étage de **typage**, en aval d'une détection, jamais en premier niveau.

## Limites

- Négatifs issus d'étiquettes déclarées (~98 % fiables) : quelques dégâts discrets peuvent s'y
  glisser.
- **Biais de source inversé, assumé** : positifs en style CarDD, négatifs en style leboncoin. Le
  0,706 mesure autant le décalage de style que la capacité de détection.
- **Licence** : CarDD est distribué pour la recherche et l'éducation uniquement, ses images venant
  de Flickr et Shutterstock — ses auteurs n'en détiennent pas le copyright. Utilisable pour la
  certification, **pas pour un produit commercial**. À trancher avant toute mise en service.
