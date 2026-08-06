# 0003 — L'état déclaré par le vendeur : exploitable côté intact, inutilisable côté abîmé

- **Date** : 2026-07-31
- **Statut** : Acceptée
- **Pilier** : État (carrosserie) — protocole d'étiquetage

## Contexte

La collecte de juillet 2026 (`collecte/scraper`) a rapporté **22 204 photos** d'annonces leboncoin
de particuliers, rangées par **état déclaré par le vendeur** : le label est dans le chemin
(`raw/images/<état>/<ad_id>/NN.jpg`), 8 valeurs, ~3,2 photos par annonce.

Il fallait des exemples de voitures **intactes**, absents de CarDD
([0001](0001-cible-multi-etiquette-cardd.md)). L'état déclaré est la seule étiquette disponible à
grande échelle — d'où la question : décrit-il ce que la photo montre ?

Les quatre états intermédiaires (`good_overall_condition`, `normal_wear_and_tear`,
`minor_repairs_needed`, `major_repairs_needed`) ont été écartés d'emblée : « réparations mineures »
désigne aussi bien une révision qu'une aile froissée. Restent les deux extrémités de l'échelle :
`undamaged` + `excellent_condition` (5 355 photos) contre `damaged` + `not_drivable` (5 718).

## Décision

**Protocole d'annotation aveugle**, puis usage asymétrique de l'étiquette déclarée :

1. échantillon de 800 annonces (200 par dossier), **une photo par annonce** tirée au hasard — pas
   la première, qui est presque toujours le plan de trois-quarts valorisant ;
2. copie sous **nom neutre** (`0001.jpg`), redimensionnée à 1 024 px, **ordre mélangé** : le chemin
   ne révèle plus l'état déclaré ;
3. **filtre automatique en amont** (`detect.py`, détection de véhicule, `fraction_cadre` ≥ 0,3) —
   le détecteur ne voit jamais l'étiquette ;
4. annotation à trois verdicts : `intact`, `abîmé`, **`à jeter`** (intérieur, document, montage,
   capture d'écran, photo floue, ou simple hésitation) ;
5. **usage** : les négatifs déclarés sont exploitables à grande échelle, les positifs déclarés ne
   servent que de vivier à annoter — jamais de cible directe.

Outil : `dl/src/annoter.py` (page HTML autonome, zéro dépendance, raccourcis clavier, sauvegarde
navigateur, export CSV).

## Mesures

Filtre automatique : **648 photos gardées sur 800**, avec des taux de rejet homogènes entre les
quatre dossiers (16 % à 21 %). Le filtre nettoie **sans déformer** la composition du jeu — vérifié,
non supposé.

Annotation : 648 photos jugées → **366 intactes, 105 abîmées, 177 à jeter (27 %)**.
Accord global avec l'état déclaré sur les 471 jugeables : **66,7 %**. Mais l'asymétrie est totale :

| dossier déclaré | jugé « abîmé » | jugé « intact » | fiabilité |
|---|---|---|---|
| `undamaged` | 1 | 100 | **99 %** |
| `excellent_condition` | 2 | 112 | **98 %** |
| `damaged` | 66 | 55 | 55 % |
| `not_drivable` | 36 | 99 | **27 %** |

## Cause, et pourquoi elle est structurelle

**L'état déclaré porte sur le véhicule ; la photo montre une vue.** Une annonce `damaged` contient
trois ou quatre photos dont une seule cadre l'aile froissée ; les autres montrent l'avant intact, le
profil, les jantes. La déclaration est vraie au niveau de l'annonce et fausse au niveau de trois
photos sur quatre.

`not_drivable` est le cas extrême, avec 73 % de photos sans dégât visible : « non roulant » désigne
le plus souvent un moteur ou une boîte hors service, sur une carrosserie intacte. L'étiquette est
juste — elle ne parle simplement pas de ce que le modèle regarde.

Ce bruit n'est donc **pas réductible** par un meilleur filtrage : il vient de la différence de
granularité entre la déclaration (annonce) et la cible (photo).

## Alternatives écartées

| alternative | raison du rejet |
|---|---|
| entraîner directement sur l'étiquette déclarée | 60 % des positifs déclarés ne montrent aucun dégât : le réseau apprendrait que des voitures visiblement intactes sont abîmées, ce qui détruirait sa précision |
| annoter sans masquer l'état déclaré | biais d'ancrage : le jugement s'aligne sur l'étiquette, et l'écart déclaration/réalité — l'objet même de la mesure — devient inobservable |
| prendre la première photo de chaque annonce | c'est le plan valorisant du vendeur, systématiquement le plus flatteur : biais de cadrage garanti |
| prendre toutes les photos de chaque annonce | à temps d'annotation égal, moins de voitures différentes, et une gestion de fuite à assurer entre découpages |
| deux verdicts au lieu de trois | une photo d'intérieur classée « intact » apprend au réseau que « tableau de bord = pas de dégât » — raccourci parfait sur le jeu, effondrement en production |
| Label Studio (docker) | ~1 Go à installer et une remise en place, pour un besoin couvert par une page HTML de 100 lignes |

## Conséquences

- **Négatifs disponibles à grande échelle** : 5 355 photos déclarées intactes, fiables à ~98 %,
  utilisables sans annotation.
- **Positifs à annoter à la main.** Rendement mesuré : ~40 % dans `damaged`, ~27 % dans
  `not_drivable`. Pour 100 positifs, compter ~300 photos à trier.
- **27 % des photos ne sont pas notables.** Le tri des vues (extérieur / intérieur / document) est
  un prérequis de la chaîne produit, pas une option.
- Cas signalés à traiter en amont : **montages** (plusieurs photos en une image — la grille
  elle-même devient un indice si elle corrèle avec l'état) et **captures d'écran de smartphone**
  (photo de seconde main : recopie d'annonce, donc risque de doublon de véhicule entre deux `ad_id`,
  ce qui casserait le découpage par annonce).

**Limites.** Un seul annotateur, non expert, aucun accord inter-annotateurs mesuré. Le critère
retenu est « dégât de carrosserie visible sur *cette* photo », hors usure normale, saleté, dégâts
mécaniques et d'intérieur.

**RGPD.** Photos issues d'annonces en ligne, pouvant porter plaques et visages. Contenu exclu de
`git`, usage interne de recherche, jamais rediffusé ; les copies d'annotation suivent le même
régime. Une plaque rattachable à une personne est une donnée personnelle : masquage requis avant
tout livrable.
