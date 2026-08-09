# ML 0005 — Contrat d'entrée : quelles questions on pose au vendeur

- **Date** : 2026-08-08
- **Statut** : Accepté
- **Pilier** : Prix

## Contexte

Le modèle du carnet 04 utilisait 15 variables, choisies pour leur pouvoir prédictif. Le
formulaire, lui, s'adresse à un particulier qui remplit une page web : chaque champ demandé
est une occasion d'abandonner, et certaines variables (puissance fiscale, Crit'Air, validité
du contrôle technique) exigent d'aller chercher la carte grise.

Le jeu de variables n'est donc **pas** un choix de modélisation seul : c'est le contrat entre
le produit et le modèle. Chaque modification a été **mesurée en validation croisée à 5 plis
avant d'être appliquée**, jamais décidée au jugé.

## Décision

### Les champs

**Obligatoires** : marque, modèle, année de mise en circulation, kilométrage, énergie, boîte
de vitesses, état déclaré.

**Facultatifs** (repliés dans un panneau) : puissance DIN, puissance fiscale, portes, places,
Crit'Air, validité du contrôle technique, couleur.

Un champ facultatif vide devient une **valeur manquante**, traitée nativement par
`HistGradientBoosting` : à chaque nœud, il apprend de quel côté envoyer les trous. Le vide est
une information, pas un bouchon. C'est ce qui permet un formulaire court sans casser le
modèle — et c'est la raison technique pour laquelle l'imputation par la médiane du carnet 03
a été retirée.

### Les changements, avec leur coût mesuré

| Changement | Effet sur la MAE |
|---|---|
| `region` **retirée** | **−14 €** |
| Mois de mise en circulation **retiré** (année seule) | **−4 €** |
| `modele` **ajouté** (encodage par fréquence) | **−35 €** |
| États regroupés de 8 à 5 crans | **0 €** |
| **Bilan** | **1 521 € → 1 471 €**, avec un champ de moins |

Retirer `region` **améliore** le modèle. Ce n'est pas un paradoxe : son importance par
permutation valait 0,0014 (fiche 0004). Une variable sans information n'est pas neutre — elle
fournit du bruit dans lequel l'arbre peut creuser des coupes qui ne généralisent pas. La
demande produit et l'intérêt du modèle allaient ici dans le même sens.

### Les états, regroupés d'après les prix et non d'après le sens commun

| Cran retenu | États d'origine fusionnés | Prix médians | n |
|---|---|---|---|
| Ne roule pas ou grosses réparations | `not_drivable`, `damaged`, `major_repairs_needed` | 1 300 / 1 500 / 1 500 € | 4 868 |
| Petites réparations à prévoir | `minor_repairs_needed` | 2 500 € | 3 248 |
| Usure normale | `normal_wear_and_tear` | 4 500 € | 3 305 |
| Bon état | `good_overall_condition`, `undamaged` | 7 000 / 9 000 € | 6 549 |
| Excellent état | `excellent_condition` | 16 000 € | 1 912 |

Les trois pires crans sont **indiscernables en prix** : les fusionner ne perd rien.
`undamaged` (9 000 €) et `excellent_condition` (16 000 €) sont séparés par 7 000 € : les
fusionner en perdrait — ce que confirme la mesure, un regroupement à 3 crans coûtant +28 €
de MAE contre 0 € à 5 crans.

L'échelle d'origine n'est d'ailleurs pas ordonnée en prix : `not_drivable` (1 300 €) est
*moins* cher que `damaged` (1 500 €). Le carnet 03 avait déjà mesuré que l'encodage ordinal
y perdait face au one-hot pour cette raison.

### Le modèle exact, encodé par fréquence

804 modalités, bien au-delà des 255 que le catégoriel natif accepte. Chaque modèle est
remplacé par son **nombre d'occurrences dans le jeu d'ajustement** — une mesure de popularité,
qui ne regarde jamais le prix, donc sans fuite de la cible.

**Limite importante, mesurée** : cet encodage ne dit pas au modèle *quel* véhicule c'est,
seulement *à quel point ce modèle est courant*. Une Clio et une Twingo aux caractéristiques
identiques ne diffèrent que de **35 €** dans la prédiction, parce que leurs fréquences sont
voisines. Le gain de 35 € de MAE est donc indirect : l'information apprise est « les modèles
rares se vendent différemment des modèles courants ».

## Alternatives écartées

- **Encodage par prix moyen** (*target encoding*) : mesuré meilleur au carnet 04 (1 465 €
  contre 1 485 €) et il identifierait vraiment le modèle. Écarté à ce stade parce qu'il
  regarde la cible : 341 modèles sur 758 sont vus trois fois ou moins, l'encodage y devient
  une mémorisation du prix de ces quelques annonces. **Décision non figée** — à reprendre avec
  un lissage vers la moyenne globale.
- **Formulaire complet à 15 champs obligatoires** : Crit'Air, puissance fiscale et validité du
  contrôle technique imposent de sortir la carte grise. Friction jugée rédhibitoire.
- **Socle court strict** (sans champs facultatifs) : perdrait `puissance_din`, deuxième
  variable la plus importante (chute de R² 0,208). Le compromis retenu — facultatifs en
  valeurs manquantes natives — évite d'avoir à choisir.
- **Table de correspondance** remplissant automatiquement puissance et portes depuis
  marque + modèle + année : imputerait une valeur plausible à la place d'une valeur réelle,
  biais supplémentaire pour un gain non mesuré.

## Conséquences

- **Le contrat ne peut pas changer d'un seul côté.** Les listes de variables vivent dans
  `app/src/preparation.py`, importé par l'entraînement *et* par le service. Les modifier
  impose de ré-entraîner ; ne pas le faire produirait des prix faux sans lever d'erreur.
- **Le contrat d'API est verrouillé.** Un champ inconnu, ou une valeur d'état de l'ancienne
  échelle à 8 crans, renvoie **422** au lieu d'être silencieusement ignoré. Constaté avant
  correction : `etat="normal_wear_and_tear"` passait en 200 et retombait sur « inconnu », le
  client recevant un prix calculé comme si l'état n'était pas renseigné.
- Les bornes numériques (kilométrage ≥ 0, Crit'Air ≤ 5, …) ne protègent pas le modèle — un
  arbre ne plante pas sur une valeur extrême — mais évitent de rendre un prix crédible sur une
  saisie absurde. Constaté avant correction : un kilométrage de −5 000 produisait une
  estimation parfaitement plausible.
- Le modèle exact n'est pas encore correctement exploité (cf. limite ci-dessus). C'est le
  premier chantier ouvert du pilier.
