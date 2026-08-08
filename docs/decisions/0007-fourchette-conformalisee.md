# 0007 — Fourchette de prix : régression quantile conformalisée, et jusqu'où la promesse tient

- **Date** : 2026-08-08
- **Statut** : Accepté
- **Pilier** : Prix

## Contexte

`ml/AGENTS.md` pose une exigence produit indépendante du dataset : **une fourchette avec
incertitude, jamais un prix ponctuel**. Un vendeur à qui l'on annonce « votre voiture vaut
6 444 € » reçoit une promesse que le modèle ne peut pas tenir. L'AI Act attend par ailleurs
que l'incertitude soit affichée et que le système reste une aide à la décision.

La première mise en ligne servait `prédiction ± MAE de la tranche` : le prix prédit, élargi de
l'erreur moyenne mesurée sur sa tranche de prix. Raccourci assumé pour livrer, avec trois
défauts qui se ramènent au même — **il ne promettait rien de vérifiable** :

1. la largeur était choisie d'après le prix *prédit* ; un modèle qui se trompe de tranche se
   trompe aussi de largeur ;
2. une erreur *moyenne* n'est pas un intervalle : une MAE de 1 520 € ne dit pas quelle
   proportion des véhicules tombe dans `± 1 520 €` ;
3. la largeur ne dépendait que du prix — une Clio vue 400 fois et un modèle vu 2 fois
   recevaient la même incertitude.

## Décision

**Régression quantile + conformalisation (CQR), α = 0,2**, soit une couverture cible de 80 %.

### Le mécanisme, en deux temps

**Temps 1 — trois modèles quantiles.** `HistGradientBoostingRegressor(loss="quantile")` aux
quantiles 0,1 / 0,5 / 0,9. Seule la fonction de perte change (perte pinball, asymétrique), pas
le type de modèle ni les hyperparamètres. La largeur devient **adaptative** : elle s'élargit
d'elle-même là où le modèle est mal à l'aise.

**Temps 2 — conformalisation.** Les quantiles appris sont optimistes. Un **jeu de calibration
jamais vu à l'ajustement** mesure de combien les bornes ratent :

```
score de non-conformité   E_i = max(borne_basse(x_i) − y_i , y_i − borne_haute(x_i))
constante                 Q   = quantile ⌈(n+1)(1−α)⌉/n des E_i
fourchette servie         [borne_basse − Q , borne_haute + Q]
```

Un `E_i` négatif signifie que le point était *dans* l'intervalle avec de la marge : la
correction joue dans les deux sens. Ici **Q = +145 €** — les bornes brutes étaient trop
étroites. Le niveau `⌈(n+1)(1−α)⌉/n`, et non `1−α`, est la correction de population finie qui
rend la garantie valable et non seulement asymptotique.

### Le découpage, et son coût

| Jeu | Taille | Rôle |
|---|---|---|
| Ajustement | 11 928 | apprendre les trois quantiles |
| **Calibration** | **3 977** | mesurer `Q` — jamais vu à l'ajustement |
| Test | 3 977 | évaluer ; **identique** aux carnets 03 et 04 |

Le jeu de test n'a pas bougé, la calibration a été prélevée dans l'ancien jeu
d'entraînement : les MAE restent comparables.

**Coût mesuré : MAE 1 448 € contre 1 425 €, soit +23 €** pour 25 % de données
d'apprentissage en moins. Échange volontaire : un peu de précision ponctuelle contre une
promesse démontrable.

### Ce qui est mesuré

| Indicateur | Valeur |
|---|---|
| Couverture cible | 80 % |
| **Couverture mesurée sur le test** | **79,5 %** |
| Correction conforme `Q` | +145 € |
| Largeur médiane | 3 399 € |
| MAE du modèle central | 1 448 € · R² 0,875 · 0 prix négatif |

**Adaptativité confirmée** : largeur médiane de 1 914 € en entrée de gamme contre 13 744 € au-
dessus de 20 000 €. Une largeur unique ne peut pas produire cet écart.

## Les deux limites, déclarées

### 1. La couverture mesurée est 79,5 %, pas 80 %

Sous la cible. L'écart-type d'échantillonnage sur 3 977 points vaut 0,63 % : 79,5 % est à
0,8 σ de la cible, cohérent avec une garantie qui porte sur l'espérance et non sur un tirage
particulier.

**C'est le chiffre mesuré qui est affiché**, pas la cible. Annoncer 80 % serait promettre un
résultat qu'on n'a pas constaté. Un ajustement de α pour atteindre exactement 80 % sur ce
test serait une sélection sur le jeu de test — refusé.

### 2. La garantie est marginale, pas conditionnelle

| Tranche de prix | n | Couverture | Largeur médiane |
|---|---|---|---|
| 500 – 2 000 € | 1 061 | **83,8 %** | 1 914 € |
| 2 000 – 5 000 € | 1 101 | 81,9 % | 2 874 € |
| 5 000 – 10 000 € | 831 | 78,5 % | 4 214 € |
| 10 000 – 20 000 € | 657 | 75,2 % | 6 072 € |
| 20 000 – 50 000 € | 250 | **72,0 %** | 13 744 € |

CQR garantit la couverture **en moyenne sur la population**, pas sur chaque segment. Le haut
de gamme est sous-couvert de 8 points : moins d'exemples, plus de variabilité. La promesse est
donc meilleure que 80 % pour l'entrée de gamme et moins bonne au-delà de 20 000 €.

Cette limite est **affichée au vendeur** et le tableau complet est exposé sur
`GET /prix/contrat`, pour que la promesse soit contrôlable de l'extérieur.

## Un défaut trouvé et corrigé : le croisement de quantiles

Les trois modèles sont ajustés **indépendamment** : rien ne les contraint à rester ordonnés.
Mesuré sur les 19 882 annonces — les bornes ne s'inversent jamais, mais l'estimation centrale
sort de la fourchette dans **1,20 % des cas** (dépassement médian 275 €, maximum 2 647 €).
Détecté par le test de parité service/modèle, sur une ligne où le central valait 24 497 € pour
une borne haute à 24 291 €.

**Correction : on rabat le central dans l'intervalle, pas l'inverse.** Ce sont les bornes qui
portent la garantie de couverture ; les déformer l'invaliderait. Le central n'est qu'une
estimation ponctuelle. La même règle est appliquée à l'entraînement, faute de quoi la MAE
rapportée décrirait un modèle que le service ne sert pas.

## Alternatives écartées

- **Régression quantile seule, sans conformalisation** : plus simple, mais la couverture
  réelle dérive (un modèle visant 90 % en couvre souvent 84 %). On afficherait une promesse
  invérifiable — exactement le défaut qu'on corrige.
- **Fourchette relative depuis l'erreur** (le mécanisme précédent) : la largeur ne s'adapte
  qu'au prix, pas au véhicule.
- **Contraindre les quantiles à ne pas se croiser** à l'entraînement : plus propre en théorie.
  À 1,2 % de croisements, le rabattement suffit ; à reconsidérer si ce taux montait — d'où sa
  publication dans les métriques de l'artefact.
- **Couverture à 90 %** : fourchettes nettement plus larges, donc moins actionnables pour un
  vendeur. 80 % retenu comme compromis entre fiabilité et utilité.

## Conséquences

- L'artefact `app/models/prix.joblib` porte désormais **trois modèles**, et `prix.json` la
  constante `Q` — la seule valeur dont le service ait besoin pour reconstruire les bornes.
- La fourchette est **plus large** que celle du mécanisme précédent (5 478 € contre 3 040 €
  pour une Clio de référence). Elle n'est pas moins bonne : l'ancienne était trop étroite pour
  ce qu'elle prétendait couvrir.
- Rappel de la fiche 0003 : la couverture porte sur le **prix demandé** d'annonces
  comparables, pas sur un prix de vente conclu.
- Une couverture conditionnelle (garantie par segment, et non en moyenne) est le prolongement
  naturel. Non instruite.
