# ML 0004 — Modèle retenu : arbres plutôt que linéaire, et ce que ça coûte en explicabilité

- **Date** : 2026-08-08
- **Statut** : Accepté
- **Pilier** : Prix

## Contexte

`ml/AGENTS.md` inscrit l'**explicabilité** parmi les attendus transverses : « fourchette et
incertitude, pas une boîte noire ». Un modèle linéaire est lisible par construction — chaque
coefficient se lit comme un effet en euros. Un modèle à arbres ne l'est pas. Le choix entre
les deux est donc un arbitrage, pas une optimisation.

Trois carnets l'ont instruit : `03_baseline_regression.ipynb` (linéaire),
`04_modeles_arbres.ipynb` (forêt aléatoire et gradient boosting), sur le **même découpage**
(`test_size=0.2`, `random_state=42`) pour que les chiffres soient comparables.

## Décision

**`HistGradientBoostingRegressor` retenu.** Le linéaire est conservé comme **référence
explicable**, pas remplacé : les deux répondent à des questions différentes.

| Modèle | MAE | R² | Prix négatifs | Lisibilité |
|---|---|---|---|---|
| Toujours la médiane (repère nul) | 5 048 € | −0,12 | — | totale, et inutile |
| Régression linéaire | 2 301 € | 0,789 | **425 / 3 977** | effets en euros… mais voir ci-dessous |
| Forêt aléatoire | 1 625 € | 0,856 | 0 | importance relative |
| **HistGradientBoosting** | **1 475 €** | **0,890** | **0** | importance relative |

Trois raisons, dans l'ordre de poids :

1. **La dépréciation n'est pas linéaire.** Une voiture perd fortement de la valeur les
   trois premières années puis s'aplatit. Un arbre capte cette courbure, une droite non.
2. **Le linéaire produit des prix négatifs** — 425 sur 3 977 prédictions, jusqu'à −9 122 €.
   Symptôme direct du point 1, et rédhibitoire pour un produit destiné au grand public.
3. **Le catégoriel natif.** `HistGradientBoosting` traite les variables catégorielles
   directement, sans les transformer en colonnes 0/1 : 15 colonnes au lieu de 896. Il gère
   aussi nativement les valeurs manquantes, ce qui permet des champs de formulaire facultatifs
   (cf. fiche 0005).

### Le fait qui a tranché : le linéaire n'était pas lisible non plus

Le carnet 03 a mis en évidence que ses coefficients étaient **inexploitables** : les modèles
les plus « négatifs » étaient tous des Opel, pendant que `marque_OPEL` figurait parmi les plus
positifs.

Cause : `marque` et `modele` sont **emboîtés** — un modèle appartient à une seule marque
(787 sur 804, soit 95,9 % des lignes). La colonne `marque_OPEL` est donc presque la somme des
colonnes `modele_*` des Opel. Le système devient quasi singulier : la régression peut poser
+20 000 d'un côté et −20 000 de l'autre sans changer une seule prédiction. Les coefficients
deviennent arbitraires.

**Les prédictions restaient bonnes, leur lecture ne l'était pas.** Retirer le bloc `marque`
rendait les coefficients cohérents pour un coût quasi nul (MAE 2 329 € contre 2 301 €) — ce
qui montre au passage que `marque` n'apportait presque rien que `modele` n'ait déjà.

L'arbitrage réel n'était donc pas « performance contre explicabilité », mais
« performance contre une explicabilité **qu'il fallait réparer d'abord**. »

## Ce qui remplace les coefficients

L'importance par **permutation** : on mélange une colonne au hasard et on mesure la chute du
R². Plus honnête que l'importance interne des arbres, qui surestime les variables à forte
cardinalité.

| Variable | Chute de R² |
|---|---|
| `age` | 0,414 |
| `puissance_din` | 0,208 |
| `kilometrage` | 0,157 |
| `etat` | 0,092 |
| `marque` | 0,087 |
| `puissance_fisc` | 0,045 |
| `energie_grp`, `places`, `portes`, `boite_auto` | < 0,02 |
| `niveau`, `region`, `critair`, `couleur`, `ct_valide_jusqu_a` | ≈ 0 |

C'est une **hiérarchie**, pas un effet en euros : elle dit ce qui compte, pas de combien. La
perte d'explicabilité est réelle et n'est pas compensée.

## Alternatives écartées

- **Cible en logarithme** : mesurée au carnet 03. Améliore nettement le bas de la distribution
  (MAE 715 € contre 1 876 € sous 2 000 €) mais explose sur le haut (12 465 € contre 6 535 €
  au-dessus de 20 000 €) — R² global −2,59. Sur les arbres, le gain disparaît (R² 0,866 contre
  0,879). Écartée.
- **Forêt aléatoire** : correcte (1 625 €) mais dominée, et sans catégoriel natif.
- **`TargetEncoder` sur `modele`** : dégradait le linéaire (2 600 € contre 2 301 €) ;
  341 modèles sur 758 sont vus trois fois ou moins au train. Cf. fiche 0005 pour l'encodage
  finalement retenu.
- **Réseau de neurones** : YAGNI sur 19 882 lignes tabulaires, où le gradient boosting est
  l'état de l'art. Aurait aggravé l'explicabilité sans contrepartie mesurée.

## Conséquences

- Le modèle **prédit bien sans expliquer précisément**. C'est une limite assumée, compensée
  au niveau du produit par la fourchette d'incertitude (fiche 0006) plutôt que par une
  explication de la prédiction.
- Le carnet 03 reste la pièce d'explicabilité du dossier : il donne les ordres de grandeur
  (un écart-type d'âge ≈ −3 325 €, de kilométrage ≈ −1 752 €, palier premium ≈ +3 162 €).
- Une explication par prédiction (valeurs de Shapley) n'a pas été instruite. Ce serait la
  suite logique si l'explicabilité devenait un attendu produit et non seulement documentaire.
