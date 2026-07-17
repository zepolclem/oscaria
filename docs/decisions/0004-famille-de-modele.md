# ADR 0004 — Famille de modèle et arbitrage interprétabilité / performance

- **Statut** : Accepté
- **Date** : 2026-07-17 (chiffres mis à jour le 2026-07-17 après correctif de parsing)

## Contexte

On a construit la modélisation par étapes, du plus simple au plus performant, en mesurant
chaque gain (carnets `02` et `03`). Rappel des termes : **coefficient de détermination** (R²) =
part des variations de prix expliquée ; **erreur absolue moyenne** = écart moyen en euros ;
**validation croisée** (*cross-validation*) = moyenne du score sur 5 découpes différentes, pour
un chiffre plus fiable qu'un seul découpage.

| Étape | Modèle | Coefficient de détermination | Erreur absolue moyenne |
|-------|--------|------------------------------|------------------------|
| Régression linéaire (17 variables dont carburant en one-hot) | LinearRegression | 0,76 | 8 303 € |
| + régularisation | ElasticNet (linéaire régularisé) | 0,76 | 7 800 € |
| + cible en logarithme | ElasticNet (log) | 0,82 (validation croisée) | 6 174 € |
| + arbres | HistGradientBoosting (renforcement d'arbres, log) | **0,89** (validation croisée) | 4 841 € |
| + fréquence du modèle exact | HistGradientBoosting (log) | **0,89** (validation croisée par pli) | 4 815 € |

Le passage aux arbres (*gradient boosting*, renforcement de gradient) capture la dépréciation
non linéaire d'une voiture, que le modèle linéaire ne peut pas modéliser.

## Décision

- **Modèle de production : HistGradientBoosting** (renforcement d'arbres), le plus performant,
  avec cible en logarithme et gestion native des valeurs manquantes.
- **Modèle de référence conservé : ElasticNet** (régression linéaire régularisée), gardé comme
  modèle **explicable** — ses coefficients se lisent directement en euros par variable.
- **Ingénierie des variables** centralisée dans `ml/src/features.py` (`clean_cars`,
  `add_brand_features`) : marque extraite de façon robuste aux noms multi-mots (Alfa Romeo,
  Land Rover…), classée en 3 paliers premium (table `ml/references/premium_brand.csv`), âge
  dérivé (2023 − année), variables motorisation (carburant, vignette crit'air, consommation),
  fréquence du modèle exact (`modele_freq`, calculée sur le train uniquement).

**Incident de parsing documenté** : le parseur générique initial concaténait tous les chiffres
d'une chaîne, corrompant la consommation (« 4 l/100km » → 4100) et rejetant 49 prix au format
« T.T.C. (H.T.) ». Corrigé (extraction du premier nombre) le 2026-07-17 ; toutes les métriques
ont été recalculées. Illustre l'importance des contrôles de vraisemblance après chaque parsing.

## Alternatives écartées

- **Rester au tout-linéaire** : plafonne à un coefficient de détermination de ~0,82, sous-estime
  systématiquement le haut de gamme (relation non linéaire).
- **Encodage de la marque par prix moyen** (*target encoding*) : plus performant seul mais
  risque de fuite de données (*data leakage*) et moins explicable → écarté au profit du palier
  premium fait main.
- **Viser 0,90+ à tout prix** (réglage fin des arbres + variables sensibles) : gain marginal au
  prix de l'interprétabilité, contraire à l'esprit BC04.

## Conséquences

- Deux modèles maintenus : un **performant** (arbres) et un **explicable** (linéaire) — c'est
  l'arbitrage **interprétabilité / performance** assumé et documenté.
- La performance vient en partie des voitures chères (forte variance) ; combinée à l'ADR 0002
  (périmètre ≤ 50 000 €), on privilégie un modèle honnête sur le marché courant plutôt qu'un
  score flatteur.
- Prochaine brique produit indépendante du choix de modèle : la fourchette (ADR 0003).
