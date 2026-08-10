# ML 0008 — Précision du central : hyperparamètres réglés, bornes préservées

- **Date** : 2026-08-10
- **Statut** : Accepté
- **Pilier** : Prix
- **Complète** : ML 0006 (la fourchette conformalisée), ML 0007 (le contrat v2)

## Contexte

Après le contrat v2 : MAE du central 1 482 €, couverture 79,98 %, largeur médiane 3 605 €.
Les trois modèles quantiles tournaient depuis le carnet 04 avec les **réglages par défaut**
de scikit-learn (100 itérations, taux d'apprentissage 0,1, 31 feuilles), jamais remis en
cause. Deux leviers instruits, chacun chiffré séparément
(plan `2026-08-10-precision-central-hyperparametres.md`, carnet 07).

## Décision

**Hyperparamètres réglés pour le modèle central uniquement, entraîné sur
ajustement + calibration ; les bornes q10/q90 restent aux défauts, sur l'ajustement seul.**

Réglages retenus (`HYPERPARAMETRES_CENTRAL`, `ml/src/entrainement.py`) — recherche
aléatoire 50 tirages, validation croisée 5 plis sur le jeu d'ajustement seul, seuil
d'adoption fixé à +15 € (bruit mesuré : ± 16 €) :

| | défaut | retenu |
|---|---|---|
| `learning_rate` | 0,1 | **0,037** |
| `max_iter` | 100 | **750** |
| `max_leaf_nodes` | 31 | **96** |
| `min_samples_leaf` | 20 | **10** |
| `l2_regularization` | 0 | 0,025 |
| `early_stopping` | « auto » | **False** (le budget se règle via `max_iter`) |

Gain en validation croisée : **−84 € de MAE** (1 590 → 1 506 €). Le motif : apprentissage
lent et long avec des arbres plus riches — les défauts s'arrêtaient trop tôt, trop
grossièrement.

## Le garde-fou qui a tout décidé

Appliqués aux **trois** quantiles, ces réglages donnaient une MAE équivalente (1 368 €)
mais une **couverture par tranche effondrée** : 64,4 % au-delà de 20 k€ (contre 72,8 %),
69,6 % entre 10 et 20 k€. Des quantiles plus précis sont aussi plus « sûrs d'eux », et la
constante conformale `Q` (bondie de +144 à +451 €) est **globale** : elle ne répare qu'en
moyenne, en gonflant l'entrée de gamme (90,6 %) sans réparer le haut. La garantie marginale
tenait (81,2 %) mais la promesse s'affaiblissait là où elle était déjà la plus fragile
(ML 0006) — refusé. Les bornes gardent les défauts : garantie **bit-pour-bit identique** à
la v2.

## Le central sur ajustement + calibration — légitime, et payant seulement réglé

Seules les bornes doivent ignorer la calibration (`Q` doit être mesuré sur du jamais-vu
d'elles). Le central ne porte aucune garantie — il est même rabattu dans la fourchette —
donc il peut apprendre sur les 15 905 lignes. Mesure honnête des deux temps :

- avec les **défauts** : 1 485 € contre 1 482 € — **aucun gain** (résultat négatif
  consigné) : à 100 itérations, le modèle est bridé avant que +33 % de données ne comptent ;
- avec les **réglages longs** : 1 382 € (fit seul) contre **1 366 €** (fit + calibration) —
  le levier vaut −16 € une fois le modèle assez grand pour s'en servir.

## Résultat servi (même jeu de test que depuis le carnet 03)

| | v2 (ML 0007) | v2 réglé |
|---|---|---|
| **MAE du central** | 1 482 € | **1 366 €** (−116 €) |
| R² | 0,871 | 0,897 |
| Couverture mesurée (cible 80 %) | 79,98 % | **79,98 %** (bornes identiques) |
| Pire tranche | 72,8 % | 72,8 % |
| Largeur médiane / `Q` | 3 605 € / +144 € | 3 605 € / +144 € |
| Croisements de quantiles | 50 (1,3 %) | 51 (1,3 %) |

Meilleure MAE de tout le projet (1 425 € pour l'ancien contrat 14 variables, sans garantie
d'aucune sorte) — obtenue avec 8 champs et la garantie intacte.

## Alternatives écartées

- **Réglages partagés par les trois quantiles** : le principe historique (« seule la perte
  change ») cède devant la mesure — cf. garde-fou ci-dessus.
- **Recherche dédiée pour les bornes** (réglées sur leur propre perte pinball 0,1/0,9 avec
  contrainte de couverture par tranche) : prolongement naturel, non instruit — le gain
  irait à la largeur, pas à la MAE.
- **Early stopping « auto » conservé** : aurait fait servir un modèle différent de celui
  mesuré par la recherche.

## Conséquences

- `ml/src/entrainement.py` : `HYPERPARAMETRES_CENTRAL`, jeux d'apprentissage asymétriques
  (bornes sur fit, central sur fit + calibration) — l'asymétrie est commentée dans le code.
- Artefacts ré-entraînés ; service inchangé (les bornes servies sont les mêmes, seul le
  central bouge — vérifié sur la Clio de référence : 5 877 — 8 275 €, central 8 140 → 8 272 €).
- La limite « garantie marginale, pas conditionnelle » (ML 0006) demeure, inchangée.
