# Précision du modèle Prix : central sur fit+cal, puis hyperparamètres

- **Date** : 2026-08-10
- **Pilier** : Prix
- **Statut** : livré
- **Amont** : `2026-08-10-contrat-entree-v2.md` (ADR ML 0007 — MAE 1 482 €, couverture
  79,98 %, largeur 3 605 €)

## Pourquoi

Récupérer de la précision sans toucher à la garantie de couverture. Deux leviers, une marche
chacun — un changement = un chiffre (règle du carnet 06).

## Marche 1 — le central apprend sur ajustement + calibration

Seuls les modèles de **bornes** (quantiles 0,1 et 0,9) portent la garantie conformale : la
constante `Q` doit être mesurée sur des données qu'**eux** n'ont jamais vues. Le modèle
**central** (quantile 0,5) ne participe pas à la garantie — il est même rabattu dans la
fourchette quand il en sort. Il peut donc apprendre sur ajustement + calibration
(15 905 lignes, +33 %) sans rien invalider. La table `modele_freq` reste comptée sur le fit
seul : c'est la table servie, et les bornes l'exigent.

Changement local dans `ml/src/entrainement.py` ; bornes bit-pour-bit identiques, donc
couverture et largeur inchangées — seuls MAE/R²/croisements du central bougent.

## Marche 2 — hyperparamètres (carnet 07)

Défauts scikit-learn jamais remis en cause. Recherche `RandomizedSearchCV` (~60 tirages,
5 plis, seed 42) sur le **jeu d'ajustement seul**, modèle central (la perte pinball à 0,5
est la MAE). Espace : `learning_rate` 0,03–0,3 (log), `max_iter` 200–800, `max_leaf_nodes`
15–127, `min_samples_leaf` 10–80, `l2_regularization` 1e-3–10 (log), `max_depth`
{None, 6, 12}. `early_stopping=False` explicite : le défaut « auto » s'active au-delà de
10 000 lignes et prélèverait sa propre validation interne — on réglerait un autre modèle
que celui qu'on croit.

Hyperparamètres retenus **partagés par les trois quantiles** (comme aujourd'hui : seule la
perte change). **Seuil d'adoption : gain CV > ~15 €** (le bruit mesuré au carnet 06 vaut
± 16 €) — sinon résultat négatif consigné, défauts conservés. Après ré-entraînement,
garde-fous : couverture test ≥ ~79 %, largeur non dégradée franchement ; repli possible :
réglage pour le central seul, défauts pour les bornes.

## Résultats

### Marche 1

**Résultat négatif (consigné comme tel)** : MAE 1 485 € contre 1 482 €, R² 0,865 contre
0,871 — aucun gain, écart dans le bruit. Bornes bit-pour-bit inchangées comme attendu
(couverture 80,0 %, largeur 3 605 €, Q +144 €). Lecture : avec les réglages par défaut
(100 itérations max, early stopping « auto »), le modèle est bridé bien avant que +33 % de
données ne puissent compter. Le changement de code est conservé provisoirement ; décision
finale à la marche 2, en re-mesurant fit vs fit+cal avec les hyperparamètres retenus.

### Marche 2

**Recherche (carnet 07)** : −84 € de MAE en CV (1 590 → 1 506 €) — adopté. Motif gagnant :
apprentissage lent et long (0,037 × 750 itérations) avec arbres riches (96 feuilles).
L'early stopping « auto » ne coûtait rien ; le frein était le plafond de 100 itérations.

**Garde-fou déclenché** : réglages appliqués aux trois quantiles, la couverture par tranche
s'effondrait (64,4 % au-delà de 20 k€ contre 72,8 %, Q bondissant à +451 €). Repli prévu au
plan adopté : **bornes aux défauts** (garantie bit-pour-bit identique à v2), **central
réglé** — et la marche 1 devient payante avec ces réglages (fit + calibration : −16 €).

**Configuration finale servie** :

| | v2 | v2 réglé |
|---|---|---|
| MAE | 1 482 € | **1 366 €** |
| R² | 0,871 | 0,897 |
| Couverture / pire tranche | 79,98 % / 72,8 % | 79,98 % / 72,8 % (identiques) |
| Largeur / Q | 3 605 € / +144 € | 3 605 € / +144 € |
| Croisements | 50 (1,3 %) | 51 (1,3 %) |

Service vérifié (Clio de référence : bornes inchangées 5 877 — 8 275 €, central
8 140 → 8 272 €). Décision complète : ADR ML 0008.
