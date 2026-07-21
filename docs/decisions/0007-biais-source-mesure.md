# ADR 0007 — Validation externe : le modèle d'état ne généralise pas aux photos d'annonce réelles

- **Statut** : Accepté (limite mesurée)
- **Date** : 2026-07-21

## Contexte

La baseline multi-label (ADR 0006) atteint **macro-F1 0.69 / micro-F1 0.71** sur le **test CarDD**.
L'ADR 0006 avait *théorisé* un **biais de source** (CarDD = photos stock Flickr/Shutterstock
"propres", pas des photos d'annonce de particulier). Il fallait le **mesurer**.

Test externe (`dl/notebooks/cardd/03_test_biais_source.ipynb`) sur **3 vraies photos d'annonce**
(leboncoin / paruvendu), à vérité terrain connue : 1 voiture **intacte**, 1 avec **dent** marquée,
1 avec **rayures**.

## Constat (résultats)

- **Intacte** → le modèle prédit `dent` (0.67), `lamp broken` (0.65), `glass shatter` (0.55) :
  il **invente plusieurs dégâts** sur une voiture propre.
- **Dent réelle** → `dent` (0.82) correct, mais **faux positifs** `glass shatter` (0.75),
  `tire flat` (0.67).
- **Rayures** → **manque le `scratch`** (0.30) et prédit `dent` (0.75) à la place.

Les probabilités deviennent **molles** (0.4–0.8) alors qu'elles étaient tranchées sur CarDD
(ex. `glass shatter` = 1.0). Signature d'un **décalage de distribution** (domain shift) : le modèle
a appris le **style** CarDD (cadrage, lumière, fond stock), pas le dégât en soi. `glass shatter`,
meilleure classe sur CarDD, se **sur-déclenche** systématiquement sur le réel.

## Décision

1. **Le modèle CarDD actuel n'est pas déployable tel quel** sur des photos d'annonce réelles.
2. **Le macro-F1 0.69 n'est pas une performance produit** : il vaut sur la distribution CarDD, pas
   sur le réel. Toute restitution doit l'afficher explicitement (transparence, cœur BC04).
3. **La prochaine amélioration passe par des données réelles**, pas par plus d'epochs sur CarDD :
   fine-tuning / domain adaptation sur des photos d'annonce annotées. À arbitrer plus tard.

## Alternatives écartées

- **Présenter 0.69 comme perf produit** : malhonnête au vu du test externe.
- **Ignorer le test** : contraire à l'exigence de transparence sur les limites (bloc BC04).
- **Empiler des epochs / augmenter le modèle** : ne corrige pas un biais de *données* (domain shift).

## Conséquences

- **Méthode à généraliser** : toute métrique interne (sur le dataset d'entraînement) doit être
  confrontée à une **validation hors-distribution** avant tout claim produit.
- **Limite de portée** : N = 3 photos = **indicatif, pas statistique**. Élargir l'échantillon
  (plus de vraies photos, intactes et abîmées) pour quantifier l'écart réel.
- **Impact produit (démo URL → estimation)** : la brique DL état ne peut pas, en l'état, ajuster
  un prix de façon fiable sur de vraies annonces — à intégrer dans le cadrage de l'itération démo.
