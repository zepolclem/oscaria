# ADR 0006 — Dataset état carrosserie (CarDD) et objectif d'entraînement (multi-label type de dégât)

- **Statut** : Accepté
- **Date** : 2026-07-21

## Contexte

Démarrage du **pilier DL — état carrosserie** (AGENTS.md §Périmètre). Il faut arrêter un
**dataset** et un **objectif d'entraînement**, en miroir du process tabulaire (EDA d'inspection
→ Gate → ADR).

Candidat inspecté et retenu : **CarDD** (USTC, IEEE T-ITS 2023), dataset public de dommages
carrosserie. Récupéré en version **officielle** (Google Drive, format **COCO**), inspecté :

- **4000 images**, splits officiels **train 2816 / val 810 / test 374**.
- Annotations **boîtes + masques de segmentation**, **6 classes** : `dent`, `scratch`, `crack`,
  `glass shatter`, `lamp broken`, `tire flat`.
- **0 image intacte** (toutes ont ≥1 dégât ; 0 image sans annotation dans les 3 splits).
- Par image : 1 à 2 types de dégât en général (max 5).
- Machine de travail : **MacBook Pro M1 Pro, backend MPS** (cf. plan DL). Une RTX 5070 Ti est
  disponible pour un objectif plus lourd ultérieur.

Contrainte structurante découverte à l'EDA : **sans image intacte, un classifieur binaire
intact/abîmé est infaisable** sur CarDD seul.

## Décision

1. **Dataset retenu = CarDD officiel** (COCO), avec ses **splits officiels** train/val/test
   (2816/810/374). Évaluation comparable à la littérature, sans re-split maison discutable.
2. **Objectif step 1 = classification multi-label du type de dégât** : 6 sorties (une par
   classe, sigmoïde), le modèle prédit quels types de dégât sont présents sur l'image.
3. **Méthode** : transfer learning **ResNet pré-entraîné, corps gelé + tête** remplacée, sur
   **MPS**. Défenses anti-surentraînement : gel du corps, augmentation, early stopping,
   surveillance train/val.
4. **Métriques honnêtes par classe** (precision/recall/F1, éventuellement AP par label), pas
   l'accuracy globale ; gestion du **déséquilibre** (pondération / `pos_weight`).

## Alternatives écartées

- **Binaire intact/abîmé maintenant** : CarDD = 0 intacte. Il faudrait une 2e source d'intactes
  (ex. Kaggle `anujms/car-damage-detection`, dossier `whole/`) — **parké pour plus tard**.
  Mélanger deux sources différentes (abîmées CarDD stock + intactes web) exposerait à un **biais
  de source** (le modèle apprend le dataset, pas le dégât). Reporté à un step ultérieur.
- **Détection / segmentation de zones directement** : objectif produit visé ("zones abîmées"),
  mais tête de détection (Mask/Faster R-CNN) plus lourde, MPS partiellement supporté →
  **step 2** sur la 5070 Ti. On pose d'abord une baseline de classification.
- **Mirror Hugging Face (FiftyOne)** comme source : mêmes images que le train officiel, mais
  **sans val/test** → écarté au profit de l'officiel (splits fournis).

## Conséquences

- Évaluation sur **splits officiels** → honnête et comparable au papier CarDD.
- **Limites assumées** (transparence BC04) :
  - **Déséquilibre fort** : `scratch` 1507 images vs `tire flat` 219 (~7× au niveau image,
    ~11× au niveau instance) → métriques par classe obligatoires, pondération.
  - **Résolution plafonnée ~1000px** : contrairement à l'attendu, l'officiel n'est **pas** plus
    haute résolution que le mirror (W médian 1000). Les petits dégâts (rayures fines) restent
    limités par cette résolution.
  - **Biais de source** : photos Flickr/Shutterstock = images "propres", bien cadrées — **pas**
    des photos d'annonce de particulier. Le modèle pourrait mal généraliser aux photos réelles
    type leboncoin → à **mesurer et documenter** (ne pas proclamer une perf transférable).
- **Multi-label = "points d'attention" produit** (quels types de dégât présents), et **baseline**
  avant l'objectif zones (step 2).
- **Licence** CarDD **recherche/éducation non commerciale** : compatible cadre scolaire ; données
  **non republiées** (contenu `dl/data/**` gitignored).
