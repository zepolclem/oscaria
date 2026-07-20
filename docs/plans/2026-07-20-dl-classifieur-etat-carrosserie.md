# Plan — Pilier DL : classifieur d'état carrosserie (intact / abîmé)

## Contexte

OscarIA a aujourd'hui un pilier **Prix** complet (régression tabulaire scikit-learn,
fourchette + incertitude via CQR, 5 ADR). Le second pilier **DL — classifieur d'état**
est écrit dans `AGENTS.md` comme "plus tard" : reconnaissance binaire **intact / abîmé**
sur photos. Rien n'est construit — pas de dossier `dl/`, aucune image, aucune lib DL.

Ce plan démarre ce pilier. Décisions cadrées avec l'utilisateur en brainstorm :

- **Objectif** : classifieur binaire **intact / abîmé** (carrosserie). Pas de reconnaissance
  marque/modèle (hors produit : le vendeur connaît sa voiture). Pas de note multi-niveaux
  (pas de data labellisée propre).
- **Méthode** : **transfer learning PyTorch** (CNN pré-entraîné, tête remplacée). Jamais
  from scratch — volume de data insuffisant.
- **Données** : aucune en main → **multi-candidats publics** puis choix, en miroir exact du
  process tabulaire (EDA d'inspection → Gate → ADR de choix). Rien n'est nommé/figé avant décision.
- **Périmètre itération 1** : **classifieur seul, bien fait**. Le lien avec le prix (décote
  selon état prédit) est **documenté (schéma), pas codé** — c'est l'itération 2.
- **GPU** : RTX 5070 Ti = archi **Blackwell (sm_120)**. **PyTorch déjà installé par l'utilisateur
  (comme dans le TP)** → reste à **vérifier que ce build voit bien la carte** (si le torch du TP
  est trop ancien pour sm_120, le smoke test le révèle et on re-installe en CUDA 12.8).
- **Méthode alignée sur le TP** `transfer-learning-alyra` (support de cours, lecture seule) :
  réutiliser sa trame transfer learning plutôt que réinventer.
- **Cadre BC04** : noté sur rigueur / transparence limites data / gestion biais, PAS sur la
  perf brute. Les limites data et le biais de source sont des **livrables** (ADR), pas des échecs.

## Conventions à respecter (repo existant)

- Structure `dl/` en miroir de `ml/` (voir `AGENTS.md` §Structure cible) :
  `dl/AGENTS.md`, `dl/pyproject.toml`, `dl/data/<slug>/raw/`, `dl/notebooks/<slug>/`,
  `dl/src/`, `dl/models/` (gitignored comme `ml/models/`).
- **Workspace uv** : ajouter `"dl"` aux `members` de `pyproject.toml` racine.
- **Gate EDA** (`ml/AGENTS.md`) : EDA d'inspection → **STOP validation utilisateur** → verdict.
  Aucun nettoyage/modélisation avant dataset retenu.
- **ADR light** dans `docs/decisions/` (format : contexte / décision / alternatives écartées).
  Prochain numéro = **0006**. Miroir des ADR 0001-0005 existants.
- **Jamais installer sans accord explicite** ; **jamais modifier un `raw/`**.
- Notebooks numérotés en miroir de `ml/notebooks/` (`01_eda_inspection.ipynb`, etc.).
- Code stabilisé extrait des notebooks vers `dl/src/` (comme `ml/src/features.py`).

## Étapes

### Stage 0 — Scaffolding `dl/` + vérif GPU (install déjà faite)
- Créer l'arbo `dl/` (dossiers + `.gitkeep` où utile), `dl/AGENTS.md` (process DL en miroir
  de `ml/AGENTS.md` : itération datasets, Gate EDA, YAGNI deps).
- `dl/pyproject.toml` (member uv) : figer les deps déjà installées — `torch`, `torchvision`,
  `numpy`, `matplotlib`, `scikit-learn` (métriques), `pillow`, `jupyterlab`, `ipykernel`.
  Aligner les versions sur ce que l'utilisateur a posé (TP), pas réinstaller.
- Étendre `.gitignore` pour `dl/models/` et `dl/data/**/raw/` (contenu), en miroir de `ml/`.
- **Vérif (bloquante)** : notebook `00_smoke_gpu.ipynb` → `torch.cuda.is_available()`,
  `torch.cuda.get_device_name()`, petit tenseur sur GPU + une conv. Si le torch du TP ne voit
  pas la 5070 Ti (sm_120), **re-installer en CUDA 12.8** (`--index-url .../whl/cu128`, nightly
  au besoin) — **avec accord install**. Ne pas avancer tant que le GPU n'est pas exploitable.

### Stage 1 — Shortlist candidats + EDA d'inspection (Gate)
- Shortlister **2 datasets publics** "car damage" à inspecter (candidats, non figés). Pistes :
  **CarDD** (académique, ~4k images, labels propres) et un **Kaggle intact/damaged** (ex.
  ~1.5-2k images, plus bruité/déséquilibré). Slugs `dl/data/<slug>/raw/`.
- Un notebook `dl/notebooks/<slug>/01_eda_inspection.ipynb` par candidat : nb d'images,
  **équilibre des classes**, résolutions/formats, doublons, exemples visuels, **qualité des
  labels**, **biais de source** (assurance = gros accidents vs annonce = voiture présentable ;
  le modèle risque d'apprendre le *style de photo*).
- **STOP — validation utilisateur** (Gate EDA) : résumé chiffré remonté, pas d'étape suivante
  sans accord.
- **ADR 0006** — choix du dataset d'état : garder / écarter, raisons, limites & biais assumés.

### Stage 2 — Préparation données (dataset retenu)
- Split **train/val/test propre, sans data leakage** : jamais la même voiture (ou même série de
  photos) des deux côtés du split. Documenter la règle de split.
- Transforms torchvision : resize/normalize ImageNet, **augmentation** légère (flip, crop,
  jitter) pour compenser le petit volume et réduire l'overfitting.
- Extraire dataloaders + transforms réutilisables vers `dl/src/` (l'équivalent DL de
  `ml/src/features.py`).

### Stage 3 — Transfer learning baseline
- CNN pré-entraîné (**ResNet18** ou **EfficientNet-B0**), corps **gelé**, tête remplacée
  (2 classes / 1 logit binaire).
- Boucle d'entraînement PyTorch sur GPU. Notebook `03_transfer_learning.ipynb`.
- Class weights / gestion du déséquilibre si l'EDA le montre.

### Stage 4 — Évaluation honnête + fine-tuning éventuel
- **Pas l'accuracy seule** (piège classes déséquilibrées, cf. philosophie CQR du pilier Prix) :
  **precision / recall / F1 par classe + matrice de confusion**, sur le test tenu à l'écart.
- Contrôle **overfitting** (courbes train vs val).
- Optionnel : dégeler quelques couches (fine-tuning) si le baseline le justifie.
- **ADR 0007** — modèle d'état : archi retenue, métriques réelles, limites, biais résiduel.

### Stage 5 — Intégration documentée + note légale/éthique
- **Schéma** du branchement produit (documenté, **pas codé**) :
  `photo → CNN état → proba abîmé → décote → ajuste la fourchette de prix`. C'est l'itération 2.
- **Note RGPD/AI Act** (attendu transverse BC04) : photos = données perso possibles (plaques,
  visages, EXIF géoloc) → à traiter ; afficher l'usage IA + l'incertitude (aide à la décision).

## Vérification (bout en bout)
1. **GPU** : `00_smoke_gpu.ipynb` voit la RTX 5070 Ti, tenseur GPU OK.
2. **Gate EDA** : les deux notebooks `01_eda_inspection` tournent, sortent des chiffres
   (volumes, équilibre classes) ; STOP respecté avant modélisation.
3. **Entraînement** : au moins 1 epoch tourne sur GPU sans erreur, la loss baisse.
4. **Éval** : matrice de confusion + precision/recall/F1 sur le test ; cohérence
   (pas 100% = leak, pas 50% = modèle mort) ; courbes train/val pour l'overfitting.
5. **Livrables doc** : ADR 0006 (dataset) et 0007 (modèle) écrits façon 0001-0005 ;
   schéma d'intégration + note légale présents.

## Hors périmètre (itération 2+)
- Coder la décote qui ajuste réellement la fourchette de prix.
- Reconnaissance marque/modèle.
- Note d'état multi-niveaux, localisation/type de dégât.
