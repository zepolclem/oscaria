# Plan — Pilier DL : classifieur d'état carrosserie (intact / abîmé)

## Contexte

OscarIA a aujourd'hui un pilier **Prix** complet (régression tabulaire scikit-learn,
fourchette + incertitude via CQR, 5 ADR). Le second pilier **DL — classifieur d'état**
est écrit dans `AGENTS.md` comme "plus tard" : reconnaissance binaire **intact / abîmé**
sur photos. Rien n'est construit — pas de dossier `dl/`, aucune image, aucune lib DL.

Ce plan démarre ce pilier. Décisions cadrées avec l'utilisateur en brainstorm :

- **Objectif** : détecter l'**état carrosserie** depuis photos. La **forme exacte** de la cible —
  classification binaire intact/abîmé, ou **localisation de zones** (détection/segmentation),
  ou binaire + carte d'attention Grad-CAM — n'est **pas figée** : elle est **décidée par l'ADR
  0006 après inspection des labels** des datasets (les labels disponibles commandent l'objectif).
  Le binaire reste le **filet de sécurité** garanti faisable ; la localisation de zones est
  l'ambition tirée par le produit (« points d'attention »). Pas de reconnaissance marque/modèle
  (hors produit : le vendeur connaît sa voiture). Pas de note d'état subjective multi-niveaux.
- **Méthode** : **transfer learning PyTorch** (CNN pré-entraîné, tête remplacée). Jamais
  from scratch — volume de data insuffisant.
- **Données** : aucune en main → **multi-candidats publics** puis choix, en miroir exact du
  process tabulaire (EDA d'inspection → Gate → ADR de choix). Rien n'est nommé/figé avant décision.
- **Périmètre itération 1** : **classifieur seul, bien fait**. Le lien avec le prix (décote
  selon état prédit) est **documenté (schéma), pas codé** — c'est l'itération 2.
- **Machine : MacBook Pro M1 Pro 32 Go, backend PyTorch MPS.** Une RTX 5070 Ti existe sur une
  autre machine, mais elle est **écartée pour l'itération 1** : le repo vit sur le Mac, et le
  workload est petit (ResNet18 **corps gelé**, ~2-4k images en 224×224 → entraînement de l'ordre
  de 20-40 min sur MPS contre 1-3 min sur la 5070 Ti). Le coût de synchro deux machines
  (code + images + notebooks) dépasse le temps gagné. Mono-machine = plus simple, assez rapide.
  La 5070 Ti reste en réserve **si** on passe au fine-tuning complet (dégel du corps), à un
  objectif de **détection/segmentation** (plus lourd), ou à un dataset nettement plus gros.
  Bascule **quasi gratuite** : PyTorch + drivers CUDA déjà installés sur le PC (façon TP
  transfer learning), sync du repo par git, données déposées à part. Donc le choix d'objectif
  (ADR 0006) n'est **pas contraint par la machine**.
- **Méthode alignée sur le TP** `transfer-learning-alyra` (support de cours, lecture seule) :
  réutiliser sa trame transfer learning plutôt que réinventer.
- **Cadre BC04** : noté sur rigueur / transparence limites data / gestion biais, PAS sur la
  perf brute. Les limites data et le biais de source sont des **livrables** (ADR), pas des échecs.

## Vision cible — démo « URL → estimation » (itérations futures, pas ce jet)

Objectif produit à terme : l'utilisateur colle l'**URL d'une annonce leboncoin**, une **API**
sur le VPS (Docker/Coolify) renvoie une **fourchette de prix** (ML texte), des **zones abîmées /
points d'attention** (DL photos) et un **score** de synthèse.

```
URL annonce ─► collecte ─► JSON annonce (prix, km, année, énergie, boîte, marque/modèle,
                │            description, owner.type pro/particulier) + URLs photos
                ▼
        ┌─ champs texte ──► modèle Prix (HistGradientBoosting + quantiles CQR) ─► fourchette
        │                                                                            │
        └─ photos ────────► modèle État (DL) ─► zones/points d'attention ─► décote ─┤
                                                                                     ▼
                                                          réponse API : fourchette + score + points
```

**Collecte (le vrai point dur, résolu par la recherche web) :**
- Leboncoin est protégé par **DataDome** : un fetch serveur « nu » depuis le VPS est bloqué
  (403, empreinte TLS, headless vanilla détecté). Pas d'API publique.
- **Atout dispo** : `n8n-playwright` (github.com/toema/n8n-playwright) déjà hébergé sur le même
  VPS + crédits **DataImpulse** (proxies résidentiels FR) = le combo « navigateur réel + IP
  résidentielle FR » identifié comme seule voie serveur viable. **Voie principale** : Playwright
  (n8n) + proxy FR → charge la page → extrait le JSON `__NEXT_DATA__`. Réserves : Playwright
  vanilla détecté (stealth requis), scoring comportemental → **1 page par action utilisateur**,
  robustesse fragile (casse à chaque évolution DataDome/`buildId`).
- **Replis ordonnés** : bookmarklet/extension côté client (le navigateur de l'utilisateur passe
  DataDome trivialement), puis saisie manuelle photos + formulaire.
- Photos : CDN `img.leboncoin.fr` hors DataDome → probablement téléchargeables par le VPS
  (à tester ; repli = le client envoie les blobs).

**Légal (résumé, ADR dédié à rédiger avec l'itération démo) :** un fetch **ponctuel, à la
demande, transitoire, non republié, non commercial** est hors du schéma condamné (Entreparticuliers
Cass. 2022 ; La Centrale Cass. 15/10/2025 = extraction massive + service concurrent). Exposition
résiduelle = **CGU** (contractuelle, pas délictuelle). Données perso vendeur traitées
transitoirement, jamais stockées ; photos = droit d'auteur → pas de corpus persistant.

**Bonus BC04 :** leboncoin = vendeurs **particuliers** → la démo **confronte en vrai le biais
ADR 0005** (modèle prix entraîné sur 100 % pros, ~10-20 % au-dessus des ventes particulier).

**Découpage en itérations :** 1 = entraînement modèle état (ce plan) · 2 = décote état→prix codée ·
3 = spike collecte (n8n-playwright/DataImpulse) + API · 4 = déploiement Coolify. Inférence CPU
suffisante sur le VPS pour servir les deux modèles.

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

### Stage 0 — Scaffolding `dl/` + vérif MPS (install déjà faite)
- Créer l'arbo `dl/` (dossiers + `.gitkeep` où utile), `dl/AGENTS.md` (process DL en miroir
  de `ml/AGENTS.md` : itération datasets, Gate EDA, YAGNI deps).
- `dl/pyproject.toml` (member uv) : figer les deps déjà installées — `torch`, `torchvision`,
  `numpy`, `matplotlib`, `scikit-learn` (métriques), `pillow`, `jupyterlab`, `ipykernel`.
  Aligner les versions sur ce que l'utilisateur a posé (TP), pas réinstaller.
- Étendre `.gitignore` pour `dl/models/` et `dl/data/**/raw/` (contenu), en miroir de `ml/`.
- **Vérif (bloquante)** : notebook `00_smoke_device.ipynb` → `torch.backends.mps.is_available()`,
  tenseur + une convolution sur `device="mps"`, et un mini-chrono comparatif CPU vs MPS pour
  confirmer que l'accélération est réelle. Prévoir le repli `cpu` proprement (helper
  `get_device()` extrait vers `dl/src/`). Ne pas avancer tant que le device n'est pas exploitable.
- Noter les limites MPS connues : certaines ops retombent sur CPU, précision mixte peu/pas
  supportée → ne pas s'appuyer dessus.

### Stage 1 — Candidats + EDA d'inspection (Gate)
- **2 candidats** à inspecter (slugs `dl/data/<slug>/raw/`), choisis pour **contraster les
  objectifs possibles** :
  - `cardd` — **CarDD** (USTC, ~4k images, masques **COCO** + boîtes, 6 classes ; classes
    déséquilibrées scratch/dent dominants ; photos Flickr/Shutterstock = biais « photos
    propres »). Licence recherche/éducation non commerciale (mirror HF `harpreetsahota/CarDD`
    ou formulaire officiel — provenance à documenter). Les masques ouvrent **détection/
    segmentation ET binaire dérivable** (≥1 masque ⇒ abîmée).
  - `kaggle-car-damage` — **Kaggle anujms/car-damage-detection** (~2,3k images, dossiers
    binaires damaged/whole ; **licence non spécifiée** → faiblesse rigueur à documenter ;
    web-scrapé bruité). Le candidat « binaire simple ».
  - Secours si `cardd` inaccessible : Roboflow « Car Damaged Severity Detection » (~3k,
    segmentation, CC BY 4.0).
- Un notebook `dl/notebooks/<slug>/01_eda_inspection.ipynb` par candidat (trame commune,
  utilitaires partagés → `dl/src/`) : volumes & splits, **classes & équilibre** (par image
  et **par instance** pour CarDD), **type et qualité des labels** (masques cohérents ? boîtes
  vides ?), résolutions/formats, **doublons** (hash perceptuel), **licence & provenance**,
  **biais de source** (crash vs annonce présentable → risque d'apprendre le *style de photo*),
  échantillons visuels annotés.
- **STOP — validation utilisateur** (Gate EDA) : résumé chiffré remonté, pas d'étape suivante
  sans accord.
- **ADR 0006** — **dataset retenu ET objectif d'entraînement** (binaire / détection /
  segmentation / binaire+Grad-CAM), déduit des labels ; garder/écarter, raisons, limites &
  biais assumés, conséquence machine (binaire → Mac/MPS ; détection/segmentation → 5070 Ti).

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
- Boucle d'entraînement PyTorch sur MPS. Notebook `03_transfer_learning.ipynb`.
  Soigner le `DataLoader` (`num_workers`, décodage JPEG) : corps gelé → le chargement d'images
  devient facilement le goulot d'étranglement, pas le calcul.
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
1. **Device** : `00_smoke_device.ipynb` → MPS disponible, tenseur + conv OK, MPS plus rapide que CPU.
2. **Gate EDA** : les deux notebooks `01_eda_inspection` tournent, sortent des chiffres
   (volumes, équilibre classes) ; STOP respecté avant modélisation.
3. **Entraînement** : au moins 1 epoch tourne sur MPS sans erreur, la loss baisse.
4. **Éval** : matrice de confusion + precision/recall/F1 sur le test ; cohérence
   (pas 100% = leak, pas 50% = modèle mort) ; courbes train/val pour l'overfitting.
5. **Livrables doc** : ADR 0006 (dataset) et 0007 (modèle) écrits façon 0001-0005 ;
   schéma d'intégration + note légale présents.

## Hors périmètre (itération 2+)
- Coder la décote qui ajuste réellement la fourchette de prix.
- Spike collecte (n8n-playwright/DataImpulse), API, bookmarklet, déploiement Coolify.
- ADR légal détaillé leboncoin (avec l'itération démo).
- Reconnaissance marque/modèle. Note d'état subjective multi-niveaux (neuf/bon/moyen/mauvais).

*(La localisation de zones/type de dégât n'est PLUS hors périmètre : elle devient un objectif
possible du modèle, tranché par l'ADR 0006 selon les labels disponibles.)*
