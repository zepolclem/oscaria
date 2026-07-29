# Remise à zéro — nettoyage, bascule sur PC, entraînement ResNet sur CarDD

## Contexte

Le projet est parti dans trop de directions : dix fiches de décision dont la moitié décrivent un
dataset abandonné, deux modèles entraînés dans des branches divergentes, une campagne d'annotation
de type de vue devenue inutile. Sur le conseil du prof d'Alyra, on repart de zéro en gardant la
**structure du dépôt** et les **données brutes**.

Ce qui est acquis et ne bouge pas :

- **`ml/`, pilier Prix** — dataset d'annonces leboncoin et carnets de régression, intacts.
- **Les 22 204 photos leboncoin** — plus une source d'entraînement, mais le futur **jeu de test réel**.
- **L'outil de détourage** (`dl/src/detect.py`) — validé, il reste la brique de zoom.

**Objectif unique : entraîner correctement un ResNet sur CarDD**, avec des métriques honnêtes et un
tableau qui dit quel choix apporte quoi. Le test sur photos leboncoin et l'app viennent après.

Le travail se poursuit sur le **PC équipé d'un RTX 5070 Ti**, d'où le nettoyage puis le commit sur
branche avant bascule.

---

## Phase 0 — Nettoyage, branche, bascule

### 0a. Supprimer

| quoi | chemin | note |
|---|---|---|
| les 10 fiches de décision | `docs/decisions/0*.md` | 0001-0007 restent récupérables par `git` ; 0008-0010 sont perdues |
| le plan périmé | `docs/plans/2026-07-20-dl-classifieur-etat-carrosserie.md` | |
| les modèles entraînés | `dl/models/*.pt` | 2 fichiers, ~45 Mo chacun |
| les carnets DL | `dl/notebooks/cardd/*.ipynb`, `dl/notebooks/leboncoin-private/*.ipynb` | 6 carnets |
| la branche « type de vue » | `dl/src/vues.py`, `dl/src/train_vues.py` | |
| les artefacts d'annotation | `dl/data/leboncoin-private/processed/*.parquet` | régénérables |
| le dossier Label Studio | `dl/data/labelstudio/` + `docker rm -f oscaria-labelstudio` | |
| le collecteur largus | `collecte/largus/` | suppression déjà en cours dans l'arbre de travail, à finaliser |

**Conservés** : `dl/src/cardd.py` (chargeur multi-étiquette), `device.py` (sélection MPS/CUDA),
`detect.py` (détourage), `localize.py` (Grad-CAM et fenêtre glissante, utiles plus tard),
`infer.py` (à réécrire en phase 3), `docs/decisions/README.md` (la convention, vidée de ses fiches),
tout `ml/`, tout `collecte/scraper` et `collecte/infra`, et **toutes les données brutes**.

`dl/data/car_damages` reste sur le disque mais **n'est pas utilisé** : inspection du 2026-07-29 —
aucun véhicule intact dans le jeu, étiquette binaire non documentée.

> Les carnets des candidats ML écartés (`french-second-hand-cars`, `la-centrale-fr`) sont **gardés** :
> l'itération multi-datasets est un attendu explicite de `ml/AGENTS.md`. À dire si tu veux les retirer.

### 0b. Déposer le plan

Copier ce plan dans **`docs/plans/2026-07-29-cardd-remise-a-zero.md`** — suivi par git, donc il
traverse jusqu'au PC. (`.local/` est ignoré par git, un plan déposé là resterait sur le Mac.)

### 0c. Branche et commit

```
git switch -c reset/cardd-baseline
git add -A && git commit          # nettoyage + plan, en un commit
git push -u origin reset/cardd-baseline
```

### 0d. Machine — MacBook (MPS), bascule PC en réserve

**On reste sur le MacBook**, en MPS. Le carnet `dl/notebooks/00_smoke_device.ipynb` est conservé
pour vérifier que le device attendu calcule vraiment.

La bascule sur le PC équipé d'un **RTX 5070 Ti** reste ouverte si la phase 2 devient trop lente.
`dl/src/device.py` traite déjà le cas via `recommend_install()` : c'est une puce **Blackwell
(sm_120)**, non couverte par les wheels PyTorch par défaut — il faudrait l'index **CUDA 12.8**
(`uv add torch torchvision --index-url https://download.pytorch.org/whl/cu128`), puis vérifier que
`get_device()` renvoie `cuda`.

---

## Phase 1 — Baseline CarDD

**Code** : `dl/src/train_cardd.py` (neuf) — **Carnet** : `dl/notebooks/cardd/01_baseline.ipynb`

CarDD fournit ses propres découpages — **2 816 / 810 / 374** images en `train2017` / `val2017` /
`test2017`, annotations COCO. On les utilise tels quels : aucun risque de fuite, et les résultats
restent comparables à la littérature.

Six classes, tâche **multi-étiquette** (une image peut en porter plusieurs) : `dent`, `scratch`,
`crack`, `glass shatter`, `lamp broken`, `tire flat`. Le chargeur existe :
`CarDDMultiLabel` dans `dl/src/cardd.py`, avec sa méthode `pos_weight()`.

Configuration de référence, volontairement simple :

- `ResNet18` pré-entraîné ImageNet, affinage complet, tête à 6 sorties ;
- entrée **224 px**, normalisation ImageNet ;
- perte `BCEWithLogitsLoss` pondérée par `pos_weight` — les classes sont déséquilibrées
  (`scratch` est environ 7 fois plus fréquent que `tire flat`) ;
- **sélection de l'époque sur `val2017`**, jamais sur le test ;
- seuil de décision à 0,5.

**Métriques** (règle `dl/AGENTS.md`) : precision / rappel / F1 **par classe**, plus macro-F1 et
micro-F1. Jamais l'exactitude seule. Plus la matrice de co-occurrence prédiction ↔ vérité, qui
remplace la matrice de confusion en multi-étiquette.

> **Gate.** S'arrêter, remonter le tableau par classe. C'est la référence contre laquelle tout se mesure.

---

## Phase 2 — Les leviers, un par un

**Carnet** : `dl/notebooks/cardd/02_leviers.ipynb`

Quatre expériences, **chacune isolée**, toutes évaluées sur `val2017` — le test reste intact.

| levier | hypothèse | coût relatif sur MPS |
|---|---|---|
| **A. résolution 224 → 384** | `crack` et `scratch` sont des dégâts fins ; sur une image native en 1000 × 667, une rayure fait quelques pixels et disparaît à 224. Levier le plus prometteur. | ~3× |
| **B. seuils par classe** | fixer 0,5 partout est arbitraire en multi-étiquette. Optimiser le seuil de chaque classe **sur la validation** gagne souvent plusieurs points de F1 sans toucher au modèle. | quasi nul — se calcule sur des prédictions déjà faites |
| **C. `ResNet18` → `ResNet50`** | plus de capacité, mais 2 816 images d'entraînement seulement : risque réel de sur-apprentissage. | ~2,5× |
| **D. augmentation plus riche** | recadrage aléatoire, retournement horizontal, variation colorimétrique. Le retournement est **sûr ici** (un dégât reste un dégât en miroir) ; il ne le serait pas pour une tâche latéralisée. | nul |

Le coût de référence est mesuré à la phase 1 (durée d'une époque de la baseline). Chronométrer la
première époque **avant** de lancer une série : si le levier A dépasse ~5 minutes par époque, la
bascule sur le PC devient rentable et se décide à ce moment-là, sur du mesuré.

Livrable : un **tableau comparatif** baseline / A / B / C / D, en macro-F1 et en F1 par classe.
C'est la pièce qui montre qu'on a mesuré au lieu d'empiler.

> **Gate.** Choisir la combinaison des leviers gagnants avant la phase 3.

---

## Phase 3 — Modèle final et évaluation

**Carnet** : `dl/notebooks/cardd/03_modele_final.ipynb`

Ré-entraîner avec les leviers retenus, sélectionner l'époque sur `val2017`, puis **évaluer une seule
fois sur `test2017`**. Mesurer le test plusieurs fois en ajustant entre deux le transformerait en
second jeu de validation, et le chiffre annoncé serait optimiste.

**Sortie** : `dl/models/cardd_resnet.pt` — `state_dict`, liste des classes, seuils par classe et
configuration, pour que l'inférence n'ait rien à deviner.

Réécrire ensuite `dl/src/infer.py` pour ce format, avec `predire(image) -> dict[classe, proba]`
utilisable telle quelle par la suite.

---

## Phases suivantes — hors périmètre immédiat

**Test sur photos leboncoin réelles.** Tu choisis des annonces, on applique la chaîne : détourage
(`detect.py`) → zoom → découpage en grille → modèle. C'est là que se mesure l'écart entre la
performance CarDD et le réel.

> Note utile : un essai antérieur avait trouvé le découpage en tuiles inférieur à Grad-CAM — mais il
> portait **sur des images CarDD, qui sont déjà des gros plans**. Y découper des tuiles n'apporte
> rien. Sur une photo de voiture entière, la grille attaque directement l'écart d'échelle.
> **Ce résultat ne se transpose pas** ; la grille reste à tester.

**L'application.** URL leboncoin → photos → tri des vues extérieures → détourage → modèle →
localisation. La récupération par URL réutilisera la technique `curl_cffi` déjà éprouvée dans
`collecte/scraper` (mesuré : `urllib` = 403, `curl_cffi` en empreinte Chrome = 200).

---

## Vérification

| phase | comment on sait que c'est bon |
|---|---|
| 0 | `docs/decisions/` ne contient que le README ; `dl/models/` est vide ; `docker ps -a` ne liste plus le conteneur ; la branche est poussée et visible sur `origin` |
| 0d | `get_device()` renvoie `cuda` sur le PC, et un produit matriciel sur GPU s'exécute sans erreur |
| 1 | tableau F1 par classe sur `val2017` ; effectifs cohérents avec les annotations COCO ; aucune évaluation sur le test |
| 2 | tableau comparatif à 5 lignes ; chaque levier testé **seul** ; toutes les mesures sur la même validation |
| 3 | métriques test rapportées **une seule fois** ; le checkpoint se recharge et prédit sur une image CarDD tirée au hasard |

Contraintes tenues : aucun fichier de `raw/` modifié, aucune installation sans accord.
Le commit de la phase 0c est explicitement demandé ; aucun autre commit sans nouvelle demande.
