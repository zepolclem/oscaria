# Rétrospective — du carnet Jupyter à l'application déployable (pilier Prix)

> Fiche écrite **après** coup, le 2026-08-08, pour consigner un travail réalisé sur deux
> journées sans plan déposé. Les chiffres sont ceux effectivement mesurés en session, pas des
> estimations. Elle complète les plans `2026-07-29` et `2026-08-06`, et précède le plan
> `2026-08-08-fourchette-conformalisee.md`.

## Point de départ

Le pilier Prix disposait de quatre carnets aboutis (`ml/notebooks/leboncoin-private/`) et
d'un modèle validé — `HistGradientBoostingRegressor`, MAE 1 475 €, R² 0,890, zéro prédiction
négative, 36 % mieux que la régression linéaire du carnet 03.

Ce modèle n'existait nulle part ailleurs que dans la mémoire d'un noyau Jupyter. Rien n'était
persisté, et `app/src/main.py` était un squelette Gradio « echo ».

**Méthode retenue avec l'utilisateur** : livrer d'abord la chaîne la plus simple qui tourne
de bout en bout, vérifier, puis affiner par marches. Ce qui est repoussé est listé, pas
oublié.

---

## Marche 1 — sortir le modèle du carnet

### Ce qui a été fait

| étape | livrable |
|---|---|
| Rapatriement | `collecte/scraper/dataset.py annonces` → 20 915 annonces depuis Postgres |
| Entraînement | `ml/src/entrainement.py` — part du `raw/`, rappelle `clean_leboncoin` |
| Artefacts | `app/models/prix.joblib` + `prix.json` |
| Service | `app/src/prix.py` — dérivations miroir du nettoyage |
| Surface | `POST /prix`, `GET /prix/contrat`, formulaire Gradio |

**Résultat** : MAE 1 476 € contre 1 475 € au carnet 04, R² 0,889 contre 0,890.

### Le module partagé, décision structurante

`app/src/preparation.py` porte `construire_jeu()` et les listes de variables. Il est importé
par `ml/src/entrainement.py` **et** par `app/src/prix.py`. Le sens de la dépendance surprend
(`ml/` → `app/`), mais il suit l'artefact : c'est l'image de l'application qui doit embarquer
ce code, `ml/` n'y est jamais copié.

Motif : le risque principal d'un service de *machine learning* n'est pas la performance du
modèle, c'est le **décalage entraînement/service** — quand le code qui prépare les données à
l'entraînement diverge de celui qui les prépare en production. Il ne lève aucune exception ;
il rend des prix faux qui ont l'air normaux.

### L'écart d'un euro, expliqué

Le carnet 04 fait `.astype("category")`, qui ordonne les modalités par apparition ;
`construire_jeu` passe par `astype("string")`, qui les trie alphabétiquement.
`HistGradientBoosting` réordonne ensuite par cible moyenne, mais départage les égalités par
l'index d'origine — d'où quelques coupes différentes. Le détour par `string` n'est pas
gratuit : c'est lui qui permet de figer le vocabulaire des catégories, donc de garantir
qu'une `RENAULT` saisie au formulaire porte le même code qu'à l'entraînement.

### La vérification qui compte

Rejouer des annonces réelles à travers deux chemins — le chemin d'entraînement et le chemin
du service, ce dernier ne recevant que ce qu'un vendeur saisirait — et exiger la même
prédiction.

**Écart maximal mesuré : 0,4 €**, l'arrondi à l'euro du service.

C'est le seul test qui attrape ce genre de bug. Un oubli de regroupement du GPL, par exemple,
donnerait un formulaire qui répond, un prix plausible, et une erreur uniquement sur les
véhicules GPL. Aucun test unitaire classique ne le voit : il n'y a rien à faire planter.

---

## Fusion des piliers Prix et Plaques

### Le conflit

Deux branches réécrivaient **le même `app/src/main.py`**, le même `deploy/Dockerfile` et le
même `.dockerignore` — l'une pour le formulaire de prix, l'autre pour le floutage de plaques.

La branche `zepolclem/ML` était exactement le point de départ de `zepolclem/no-plan-question`
et les changements Prix n'existaient qu'en *working tree* : sauvegarde des fichiers, retour à
l'état de base, `git merge --ff-only` (aucun commit créé), réapplication à la main.

### La correction de fond

Un fichier par pilier, `main.py` réduit à un assembleur :

```
app/src/main.py          assemble les onglets et monte les routeurs — aucune logique métier
app/src/page_prix.py     contrat d'entrée, route /prix, formulaire
app/src/page_plaques.py  route /plaques, démo de floutage
```

Ajouter un pilier = déposer `page_<nom>.py` et une ligne dans `PAGES`. Personne n'a plus
besoin de modifier `main.py` pour livrer sa page.

### Démarrage dégradé

Chaque page est importée dans un `try/except`. Un pilier absent ou dont le modèle ne charge
pas n'entraîne pas les autres : son onglet affiche l'indisponibilité, `/sante` reste vert et
détaille l'état pilier par pilier.

```json
{"statut": "ok", "piliers": {"page_prix": "charge", "page_plaques": "absent"}}
```

Un healthcheck qui rougit parce qu'un pilier sur deux manque fait redémarrer le conteneur en
boucle chez Coolify — et on perd aussi celui qui fonctionnait. D'où la séparation entre
« l'application répond » et « tous les piliers sont chargés ». Les piliers qui savent
distinguer *module chargé* de *modèle chargé* l'exposent par une fonction `etat()`.

---

## Allègement du formulaire

Demande produit : retirer la région, ne garder que l'année de mise en circulation, réduire
les états, ajouter le modèle du véhicule. Chaque changement a été **chiffré en validation
croisée 5 plis avant d'être appliqué**, pas décidé au jugé.

| changement | effet sur la MAE |
|---|---|
| retirer `region` | **−14 €** |
| garder l'année seule (perdre le mois) | **−4 €** |
| ajouter `modele` (encodage par fréquence) | **−35 €** |
| regrouper les états de 8 à 5 crans | **0 €** |

**Résultat : MAE 1 425 €, R² 0,898, zéro prédiction négative — avec un champ de moins.**

### Retirer de l'information peut améliorer un modèle

`region` avait une importance par permutation de 0,0014 au carnet 04. Une variable sans
information n'est pas neutre : elle fournit du bruit dans lequel l'arbre peut creuser des
coupes qui ne généralisent pas. La supprimer réduit l'espace de recherche.

### Le regroupement des états, calé sur les prix

Prix médians observés par cran d'origine :

| état déclaré | n | prix médian |
|---|---|---|
| `not_drivable` | 1 006 | 1 300 € |
| `damaged` | 681 | 1 500 € |
| `major_repairs_needed` | 3 181 | 1 500 € |
| `minor_repairs_needed` | 3 248 | 2 500 € |
| `normal_wear_and_tear` | 3 305 | 4 500 € |
| `good_overall_condition` | 3 301 | 7 000 € |
| `undamaged` | 3 248 | 9 000 € |
| `excellent_condition` | 1 912 | 16 000 € |

Les trois pires crans sont **indiscernables en prix** : les fusionner ne perd rien.
`undamaged` et `excellent_condition` sont séparés par 7 000 € : les fusionner en perdrait.
D'où 5 crans, mesurés à 1 467 € contre 1 471 € à 8 crans — le regroupement est très
légèrement gagnant. Un regroupement à 3 crans coûtait +28 €.

### Ce que l'encodage par fréquence apprend vraiment

804 modèles de véhicules dépassent la limite de 255 modalités du catégoriel natif. Chaque
modèle est donc remplacé par son **nombre d'occurrences dans le jeu d'ajustement** — une
mesure de popularité, qui ne regarde jamais le prix, donc sans fuite de la cible.

Conséquence à connaître, mesurée :

```
Clio   2015, 120 000 km, diesel  →  6 444 €
Twingo 2015, 120 000 km, diesel  →  6 409 €
sans modèle renseigné            →  7 955 €
```

Le modèle ne sait pas *quel* véhicule c'est, seulement *à quel point ce modèle est courant*.
Le gain de 35 € est réel mais indirect : « les modèles rares se vendent différemment des
modèles courants ». L'alternative qui identifierait vraiment le modèle — l'encodage par prix
moyen — a été mesurée meilleure au carnet 04 (1 465 € contre 1 485 €) mais regarde la cible,
donc fuit sur les 341 modèles vus trois fois ou moins. **Décision laissée ouverte.**

### Deux silences corrigés

Les tests de cas dégradés ont montré que l'API acceptait en HTTP 200 :

- une valeur d'état de l'ancienne échelle à 8 crans, qui retombait sur « inconnu » — le
  client recevait un prix calculé comme si l'état n'était pas renseigné ;
- les champs `region` et `mois_mec`, disparus du contrat, silencieusement ignorés ;
- un kilométrage de −5 000, qui produisait une estimation parfaitement plausible.

Corrigés par `model_config = ConfigDict(extra="forbid")`, un `Literal` sur les cinq crans
d'état, et des bornes tirées du domaine réellement observé. Ces trois cas renvoient désormais
**422**. Un refus explicite vaut mieux qu'une réponse trompeuse.

---

## Image Docker sans pilotes CUDA

### Le constat

```
image : 5,53 Go
  2,9 G  nvidia/     pilotes CUDA
  910 M  torch/
  651 M  triton/     compilateur GPU
```

Le conteneur tourne sur processeur (`page_plaques.py` fixe `torch.device("cpu")`) et le
serveur cible a 2 vCPU et 4 Go de RAM. Ces 3,5 Go étaient poussés à chaque déploiement pour
rien.

### Trois faits qui ont cadré la solution

1. **Le Mac de développement n'a déjà aucun paquet CUDA** — 0 paquet `nvidia-*`,
   `torch.cuda.is_available()` à `False`, c'est MPS qui travaille. Les wheels `nvidia-*`
   n'existent que pour Linux x86_64.
2. **L'image Docker et le PC GPU sont tous deux sous Linux** (RTX 5070 Ti, Blackwell
   `sm_120`, CUDA 12.8). Aucun marqueur de plateforme ne peut les distinguer.
3. **`--torch-backend=cpu` est réservé à l'interface `uv pip`** ; le Dockerfile utilise
   `uv sync --frozen`, qui ne l'accepte pas (uv 0.9.17).

### Résultat négatif — la configuration documentée par uv ne marche pas ici

La forme officielle — deux extras `cpu` / `cu128` plus `[tool.uv] conflicts` — a été essayée
en cinq variantes. Toutes échouent sur :

```
Requirements contain conflicting indexes for package `torch`
```

**Le test qui a tranché** : remplacer `oscaria-app` par un nom de paquet volontairement
inexistant dans la déclaration `conflicts`. uv n'a produit **aucune erreur** — message
identique. Il ne lit donc pas ce bloc du tout. Cause probable : `package = false` sur la
racine comme sur les membres de l'espace de travail, alors que l'exemple de la documentation
porte sur un projet installable ordinaire.

C'est une conclusion qu'on n'atteint pas en relisant sa configuration : il fallait injecter
une faute volontaire pour voir si l'outil la remarquait.

### La solution retenue

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]
```

Le marqueur `sys_platform == 'linux'` protège macOS : le Mac reste sur PyPI, et le backend
MPS avec lui. Router tout le monde vers l'index CPU aurait été plus court à écrire et aurait
fait basculer l'entraînement local sur processeur **sans aucun message** — pas d'erreur,
juste dix fois plus lent. La vérification `torch.backends.mps.is_available()` a donc été
placée en tout premier, avant même la mesure de la taille de l'image.

**Résultat : 5,53 Go → 1,5 Go**, zéro paquet `nvidia`, zéro `triton`, MPS local intact
(vérifié par un calcul sur `device='mps'`), prix servi identique au centime.

### Le chemin GPU, en deux temps

Documenté dans `dl/src/device.py:recommend_install()` :

```
1. uv sync --all-packages                                  # variante CPU
2. uv pip install --torch-backend=cu128 torch torchvision  # bascule CUDA
```

**Limite assumée** : l'étape 2 installe une version que `uv.lock` ne décrit pas ; un
`uv sync` ultérieur la réécrasera par la variante CPU. La sortie propre serait de rendre
`app` et `dl` installables pour débloquer `conflicts` — refonte de l'espace de travail, pas
un réglage.

---

## État à la clôture de ces marches

| | valeur |
|---|---|
| Annonces | 20 915 brutes → 19 882 après nettoyage |
| Modèle | MAE 1 425 € · R² 0,898 · 0 prédiction négative |
| Parité service/modèle | vérifiée, écart maximal 0,4 € |
| API | `/sante`, `/prix`, `/prix/contrat`, `/plaques` |
| Image Docker | 1,5 Go, deux modèles embarqués |

## Ce qui reste ouvert

1. **La fourchette n'est pas encore garantie** — elle vaut `prédiction ± MAE de la tranche`,
   un raccourci assumé. Objet du plan `2026-08-08-fourchette-conformalisee.md`.
2. **Encodage du modèle exact par prix moyen** plutôt que par fréquence : mesuré meilleur,
   mais fuit sur les modèles rares. Décision non prise.
3. **Fiches de décision à reconstruire** — `ml/src/leboncoin.py` cite encore « ADR 0002 » et
   le carnet 04 « ADR 0003 », références orphelines depuis la remise à zéro du 2026-08-06.
