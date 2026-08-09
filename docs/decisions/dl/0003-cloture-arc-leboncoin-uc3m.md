# DL 0003 — Clôture de l'arc leboncoin/UC3M-LP : ce qui est abandonné, et à qui appartiennent les chiffres

- **Date** : 2026-08-09
- **Statut** : Accepté
- **Pilier** : plaques (fiche de clôture — complète les fiches 0001 et 0002)

## Contexte

Le pilier Plaques s'est développé sur **deux lignes de travail parallèles** : la branche
locale `reset/plaques` (jamais poussée) et `main`. Chaque ligne a écrit ses propres fiches
0001/0002 et entraîné son propre modèle. Un audit de cohérence (2026-08-09, plan
`docs/plans/2026-08-09-cloture-arc-leboncoin-uc3m.md`) a montré que, lus depuis `main`
seul, trois fils restaient pendants : le sort des photos leboncoin, le sort de la piste
d'évaluation UC3M-LP, et l'attribution du chiffre « AP 0,932 ». Cette fiche solde les trois.

## Décision

### 1. Les photos leboncoin sortent définitivement du pilier

Les **22 204 photos** collectées via le scraper (`collecte/scraper/`) devaient servir à
entraîner et évaluer la détection de plaques. Elles ne servent à rien pour ce pilier :
leboncoin **floute les plaques à la source** dans la très grande majorité des annonces
(~95 % estimés, constat de terrain vérifié sur échantillon — raisonnement complet dans la
fiche [0002](0002-transfert-plaques-photos-reelles.md)). Par ailleurs le dossier
`dl/data/leboncoin-private/raw/` a été constaté **vide** le 2026-08-08.

**Le pilier Plaques ne s'appuie plus sur aucune image leboncoin.** Ceci tranche la
contradiction entre l'ADR DL 0002 (« dossier vide ») et l'ADR ML 0001 (« les photos
alimentent le pilier plaques ») — cette dernière ligne reçoit un erratum. Le pilier Prix
n'est pas concerné : il n'utilise que les annonces (texte et chiffres), pas les photos.

### 2. L'arc d'évaluation UC3M-LP est clos, sans verdict

La branche `reset/plaques` avait choisi UC3M-LP (Universidad Carlos III de Madrid —
*License Plates* : 1 975 photos de voitures espagnoles, plaques annotées, licence CC BY
4.0) comme domaine d'évaluation de remplacement, les plaques espagnoles ayant le même
gabarit long que les françaises (520 × 110 mm). État réel de cet arc au moment de la
clôture :

| Étape | État |
|---|---|
| Téléchargement (8,6 Go) et analyse exploratoire des données | faits (`dl/notebooks/plaques/03_uc3m_eda.ipynb`, branche tombeau) |
| Script d'inférence `transfert_uc3m.py` (deux fenêtres de redimensionnement) | écrit, jamais exécuté |
| Mesure du transfert (rappel/précision sur les 395 images test) | **jamais produite** |
| Verdict (la fiche 0003 promise par le plan de bascule) | **jamais écrit** |

Entre-temps, `main` a répondu à la même question — « le modèle tient-il hors de son
dataset d'entraînement ? » — par un **meilleur examen** : 87 photos réelles françaises,
annotées en aveugle, rappel mesuré 0,895 (fiche 0002). Mesurer UC3M-LP n'apporterait
qu'un verdict sur un domaine moins proche du produit (contexte urbain espagnol, photos
d'appareil reflex 5184 × 3888 px, loin du cadrage annonce).

**Idée conservée pour l'avenir** (seul acquis technique de l'arc) : la fenêtre de
redimensionnement du modèle (`transform.min_size`/`max_size`) est un paramètre
d'**inférence**, pas des poids — elle peut être élargie sans ré-entraîner si un jour des
plaques trop petites à l'image font chuter le rappel.

### 3. Généalogie des deux entraînements — à qui appartient l'AP 0,932

Deux checkpoints distincts (empreintes git différentes, vérifié) ont porté le même nom
`plaques_baseline.pt` :

| | Entraînement abandonné (branche tombeau) | **Modèle livré** (`main`, tag `modele-plaques-v1`) |
|---|---|---|
| Taux d'apprentissage | 1e-3 | 0,005 |
| Précision moyenne (AP) en validation | 0,932 (époque 9) | non mesurée / non consignée |
| Rappel / précision en validation | 0,918 / 0,927 (seuil 0,5) | 0,876 / 0,904 (seuil 0,3, fiche 0002) |
| Mesure sur domaine réel | aucune | rappel 0,895 sur 87 photos (fiche 0002) |

**Règle : le chiffre « AP 0,932 » ne doit jamais être cité pour le modèle déployé.** Il
appartient à un entraînement qui n'existe plus dans la chaîne livrée. Les chiffres du
modèle livré sont ceux de la fiche 0002.

### 4. Traces et ménage

- La branche `reset/plaques` devient un **tombeau local** (archive non poussée, pattern de
  `reset/cardd-baseline`) : elle conserve les fiches d'origine, l'EDA UC3M-LP et le script
  d'inférence, consultables par `git show`.
- `dl/data/uc3m-lp/raw/UC3M-LP.zip` (4,2 Go, doublon de l'archive extraite) est supprimé ;
  le dataset extrait (4,3 Go) est conservé, re-téléchargeable sur Zenodo au besoin.

## Alternatives écartées

| alternative | raison du rejet |
|---|---|
| lancer enfin la mesure UC3M-LP pour « finir » l'arc | le verdict n'informerait aucune décision : le transfert est déjà validé sur un domaine plus proche du produit |
| supprimer la branche `reset/plaques` | perdrait les seuls exemplaires des mesures de l'entraînement 1e-3 et de l'EDA UC3M-LP (commits jamais poussés) |
| ré-entraîner le modèle de `main` pour retrouver une AP comparable à 0,932 | chiffre de validation interne sans valeur produit ; la mesure qui compte (domaine réel) existe déjà |
| réécrire les fiches 0001/0002 pour gommer les traces des deux lignes | contraire à la doctrine d'immuabilité des fiches ; l'histoire réelle est une pièce du dossier BC04, pas un défaut |

## Conséquences

- Toute l'histoire du pilier Plaques se lit désormais depuis `main` seul ; les renvois vers
  la branche locale sont annotés comme pointant vers un tombeau.
- Les errata datés sont posés dans l'ADR ML 0001 (photos) et l'ADR DL 0002 (renvoi).
- Prochaine étape du pilier inchangée : Phase 4 — CLI de floutage par dossier (future
  fiche 0004).

## Limites

- La disparition du contenu de `dl/data/leboncoin-private/raw/` (22 204 photos, 2,4 Go,
  non reconstructibles) n'est **pas expliquée** — constatée, pas élucidée. Sans objet pour
  le pilier depuis la présente clôture, mais consigné honnêtement.
- Le modèle livré n'a pas d'AP consignée sur son propre entraînement ; sa qualité n'est
  établie que par la validation interne (rappel 0,876) et le lot réel (fiche 0002). Jugé
  suffisant : c'est la mesure de domaine réel qui engage le produit.
