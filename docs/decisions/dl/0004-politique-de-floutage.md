# DL 0004 — Politique de floutage : réglages par défaut, garde-fous, couverture mesurée

- **Date** : 2026-08-10
- **Statut** : Accepté
- **Pilier** : plaques (Phase 4 — clôt le programme annoncé par les fiches 0001 à 0003)

## Contexte

Les fiches précédentes ont livré le détecteur (0001), validé son transfert sur domaine réel
(0002) et remis l'historique en cohérence (0003). Restait la dernière brique du plan
(`docs/plans/2026-08-06-remise-a-zero-plaques.md`, Phase 4) : transformer la démo web en
**outil de traitement par lot**, et figer une **politique de floutage** — quels réglages,
pourquoi, et ce qu'ils garantissent (ou pas).

## Décision

### Réglages par défaut (CLI `dl/src/flouter.py`)

| Paramètre | Valeur | Justification mesurée |
|---|---|---|
| Seuil de score | **0,3** | le plus permissif utile : les résultats sont identiques aux seuils 0,2/0,3/0,5 (fiche 0002 — le modèle est confiant, ses échecs sont francs) ; rappel prioritaire (fiche 0001) |
| Marge d'élargissement | **15 %** | couvre les bords de plaque que la boîte prédite rogne ; c'est la marge de la mesure de couverture de la fiche 0002 |
| Rayon de flou | **0,4 × hauteur de boîte**, plancher 6 px | un rayon fixe laisse lisibles les plaques en gros plan ; proportionnel = illisibilité constante quelle que soit la distance |

### Couverture garantie par ces réglages — chiffres de la fiche 0002

Mesurée sur le domaine réel (87 photos, 95 plaques vérité), **aux mêmes seuil et marge** —
aucune nouvelle mesure nécessaire :

- couverture moyenne après marge : **95,3 %** ;
- plaques couvertes à ≥ 99 % (illisibles après flou) : **91,6 %** ;
- plaques sans aucun flou : **4,2 % (4/95)** — fuite résiduelle assumée et affichée.

### Garde-fous du traitement par lot

- `flouter_dossier()` **refuse** d'écrire dans un dossier `raw/` ou dans le dossier
  d'entrée : les originaux sont intouchables, les images floutées vont ailleurs.
- Règle livrable inchangée : **seules les images floutées sortent du poste.**
- Sans vérité terrain, le bilan du lot ne compte que les détections — il ne peut pas dire
  combien de plaques ont été ratées ; ce chiffre-là vient de la mesure 0002.

### Démo

`dl/notebooks/plaques/04_demo_floutage.ipynb` : grille avant/après sur 12 images du split
test **UC3M-LP** (CC BY 4.0 — publiables dans le dossier de certification avec attribution),
11/12 plaques détectées et floutées. Démo **visuelle**, pas une mesure : le domaine UC3M-LP
n'a jamais été évalué (fiche 0003). Le lot réel de la fiche 0002 n'est pas montrable :
données personnelles, une grille « avant » à plaques lisibles violerait la règle livrable.

## Alternatives écartées

| alternative | raison du rejet |
|---|---|
| rectangle noir plein au lieu du flou | équivalent en protection mais plus brutal visuellement pour une annonce ; le flou gaussien fort rend la plaque illisible (contrôle visuel carnet 04) sans défigurer la photo |
| pixelisation | réversibilité partielle documentée sur les mosaïques à gros blocs ; le flou gaussien à rayon proportionnel n'a pas ce défaut à nos rayons |
| baisser le seuil sous 0,2 | aucun gain mesuré (fiche 0002), du bruit en plus |
| inpainting (effacement génératif de la plaque) | dépendance lourde et fabrication de contenu — hors périmètre d'un floutage de minimisation |

## Conséquences

- Le pilier Plaques est **complet au sens du plan** : dataset → entraînement → transfert
  mesuré → politique de floutage outillée. Les évolutions futures (lot multi-sources,
  fine-tuning si un domaine plus dur apparaît) donneront de nouvelles fiches.
- La démo web (`app/src/page_plaques.py`) et la CLI partagent les mêmes fonctions et les
  mêmes réglages par défaut — un seul endroit à changer si une fiche future les révise.

## Limites

- **Le floutage n'est pas garanti à 100 %** (4,2 % de plaques sans flou sur le domaine
  réel) : tout livrable qui montre des photos floutées doit continuer à l'afficher.
- La couverture est mesurée sur un lot mono-photographe (fiche 0002) ; un lot multi-sources
  reste souhaitable avant toute promesse plus forte.
- Le bilan de `flouter_dossier()` sur un lot non annoté est un compteur de détections, pas
  une mesure de rappel.
