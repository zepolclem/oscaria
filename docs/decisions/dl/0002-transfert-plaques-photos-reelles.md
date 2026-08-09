# DL 0002 — Transfert de la baseline plaques sur photos réelles françaises

- **Date** : 2026-08-08
- **Statut** : Accepté
- **Pilier** : plaques (suite de la fiche 0001)

## Contexte

La baseline (fiche 0001) est entraînée sur un dataset international Kaggle. Question de la
phase-leçon : tient-elle sur le **domaine cible réel** ? L'arc dégâts est mort ici (ROC 0,509).
Lot de mesure : **87 photos personnelles de l'utilisateur** (iPhone 13, type annonce leboncoin,
plaques françaises), annotées **en aveugle** via `dl/src/annoter_plaques.py` (copies neutres
mélangées, EXIF/GPS supprimés, la page ne montre aucune prédiction du modèle) : **95 boîtes
vérité, 4 photos sans plaque**. Mesures : `dl/notebooks/plaques/03_transfert_reel.ipynb`.

Note : `dl/data/leboncoin-private/raw/` s'est révélé vide (les 22 204 photos du plan de
remise à zéro n'y sont plus) — ce lot personnel le remplace pour cette mesure.

## Chiffres mesurés

| Mesure (seuil de score 0,3) | Interne (Kaggle) | Réel (87 photos) |
|---|---|---|
| Rappel @IoU 0,5 | 0,876 | **0,895** (85/95, 10 ratées) |
| Rappel @IoU 0,3 (boîte décalée mais utile au flou) | — | **0,958** (4 ratées) |
| Précision @IoU 0,5 | 0,904 | 0,904 (9 faux positifs) |
| Couverture après marge 15 % — moyenne | — | **95,3 %** |
| Plaques couvertes ≥ 99 % (illisibles après flou) | — | **91,6 %** |
| Plaques sans aucun flou | — | **4,2 % (4/95)** |
| Faux positifs sur les 4 photos sans plaque | — | 1 |

Résultats identiques aux seuils 0,2 / 0,3 / 0,5 : le modèle est très confiant ; ses échecs
sont francs (plaques minuscules ou très inclinées), pas des détections timides sous le seuil.

## Décision

- **Transfert validé** : rappel réel 0,895 ≥ gate 0,80 du plan → **pas de fine-tuning** sur
  photos françaises à ce stade.
- **Seuil de floutage : 0,3 confirmé** (baisser à 0,2 n'apporte rien de mesuré).
- **Bandeau de la démo mis à jour** : « démo non validée » → chiffres réels affichés
  (9 plaques sur 10 détectées, floutage non garanti à 100 %).

## Alternatives écartées

- **Fine-tuning sur la moitié du lot français** : prévu par le plan si rappel < 0,80 — non
  déclenché (0,895). Reste l'option de première main si un lot plus varié fait chuter le rappel.
- **Baisser le seuil sous 0,2** : aucun gain mesuré, du bruit en plus.

## Conséquences et limites honnêtes

- **Fuite résiduelle assumée et affichée : 4 plaques sur 95 sans aucun flou.** Le livrable ne
  doit jamais prétendre à un floutage garanti.
- **Portée de la mesure** : un seul photographe, un seul appareil, plaques majoritairement UE
  longues — bonne mesure de domaine, pas une preuve universelle. Un lot multi-sources
  (leboncoin restauré ou re-scrapé) reste souhaitable pour une fiche 0004 éventuelle.
- Rappel interne ≤ rappel réel : photos iPhone plus nettes et mieux cadrées que le Kaggle
  moyen — le « domaine réel » de ce lot est plus facile, pas plus dur. À garder en tête.
- Prochaine étape du pilier : Phase 4 — CLI de floutage par dossier + politique consignée
  (fiche 0003).
