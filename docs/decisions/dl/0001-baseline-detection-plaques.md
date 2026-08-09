# DL 0001 — Baseline détection de plaques : dataset, modèle, device, métriques

- **Date** : 2026-08-06
- **Statut** : Accepté
- **Pilier** : plaques (détection + floutage, pas d'OCR — minimisation RGPD)

## Contexte

Premier modèle du pilier plaques. L'EDA du dataset candidat Kaggle
`andrewmvd/car-plate-detection` (433 images, 471 boîtes PASCAL VOC, licence CC0) est validée
(`dl/notebooks/plaques/01_eda.ipynb`) : intégrité parfaite, résolution médiane 400×270 px,
30 % de plaques sous 1 % de l'image (petits objets), ratios d'aspect internationaux
(médiane 2,86 ; 48 % ≥ 3 « allure UE longue »). Verdict utilisateur : **gardé**.

## Décision

| Sujet | Choix | Pourquoi |
|---|---|---|
| Dataset | `car-plate-detection` (433 img) | seul candidat testé à ce stade, propre, CC0, format standard ; l'écart au domaine français sera **mesuré** en Phase 3, pas présumé |
| Cible | 1 classe « plaque », multi-boîtes | 5,5 % d'images multi-plaques (EDA) ; pas d'OCR |
| Modèle | `fasterrcnn_mobilenet_v3_large_fpn` pré-entraîné COCO, tête remplacée (2 classes : fond + plaque) | candidat léger du plan ; 433 images imposent le fine-tuning d'un pré-entraîné, pas un entraînement de zéro |
| Device | **CPU pour l'entraînement**, MPS pour l'inférence | fait mesuré du 2026-08-06 (torch 2.13) : le backward détection sur MPS ne plante plus mais produit des **gradients corrompus en silence** (norme 2×10⁸ puis NaN au pas 1 ; CPU sain, mêmes données/hyperparamètres). Époque CPU mesurée ≈ 4,4 min < seuil 15 min → pas de bascule PC |
| Hyperparamètres | SGD lr 0,005, momentum 0,9, weight decay 5e-4, batch 4, 10 époques, graine 42 | réglages standard torchvision detection ; consignés dans le checkpoint autoporteur |
| Split | train/val 346/87 **par image**, graine 42 | identité voiture non documentée dans le dataset → limite consignée (risque résiduel de fuite si une même voiture apparaît deux fois) |
| Métriques | **rappel prioritaire** à IoU ≥ 0,5, précision en second, courbe précision/rappel ; éval maison lisible (`dl/src/plaques.py`) | une plaque ratée = fuite RGPD ; un faux positif = un flou en trop. Pas d'accuracy seule |
| Jeu de test | **aucun jeu de test interne** | leçon de l'arc dégâts (ROC 0,509 sur le domaine cible) : le vrai test est le lot leboncoin annoté en aveugle (Phase 3, ADR 0002) |

## Alternatives écartées

- **Entraînement MPS** : gradients corrompus (mesuré, cf. ci-dessus).
- **Bascule PC RTX 5070 Ti** : inutile, époque CPU sous le seuil.
- **Modèle plus lourd (`fasterrcnn_resnet50_fpn_v2`)** : réservé à une itération suivante si
  la baseline plafonne ; sur 433 images le léger suffit pour établir le plancher.
- **YOLO/ultralytics** : dépendance nouvelle + licence AGPL à instruire — YAGNI pour une
  baseline ; à réexaminer si le rappel est insuffisant.

## Conséquences

- La qualité réelle se jugera en Phase 3 sur photos leboncoin (transfert de domaine) ;
  la fiche 0002 consignera ces chiffres et le choix du seuil bas de floutage.
- Vérité d'entraînement : le smoke test « pas d'exception » ne suffit pas sur MPS ;
  tout futur smoke test d'entraînement vérifie la **norme des gradients**.
- Checkpoint : `dl/models/plaques_baseline.pt` (gitignoré), autoporteur
  (state_dict + config + historique des pertes), rechargé via
  `train_plaques.charger_checkpoint()` (`weights_only=True`).
