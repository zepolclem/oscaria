# Décisions — pilier DL (Plaques)

Fiches de décision du pilier **Plaques** : détection des plaques d'immatriculation et floutage
avant tout livrable, au titre de la minimisation RGPD. Périmètre acté : détection de boîte et
flou gaussien, **pas de lecture ni d'OCR**.

Format et règles communes : voir le [journal général](../README.md). Numérotation propre à cet
espace ; une fiche se cite « ADR DL 0002 ».

## Index

| # | fiche | date | en une ligne |
|---|---|---|---|
| 0001 | [Baseline détection plaques](0001-baseline-detection-plaques.md) | 2026-08-06 | dataset car-plate-detection gardé ; Faster R-CNN MobileNetV3 fine-tuné sur CPU (backward MPS corrompu — mesuré) ; rappel prioritaire, pas de test interne |
| 0002 | [Transfert photos réelles](0002-transfert-plaques-photos-reelles.md) | 2026-08-08 | 87 photos perso annotées en aveugle : rappel réel 0,895, couverture 95,3 %, 4/95 plaques sans flou ; transfert validé, seuil 0,3 confirmé, pas de fine-tuning |
