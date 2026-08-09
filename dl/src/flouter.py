"""Floutage des plaques d'immatriculation — cœur du pilier plaques (RGPD).

Deux fonctions : `detecter_plaques()` (inférence du checkpoint baseline sur UNE image)
et `flouter_image()` (flou gaussien sur les boîtes, élargies d'une marge). La CLI de
traitement par dossier et la politique de floutage définitive (ADR DL 0003) viendront en
Phase 4 ; ce module porte le minimum nécessaire à la démo web.

Seuil de score par défaut : 0,3 — le plus permissif mesuré au carnet 02 (rappel 0,876,
précision 0,904 en validation interne). Hiérarchie ADR DL 0001 : rappel prioritaire, un
faux positif ne coûte qu'un flou en trop. Ce seuil sera re-jugé par la Phase 3.
"""

from __future__ import annotations

import torch
from PIL import Image, ImageFilter
from torchvision import transforms

SEUIL_DEMO = 0.3

_TO_TENSOR = transforms.ToTensor()


@torch.no_grad()
def detecter_plaques(modele, device, img: Image.Image, seuil: float = SEUIL_DEMO) -> list[list[float]]:
    """Boîtes `[xmin, ymin, xmax, ymax]` des plaques détectées au-dessus du seuil."""
    tenseur = _TO_TENSOR(img.convert("RGB")).to(device)
    sortie = modele([tenseur])[0]
    return [b.tolist() for b, s in zip(sortie["boxes"].cpu(), sortie["scores"].cpu())
            if float(s) >= seuil]


def flouter_image(img: Image.Image, boites, marge: float = 0.15, rayon_min: int = 6) -> Image.Image:
    """Renvoie une copie de l'image avec chaque boîte floutée (l'original n'est pas modifié).

    - `marge` : élargissement relatif de la boîte (esprit `recadrer()` de detect.py) —
      couvre les bords de plaque que la boîte prédite rogne souvent.
    - Rayon du flou **proportionnel à la hauteur de boîte** (plancher `rayon_min`) :
      une grande plaque en gros plan exige un flou bien plus fort qu'une plaque lointaine.
    """
    resultat = img.convert("RGB").copy()
    W, H = resultat.size
    for xmin, ymin, xmax, ymax in boites:
        mx, my = (xmax - xmin) * marge, (ymax - ymin) * marge
        zone = (max(0, int(xmin - mx)), max(0, int(ymin - my)),
                min(W, int(xmax + mx)), min(H, int(ymax + my)))
        if zone[2] <= zone[0] or zone[3] <= zone[1]:
            continue
        rayon = max(rayon_min, int((zone[3] - zone[1]) * 0.4))
        resultat.paste(resultat.crop(zone).filter(ImageFilter.GaussianBlur(rayon)), zone)
    return resultat
