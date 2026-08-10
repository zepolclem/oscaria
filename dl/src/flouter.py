"""Floutage des plaques d'immatriculation — cœur du pilier plaques (RGPD).

Trois fonctions : `detecter_plaques()` (inférence du checkpoint baseline sur UNE image),
`flouter_image()` (flou gaussien sur les boîtes, élargies d'une marge) et
`flouter_dossier()` (traitement par lot). La politique de floutage — réglages par défaut
et couverture mesurée — est consignée en ADR DL 0004.

Seuil de score par défaut : 0,3 — le plus permissif mesuré, confirmé sur domaine réel
(ADR DL 0002 : rappel 0,895, couverture après marge 95,3 % sur 87 photos). Hiérarchie
ADR DL 0001 : rappel prioritaire, un faux positif ne coûte qu'un flou en trop.

CLI :
    uv run python src/flouter.py --entree <dossier> --sortie <dossier> [--seuil 0.3 --marge 0.15]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageFilter
from torchvision import transforms

SEUIL_DEMO = 0.3
MARGE_DEFAUT = 0.15
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

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


def flouter_dossier(entree: Path, sortie: Path, seuil: float = SEUIL_DEMO,
                    marge: float = MARGE_DEFAUT, journal=print) -> dict:
    """Floute toutes les images de `entree` vers `sortie` (mêmes noms de fichiers).

    `sortie` ne doit jamais être un dossier `raw/` ni `entree` lui-même : les originaux
    sont intouchables (règle du pilier). Renvoie un bilan chiffré — sans vérité terrain,
    on ne peut compter que les détections, pas les plaques ratées.
    """
    if sortie.resolve() == entree.resolve() or sortie.name == "raw" or "raw" in sortie.resolve().parts:
        raise ValueError(f"sortie interdite ({sortie}) : jamais dans raw/ ni sur l'entrée")

    from device import get_device
    from train_plaques import charger_checkpoint

    device = get_device()  # inférence seule : MPS sain (ADR DL 0001)
    checkpoint = Path(__file__).resolve().parents[1] / "models" / "plaques_baseline.pt"
    modele, _ = charger_checkpoint(checkpoint, device)

    images = sorted(p for p in entree.iterdir() if p.suffix.lower() in EXTENSIONS)
    sortie.mkdir(parents=True, exist_ok=True)
    bilan = {"images": len(images), "avec_plaque": 0, "boites": 0}
    for k, chemin in enumerate(images, 1):
        img = Image.open(chemin)
        boites = detecter_plaques(modele, device, img, seuil)
        flouter_image(img, boites, marge).save(sortie / chemin.name)
        bilan["avec_plaque"] += bool(boites)
        bilan["boites"] += len(boites)
        if k % 20 == 0:
            journal(f"{k}/{len(images)} images traitées")
    journal(f"{bilan['images']} images -> {sortie} : {bilan['boites']} plaques floutées "
            f"sur {bilan['avec_plaque']} images ({bilan['images'] - bilan['avec_plaque']} sans détection)")
    return bilan


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Floutage des plaques par dossier (ADR DL 0004)")
    p.add_argument("--entree", required=True, type=Path)
    p.add_argument("--sortie", required=True, type=Path)
    p.add_argument("--seuil", type=float, default=SEUIL_DEMO)
    p.add_argument("--marge", type=float, default=MARGE_DEFAUT)
    a = p.parse_args()
    flouter_dossier(a.entree, a.sortie, a.seuil, a.marge)
