"""Dataset plaques (PASCAL VOC) et évaluation IoU maison — pilier plaques OscarIA.

Trois briques, dans l'ordre où la Phase 2 les consomme :
- `charger_annotations()` : lit les XML PASCAL VOC → une liste d'exemples `(image, boîtes)`.
- `PlaquesDataset` : dataset PyTorch au format détection torchvision
  (`{"boxes": [N, 4], "labels": [N]}`, une seule classe « plaque »).
- `evaluer()` : rappel/précision à IoU et seuil de score donnés, par appariement glouton,
  plus la courbe précision/rappel. Écrite à la main (~50 lignes) pour rester lisible par
  un novice — aucune dépendance nouvelle.

IoU (*Intersection over Union*) : aire de l'intersection de deux boîtes divisée par l'aire
de leur union. 1.0 = boîtes identiques, 0.0 = disjointes. Une prédiction « touche » une
vérité quand leur IoU dépasse un seuil (0.5 par convention).
"""

from __future__ import annotations

import random
from pathlib import Path

import defusedxml.ElementTree as ET
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

_TO_TENSOR = transforms.ToTensor()


def charger_annotations(racine) -> list[dict]:
    """Lit `annotations/*.xml` (PASCAL VOC) et renvoie une liste d'exemples triée par nom.

    Chaque exemple : `{"image": Path, "boites": [[xmin, ymin, xmax, ymax], ...]}`.
    Le tri par nom rend l'ordre indépendant du système de fichiers (split reproductible).
    """
    racine = Path(racine)
    exemples = []
    for xml in sorted(racine.glob("annotations/*.xml")):
        noeud = ET.parse(xml).getroot()
        boites = []
        for obj in noeud.findall("object"):
            b = obj.find("bndbox")
            boites.append([int(b.findtext(c)) for c in ("xmin", "ymin", "xmax", "ymax")])
        exemples.append({"image": racine / "images" / f"{xml.stem}.png", "boites": boites})
    return exemples


def split_train_val(exemples, part_val: float = 0.2, graine: int = 42):
    """Split par image, mélangé avec une graine fixe → reproductible (~346/87 sur 433).

    Split par image et non par voiture : l'identité des véhicules n'est pas documentée
    dans ce dataset (limite consignée dans l'EDA et l'ADR DL 0001).
    """
    indices = list(range(len(exemples)))
    random.Random(graine).shuffle(indices)
    n_val = round(len(exemples) * part_val)
    val = [exemples[i] for i in indices[:n_val]]
    train = [exemples[i] for i in indices[n_val:]]
    return train, val


class PlaquesDataset(Dataset):
    """Dataset au format attendu par les détecteurs torchvision.

    `__getitem__` renvoie `(image_tensor, cible)` avec
    `cible = {"boxes": FloatTensor [N, 4], "labels": LongTensor [N]}` — labels tous à 1 :
    une seule classe « plaque », le 0 étant réservé au fond par convention torchvision.
    """

    def __init__(self, exemples: list[dict]):
        self.exemples = exemples

    def __len__(self) -> int:
        return len(self.exemples)

    def __getitem__(self, i: int):
        ex = self.exemples[i]
        image = _TO_TENSOR(Image.open(ex["image"]).convert("RGB"))
        boites = torch.tensor(ex["boites"], dtype=torch.float32)
        cible = {"boxes": boites, "labels": torch.ones(len(boites), dtype=torch.int64)}
        return image, cible

    @staticmethod
    def collate(lot):
        """Les images ont des tailles différentes : pas d'empilement, on garde des listes."""
        return tuple(zip(*lot))


def iou(a, b) -> float:
    """IoU de deux boîtes `[xmin, ymin, xmax, ymax]`."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aire_a = (a[2] - a[0]) * (a[3] - a[1])
    aire_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aire_a + aire_b - inter) if inter else 0.0


def apparier(predictions, verites, seuil_iou: float = 0.5):
    """Appariement glouton : chaque prédiction (score décroissant) prend la vérité libre
    de meilleur IoU si celui-ci atteint le seuil. Renvoie `(vrais_positifs, faux_positifs,
    faux_negatifs)` pour UNE image.

    `predictions` : liste de `(boite, score)` ; `verites` : liste de boîtes.
    """
    libres = list(range(len(verites)))
    vp = 0
    for boite, _ in sorted(predictions, key=lambda p: -p[1]):
        if not libres:
            break
        meilleur = max(libres, key=lambda j: iou(boite, verites[j]))
        if iou(boite, verites[meilleur]) >= seuil_iou:
            libres.remove(meilleur)
            vp += 1
    return vp, len(predictions) - vp, len(libres)


def evaluer(predictions_par_image, verites_par_image, seuil_iou=0.5, seuil_score=0.5):
    """Rappel et précision globaux sur un jeu d'images, à IoU et score donnés.

    - **rappel** = plaques vérité retrouvées / plaques vérité totales — LA métrique RGPD
      (une plaque ratée = fuite de donnée personnelle) ;
    - **précision** = prédictions correctes / prédictions émises (un faux positif ne coûte
      qu'un flou en trop).
    """
    VP = FP = FN = 0
    for preds, verites in zip(predictions_par_image, verites_par_image):
        retenues = [(b, s) for b, s in preds if s >= seuil_score]
        vp, fp, fn = apparier(retenues, verites, seuil_iou)
        VP, FP, FN = VP + vp, FP + fp, FN + fn
    rappel = VP / (VP + FN) if VP + FN else 0.0
    precision = VP / (VP + FP) if VP + FP else 0.0
    return {"rappel": rappel, "precision": precision, "VP": VP, "FP": FP, "FN": FN}


def courbe_precision_rappel(predictions_par_image, verites_par_image, seuil_iou=0.5, pas=0.05):
    """Balaye le seuil de score de 0 à 0.95 → liste de points `(seuil, rappel, précision)`.

    C'est cette courbe qui fixera le « seuil bas » du floutage (Phase 3/4) : on y lit le
    seuil le plus permissif qui maximise le rappel sans noyer l'image de faux positifs.
    """
    points = []
    seuil = 0.0
    while seuil < 1.0:
        r = evaluer(predictions_par_image, verites_par_image, seuil_iou, seuil)
        points.append((round(seuil, 2), r["rappel"], r["precision"]))
        seuil += pas
    return points
