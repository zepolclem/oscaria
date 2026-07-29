"""Détourage du véhicule sur les photos d'annonce — pré-traitement du pilier État.

Motivation (idée utilisateur, ADR 0009) : les photos leboncoin cadrent la voiture entière à
plusieurs mètres, là où CarDD est fait de gros plans de dégâts. Recadrer sur le véhicule sert
**deux** objectifs, le second étant le plus important :

1. **Résolution** — on récupère les pixels perdus dans le ciel, le gravier et la haie.
2. **Suppression du confondant** — l'EDA d'inspection a mesuré un écart de cadrage entre classes
   déclarées (ratio médian 0,75 pour `damaged` contre 1,0 pour `excellent_condition`). Tant qu'on
   garde le fond, un réseau peut apprendre « allée de gravier = abîmé » au lieu de regarder la
   tôle. Le détourage retire ce raccourci visuel.

Effet de bord utile : **l'absence de détection est déjà une information de type de vue.** Un
document, un habitacle ou une pièce détachée seule ne déclenchent pas un détecteur de véhicules.

Méthode : détecteur pré-entraîné sur COCO (`Faster R-CNN`), en **inférence pure** — aucun
ré-entraînement. L'ADR 0008 avait établi que seul le *backward* de détection est cassé sur MPS ;
l'inférence, elle, fonctionne.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import detection

# Catégories COCO retenues comme « véhicule ». `motorcycle` et `bicycle` sont volontairement
# exclues : une annonce de voiture qui en contient une au second plan ne doit pas être recadrée
# dessus. `truck` et `bus` sont gardées car COCO classe souvent un utilitaire ou un van ainsi.
VEHICULE_COCO = {"car", "truck", "bus"}

_TO_TENSOR = transforms.ToTensor()


def load_detector(device, poids=None):
    """Charge un détecteur COCO pré-entraîné et renvoie `(modèle, catégories)`.

    `catégories` est la liste indexée par `label` que renvoie le modèle — on y lit le **nom** de
    la classe plutôt que de coder en dur un identifiant COCO, dont la numérotation varie selon
    les versions du jeu (80 catégories utiles réparties sur 91 identifiants).

    Le premier appel télécharge les poids (~167 Mo) dans le cache torch.
    """
    poids = poids or detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    model = detection.fasterrcnn_resnet50_fpn_v2(weights=poids)
    model.eval().to(device)
    return model, poids.meta["categories"]


@torch.no_grad()
def detecter(model, categories, device, chemins, score_min=0.5, taille_lot=8):
    """Détecte les véhicules sur une liste de chemins d'images.

    Renvoie une liste de dictionnaires, un par image, dans l'ordre d'entrée :
    `chemin`, `largeur`, `hauteur`, `n_vehicules`, et — si au moins un véhicule dépasse
    `score_min` — la **plus grande** boîte trouvée (`boite`, `score`, `classe`, `fraction_cadre`).

    On retient la plus grande et non la mieux notée : sur une annonce, la voiture vendue est au
    premier plan ; celles du fond (voisines, parking) sont plus petites, parfois mieux notées.
    """
    resultats: list[dict] = []
    for i in range(0, len(chemins), taille_lot):
        lot = [Path(c) for c in chemins[i : i + taille_lot]]
        images = [Image.open(c).convert("RGB") for c in lot]
        tenseurs = [_TO_TENSOR(im).to(device) for im in images]
        sorties = model(tenseurs)

        for chemin, img, sortie in zip(lot, images, sorties):
            W, H = img.size
            ligne = {"chemin": str(chemin), "largeur": W, "hauteur": H, "n_vehicules": 0,
                     "boite": None, "score": None, "classe": None, "fraction_cadre": None}

            boites = sortie["boxes"].cpu()
            scores = sortie["scores"].cpu()
            noms = [categories[int(l)] for l in sortie["labels"].cpu()]

            retenus = [
                (b.tolist(), float(s), n)
                for b, s, n in zip(boites, scores, noms)
                if n in VEHICULE_COCO and s >= score_min
            ]
            ligne["n_vehicules"] = len(retenus)
            if retenus:
                b, s, n = max(retenus, key=lambda r: (r[0][2] - r[0][0]) * (r[0][3] - r[0][1]))
                ligne.update(boite=b, score=s, classe=n,
                             fraction_cadre=((b[2] - b[0]) * (b[3] - b[1])) / (W * H))
            resultats.append(ligne)
    return resultats


def recadrer(img: Image.Image, boite, marge: float = 0.05) -> Image.Image:
    """Recadre l'image sur la boîte, avec une marge relative à la taille de la boîte.

    La marge évite de couper au ras de la carrosserie : un pare-chocs arraché ou une aile
    enfoncée dépasse souvent la boîte que le détecteur trace autour du véhicule intact.
    """
    W, H = img.size
    x1, y1, x2, y2 = boite
    mx, my = (x2 - x1) * marge, (y2 - y1) * marge
    return img.crop((max(0, int(x1 - mx)), max(0, int(y1 - my)),
                     min(W, int(x2 + mx)), min(H, int(y2 + my))))
