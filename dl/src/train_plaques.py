"""Entraînement de la baseline détection de plaques — pilier plaques OscarIA.

Fine-tuning d'un détecteur torchvision pré-entraîné COCO, tête remplacée pour 2 classes
(fond + plaque). Choix du modèle et du device consignés dans l'ADR 0001.

Fait mesuré le 2026-08-06 (torch 2.13) : le backward des détecteurs torchvision sur MPS
ne plante plus mais produit des **gradients corrompus en silence** (norme 2e8 puis NaN dès
le pas 1 sur données réelles ; CPU sain avec les mêmes données et hyperparamètres).
→ **Entraînement sur CPU** (~4,4 min/époque mesurées, 346 images, batch 4) ;
l'inférence MPS, elle, fonctionne (fait `detect.py`).

Le checkpoint écrit est **autoporteur** : state_dict + config (modèle, classes, split,
hyperparamètres) + seuil de score par défaut — rechargeable sans relire ce fichier.

CLI :
    uv run --project dl python dl/src/train_plaques.py \
        --donnees dl/data/plaques/raw --sortie dl/models/plaques_baseline.pt \
        [--epoques 10 --lr 0.005 --graine 42]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from device import get_device

CONFIG_DEFAUT = {
    "modele": "fasterrcnn_mobilenet_v3_large_fpn",
    "num_classes": 2,  # fond + plaque
    "part_val": 0.2,
    "graine": 42,
    "epoques": 10,
    "lr": 0.005,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "taille_lot": 4,
    "seuil_score_defaut": 0.5,  # réévalué en Phase 3 (courbe rappel/seuil)
}


def construire_modele(num_classes: int = 2):
    """Détecteur pré-entraîné COCO, tête de classification remplacée pour nos classes."""
    modele = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    in_features = modele.roi_heads.box_predictor.cls_score.in_features
    modele.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return modele


def entrainer(donnees, sortie, config=None, device=None, journal=print):
    """Boucle d'entraînement complète ; renvoie `(modèle, historique des pertes)`."""
    # Import différé : le parsing XML (defusedxml) n'est requis qu'à l'entraînement —
    # l'app web n'embarque que l'inférence (charger_checkpoint/predire).
    from plaques import PlaquesDataset, charger_annotations, split_train_val

    config = {**CONFIG_DEFAUT, **(config or {})}
    # CPU par défaut : backward MPS corrompu (gradients NaN silencieux), cf. docstring.
    device = device or torch.device("cpu")
    torch.manual_seed(config["graine"])

    exemples = charger_annotations(donnees)
    train, val = split_train_val(exemples, config["part_val"], config["graine"])
    journal(f"split : {len(train)} train / {len(val)} val | device : {device}")

    chargeur = DataLoader(
        PlaquesDataset(train), batch_size=config["taille_lot"], shuffle=True,
        collate_fn=PlaquesDataset.collate,
    )
    modele = construire_modele(config["num_classes"]).train().to(device)
    optimiseur = torch.optim.SGD(
        modele.parameters(), lr=config["lr"],
        momentum=config["momentum"], weight_decay=config["weight_decay"],
    )

    historique = []
    for epoque in range(1, config["epoques"] + 1):
        t0, cumul = time.time(), 0.0
        for images, cibles in chargeur:
            images = [im.to(device) for im in images]
            cibles = [{k: v.to(device) for k, v in c.items()} for c in cibles]
            pertes = modele(images, cibles)
            total = sum(pertes.values())
            optimiseur.zero_grad()
            total.backward()
            optimiseur.step()
            cumul += float(total.detach())
        moyenne = cumul / len(chargeur)
        historique.append(moyenne)
        journal(f"époque {epoque:2d}/{config['epoques']} — perte {moyenne:.4f} "
                f"({time.time() - t0:.0f}s)")

    if sortie:
        sortie = Path(sortie)
        sortie.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": modele.state_dict(), "config": config,
                    "historique_pertes": historique}, sortie)
        journal(f"checkpoint → {sortie}")
    return modele, historique


def charger_checkpoint(chemin, device=None):
    """Recharge un checkpoint autoporteur → `(modèle en eval, config)`."""
    device = device or get_device()
    # weights_only=True : refuse tout objet Python arbitraire dans le .pt (sécurité) —
    # notre checkpoint n'a que des tenseurs et des types simples.
    ckpt = torch.load(chemin, map_location=device, weights_only=True)
    modele = construire_modele(ckpt["config"]["num_classes"])
    modele.load_state_dict(ckpt["state_dict"])
    modele.eval().to(device)
    return modele, ckpt["config"]


@torch.no_grad()
def predire(modele, device, exemples, taille_lot: int = 8):
    """Inférence sur une liste d'exemples → liste de listes `(boite, score)` par image."""
    from PIL import Image
    from torchvision import transforms

    to_tensor = transforms.ToTensor()
    modele.eval()
    resultats = []
    for i in range(0, len(exemples), taille_lot):
        lot = exemples[i : i + taille_lot]
        tenseurs = [to_tensor(Image.open(ex["image"]).convert("RGB")).to(device) for ex in lot]
        for sortie in modele(tenseurs):
            resultats.append(list(zip(sortie["boxes"].cpu().tolist(),
                                      sortie["scores"].cpu().tolist())))
    return resultats


if __name__ == "__main__":
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--donnees", default="dl/data/plaques/raw")
    parseur.add_argument("--sortie", default="dl/models/plaques_baseline.pt")
    parseur.add_argument("--epoques", type=int, default=CONFIG_DEFAUT["epoques"])
    parseur.add_argument("--lr", type=float, default=CONFIG_DEFAUT["lr"])
    parseur.add_argument("--graine", type=int, default=CONFIG_DEFAUT["graine"])
    args = parseur.parse_args()
    entrainer(args.donnees, args.sortie,
              {"epoques": args.epoques, "lr": args.lr, "graine": args.graine})
