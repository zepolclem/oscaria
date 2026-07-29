"""Localisation des dégâts SANS ré-entraînement — deux méthodes en duel (ADR 0008).

1. `tile_scores`  : fenêtre glissante multi-échelle (idée utilisateur) — chaque tuile passe
   dans le classifieur multi-label ; les probas s'accumulent en cartes de score par classe.
2. `gradcam_map`  : Grad-CAM (carte d'activation par gradients) sur `layer4` du ResNet18 —
   une passe, pas de découpage.

Les deux produisent une carte [0..1] à la taille de l'image parente ; `map_to_box` en tire
une boîte englobante. Évaluation contre les vraies boîtes CarDD test dans le notebook 04.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as TF
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

_NORM = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
_TO_TENSOR = transforms.ToTensor()


def _positions(full: int, size: int, stride: int) -> list[int]:
    """Positions de fenêtres couvrant [0, full] — la dernière colle au bord."""
    if full <= size:
        return [0]
    xs = list(range(0, full - size + 1, stride))
    if xs[-1] != full - size:
        xs.append(full - size)
    return xs


@torch.no_grad()
def tile_scores(
    model, classes, device, img: Image.Image,
    tile_sizes=(224, 448), overlap: float = 0.5, batch_size: int = 64,
) -> np.ndarray:
    """Cartes de score par classe via fenêtre glissante multi-échelle.

    Renvoie un tableau (n_classes, H, W) : moyenne des probabilités des tuiles couvrant
    chaque pixel. Les tuiles sont redimensionnées à 224 avant de passer dans le modèle.
    """
    model.eval()
    rgb = img.convert("RGB")
    W, H = rgb.size
    x_full = _TO_TENSOR(rgb)  # (3, H, W) en [0,1]

    crops, boxes = [], []
    for s in tile_sizes:
        if s > max(W, H):
            continue
        stride = max(1, int(s * (1 - overlap)))
        for y in _positions(H, min(s, H), stride):
            for x in _positions(W, min(s, W), stride):
                w, h = min(s, W), min(s, H)
                tile = x_full[:, y : y + h, x : x + w].unsqueeze(0)
                crops.append(TF.interpolate(tile, size=(224, 224), mode="bilinear",
                                            align_corners=False)[0])
                boxes.append((x, y, w, h))

    score = np.zeros((len(classes), H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.float64)
    for i in range(0, len(crops), batch_size):
        batch = torch.stack([_NORM(c) for c in crops[i : i + batch_size]]).to(device)
        probs = torch.sigmoid(model(batch)).cpu().numpy()
        for p, (x, y, w, h) in zip(probs, boxes[i : i + batch_size]):
            score[:, y : y + h, x : x + w] += p[:, None, None]
            count[y : y + h, x : x + w] += 1
    return score / np.maximum(count, 1)


def gradcam_map(model, classes, device, img: Image.Image, class_idx: int) -> np.ndarray:
    """Grad-CAM sur `layer4` — carte (H, W) normalisée [0..1] à la taille de l'image parente.

    Le forward se fait sur l'image redimensionnée 224 (distribution d'entraînement),
    la carte 7x7 est ré-agrandie à la taille d'origine.
    """
    model.eval()
    rgb = img.convert("RGB")
    W, H = rgb.size
    x = _NORM(_TO_TENSOR(rgb.resize((224, 224)))).unsqueeze(0).to(device)

    acts: list[torch.Tensor] = []
    grads: list[torch.Tensor] = []
    h1 = model.layer4.register_forward_hook(lambda m, i, o: acts.append(o))
    h2 = model.layer4.register_full_backward_hook(lambda m, gi, go: grads.append(go[0]))
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        logits[0, class_idx].backward()
    finally:
        h1.remove(); h2.remove()

    a, g = acts[0][0], grads[0][0]              # (C, 7, 7)
    weights = g.mean(dim=(1, 2), keepdim=True)   # importance par canal
    cam = torch.relu((weights * a).sum(0))       # (7, 7)
    cam = TF.interpolate(cam[None, None], size=(H, W), mode="bilinear",
                         align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    return cam / cam.max() if cam.max() > 0 else cam


def map_to_box(score_map: np.ndarray, thr_frac: float = 0.5):
    """Boîte englobante (x1, y1, x2, y2) des pixels >= thr_frac * max. None si carte nulle."""
    m = float(score_map.max())
    if m <= 0:
        return None
    ys, xs = np.nonzero(score_map >= thr_frac * m)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def box_iou_xyxy(a, b) -> float:
    """IoU de deux boîtes (x1, y1, x2, y2)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0
