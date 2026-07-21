"""Inférence multi-label CarDD — charge le modèle sauvé et prédit sur des images.

Réutilisable par les notebooks ET, plus tard, l'API produit (step 3/4).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

from device import get_device

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_model(ckpt_path: str | Path, device=None):
    """Recharge le checkpoint (`state_dict` + `classes`) et renvoie (model, classes, device)."""
    device = device or get_device()
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    classes = ck["classes"]
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(ck["state_dict"])
    model.eval().to(device)
    return model, classes, device


@torch.no_grad()
def predict(model, classes, device, img: Image.Image) -> dict[str, float]:
    """Probabilité (sigmoïde) de présence de chaque type de dégât sur l'image."""
    x = _TF(img.convert("RGB")).unsqueeze(0).to(device)
    probs = torch.sigmoid(model(x))[0].cpu()
    return {c: float(p) for c, p in zip(classes, probs)}
