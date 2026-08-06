"""Classifieur binaire « carrosserie abîmée / intacte » sur photos d'annonces.

Pourquoi ce module existe : le modèle CarDD (`cardd_baseline.pt`, macro-F1 0,815 sur gros plans)
ne transfère pas sur des photos d'annonce — aire sous la courbe ROC mesurée à **0,509** sur 471
photos annotées à la main, soit un classement aléatoire. Le décalage de domaine est trop grand pour
être rattrapé par un seuil. Il faut donc entraîner sur des photos du bon domaine.

Étiquettes : verdicts humains (`intact` / `abime`), annotation aveugle — l'état déclaré par le
vendeur n'a pas été montré à l'annotateur. Mesure associée : l'étiquette déclarée est fiable à 98 %
côté intact, mais seulement 40 % côté abîmé (une annonce « abîmée » contient surtout des photos de
faces intactes), ce qui interdit de l'utiliser telle quelle comme cible.

Une sortie unique (logit) plutôt que deux : en binaire, deux sorties softmax sont redondantes, et
un seul logit permet `pos_weight` pour compenser le déséquilibre 3,5 pour 1.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def transforms_binaire(taille: int = 224, augmenter: bool = False):
    """Prétraitement. `augmenter` n'est activé que sur les blocs d'entraînement.

    Retournement horizontal : sûr ici, une aile froissée reste un dégât en miroir. Variation
    colorimétrique légère : les photos d'annonce vont du plein soleil au sous-sol de parking.
    Avec 471 images, l'augmentation n'est pas un levier de confort — c'est ce qui limite la
    mémorisation.
    """
    etapes = [transforms.Resize((taille, taille))]
    if augmenter:
        etapes += [transforms.RandomHorizontalFlip(),
                   transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2)]
    etapes += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(etapes)


class PhotosAnnotees(Dataset):
    """Photos anonymisées + verdict humain. Cible : 1.0 = abîmé, 0.0 = intact."""

    def __init__(self, df: pd.DataFrame, dossier_img, transform):
        self.fichiers = list(df["fichier"])
        self.cibles = [1.0 if v == "abime" else 0.0 for v in df["verdict"]]
        self.dossier = Path(dossier_img)
        self.transform = transform

    def __len__(self):
        return len(self.fichiers)

    def __getitem__(self, i):
        img = Image.open(self.dossier / self.fichiers[i]).convert("RGB")
        return self.transform(img), torch.tensor(self.cibles[i])


def build_model_binaire(init: str = "imagenet", checkpoint_cardd=None):
    """ResNet18 à une sortie.

    - `init="imagenet"` : poids ImageNet, tête neuve. La référence.
    - `init="cardd"` : poids affinés sur CarDD (le tronc a vu des dégâts de carrosserie), tête
      neuve. Teste si l'entraînement CarDD a laissé quelque chose d'utilisable malgré son
      classement aléatoire sur ce domaine — les deux questions sont distinctes.
    """
    if init == "imagenet":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif init == "cardd":
        model = models.resnet18(weights=None)
        ck = torch.load(checkpoint_cardd, map_location="cpu", weights_only=True)
        model.fc = nn.Linear(model.fc.in_features, len(ck["classes"]))
        model.load_state_dict(ck["state_dict"])
    else:
        raise ValueError(f"init inconnu : {init}")
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def evaluer_binaire(y, probas, nom: str = "") -> dict:
    """Aire sous la courbe ROC, précision moyenne, et meilleur compromis précision/rappel.

    La **précision moyenne** (aire sous la courbe précision/rappel) est la mesure honnête en
    déséquilibre : elle ne regarde que la classe rare. Son plancher n'est pas 0,5 mais la
    prévalence — ici 105/471 = 0,223.
    """
    y, probas = np.asarray(y), np.asarray(probas)
    p, r, seuils = precision_recall_curve(y, probas)
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    k = int(np.argmax(f1))
    return {
        "modèle": nom,
        "aire ROC": round(float(roc_auc_score(y, probas)), 3),
        "précision moyenne": round(float(average_precision_score(y, probas)), 3),
        "prévalence (plancher)": round(float(y.mean()), 3),
        "meilleur F1": round(float(f1[k]), 3),
        "précision à ce seuil": round(float(p[k]), 3),
        "rappel à ce seuil": round(float(r[k]), 3),
        "seuil": round(float(seuils[min(k, len(seuils) - 1)]), 3),
    }


def _entrainer_un_bloc(model, train_ds, val_ds, device, epochs, lr, batch_size, pos_weight):
    """Entraîne sur un bloc et renvoie les probabilités de validation de la dernière époque.

    Pas de sélection d'époque sur la validation ici : ce bloc de validation sert à **mesurer**.
    Choisir l'époque dessus le transformerait en jeu de réglage et gonflerait le résultat. Le
    budget d'époques est donc fixe et identique pour toutes les configurations.
    """
    model = model.to(device)
    # float32 explicite : MPS refuse le float64, que numpy produit par défaut.
    crit = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([float(pos_weight)], dtype=torch.float32, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    charge_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    charge_val = DataLoader(val_ds, batch_size=batch_size, num_workers=0)

    for _ in range(epochs):
        model.train()
        for x, y in charge_train:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(model(x).squeeze(1), y).backward()
            opt.step()

    model.eval()
    probas = []
    with torch.no_grad():
        for x, _ in charge_val:
            probas.append(torch.sigmoid(model(x.to(device)).squeeze(1)).cpu().numpy())
    return np.concatenate(probas)


def validation_croisee(df, dossier_img, device, init="imagenet", checkpoint_cardd=None,
                       n_blocs=5, epochs=8, lr=1e-4, batch_size=32, taille=224,
                       augmenter=True, fraction=1.0, seed=0, verbose=True):
    """Validation croisée stratifiée : chaque photo est prédite par un modèle qui ne l'a pas vue.

    `fraction` < 1 réduit la taille du jeu d'entraînement de chaque bloc (la validation reste
    entière) : en faisant varier ce paramètre on trace la **courbe d'apprentissage**, qui dit si
    annoter davantage vaut le coup ou si la performance plafonne déjà.

    Renvoie `(probas_hors_bloc, y)` — chaque probabilité vient du bloc où la photo était en
    validation, donc l'évaluation globale est honnête.
    """
    df = df.reset_index(drop=True)
    y = (df["verdict"] == "abime").astype(int).values
    probas = np.zeros(len(df))
    tf_train = transforms_binaire(taille, augmenter)   # augmentation : entraînement seulement
    tf_eval = transforms_binaire(taille, False)        # évaluation toujours déterministe

    skf = StratifiedKFold(n_splits=n_blocs, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)
    for bloc, (i_train, i_val) in enumerate(skf.split(df, y), 1):
        if fraction < 1.0:
            i_train = rng.choice(i_train, size=max(2, int(len(i_train) * fraction)),
                                 replace=False)
        train_ds = PhotosAnnotees(df.iloc[i_train], dossier_img, tf_train)
        val_ds = PhotosAnnotees(df.iloc[i_val], dossier_img, tf_eval)
        pos = y[i_train].sum()
        pos_weight = (len(i_train) - pos) / max(pos, 1)

        torch.manual_seed(seed + bloc)
        model = build_model_binaire(init, checkpoint_cardd)
        probas[i_val] = _entrainer_un_bloc(model, train_ds, val_ds, device,
                                           epochs, lr, batch_size, pos_weight)
        if verbose:
            print(f"  bloc {bloc}/{n_blocs} — {len(i_train)} entraînement / {len(i_val)} "
                  f"validation — aire ROC du bloc : {roc_auc_score(y[i_val], probas[i_val]):.3f}")
    return probas, y
