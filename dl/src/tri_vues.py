"""Tri des vues : une photo d'annonce est-elle **exploitable** pour juger la carrosserie ?

Raison d'être, mesurée et non supposée : sur 800 photos d'annonce tirées au hasard, **27 %** ont été
jugées non exploitables à l'annotation (intérieur, compartiment moteur, document, montage, capture
d'écran de smartphone, photo floue ou de nuit), et **19 %** avaient déjà été écartées en amont par
`detect.py` faute de véhicule visible. Au total, environ quatre photos sur dix ne portent aucune
information sur l'état de la carrosserie.

La fiche 0003 conclut : « le tri des vues est un prérequis de la chaîne produit, pas une option ».
Ce module l'implémente.

Étiquettes : **aucune annotation nouvelle**. Elles existent déjà dans `verdicts.csv` — le verdict
`jeter` contre les verdicts `intact` / `abime`.

Limite à déclarer : la classe `jeter` agrège des causes hétérogènes (un document et une photo floue
n'ont rien en commun visuellement) et intègre un critère subjectif — l'instruction d'annotation
disait « dans le doute, jeter ». Le modèle apprend donc aussi à imiter l'hésitation d'**un seul**
annotateur. On promet « exploitable / non exploitable », jamais le sous-type : avec 177 exemples
répartis sur cinq causes, il resterait ~35 exemples par cause.
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

from binaire import build_model_binaire, transforms_binaire


class PhotosTri(Dataset):
    """Photos annotées. Cible : 1.0 = **non exploitable** (verdict `jeter`), 0.0 = exploitable."""

    def __init__(self, df: pd.DataFrame, dossier_img, transform):
        self.fichiers = list(df["fichier"])
        self.cibles = [1.0 if v == "jeter" else 0.0 for v in df["verdict"]]
        self.dossier = Path(dossier_img)
        self.transform = transform

    def __len__(self):
        return len(self.fichiers)

    def __getitem__(self, i):
        img = Image.open(self.dossier / self.fichiers[i]).convert("RGB")
        return self.transform(img), torch.tensor(self.cibles[i])


def _entrainer_bloc(model, train_ds, val_ds, device, epochs, lr, batch_size, pos_weight):
    """Entraîne sur un bloc, renvoie les probabilités de validation. Budget d'époques fixe.

    Aucune sélection d'époque sur la validation : ce bloc sert à **mesurer**, pas à régler.
    """
    model = model.to(device)
    crit = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([float(pos_weight)], dtype=torch.float32, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    charge = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    for _ in range(epochs):
        model.train()
        for x, y in charge:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(model(x).squeeze(1), y).backward()
            opt.step()

    model.eval()
    probas = []
    with torch.no_grad():
        for x, _ in DataLoader(val_ds, batch_size=batch_size, num_workers=0):
            probas.append(torch.sigmoid(model(x.to(device)).squeeze(1)).cpu().numpy())
    return np.concatenate(probas)


def validation_croisee(df, dossier_img, device, n_blocs=5, epochs=8, lr=1e-4, batch_size=32,
                       taille=224, seed=0, verbose=True):
    """Validation croisée stratifiée. Renvoie `(probas_hors_bloc, y)`.

    Entrée à 224 px et non 384 : reconnaître un document, un habitacle ou une photo floue est une
    décision **globale** sur l'image, pas une affaire de détail fin — contrairement à la détection
    de dégât, où la résolution avait apporté un gain mesuré (fiche 0004).
    """
    df = df.reset_index(drop=True)
    y = (df["verdict"] == "jeter").astype(int).values
    probas = np.zeros(len(df))
    tf_train = transforms_binaire(taille, augmenter=True)
    tf_eval = transforms_binaire(taille, augmenter=False)

    for bloc, (i_tr, i_va) in enumerate(
            StratifiedKFold(n_splits=n_blocs, shuffle=True, random_state=seed).split(df, y), 1):
        train_ds = PhotosTri(df.iloc[i_tr], dossier_img, tf_train)
        val_ds = PhotosTri(df.iloc[i_va], dossier_img, tf_eval)
        pos = y[i_tr].sum()
        torch.manual_seed(seed + bloc)
        probas[i_va] = _entrainer_bloc(build_model_binaire("imagenet"), train_ds, val_ds, device,
                                       epochs, lr, batch_size, (len(i_tr) - pos) / max(pos, 1))
        if verbose:
            print(f"  bloc {bloc}/{n_blocs} — aire ROC {roc_auc_score(y[i_va], probas[i_va]):.3f}")
    return probas, y


def evaluer(y, probas, cible_rappel: float = 0.8) -> dict:
    """Mesures du tri, plus le seuil atteignant un rappel visé.

    Le compromis n'est pas symétrique ici. **Laisser passer** une photo inexploitable pollue le
    score d'état de l'annonce ; **écarter** à tort une bonne photo ne coûte qu'un peu d'information,
    puisqu'une annonce en compte plusieurs. On vise donc un rappel élevé sur la classe « à jeter »,
    quitte à sacrifier de la précision.
    """
    y, probas = np.asarray(y), np.asarray(probas)
    p, r, seuils = precision_recall_curve(y, probas)
    ok = np.where(r[:-1] >= cible_rappel)[0]
    k = int(ok[np.argmax(p[:-1][ok])]) if len(ok) else int(np.argmax(r[:-1]))
    return {
        "aire ROC": round(float(roc_auc_score(y, probas)), 3),
        "précision moyenne": round(float(average_precision_score(y, probas)), 3),
        "prévalence (plancher)": round(float(y.mean()), 3),
        f"seuil à rappel ≥ {cible_rappel}": round(float(seuils[k]), 3),
        "précision à ce seuil": round(float(p[k]), 3),
        "rappel à ce seuil": round(float(r[k]), 3),
        "part de photos écartées": round(float((probas >= seuils[k]).mean()), 3),
    }
