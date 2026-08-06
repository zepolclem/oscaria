"""Entraînement et évaluation multi-étiquette sur CarDD — pilier État OscarIA.

Module paramétré : les leviers de la phase 2 (résolution, architecture, seuils par classe,
augmentation) se branchent sur ces mêmes fonctions, sans réécriture.

Décisions de la phase 1 (voir plan `docs/plans/`) :
- cible multi-étiquette 6 types de dégât, binaire « abîmé » dérivé à l'inférence ;
- découpages CarDD natifs, `test2017` jamais ouvert avant la phase 3 ;
- métrique arbitre = macro-F1, tableau par classe toujours produit ;
- budget fixe de 10 époques, poids de la meilleure époque sur validation.

Tout renvoie des **probabilités** ; la décision (seuil) est appliquée dans `evaluate`.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import models, transforms

from cardd import CLASSES

# Statistiques ImageNet : le backbone pré-entraîné attend cette normalisation.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int = 0) -> None:
    """Fixe les graines pour que comparer deux variantes ne mesure pas du bruit."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_transforms(taille: int = 224, augmenter: bool = False):
    """Prétraitement des images.

    `Resize((t, t))` **écrase le format d'origine** (1000x667 -> carré étiré) au lieu de
    recadrer au centre : un dégât d'aile ou de pare-chocs se trouve souvent au bord de la
    photo, et un recadrage central le supprimerait. On préfère déformer que perdre.

    `augmenter=True` (levier D de la phase 2) : retournement horizontal — sûr ici, un dégât
    reste un dégât en miroir — plus variation colorimétrique légère.
    """
    etapes = [transforms.Resize((taille, taille))]
    if augmenter:
        etapes += [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    etapes += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(etapes)


def build_model(arch: str = "resnet18", gele: bool = False, n_classes: int = len(CLASSES)):
    """Backbone pré-entraîné ImageNet + tête à `n_classes` sorties.

    Pas de fonction softmax : en multi-étiquette les 6 sorties sont indépendantes, la
    sigmoïde est appliquée plus tard (`predict_probas`).

    `gele=True` : seuls les poids de la tête sont entraînés (extraction de caractéristiques).
    `gele=False` : affinage complet, tous les poids bougent.
    """
    fabriques = {"resnet18": models.resnet18, "resnet50": models.resnet50}
    poids = {"resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
             "resnet50": models.ResNet50_Weights.IMAGENET1K_V2}
    if arch not in fabriques:
        raise ValueError(f"architecture inconnue : {arch} (attendu : {list(fabriques)})")

    model = fabriques[arch](weights=poids[arch])
    if gele:
        for p in model.parameters():
            p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, n_classes)  # tête neuve, toujours entraînable
    return model


def make_loader(dataset, batch_size: int = 32, shuffle: bool = False, num_workers: int = 4):
    """DataLoader. Si les workers bloquent dans un notebook, passer `num_workers=0`."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )


# --------------------------------------------------------------------------- #
# Planchers triviaux — la référence sans apprentissage
# --------------------------------------------------------------------------- #

def plancher_probas(train_ds, val_ds, strategie: str = "frequences", seed: int = 0):
    """Probabilités d'un modèle qui n'a rien appris, pour situer le plancher.

    - `"frequences"` : chaque classe est tirée au hasard (Bernoulli) à sa fréquence
      d'apparition dans l'entraînement. Rappel attendu ≈ fréquence, précision ≈ prévalence.
    - `"toujours_oui"` : tout est prédit positif. Rappel = 1, précision = prévalence.
      C'est le score qu'un réseau doit battre franchement, sinon il n'apprend rien.

    Renvoie `(probas, cibles)` au même format que `predict_probas`, pour passer dans le
    même `evaluate`.
    """
    y_train = train_ds.label_matrix().numpy()
    cibles = val_ds.label_matrix().numpy()
    freq = y_train.mean(axis=0)

    if strategie == "toujours_oui":
        probas = np.ones_like(cibles)
    elif strategie == "frequences":
        rng = np.random.default_rng(seed)
        probas = (rng.random(cibles.shape) < freq).astype(float)
    else:
        raise ValueError(f"stratégie inconnue : {strategie}")
    return probas, cibles


# --------------------------------------------------------------------------- #
# Entraînement
# --------------------------------------------------------------------------- #

@torch.no_grad()
def predict_probas(model, loader, device):
    """Probabilités par classe (fonction sigmoïde appliquée) et cibles. Matrices (N, 6)."""
    model.eval()
    probas, cibles = [], []
    for x, y in loader:
        logits = model(x.to(device))
        probas.append(torch.sigmoid(logits).cpu().numpy())
        cibles.append(y.numpy())
    return np.concatenate(probas), np.concatenate(cibles)


def train(
    model,
    train_ds,
    val_ds,
    device,
    epochs: int = 10,
    lr: float = 1e-4,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Boucle d'entraînement à budget fixe, sélection de l'époque sur la validation.

    Perte : entropie croisée binaire par classe, pondérée par `pos_weight` = négatifs /
    positifs (compense scratch ~7x tire flat, sinon les classes rares sont abandonnées).

    L'époque retenue est celle du **meilleur macro-F1 sur validation**, jamais la dernière :
    la fin de l'entraînement est souvent déjà en surapprentissage. Le test n'est pas touché.

    Renvoie : historique par époque, meilleurs poids, époque retenue, durée moyenne d'époque.
    """
    set_seed(seed)
    model = model.to(device)

    train_loader = make_loader(train_ds, batch_size, shuffle=True, num_workers=num_workers)
    val_loader = make_loader(val_ds, batch_size, shuffle=False, num_workers=num_workers)

    pos_weight = train_ds.pos_weight().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Seuls les paramètres entraînables : si le backbone est gelé, Adam ne voit que la tête.
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)

    historique, durees = [], []
    meilleur = {"macro_f1": -1.0, "epoch": None, "state_dict": None}

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        perte_cumul, n_lots = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            perte_cumul += loss.item()
            n_lots += 1
        duree = time.perf_counter() - t0
        durees.append(duree)

        probas, cibles = predict_probas(model, val_loader, device)
        scores = evaluate(probas, cibles)
        macro_f1 = float(scores.loc["macro", "F1"])
        historique.append({
            "epoch": epoch,
            "perte_train": perte_cumul / max(n_lots, 1),
            "macro_f1_val": macro_f1,
            "micro_f1_val": float(scores.loc["micro", "F1"]),
            "duree_s": duree,
        })

        if macro_f1 > meilleur["macro_f1"]:
            meilleur = {
                "macro_f1": macro_f1,
                "epoch": epoch,
                "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
        if verbose:
            print(f"époque {epoch:2d}/{epochs} | perte {historique[-1]['perte_train']:.4f} "
                  f"| macro-F1 val {macro_f1:.4f} | {duree:.1f} s")

    return {
        "historique": pd.DataFrame(historique),
        "meilleur_epoch": meilleur["epoch"],
        "meilleur_macro_f1": meilleur["macro_f1"],
        "best_state_dict": meilleur["state_dict"],
        "duree_epoque_s": float(np.mean(durees)),
        "model": model,
    }


# --------------------------------------------------------------------------- #
# Évaluation
# --------------------------------------------------------------------------- #

def evaluate(probas, cibles, seuils=0.5, classes=CLASSES) -> pd.DataFrame:
    """Precision / rappel / F1 par classe, plus macro et micro.

    `seuils` : un scalaire (même seuil partout) ou un vecteur de 6 valeurs (levier B de la
    phase 2 — un seuil par classe, réglé sur la validation).

    Jamais l'exactitude : sur `tire flat` (59 des 810 images de validation), ne jamais
    prédire la classe donne déjà 92,7 % d'exactitude sans rien avoir appris.
    """
    probas, cibles = np.asarray(probas), np.asarray(cibles)
    seuils = np.broadcast_to(np.asarray(seuils, dtype=float), (probas.shape[1],))
    preds = (probas >= seuils).astype(int)

    p, r, f1, support = precision_recall_fscore_support(
        cibles, preds, average=None, zero_division=0, labels=range(len(classes))
    )
    lignes = pd.DataFrame(
        {"precision": p, "rappel": r, "F1": f1, "support": support.astype(int)}, index=classes
    )

    agreges = {}
    for moyenne in ("macro", "micro"):
        pm, rm, fm, _ = precision_recall_fscore_support(
            cibles, preds, average=moyenne, zero_division=0
        )
        agreges[moyenne] = {"precision": pm, "rappel": rm, "F1": fm, "support": int(support.sum())}
    lignes = pd.concat([lignes, pd.DataFrame(agreges).T])
    lignes["support"] = lignes["support"].astype(int)
    return lignes.round(4)


def cooccurrence(probas, cibles, seuils=0.5, classes=CLASSES) -> pd.DataFrame:
    """Matrice de co-occurrence vérité (lignes) x prédiction (colonnes).

    Remplace la matrice de confusion, qui n'a pas de sens en multi-étiquette : une image
    peut porter plusieurs classes vraies et déclencher plusieurs prédictions. La diagonale
    compte les vrais positifs ; le hors-diagonale montre les confusions systématiques
    (typiquement `crack` prédit là où il y a `scratch`).
    """
    seuils = np.broadcast_to(np.asarray(seuils, dtype=float), (np.asarray(probas).shape[1],))
    preds = (np.asarray(probas) >= seuils).astype(int)
    m = np.asarray(cibles).T @ preds  # (6 vérités) x (6 prédictions)
    return pd.DataFrame(m.astype(int), index=[f"vrai:{c}" for c in classes],
                        columns=[f"préd:{c}" for c in classes])


def resume_ligne(nom: str, scores: pd.DataFrame, duree_epoque_s: float | None = None) -> dict:
    """Une ligne du tableau de synthèse : macro-F1, micro-F1, F1 par classe, coût."""
    ligne = {"modèle": nom,
             "macro-F1": scores.loc["macro", "F1"],
             "micro-F1": scores.loc["micro", "F1"]}
    ligne |= {f"F1 {c}": scores.loc[c, "F1"] for c in CLASSES}
    ligne["durée/époque (s)"] = round(duree_epoque_s, 1) if duree_epoque_s else None
    return ligne


# --------------------------------------------------------------------------- #
# Checkpoint
# --------------------------------------------------------------------------- #

def sauver_checkpoint(chemin, state_dict, config: dict, seuils=0.5, classes=CLASSES) -> Path:
    """Sauvegarde autoportante : poids + classes + seuils + configuration.

    L'inférence n'a rien à deviner (ni l'ordre des classes, ni la résolution d'entrée).
    """
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    # Types simples uniquement (pas de ndarray) : permet un rechargement en mode restreint.
    seuils = float(seuils) if np.isscalar(seuils) else [float(s) for s in np.asarray(seuils)]
    torch.save({"state_dict": state_dict, "classes": list(classes),
                "seuils": seuils, "config": config}, chemin)
    return chemin


def charger_checkpoint(chemin, device):
    """Recharge un checkpoint et reconstruit le modèle prêt à prédire.

    `weights_only=True` : le chargement n'exécute aucun code arbitraire (un fichier `.pt`
    est un pickle, donc un vecteur d'exécution si on le charge sans restriction).
    """
    ck = torch.load(chemin, map_location=device, weights_only=True)
    model = build_model(arch=ck["config"].get("arch", "resnet18"), n_classes=len(ck["classes"]))
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck
