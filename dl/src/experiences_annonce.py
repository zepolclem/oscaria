"""Chaîne « cohérence déclaration ↔ photos » — expériences rejouables en ligne de commande.

    python dl/src/experiences_annonce.py tri        # tri des vues, validation croisée (~5 min)
    python dl/src/experiences_annonce.py preparer   # lot d'évaluation par annonce (~50 min)
    python dl/src/experiences_annonce.py evaluer    # chaîne complète + agrégation (~10 min)

Résultats en CSV dans `dl/data/leboncoin-private/`, relus et commentés par le carnet
`dl/notebooks/leboncoin-private/03_chaine_coherence.ipynb`. Le contenu de `dl/data/` étant exclu de
git, les chiffres qui font foi sont ceux du carnet exécuté et de `docs/decisions/`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

import annonce  # noqa: E402
import tri_vues  # noqa: E402
from binaire import PhotosAnnotees, build_model_binaire, transforms_binaire  # noqa: E402
from device import get_device  # noqa: E402

DL = Path(__file__).resolve().parents[1]
ANNOT = DL / "data" / "leboncoin-private" / "annot"
LOT = DL / "data" / "leboncoin-private" / "annonces"
RACINE = DL / "data" / "leboncoin-private" / "raw" / "images"
CK_TRI = DL / "models" / "tri_vues.pt"


def _verdicts(jugeables_seulement=True) -> pd.DataFrame:
    man = pd.read_csv(ANNOT / "manifeste.csv", dtype={"id": str})
    man["fichier"] = man["id"] + ".jpg"
    df = man.merge(pd.read_csv(ANNOT / "verdicts.csv"), on="fichier")
    if jugeables_seulement:
        df = df[df["verdict"].isin(["intact", "abime"])]
    return df.reset_index(drop=True)


def _entrainer(model, ds, device, y, epochs=8, lr=1e-4, batch_size=32):
    """Entraînement simple à budget fixe, perte pondérée par le déséquilibre."""
    model = model.to(device)
    crit = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
        [float((len(y) - y.sum()) / max(y.sum(), 1))], dtype=torch.float32, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for x, t in DataLoader(ds, batch_size=batch_size, shuffle=True):
            opt.zero_grad()
            crit(model(x.to(device)).squeeze(1), t.to(device)).backward()
            opt.step()
    return model.eval()


def tri() -> pd.DataFrame:
    """Tri des vues en validation croisée, puis modèle final sauvé pour la chaîne."""
    df = _verdicts(jugeables_seulement=False)
    device = get_device()
    probas, y = tri_vues.validation_croisee(df, ANNOT / "img", device, epochs=8)
    res = pd.DataFrame([tri_vues.evaluer(y, probas, cible_rappel=0.8)])
    res.to_csv(ANNOT / "tri_vues_resultats.csv", index=False)
    pd.DataFrame({"fichier": df.fichier, "verdict": df.verdict,
                  "proba_jeter": probas}).to_csv(ANNOT / "probas_tri_vues.csv", index=False)

    # modèle final : entraîné sur les 648, sans validation croisée — ici on applique, on ne mesure
    # plus. La mesure honnête est celle de la validation croisée ci-dessus.
    torch.manual_seed(0)
    ds = tri_vues.PhotosTri(df, ANNOT / "img", transforms_binaire(224, augmenter=True))
    m = _entrainer(build_model_binaire("imagenet"), ds, device, y)
    CK_TRI.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in m.state_dict().items()},
                "classes": ["jeter"], "seuils": 0.5,
                "config": {"arch": "resnet18", "taille": 224, "tache": "tri des vues"}}, CK_TRI)
    return res


def preparer(n_par_classe: int = 300, max_photos: int = 5) -> pd.DataFrame:
    """Échantillonne des annonces entières et prépare photos + détourages.

    Exclusion **ciblée** : seulement les annonces du lot d'annotation, qui ont servi à entraîner le
    binaire évalué ici. Les lots de négatifs CarDD ne sont pas exclus — ils ont alimenté un autre
    modèle. L'anti-fuite se raisonne par couple modèle-évaluation, pas par « donnée déjà touchée ».
    """
    exclure = set(pd.read_csv(ANNOT / "manifeste.csv", dtype={"ad_id": str})["ad_id"])
    df = annonce.echantillonner_par_annonce(RACINE, exclure, n_par_classe=n_par_classe,
                                            max_photos=max_photos, seed=0)
    return annonce.preparer_photos(df, LOT, get_device())


def evaluer() -> pd.DataFrame:
    """Chaîne complète : tri des vues, binaire d'état, agrégation par annonce."""
    device = get_device()
    man = pd.read_csv(LOT / "manifeste_annonces.csv", dtype={"ad_id": str})

    # binaire d'état, configuration gagnante de la fiche 0004 : 384 px sur détourage
    verdicts = _verdicts()
    y_tr = (verdicts.verdict == "abime").astype(int).values
    torch.manual_seed(0)
    ds = PhotosAnnotees(verdicts, ANNOT / "img_crop", transforms_binaire(384, augmenter=True))
    etat = _entrainer(build_model_binaire("imagenet"), ds, device, y_tr)

    ck = torch.load(CK_TRI, map_location=device, weights_only=True)
    vues = build_model_binaire("imagenet")
    vues.load_state_dict(ck["state_dict"])
    vues = vues.to(device).eval()

    tf384, tf224 = transforms_binaire(384, False), transforms_binaire(224, False)
    p_etat, p_jeter = [], []
    with torch.no_grad():
        for f in man.fichier:
            crop = tf384(Image.open(LOT / "crops" / f).convert("RGB")).unsqueeze(0).to(device)
            photo = tf224(Image.open(LOT / "photos" / f).convert("RGB")).unsqueeze(0).to(device)
            p_etat.append(float(torch.sigmoid(etat(crop).squeeze()).cpu()))
            p_jeter.append(float(torch.sigmoid(vues(photo).squeeze()).cpu()))
    man["proba"], man["proba_jeter"] = p_etat, p_jeter
    man.to_csv(LOT / "probas_photos.csv", index=False)

    verite = man.groupby("ad_id").classe_declaree.first()
    y = (verite == "abime").astype(int)
    y_photo = (man.classe_declaree == "abime").astype(int)
    lignes = [{"niveau": "photo (sans agrégation)", "filtre tri": "non", "règle": "—",
               "annonces": None, "aire ROC": round(roc_auc_score(y_photo, man.proba), 3),
               "précision moyenne": round(average_precision_score(y_photo, man.proba), 3)}]
    for filtre in (False, True):
        sous = man[man.proba_jeter < 0.5] if filtre else man
        for regle in ("max", "moyenne", "moyenne_top2"):
            sc = annonce.agreger(sous, regle).set_index("ad_id")["score_annonce"]
            yy = y.loc[sc.index]
            lignes.append({"niveau": "annonce", "filtre tri": "oui" if filtre else "non",
                           "règle": regle, "annonces": len(sc),
                           "aire ROC": round(roc_auc_score(yy, sc), 3),
                           "précision moyenne": round(average_precision_score(yy, sc), 3)})
    res = pd.DataFrame(lignes)
    res.to_csv(LOT / "resultats_agregation.csv", index=False)
    return res


if __name__ == "__main__":
    commandes = {"tri": tri, "preparer": preparer, "evaluer": evaluer}
    nom = sys.argv[1] if len(sys.argv) > 1 else ""
    if nom not in commandes:
        sys.exit(f"usage : python {Path(__file__).name} {{{'|'.join(commandes)}}}")
    t0 = time.perf_counter()
    print(commandes[nom]().to_string(index=False))
    print(f"\n({time.perf_counter() - t0:.0f}s)")
