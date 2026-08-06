"""Expériences du pilier État, domaine « photos d'annonce » — rejouables en ligne de commande.

    python dl/src/experiences_binaire.py transfert   # CarDD transfère-t-il ? (~2 min)
    python dl/src/experiences_binaire.py pilote      # apprenable ? courbe d'apprentissage (~12 min)
    python dl/src/experiences_binaire.py leviers     # résolution x détourage (~20 min)

Les résultats sont écrits en CSV dans `dl/data/leboncoin-private/annot/`, que le carnet
`dl/notebooks/leboncoin-private/01_annotation_et_pilote.ipynb` relit et commente.

Le contenu de `dl/data/` est exclu de git (convention du dépôt) : les chiffres qui font foi sont
ceux affichés dans le carnet exécuté et repris dans `docs/decisions/`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))

import binaire  # noqa: E402
from detect import detecter, load_detector, recadrer  # noqa: E402
from device import get_device  # noqa: E402

ANNOT = Path(__file__).resolve().parents[1] / "data" / "leboncoin-private" / "annot"
IMG, CROP = ANNOT / "img", ANNOT / "img_crop"
CK_CARDD = Path(__file__).resolve().parents[1] / "models" / "cardd_baseline.pt"


def charger_annotations(jugeables_seulement: bool = True) -> pd.DataFrame:
    """Manifeste (état déclaré, annonce) recollé aux verdicts humains (annotation aveugle)."""
    man = pd.read_csv(ANNOT / "manifeste.csv", dtype={"id": str})
    man["fichier"] = man["id"] + ".jpg"
    df = man.merge(pd.read_csv(ANNOT / "verdicts.csv"), on="fichier")
    if jugeables_seulement:
        df = df[df["verdict"].isin(["intact", "abime"])]
    return df.reset_index(drop=True)


def mesurer_transfert_cardd() -> pd.DataFrame:
    """Le modèle CarDD sait-il **ordonner** les photos d'annonce selon le verdict humain ?

    On mesure l'aire sous la courbe ROC, qui ne dépend d'aucun seuil : si elle vaut 0,5, aucune
    calibration ne pourra rien en tirer, et la question des seuils est close.
    """
    from sklearn.metrics import roc_auc_score

    from infer import load_model, predict

    df = charger_annotations()
    device = get_device()
    modele, classes, device = load_model(CK_CARDD, device)
    probas = pd.DataFrame([
        predict(modele, classes, device, Image.open(IMG / f).convert("RGB")) for f in df.fichier
    ])
    y = (df.verdict == "abime").astype(int)
    lignes = [{"signal": "max des 6 probabilités", "aire ROC": roc_auc_score(y, probas.max(axis=1))}]
    lignes += [{"signal": c, "aire ROC": roc_auc_score(y, probas[c])} for c in classes]
    res = pd.DataFrame(lignes).round(3)
    res.to_csv(ANNOT / "transfert_cardd.csv", index=False)
    probas.assign(verdict=df.verdict).to_csv(ANNOT / "probas_cardd.csv", index=False)
    return res


def preparer_detourages() -> int:
    """Recadre chaque photo annotée sur le véhicule détecté. Repli sur l'image entière sinon.

    Le repli est délibéré : écarter les photos sans détection changerait la composition du jeu
    d'une configuration à l'autre, et la comparaison ne serait plus à jeu constant.
    """
    df = charger_annotations()
    if CROP.exists() and len(list(CROP.glob("*.jpg"))) >= len(df):
        return 0
    CROP.mkdir(parents=True, exist_ok=True)
    device = get_device()
    detecteur, cats = load_detector(device)
    res = detecter(detecteur, cats, device, [str(IMG / f) for f in df.fichier], score_min=0.5)
    sans_boite = 0
    for fichier, r in zip(df.fichier, res):
        img = Image.open(IMG / fichier).convert("RGB")
        if r["boite"] is None:
            sans_boite += 1
        else:
            img = recadrer(img, r["boite"])
        img.save(CROP / fichier, quality=88)
    pd.DataFrame(res).to_csv(ANNOT / "boites_471.csv", index=False)
    return sans_boite


def pilote() -> pd.DataFrame:
    """Deux initialisations + courbe d'apprentissage, à 224 px sur photo entière."""
    df, device, resultats = charger_annotations(), get_device(), []
    for init in ("imagenet", "cardd"):
        p, y = binaire.validation_croisee(df, IMG, device, init=init,
                                          checkpoint_cardd=CK_CARDD, epochs=8, lr=1e-4)
        resultats.append(binaire.evaluer_binaire(y, p, f"init {init}"))
    for frac in (0.25, 0.5, 0.75):
        p, y = binaire.validation_croisee(df, IMG, device, init="imagenet", epochs=8,
                                          lr=1e-4, fraction=frac, verbose=False)
        resultats.append(binaire.evaluer_binaire(y, p, f"imagenet — {int(frac*100)} % des données"))
    res = pd.DataFrame(resultats)
    res.to_csv(ANNOT / "pilote_resultats.csv", index=False)
    return res


def leviers() -> pd.DataFrame:
    """Résolution et détourage, chacun isolé puis combinés — mêmes photos, même découpage."""
    preparer_detourages()
    df, device, resultats = charger_annotations(), get_device(), []
    configs = [("224 px, photo entière", IMG, 224), ("384 px, photo entière", IMG, 384),
               ("224 px, détourée", CROP, 224), ("384 px, détourée", CROP, 384)]
    for nom, dossier, taille in configs:
        t0 = time.perf_counter()
        p, y = binaire.validation_croisee(df, dossier, device, init="imagenet", epochs=8,
                                          lr=1e-4, taille=taille, verbose=False)
        ligne = binaire.evaluer_binaire(y, p, nom)
        ligne["durée (s)"] = round(time.perf_counter() - t0)
        resultats.append(ligne)
        suffixe = "crop" if dossier == CROP else "full"
        pd.DataFrame({"fichier": df.fichier, "verdict": df.verdict, "proba": p}).to_csv(
            ANNOT / f"probas_{taille}_{suffixe}.csv", index=False)
    res = pd.DataFrame(resultats)
    res.to_csv(ANNOT / "leviers_resultats.csv", index=False)
    return res


if __name__ == "__main__":
    commandes = {"transfert": mesurer_transfert_cardd, "pilote": pilote, "leviers": leviers}
    nom = sys.argv[1] if len(sys.argv) > 1 else ""
    if nom not in commandes:
        sys.exit(f"usage : python {Path(__file__).name} {{{'|'.join(commandes)}}}")
    print(commandes[nom]().to_string(index=False))
