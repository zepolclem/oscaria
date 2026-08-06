"""Fabrication de tuiles « carrosserie intacte » — négatifs à greffer dans CarDD.

Pourquoi : CarDD ne contient aucune voiture intacte, donc le modèle multi-étiquette n'a jamais vu
la cible `[0,0,0,0,0,0]` et prédit un dégât sur tout (aire ROC 0,509 sur photos d'annonce, fiche
0002). L'architecture sait pourtant dire « rien » — il manque seulement des exemples.

Source : photos leboncoin **déclarées intactes** (`undamaged` + `excellent_condition`), fiables à
~98 % (fiche 0003). Pas telles quelles — plein cadre, mauvais domaine — mais découpées en **gros
plans de carrosserie** (détourage puis tuiles dans la boîte véhicule) pour ressembler au cadrage
CarDD. Un dataset externe aurait un style trop différent : le réseau apprendrait la source, pas le
dégât (biais de source).

Garde-fou anti-fuite : les annonces du lot d'annotation (`annot/manifeste.csv`) sont **exclues** —
le jeu des 471 verdicts reste un jeu d'évaluation vierge. Découpage train/val **par annonce**.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
from PIL import Image

from annoter import DOSSIERS_INTACT
from detect import detecter, load_detector, recadrer


def echantillonner_annonces(racine_images, exclure_ad_ids, n: int = 700,
                            seed: int = 0) -> pd.DataFrame:
    """Tire `n` annonces déclarées intactes, hors annonces exclues ; une photo par annonce.

    La photo est tirée au hasard, pas en première position : la photo 1 est presque toujours le
    plan de trois-quarts valorisant, ce qui biaiserait la diversité des vues.
    """
    rng = random.Random(seed)
    exclure = set(str(a) for a in exclure_ad_ids)
    racine = Path(racine_images)
    candidates = []
    for dossier in DOSSIERS_INTACT:
        candidates += [(p, dossier) for p in sorted((racine / dossier).iterdir())
                       if p.is_dir() and p.name not in exclure]
    lignes = []
    for annonce, dossier in rng.sample(candidates, min(n, len(candidates))):
        photos = sorted(annonce.glob("*.jpg"))
        if photos:
            lignes.append({"chemin": str(rng.choice(photos)), "ad_id": annonce.name,
                           "etat_declare": dossier})
    return pd.DataFrame(lignes)


def preparer_lot_grille(df: pd.DataFrame, dossier_sortie, device, cote_max: int = 900,
                        score_min: float = 0.5) -> pd.DataFrame:
    """Détoure les photos et prépare le lot pour l'annotation par grille cliquable.

    Différence avec `generer_tuiles` : ici **aucune tuile n'est découpée automatiquement**. On
    produit les détourages, l'humain choisit ensuite les cases qui montrent de la tôle saine.

    La grille se pose sur le **détourage** et non sur la photo brute : une case représente alors
    toujours la même fraction de véhicule, quelle que soit la distance de prise de vue. C'est ce
    qui corrige le mélange d'échelles du tirage aléatoire.

    Les photos sans véhicule détecté sont **écartées** : une grille sans voiture n'a rien à offrir.
    """
    from detect import detecter, load_detector  # import local : torch chargé au besoin

    sortie = Path(dossier_sortie)
    (sortie / "detourages").mkdir(parents=True, exist_ok=True)

    detecteur, categories = load_detector(device)
    res = detecter(detecteur, categories, device, list(df["chemin"]), score_min=score_min)

    lignes = []
    for ligne, r in zip(df.itertuples(), res):
        if r["boite"] is None:
            continue
        crop = recadrer(Image.open(ligne.chemin).convert("RGB"), r["boite"])
        if max(crop.size) > cote_max:      # borne haute : confort d'affichage, pas de perte utile
            ratio = cote_max / max(crop.size)
            crop = crop.resize((round(crop.width * ratio), round(crop.height * ratio)),
                               Image.LANCZOS)
        nom = f"{ligne.ad_id}.jpg"
        crop.save(sortie / "detourages" / nom, quality=92)
        lignes.append({"fichier": nom, "ad_id": ligne.ad_id, "etat_declare": ligne.etat_declare,
                       "photo_origine": ligne.chemin, "largeur": crop.width,
                       "hauteur": crop.height, "fraction_cadre": r["fraction_cadre"]})

    manifeste = pd.DataFrame(lignes)
    manifeste.to_csv(sortie / "manifeste_grille.csv", index=False)
    return manifeste


def decouper_cases(manifeste: pd.DataFrame, verdicts: pd.DataFrame, dossier_lot,
                   cols: int = 3, lignes_grille: int = 2, frac_val: float = 0.2,
                   seed: int = 0) -> pd.DataFrame:
    """Découpe les cases cliquées aux coordonnées de la grille. Découpage train/val par annonce.

    Les coordonnées ne sont jamais figées côté page : elles se recalculent ici depuis l'index de
    case et la taille du détourage, ce qui permet de changer la grille sans réannoter.
    """
    rng = random.Random(seed)
    lot = Path(dossier_lot)
    for s in ("train", "val"):
        (lot / s).mkdir(parents=True, exist_ok=True)

    ads = sorted(set(verdicts["fichier"]))
    rng.shuffle(ads)
    n_val = int(len(ads) * frac_val)
    split_par_fichier = {f: ("val" if i < n_val else "train") for i, f in enumerate(ads)}

    infos = manifeste.set_index("fichier")
    sorties = []
    for ligne in verdicts.itertuples():
        # case = -1 : photo examinée, aucune case utilisable — rien à découper
        if int(ligne.case) < 0 or ligne.fichier not in infos.index:
            continue
        meta = infos.loc[ligne.fichier]
        W, H = int(meta.largeur), int(meta.hauteur)
        pas_x, pas_y = W / cols, H / lignes_grille
        col, rang = int(ligne.case) % cols, int(ligne.case) // cols
        boite = (round(col * pas_x), round(rang * pas_y),
                 round((col + 1) * pas_x), round((rang + 1) * pas_y))

        split = split_par_fichier[ligne.fichier]
        nom = f"{Path(ligne.fichier).stem}_{int(ligne.case)}.jpg"
        img = Image.open(lot / "detourages" / ligne.fichier).convert("RGB").crop(boite)
        img.save(lot / split / nom, quality=92)
        sorties.append({"fichier": f"{split}/{nom}", "split": split,
                        "ad_id": str(meta.ad_id), "case": int(ligne.case),
                        "cote_x": boite[2] - boite[0], "cote_y": boite[3] - boite[1]})

    res = pd.DataFrame(sorties)
    res.to_csv(lot / "manifeste_cases.csv", index=False)
    return res


def degrader_comme_tuiles(img: Image.Image, cote_effectif: int = 180) -> Image.Image:
    """Ramène une image à la résolution effective d'une tuile, puis la remonte.

    Sert à **apparier la netteté** entre les deux classes. Sans cela, les positifs CarDD (~1000 px
    réduits vers 224, donc nets) et les négatifs leboncoin (~165 px agrandis vers 224, donc flous)
    diffèrent par un indice que le réseau peut apprendre à la place du dégât : « flou = intact ».
    """
    cote = max(img.size)
    if cote <= cote_effectif:
        return img
    ratio = cote_effectif / cote
    petite = img.resize((max(1, round(img.width * ratio)), max(1, round(img.height * ratio))),
                        Image.LANCZOS)
    return petite.resize(img.size, Image.BICUBIC)


def generer_tuiles(df: pd.DataFrame, dossier_sortie, device, tailles=(224, 448),
                   tuiles_par_photo: int = 3, frac_val: float = 0.2, cote_min: int = 96,
                   seed: int = 0) -> pd.DataFrame:
    """Détoure chaque photo puis découpe des tuiles carrées aléatoires DANS la boîte véhicule.

    - repli sur l'image entière si aucun véhicule détecté (photo conservée, pas écartée) ;
    - tuile plus grande que la boîte → rabattue à la boîte ; tuile < `cote_min` px → abandonnée
      (trop peu de matière pour ressembler à un gros plan CarDD) ;
    - découpage train/val **par annonce** (`frac_val`), jamais par tuile : deux tuiles d'une même
      voiture ne se retrouvent pas de part et d'autre.

    Écrit `<dossier_sortie>/{train,val}/<ad_id>_<k>.jpg` + `manifeste_negatifs.csv`. Renvoie le
    manifeste.
    """
    rng = random.Random(seed)
    sortie = Path(dossier_sortie)
    for s in ("train", "val"):
        (sortie / s).mkdir(parents=True, exist_ok=True)

    detecteur, categories = load_detector(device)
    res = detecter(detecteur, categories, device, list(df["chemin"]), score_min=0.5)

    # répartition train/val par annonce, avant de générer la moindre tuile
    ads = list(df["ad_id"])
    rng.shuffle(ads)
    n_val = int(len(ads) * frac_val)
    split_par_ad = {a: ("val" if i < n_val else "train") for i, a in enumerate(ads)}

    lignes = []
    for ligne, r in zip(df.itertuples(), res):
        img = Image.open(ligne.chemin).convert("RGB")
        if r["boite"] is not None:
            zone = recadrer(img, r["boite"], marge=0.0)
        else:
            zone = img
        W, H = zone.size
        split = split_par_ad[ligne.ad_id]
        for k in range(tuiles_par_photo):
            cote = min(rng.choice(tailles), W, H)
            if cote < cote_min:
                continue
            x = rng.randint(0, W - cote)
            y = rng.randint(0, H - cote)
            nom = f"{ligne.ad_id}_{k}.jpg"
            zone.crop((x, y, x + cote, y + cote)).save(sortie / split / nom, quality=90)
            lignes.append({"fichier": f"{split}/{nom}", "split": split, "ad_id": ligne.ad_id,
                           "photo_origine": ligne.chemin, "cote": cote,
                           "vehicule_detecte": r["boite"] is not None})
    manifeste = pd.DataFrame(lignes)
    manifeste.to_csv(sortie / "manifeste_negatifs.csv", index=False)
    return manifeste
