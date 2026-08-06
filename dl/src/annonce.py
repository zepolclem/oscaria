"""Pilier État au niveau **annonce** : cohérence entre l'état déclaré et ce que montrent les photos.

Pourquoi ce niveau. La fiche 0003 a mesuré que l'état déclaré par le vendeur est « vrai au niveau de
l'annonce et faux au niveau de trois photos sur quatre » : une annonce `damaged` contient trois ou
quatre photos dont une seule cadre le dégât. Évaluer photo par photo revient donc à se battre contre
une étiquette qui n'a jamais prétendu décrire une photo.

En agrégeant les photos d'une même annonce, on remet l'étiquette à la granularité pour laquelle elle
a été écrite — et le jeu évaluable passe de 471 photos annotées à la main à ~2 140 annonces
étiquetées gratuitement.

Promesse tenue en conséquence : le modèle ne dit pas « cette voiture est abîmée » (0,61 de précision
au meilleur seuil, fiche 0004 — quatre alertes sur dix seraient fausses), mais « les photos de cette
annonce ne soutiennent pas l'état déclaré ». Positionnement de tiers neutre, et barre à la hauteur
de ce qui est réellement mesuré.

`not_drivable` est **exclu** : 73 % de ses photos montrent une carrosserie intacte (panne mécanique).
L'étiquette est juste, elle ne parle pas de ce que le modèle regarde.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

DOSSIERS_INTACT = ["undamaged", "excellent_condition"]
DOSSIERS_ABIME = ["damaged"]          # `not_drivable` volontairement écarté, voir en-tête


def echantillonner_par_annonce(racine_images, exclure_ad_ids, n_par_classe: int = 300,
                               max_photos: int = 5, seed: int = 0) -> pd.DataFrame:
    """Tire des annonces entières (toutes leurs photos, plafonnées), pas des photos isolées.

    C'est la différence avec `negatifs.echantillonner_annonces` : ici l'unité d'échantillonnage
    **et** d'évaluation est l'annonce, donc on garde ses photos ensemble. Le plafond `max_photos`
    évite qu'une annonce à 20 photos pèse autant que six annonces ordinaires.
    """
    rng = random.Random(seed)
    exclure = {str(a) for a in exclure_ad_ids}
    racine = Path(racine_images)

    lignes = []
    for classe, dossiers in (("intact", DOSSIERS_INTACT), ("abime", DOSSIERS_ABIME)):
        annonces = []
        for dossier in dossiers:
            annonces += [(p, dossier) for p in sorted((racine / dossier).iterdir())
                         if p.is_dir() and p.name not in exclure]
        for annonce, dossier in rng.sample(annonces, min(n_par_classe, len(annonces))):
            photos = sorted(annonce.glob("*.jpg"))
            for rang, photo in enumerate(photos[:max_photos]):
                lignes.append({"chemin": str(photo), "ad_id": annonce.name, "rang": rang,
                               "etat_declare": dossier, "classe_declaree": classe})
    return pd.DataFrame(lignes)


def preparer_photos(df: pd.DataFrame, dossier_sortie, device, score_min: float = 0.5,
                    cote_max: int = 900) -> pd.DataFrame:
    """Copie chaque photo et son détourage véhicule, et note l'absence de détection.

    Les photos **sans véhicule détecté sont conservées** (colonne `vehicule` à faux) : en
    production elles arrivent quand même, et c'est au tri des vues de les écarter. Les retirer ici
    reviendrait à évaluer la chaîne sur une entrée déjà nettoyée par un autre modèle.
    """
    from detect import detecter, load_detector, recadrer

    sortie = Path(dossier_sortie)
    for s in ("photos", "crops"):
        (sortie / s).mkdir(parents=True, exist_ok=True)

    detecteur, categories = load_detector(device)
    res = detecter(detecteur, categories, device, list(df["chemin"]), score_min=score_min)

    lignes = []
    for ligne, r in zip(df.itertuples(), res):
        nom = f"{ligne.ad_id}_{ligne.rang}.jpg"
        img = Image.open(ligne.chemin).convert("RGB")
        img.save(sortie / "photos" / nom, quality=90)

        crop = recadrer(img, r["boite"]) if r["boite"] is not None else img
        if max(crop.size) > cote_max:
            ratio = cote_max / max(crop.size)
            crop = crop.resize((round(crop.width * ratio), round(crop.height * ratio)),
                               Image.LANCZOS)
        crop.save(sortie / "crops" / nom, quality=90)

        lignes.append({"fichier": nom, "ad_id": ligne.ad_id, "rang": ligne.rang,
                       "etat_declare": ligne.etat_declare, "classe_declaree": ligne.classe_declaree,
                       "vehicule": r["boite"] is not None, "fraction_cadre": r["fraction_cadre"]})

    manifeste = pd.DataFrame(lignes)
    manifeste.to_csv(sortie / "manifeste_annonces.csv", index=False)
    return manifeste


def agreger(probas_par_photo: pd.DataFrame, regle: str = "max",
            colonne: str = "proba") -> pd.DataFrame:
    """Passe du score par photo au score par annonce.

    Trois règles, et le choix entre elles est une **décision produit**, pas technique :

    - `max` : une seule photo suffit à alerter. Sensible, mais alarmiste — une fausse alerte coûte
      de l'argent au vendeur en négociation.
    - `moyenne` : dilue le seul cliché qui montre le dégât, précisément celui qui compte.
    - `moyenne_top2` : compromis — exige deux photos concordantes avant d'alerter.
    """
    g = probas_par_photo.groupby("ad_id")[colonne]
    if regle == "max":
        score = g.max()
    elif regle == "moyenne":
        score = g.mean()
    elif regle == "moyenne_top2":
        score = g.apply(lambda s: np.sort(s.values)[-2:].mean())
    else:
        raise ValueError(f"règle inconnue : {regle}")
    return score.rename("score_annonce").reset_index()
