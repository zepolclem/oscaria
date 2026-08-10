"""Entraînement du modèle Prix — dataset `leboncoin-private`.

Produit une **fourchette de prix dont la couverture est garantie**, pas un prix ponctuel
(exigence `ml/AGENTS.md`). Deux mécanismes empilés :

1. **Régression quantile** — trois modèles au lieu d'un, entraînés avec la perte pinball
   (`loss="quantile"`) aux quantiles 0,1 / 0,5 / 0,9. La largeur de la fourchette devient
   *adaptative* : elle s'élargit d'elle-même là où le modèle est mal à l'aise.
2. **Conformalisation (CQR)** — les quantiles appris sont optimistes (un modèle visant 80 %
   en couvre typiquement 72-76 %). Un jeu de calibration jamais vu à l'ajustement mesure de
   combien les bornes ratent, et les corrige d'une constante `Q`. La couverture devient alors
   garantie à au moins 1-α sur des données nouvelles.

Le carnet lit un Parquet intermédiaire produit par le carnet 02. Ce module s'en passe : il
part du `raw/` et rappelle `clean_leboncoin()`. Une commande, pas d'état intermédiaire à
resynchroniser.

    uv run --package oscaria-ml python ml/src/entrainement.py

Écrit `app/models/prix.joblib` + `app/models/prix.json`. Destination volontaire : `ml/models/`
est gitignoré, donc invisible du build Docker, qui ne copie que du versionné.

Historique : plan `docs/plans/2026-08-08-fourchette-conformalisee.md`, marches précédentes
dans `docs/plans/2026-08-07-chaine-bout-en-bout.md`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from leboncoin import ENERGIES_RARES, clean_leboncoin

RACINE = Path(__file__).resolve().parents[2]
PARQUET = RACINE / "ml" / "data" / "leboncoin-private" / "raw" / "annonces.parquet"
PREMIUM = RACINE / "ml" / "references" / "premium_brand.csv"
SORTIE = RACINE / "app" / "models"

# La mise en forme des colonnes est IMPORTÉE du service, pas recopiée ici : c'est la seule
# façon de garantir qu'un véhicule est encodé à l'identique à l'entraînement et à l'inférence.
# Le sens de la dépendance surprend (ml -> app), mais il suit l'artefact : c'est l'image de
# l'application qui doit embarquer ce code, `ml/` n'y est jamais copié.
sys.path.insert(0, str(RACINE / "app" / "src"))
from preparation import (  # noqa: E402
    CAT,
    ETATS_GROUPES,
    FREQUENCE_INCONNUE,
    INCONNU,
    NUM,
    TRANCHES,
    construire_jeu,
)


# Quantiles appris. 0,1 et 0,9 encadrent 80 % de la distribution conditionnelle ; 0,5 est la
# médiane, servie comme estimation centrale. La perte pinball à 0,5 revient à minimiser
# l'erreur absolue — le modèle central est donc directement optimisé pour la MAE.
QUANTILES = {"q10": 0.1, "q50": 0.5, "q90": 0.9}

# Risque toléré : 20 %, donc une couverture cible de 80 %. Plus bas donnerait des fourchettes
# plus larges, donc moins actionnables pour un vendeur ; plus haut, une promesse plus fragile.
ALPHA = 0.2

# Réglages issus de la recherche du carnet 07 (50 tirages, CV 5 plis sur le fit seul) :
# −84 € de MAE contre les défauts. Le motif : apprentissage lent et long (0,037 × 750
# itérations au lieu de 0,1 × 100) avec des arbres plus riches (96 feuilles au lieu de 31).
# `early_stopping=False` : c'est ainsi que la recherche a mesuré ces réglages — le laisser
# en « auto » entraînerait un autre modèle que celui qui a été choisi.
#
# Appliqués au CENTRAL SEULEMENT. Essayés sur les trois quantiles : MAE équivalente mais la
# couverture par tranche s'effondrait au-delà de 10 k€ (64 % sur le haut de gamme contre
# 73 % avec les bornes aux défauts) — des bornes plus « sûres d'elles » que la constante
# conformale, globale, ne répare qu'en moyenne. La précision ne vaut pas une promesse
# affaiblie là où elle était déjà la plus fragile (ADR ML 0008).
HYPERPARAMETRES_CENTRAL = {
    "learning_rate": 0.037,
    "max_iter": 750,
    "max_leaf_nodes": 96,
    "min_samples_leaf": 10,
    "l2_regularization": 0.025,
    "max_depth": None,
    "early_stopping": False,
}


def conformaliser(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, alpha: float = ALPHA) -> float:
    """Constante d'élargissement `Q` mesurée sur le jeu de calibration (méthode CQR).

    Le score de non-conformité d'un point est de combien l'intervalle l'a raté :

        E_i = max(lo_i - y_i,  y_i - hi_i)

    Il est **positif** si le point tombe hors de l'intervalle (la valeur dit de combien
    l'élargir), **négatif** s'il tombe dedans avec de la marge (la valeur dit de combien on
    pourrait resserrer). La conformalisation corrige donc dans les deux sens : `Q` négatif
    signifie des quantiles trop prudents.

    Le niveau empirique `⌈(n+1)(1-α)⌉/n` — et non `1-α` — est la correction de population
    finie qui rend la garantie valable, pas seulement asymptotique.
    """
    scores = np.maximum(lo - y, y - hi)
    n = len(scores)
    niveau = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, niveau, method="higher"))


def couverture_par_tranche(
    y_vrai: pd.Series, lo: np.ndarray, hi: np.ndarray
) -> pd.DataFrame:
    """Couverture réelle et largeur, par tranche de prix.

    Le tableau de bord de la garantie. Une couverture globale de 80 % peut parfaitement
    cacher 95 % sur les petites voitures et 60 % sur le haut de gamme : la promesse serait
    alors fausse pour un segment entier tout en paraissant tenue en moyenne. C'est ce
    découpage, pas la moyenne, qui dit si la fourchette est honnête pour tout le monde.
    """
    tranches = pd.cut(y_vrai, TRANCHES)
    dedans = pd.Series((y_vrai.to_numpy() >= lo) & (y_vrai.to_numpy() <= hi), index=y_vrai.index)
    largeur = pd.Series(hi - lo, index=y_vrai.index)
    g = lambda s: s.groupby(tranches, observed=True)  # noqa: E731
    tab = pd.DataFrame({
        "n": g(y_vrai).size(),
        "prix_median": g(y_vrai).median().round(0),
        "couverture": g(dedans).mean().round(3),
        "largeur_mediane": g(largeur).median().round(0),
    })
    tab["largeur_sur_prix_pct"] = (100 * tab["largeur_mediane"] / tab["prix_median"]).round(0)
    return tab


def erreur_par_tranche(y_vrai: pd.Series, y_pred: np.ndarray) -> pd.DataFrame:
    """MAE par tranche de prix — le tableau du carnet 04 §3.

    Ne fabrique plus la fourchette (c'était le raccourci de la marche 1, remplacé par la
    conformalisation) mais reste une mesure utile du modèle médian, et une pièce du dossier
    de certification : l'erreur en euros croît avec le prix pendant que l'erreur relative
    décroît, et les deux lectures ne désignent pas le même modèle.
    """
    tranches = pd.cut(y_vrai, TRANCHES)
    err = pd.Series(np.abs(y_vrai.to_numpy() - y_pred), index=y_vrai.index)
    tab = pd.DataFrame({
        "n": y_vrai.groupby(tranches, observed=True).size(),
        "prix_median": y_vrai.groupby(tranches, observed=True).median().round(0),
        "mae": err.groupby(tranches, observed=True).mean().round(0),
    })
    tab["mae_sur_prix_pct"] = (100 * tab["mae"] / tab["prix_median"]).round(0)
    return tab


def entrainer() -> dict:
    """Nettoie, entraîne, évalue. Renvoie le modèle et tout ce qu'il faut pour le servir."""
    df, journal = clean_leboncoin(PARQUET, PREMIUM)
    date_reference = pd.Timestamp(df["scraped_at"].max()).tz_localize(None)

    # Âge recalculé depuis la seule ANNÉE de mise en circulation. `clean_leboncoin` le dérive
    # au mois près, mais le formulaire ne demande plus le mois : entraîner sur une précision
    # que le service n'aura jamais, c'est le décalage entraînement/service par construction.
    df["age"] = (
        date_reference - pd.to_datetime(dict(year=df["annee"], month=1, day=1))
    ).dt.days / 365.25

    df["etat"] = df["etat"].map(ETATS_GROUPES)

    # Découpage en TROIS. Le premier est inchangé depuis les carnets 03 et 04 : le jeu de
    # test reste bit-pour-bit le même, donc les MAE restent comparables (1 425 / 1 475 /
    # 2 301 €). La calibration est prélevée DANS l'ancien train, pas ailleurs.
    #
    # La contrainte « jamais vu » ne pèse que sur les BORNES : `Q` doit être mesuré sur des
    # données que q10 et q90 n'ont pas apprises, sinon il mesure du connu et la garantie
    # s'évapore. Le central, lui, ne porte aucune garantie (il est même rabattu dans la
    # fourchette quand il en sort) : il a le droit d'apprendre aussi sur la calibration.
    df_train, df_test = train_test_split(df, test_size=0.20, random_state=42)
    df_fit, df_cal = train_test_split(df_train, test_size=0.25, random_state=42)

    # Encodage par fréquence du modèle exact : 804 modalités, bien au-delà des 255 que le
    # catégoriel natif accepte. La fréquence est une mesure de POPULARITÉ — elle ne regarde
    # jamais le prix, donc pas de fuite de la cible, contrairement à un encodage par prix
    # moyen (que le carnet 03 avait déjà vu dégrader le linéaire).
    #
    # Comptée sur le jeu d'AJUSTEMENT SEUL, pas sur ajustement + calibration : la calibration
    # doit rester une donnée jamais vue à tous égards, y compris à travers un encodage.
    frequence = df_fit["modele"].value_counts()
    for d in (df_fit, df_cal, df_test):
        d["modele_freq"] = d["modele"].map(frequence).fillna(FREQUENCE_INCONNUE).astype("float64")

    # Vocabulaire des catégories déduit du jeu complet : il ne dépend pas de la cible, donc
    # ne fuit rien, et le service doit connaître toutes les marques offertes au formulaire.
    categories = {
        c: sorted(set(df[c].fillna(INCONNU).astype("string")) | {INCONNU}) for c in CAT
    }

    X_fit = construire_jeu(df_fit, categories=categories)
    X_cal = construire_jeu(df_cal, categories=categories)
    X_test = construire_jeu(df_test, categories=categories)
    y_fit, y_cal, y_test = df_fit["prix_eur"], df_cal["prix_eur"], df_test["prix_eur"]

    # Trois modèles, mêmes hyperparamètres — seule la fonction de perte change.
    # `loss="quantile"` est la perte pinball : asymétrique, elle pénalise plus cher la
    # sous-estimation que la sur-estimation pour un quantile haut, et l'inverse pour un
    # quantile bas. C'est ce qui fait apprendre un quantile plutôt qu'une moyenne.
    #
    # Jeux d'apprentissage ASYMÉTRIQUES, et c'est voulu : les bornes n'apprennent que sur le
    # fit (la calibration doit leur rester inconnue pour que `Q` vaille quelque chose), le
    # central apprend sur fit + calibration (+33 % de données) puisqu'il ne promet rien —
    # mesuré : MAE 1 482 -> voir plan `2026-08-10-precision-central-hyperparametres.md`.
    X_central = pd.concat([X_fit, X_cal])
    y_central = pd.concat([y_fit, y_cal])

    def _quantile(q: float, **hyper) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            loss="quantile", quantile=q, categorical_features=CAT, random_state=42, **hyper
        )

    modeles = {
        "q10": _quantile(QUANTILES["q10"]).fit(X_fit, y_fit),
        "q50": _quantile(QUANTILES["q50"], **HYPERPARAMETRES_CENTRAL).fit(X_central, y_central),
        "q90": _quantile(QUANTILES["q90"]).fit(X_fit, y_fit),
    }

    # Calibration : de combien les bornes ratent, sur des données jamais vues.
    Q = conformaliser(
        modeles["q10"].predict(X_cal), modeles["q90"].predict(X_cal), y_cal.to_numpy()
    )

    lo_test = modeles["q10"].predict(X_test) - Q
    hi_test = modeles["q90"].predict(X_test) + Q
    pred = modeles["q50"].predict(X_test)

    # CROISEMENT DE QUANTILES — même garde-fou que dans le service, appliqué ici pour que les
    # métriques décrivent le modèle réellement servi et non un autre. Trois modèles ajustés
    # indépendamment ne sont contraints par rien à rester ordonnés : sur ce jeu, les bornes ne
    # s'inversent jamais mais le central sort de la fourchette dans 1,2 % des cas.
    lo_test, hi_test = np.minimum(lo_test, hi_test), np.maximum(lo_test, hi_test)
    n_croisements = int(((pred < lo_test) | (pred > hi_test)).sum())
    pred = np.clip(pred, lo_test, hi_test)

    couverture = float(((y_test.to_numpy() >= lo_test) & (y_test.to_numpy() <= hi_test)).mean())

    # Deux dérivations que le service devra refaire sur une ligne de formulaire, et qu'on lui
    # livre déjà résolues plutôt que de lui faire réimporter `ml/` (absent de l'image Docker)
    # ou relire `premium_brand.csv`. Réimplémenter `_normalize` côté service serait la porte
    # ouverte au décalage : MERCEDES-BENZ raterait la table premium et perdrait son palier.
    niveau_par_marque = (
        df.dropna(subset=["niveau"]).groupby("marque")["niveau"].first().astype(int).to_dict()
    )
    energies = sorted(df["energie"].dropna().unique().tolist())

    # Liste en cascade du formulaire : quels modèles proposer une fois la marque choisie.
    # Construite sur le jeu complet — c'est une offre d'interface, pas une statistique
    # apprise ; restreindre au train priverait le formulaire de modèles parfaitement valides.
    modeles_par_marque = {
        m: sorted(g["modele"].dropna().unique().tolist())
        for m, g in df.groupby("marque", observed=True)
    }

    return {
        "modeles": modeles,
        "conformal_q": Q,
        "couverture": couverture,
        "largeur_mediane": float(np.median(hi_test - lo_test)),
        "couverture_tranches": couverture_par_tranche(y_test, lo_test, hi_test),
        "n_fit": len(X_fit),
        "n_cal": len(X_cal),
        "categories": categories,
        "niveau_par_marque": niveau_par_marque,
        "energies": energies,
        "frequence_modele": {k: int(v) for k, v in frequence.items()},
        "modeles_par_marque": modeles_par_marque,
        "journal": journal,
        "n_depart": len(pd.read_parquet(PARQUET, columns=["ad_id"])),
        "date_reference": date_reference.isoformat(),
        "tranches": erreur_par_tranche(y_test, pred),
        "metriques": {
            "mae": round(float(mean_absolute_error(y_test, pred))),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, pred)))),
            "r2": round(float(r2_score(y_test, pred)), 3),
            "n_fit": len(X_fit),
            "n_cal": len(X_cal),
            "n_test": len(X_test),
            "predictions_negatives": int((pred < 0).sum()),
            # Combien de fois l'estimation centrale a dû être rabattue dans la fourchette.
            # Chiffre à surveiller : s'il montait franchement, il faudrait contraindre les
            # quantiles à ne pas se croiser plutôt que de corriger après coup.
            "croisements_quantiles": n_croisements,
        },
    }


def ecrire_artefacts(res: dict) -> None:
    """Les modèles sérialisés, et à côté tout ce que l'inférence doit savoir pour ne pas dériver."""
    SORTIE.mkdir(parents=True, exist_ok=True)
    # Un dictionnaire des trois quantiles, pas un modèle seul : ils sont indissociables, et
    # un fichier unique évite qu'un déploiement partiel n'en serve que deux sur trois.
    joblib.dump(res["modeles"], SORTIE / "prix.joblib", compress=3)

    meta = {
        "dataset": "leboncoin-private",
        "features_num": NUM,
        "features_cat": CAT,
        "categorie_inconnue": INCONNU,
        "categories": res["categories"],
        # Tables de dérivation pour le service : marque -> palier premium (0/1/2), et le
        # regroupement des énergies trop rares pour être apprises séparément.
        "niveau_par_marque": res["niveau_par_marque"],
        "energies_saisissables": res["energies"],
        "energies_rares": ENERGIES_RARES,
        # Encodage du modèle exact : le compte d'occurrences au TRAIN. Un modèle absent de
        # cette table n'a jamais été vu à l'entraînement -> `FREQUENCE_INCONNUE`.
        "frequence_modele": res["frequence_modele"],
        # Liste en cascade du formulaire : marque choisie -> modèles proposés.
        "modeles_par_marque": res["modeles_par_marque"],
        "etats_groupes": ETATS_GROUPES,
        # Sert à calculer `age` à l'inférence. Volontairement figée : utiliser la date du jour
        # ferait glisser les âges d'entrée hors de la plage apprise à mesure que le modèle
        # vieillit. C'est aussi le repère qui dit quand ré-entraîner.
        "date_reference": res["date_reference"],
        # --- fourchette conformalisée -------------------------------------------------
        # `conformal_q` est la seule valeur dont le service a besoin pour reconstruire les
        # bornes : `[q10 - Q, q90 + Q]`. Tout le reste ci-dessous est du reporting.
        "quantiles": QUANTILES,
        "alpha": ALPHA,
        "conformal_q": round(res["conformal_q"], 2),
        "couverture_test": round(res["couverture"], 4),
        "largeur_mediane_test": round(res["largeur_mediane"]),
        "couverture_par_tranche": [
            {"min": TRANCHES[i], "max": TRANCHES[i + 1], "n": int(r.n),
             "couverture": float(r.couverture), "largeur_mediane": float(r.largeur_mediane)}
            for i, r in enumerate(res["couverture_tranches"].itertuples())
        ],
        "tranches_bornes": TRANCHES,
        # Conservée bien que le service ne la consomme plus : la MAE par tranche reste une
        # mesure du modèle médian et une pièce du dossier de certification.
        "mae_par_tranche": [
            {"min": TRANCHES[i], "max": TRANCHES[i + 1], "mae": float(v)}
            for i, v in enumerate(res["tranches"]["mae"].tolist())
        ],
        "metriques": res["metriques"],
        # Un artefact joblib ne se recharge pas fiablement d'une version majeure de
        # scikit-learn à l'autre : la version qui a produit le fichier doit voyager avec lui.
        "version_sklearn": sklearn.__version__,
    }
    (SORTIE / "prix.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    res = entrainer()
    ecrire_artefacts(res)
    m = res["metriques"]

    n_total = m["n_fit"] + m["n_cal"] + m["n_test"]

    print("=" * 74)
    print("MODELE PRIX — fourchette conformalisee (quantiles + CQR)")
    print("=" * 74)
    print(f"depart {res['n_depart']} annonces -> {n_total} apres nettoyage")
    print(f"ajustement {m['n_fit']} | calibration {m['n_cal']} | test {m['n_test']}")
    print()
    print("--- modele central (quantile 0.5) ---")
    print(f"MAE  {m['mae']:>7,} EUR   (marche precedente, sur 15,905 lignes : 1,425)")
    print(f"RMSE {m['rmse']:>7,} EUR   (marche precedente : 2,417)")
    print(f"R2   {m['r2']:>7}       (marche precedente : 0.898)")
    print(f"predictions negatives : {m['predictions_negatives']}")
    print(f"central rabattu dans la fourchette (croisement de quantiles) : "
          f"{m['croisements_quantiles']} / {m['n_test']} "
          f"({100 * m['croisements_quantiles'] / m['n_test']:.1f} %)")
    print()
    print("--- garantie de couverture ---")
    print(f"cible                : {1 - ALPHA:.0%}")
    print(f"couverture MESUREE   : {res['couverture']:.1%}  <- doit etre >= {1 - ALPHA:.0%}")
    print(f"correction conforme  : {res['conformal_q']:+,.0f} EUR"
          f"   ({'bornes elargies' if res['conformal_q'] > 0 else 'bornes resserrees'})")
    print(f"largeur mediane      : {res['largeur_mediane']:,.0f} EUR")
    print()
    print("--- la garantie tient-elle PARTOUT ? ---")
    print(res["couverture_tranches"].to_string())
    creuse = res["couverture_tranches"]["couverture"].min()
    if creuse < 0.70:
        print(f"\n/!\\ tranche a {creuse:.0%} de couverture : la promesse est FAUSSE pour ce")
        print("    segment meme si la moyenne tient. A signaler avant tout deploiement.")
    print()
    print("--- MAE par tranche (modele median, reporting seul) ---")
    print(res["tranches"].to_string())
    print()
    print(f"-> {SORTIE / 'prix.joblib'}")
    print(f"-> {SORTIE / 'prix.json'}")
    print("=" * 74)


if __name__ == "__main__":
    main()
