# Marche 2 — une fourchette qu'on peut prouver (régression quantile + conformalisation)

## Contexte

La fourchette servie aujourd'hui vaut `prédiction ± MAE de la tranche` : le modèle sort un
prix, on regarde dans quelle tranche il tombe, on élargit de l'erreur moyenne mesurée sur
cette tranche. C'était le raccourci assumé de la marche 1 — cinq lignes de code pour
satisfaire l'exigence produit dès la première mise en ligne.

Ce raccourci a trois défauts, tous du même genre : **il ne promet rien de vérifiable.**

1. La largeur est choisie d'après le prix *prédit*, pas le prix réel. Si le modèle se trompe
   de tranche, il choisit aussi la mauvaise largeur.
2. L'erreur *moyenne* n'est pas un intervalle. Une MAE de 1 520 € ne dit pas quelle
   proportion des véhicules tombe dans `± 1 520 €` — ça pourrait être 50 % comme 70 %.
3. La largeur ne dépend que du prix. Une Clio vue 400 fois et un modèle vu 2 fois reçoivent
   la même incertitude, alors que la seconde est bien moins prévisible.

**Objectif de cette marche** : remplacer cette table par un intervalle dont la couverture est
**mesurée et garantie** — pouvoir écrire « 8 fois sur 10, le prix de vente tombe dans cette
fourchette » et le prouver sur le jeu de test. C'est ce que `ml/AGENTS.md` demande
(« fourchette avec incertitude »), ce que l'AI Act attend (afficher l'incertitude), et ce que
le carnet 04 laissait explicitement en dette.

## Le principe, en deux temps

**Temps 1 — régression quantile.** Au lieu d'un modèle qui apprend le prix *moyen*, trois
modèles qui apprennent des **quantiles** : le 10ᵉ centile (borne basse), la médiane (prix
central), le 90ᵉ centile (borne haute). On change la fonction de perte —
`loss="quantile"`, la perte pinball, disponible en scikit-learn 1.9 — pas le type de modèle.
La largeur devient **adaptative** : elle s'élargit toute seule là où le modèle est mal à
l'aise.

**Temps 2 — conformalisation (CQR).** Les quantiles appris sont optimistes : un modèle
entraîné à 80 % de couverture en couvre typiquement 72–76 % en vrai. On met donc de côté un
**jeu de calibration** jamais vu à l'entraînement, on y mesure de combien les bornes ratent,
et on les élargit d'une constante unique `Q`. La théorie garantit alors une couverture
d'au moins 80 % sur des données nouvelles.

```
score de non-conformité   E_i = max(borne_basse(x_i) − y_i,  y_i − borne_haute(x_i))
constante                 Q   = quantile ⌈(n+1)·0,8⌉/n des E_i
fourchette servie         [borne_basse − Q ,  borne_haute + Q]
```

Un `E_i` négatif signifie que le point était **dans** l'intervalle avec de la marge ; si
beaucoup le sont, `Q` sera négatif et les bornes se **resserreront**. La conformalisation
corrige dans les deux sens.

## Le coût à mesurer honnêtement

Le jeu de calibration doit sortir de quelque part. Découpage retenu :

| jeu | taille | rôle | change ? |
|---|---|---|---|
| entraînement | 11 929 | ajuster les trois modèles | **−25 %** (était 15 905) |
| calibration | 3 976 | mesurer `Q` | nouveau |
| test | 3 977 | évaluer | **identique** aux carnets 03/04 |

Le test reste bit-pour-bit le même (`train_test_split(df, 0.2, random_state=42)` inchangé,
la calibration est prélevée **dans** l'ancien train) : les MAE restent comparables à
1 425 € / 1 475 € / 2 301 €.

**On perd 25 % des données d'entraînement pour gagner une garantie.** La MAE va monter — de
combien, c'est une mesure du plan, pas une supposition. Si elle dépasse nettement 1 500 €, le
compromis mérite d'être reposé à l'utilisateur avant de servir le nouveau modèle.

## Étape 0 — Déposer les plans dans le dépôt (pièces de soutenance)

Le dépôt a déjà la convention : `docs/plans/YYYY-MM-DD-<slug>.md`, avec deux fiches
existantes (`2026-07-29-cardd-remise-a-zero.md`, `2026-08-06-remise-a-zero-plaques.md`).
Les plans du pilier Prix n'y sont pas — ils n'ont vécu que dans les sessions de travail.

Deux fichiers à écrire **avant** de toucher au code :

1. **`docs/plans/2026-08-08-fourchette-conformalisee.md`** — ce plan, tel quel.
2. **`docs/plans/2026-08-07-chaine-bout-en-bout.md`** — la rétrospective des trois marches
   déjà réalisées, qui n'existe nulle part alors que le travail est fait. À reconstituer
   depuis les faits mesurés, sans les réinventer :

   - **Marche 1 — sortir le modèle du carnet.** Rapatriement des 20 915 annonces depuis
     Postgres ; extraction de `ml/src/entrainement.py` ; artefacts `app/models/` ; endpoint
     `/prix` et formulaire Gradio. MAE reproduite à 1 476 € contre 1 475 € au carnet 04,
     l'écart d'un euro venant de l'ordre des catégories. Parité service/modèle vérifiée à
     0,4 € près.
   - **Fusion des piliers Prix et Plaques.** Découpage `main.py` (assembleur) +
     `page_prix.py` / `page_plaques.py`, motivé par un conflit de fusion réel entre deux
     branches qui réécrivaient le même fichier. Healthcheck qui distingue « module chargé »
     de « modèle chargé ».
   - **Allègement du formulaire.** Retrait de `region` (−14 € de MAE) et du mois de mise en
     circulation (−4 €), ajout du modèle exact encodé par fréquence (−35 €), regroupement
     des états de 8 à 5 crans calé sur les prix médians (0 €). Résultat : MAE 1 425 €,
     R² 0,898, un champ de moins au formulaire. Contrat verrouillé (`extra="forbid"`,
     `Literal` sur l'état) après avoir constaté qu'une valeur obsolète passait en 200.
   - **Image Docker sans CUDA.** 5,53 Go → 1,5 Go. Inclut le résultat négatif : la forme
     documentée par uv (deux extras `cpu`/`cu128` + `[tool.uv] conflicts`) ne fonctionne pas
     dans cet espace de travail — uv n'y lit pas `conflicts`, établi en injectant un nom de
     paquet volontairement faux et en constatant l'absence d'erreur.

   Les résultats négatifs et les compromis explicités valent autant que les réussites pour le
   bloc BC04, qui juge le pilotage et la transparence sur les limites — pas la performance.

Ces fiches sont des **plans**, pas des fiches de décision : `docs/decisions/` reste à
reconstruire séparément (cf. hors périmètre).

## Étape 1 — `ml/src/entrainement.py`

Le squelette existant est bon : `clean_leboncoin` → dérivations → `construire_jeu` →
artefacts. Ce qui change tient dans `entrainer()` et `ecrire_artefacts()`.

- **Découpage en trois** : le `train_test_split` actuel est conservé tel quel, puis un second
  découpage `train_test_split(df_train, test_size=0.25, random_state=42)` sépare ajustement
  et calibration.
- **La fréquence du modèle exact** (`modele_freq`) reste comptée sur le **jeu d'ajustement
  seul** — pas sur ajustement + calibration. Même motif qu'aujourd'hui : la calibration doit
  rester une donnée jamais vue, sinon `Q` est mesuré sur du connu et la garantie s'évapore.
- **Trois modèles** :
  `HistGradientBoostingRegressor(loss="quantile", quantile=q, categorical_features=CAT,
  random_state=42)` pour `q ∈ {0.1, 0.5, 0.9}`. Le reste des hyperparamètres ne bouge pas.
- **Nouvelle fonction `conformaliser(lo_cal, hi_cal, y_cal, alpha=0.2) -> float`** : calcule
  les scores de non-conformité et renvoie `Q`. Une quinzaine de lignes, testable seule.
- **Nouvelle fonction `couverture_par_tranche(y, lo, hi)`** : remplace `erreur_par_tranche`
  dans le rôle de tableau de bord. Colonnes : effectif, taux de couverture réel, largeur
  médiane, largeur rapportée au prix médian. C'est le tableau qui dit si la garantie tient
  **partout** ou seulement en moyenne — une couverture globale de 80 % peut cacher 95 % en
  bas de gamme et 60 % en haut.

`erreur_par_tranche` est conservée : la MAE par tranche reste une mesure utile du modèle
médian, elle ne sert simplement plus à fabriquer la fourchette.

## Étape 2 — Les artefacts

`app/models/prix.joblib` porte désormais un dictionnaire `{"q10": …, "q50": …, "q90": …}`
plutôt qu'un modèle unique. `prix.json` gagne :

```
alpha                  0.2
conformal_q            <la constante, en euros>
couverture_test        <mesurée, doit être >= 0.80>
largeur_mediane_test   <en euros>
couverture_par_tranche [{min, max, n, couverture, largeur_mediane}, …]
```

`mae_par_tranche` reste dans le fichier — plus consommé par le service, toujours utile au
dossier de certification.

## Étape 3 — `app/src/prix.py`

`estimer()` garde exactement sa signature et sa forme de sortie ; seule la fabrication des
bornes change.

```
lo = modeles["q10"].predict(X)[0] - Q
central = modeles["q50"].predict(X)[0]
hi = modeles["q90"].predict(X)[0] + Q
```

Plancher à 0 sur `bas` et `central` conservé. Les clés `largeur_mae` et `tranche` de la
réponse disparaissent au profit de `couverture` (0,8) et `largeur` — les anciennes décrivaient
un mécanisme qui n'existe plus, les garder induirait en erreur.

Un garde-fou à écrire explicitement : **si `lo > hi` après conformalisation** (possible sur
une saisie très atypique, les quantiles n'étant pas garantis ordonnés), les échanger plutôt
que de servir une fourchette inversée.

## Étape 4 — `app/src/page_prix.py`

Le texte affiché change de nature, et c'est la moitié de l'intérêt de la marche :

> **4 900 € — 8 000 €**
> Estimation centrale : 6 400 €
> Cette fourchette contient le prix de vente **8 fois sur 10**, mesuré sur 3 977 annonces
> jamais vues par le modèle.

La mention « aide à la décision, pas une expertise » reste. `/prix/contrat` expose `alpha` et
`couverture_test` pour que la promesse soit vérifiable depuis l'extérieur.

## Vérification

1. **La garantie tient** — le test central de la marche :
   ```
   PYTHONPATH=ml/src uv run --package oscaria-ml python ml/src/entrainement.py
   ```
   `couverture_test` doit valoir **≥ 0,80** (attendu 0,80–0,83 ; un chiffre très au-dessus de
   0,85 signale des bornes trop larges, donc peu informatives). La MAE du modèle médian doit
   être imprimée à côté de 1 425 € pour rendre le coût des 25 % de données visible.
2. **La garantie tient partout** : dans `couverture_par_tranche`, aucune tranche ne doit
   descendre sous ~0,70. Une tranche à 0,55 signifierait que la promesse est fausse pour ce
   segment même si la moyenne est bonne — à signaler avant tout déploiement.
3. **Parité service/modèle** — rejouer le script de vérification de la marche 1, adapté aux
   trois modèles : les bornes calculées par `prix.estimer()` doivent être identiques à celles
   calculées directement, sur 10 annonces tirées au sort. Écart attendu < 1 € (arrondi).
4. **Bout en bout, local puis conteneur** :
   ```
   CHEMIN_MODELE=dl/models/plaques_baseline.pt \
     uv run --package oscaria-app uvicorn --app-dir app/src main:app --port 8000
   curl -s -X POST localhost:8000/prix -H 'Content-Type: application/json' \
     -d '{"marque":"RENAULT","modele":"Clio","annee_mec":2015,"kilometrage":120000,
          "energie":"Diesel","boite":"Manuelle","etat":"3_usure"}'
   docker build -f deploy/Dockerfile -t oscaria-app:cqr . && docker run -d -p 8090:8000 …
   ```
   La fourchette doit être **plus étroite en bas de gamme et plus large en haut** que
   l'ancienne — c'est la signature de l'adaptativité ; si la largeur est constante, les
   modèles quantiles n'ont rien appris de spécifique.
5. **Cas dégradés** rejoués : marque inconnue, tous les champs facultatifs vides, véhicule de
   1985, modèle inconnu. Aucun 500, aucune borne basse négative, aucune fourchette inversée.

## Hors périmètre

- **Encodage du modèle exact par prix moyen** (remplacerait la fréquence, qui mesure la
  popularité et non l'identité — une Clio et une Twingo comparables ne diffèrent que de 35 €).
  Décision indépendante, à reprendre après.
- **Formulaire à socle court avec masquage aléatoire** des champs facultatifs.
- **Fiches de décision (ADR)** à réécrire : `ml/src/leboncoin.py` cite encore « ADR 0002 » et
  le carnet 04 « ADR 0003 », références orphelines depuis la remise à zéro du 2026-08-06.
