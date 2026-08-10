# Mesurer l'effet des champs facultatifs vides sur l'estimation de prix

- **Date** : 2026-08-10
- **Pilier** : Prix
- **Statut** : clos — remède choisi et livré (contrat v2, plan `2026-08-10-contrat-entree-v2.md`, ADR ML 0007)

## Constat

Sur le formulaire `/prix`, laisser vide l'accordéon « Informations complémentaires »
(puissance DIN, puissance fiscale, Crit'Air, portes, places, contrôle technique, couleur)
produit une estimation nettement plus basse ; remplir ces champs la remonte. Constaté à la
main sur le formulaire Gradio.

## Cause suspectée

Un champ vide devient une valeur manquante (NaN). `HistGradientBoostingRegressor` traite le
manquant nativement : il a appris **pendant l'entraînement** de quel côté de chaque coupe
envoyer les trous. Or dans les annonces scrapées, « champ non renseigné » corrèle avec
« annonce bâclée de véhicule pas cher » : le modèle a donc appris « manquant = bon marché ».

Au service, « vide » signifie autre chose : « le vendeur ne connaît pas la valeur ». Même
encodage, deux significations — un décalage entraînement/service porté par le **motif de
manquance** (MNAR — *Missing Not At Random*, manquance non aléatoire : le fait qu'une valeur
manque dépend de ce qu'elle aurait été, ou ici du type d'annonce).

## Décision de méthode : mesurer avant de corriger

Aucun remède à cette marche. On chiffre d'abord, sur le modèle réellement servi, puis on
choisira (imputation, avertissement d'interface, ou limite documentée) — un résultat négatif
(« effet en réalité négligeable ») serait consigné pareil.

## Les trois expériences

Toutes dans `ml/notebooks/leboncoin-private/05_champs_manquants.ipynb`, sans aucun
ré-entraînement : on interroge `app/models/prix.joblib` + `prix.json` tels quels.

1. **A — complet vs « formulaire minimal »** : chaque annonce du jeu de test (3 977 lignes,
   mêmes splits `random_state=42` que `ml/src/entrainement.py`) est prédite deux fois — telle
   quelle, puis avec les champs de l'accordéon masqués (`puissance_din`, `puissance_fisc`,
   `critair`, `portes`, `places`, `ct_valide_jusqu_a`, `couleur` → manquant). Mesures :
   décalage du central (médiane, quartiles, par tranche de prix), évolution de la largeur de
   fourchette (s'élargit-elle, comme l'adaptativité le voudrait ?), MAE et couverture en mode
   minimal (la garantie 79,5 % tient-elle quand l'info manque ?).
2. **B — l'hypothèse MNAR dans les données** : prix médian des annonces avec vs sans chaque
   champ renseigné, dans le jeu nettoyé. Confirme (ou infirme) que « manquant = pas cher »
   vient bien des données.
3. **C — le cas utilisateur rejoué** : la Clio de référence (RENAULT Clio 2015, 120 000 km,
   Diesel, boîte manuelle, usure normale) via `app/src/prix.estimer()`, accordéon vide puis
   rempli — le chiffre parlant pour la soutenance.

## Garde-fous

- Le jeu de test ne sert à **évaluer**, jamais à choisir quoi que ce soit.
- Contrôle de cohérence obligatoire : en mode « complet », le carnet doit retrouver la MAE
  (1 448 €) et la couverture (79,48 %) de `app/models/prix.json` — sinon la reproduction des
  splits est fausse et les chiffres « minimal » ne valent rien.
- La logique bornes/rabattement de `app/src/prix.py` est rejouée à l'identique pour que la
  mesure décrive le service servi, pas un autre modèle.

## Résultats

Carnet exécuté (`05_champs_manquants.ipynb`), reproduction du mode « complet » exacte
(MAE 1 448 €, couverture 79,48 % — identiques à `prix.json`).

**A — complet vs formulaire minimal** (3 977 annonces de test) :

| | complet | accordéon vide |
|---|---|---|
| MAE du central | 1 448 € | **2 544 €** |
| couverture | 79,48 % | **81,97 %** |
| largeur médiane | 3 399 € | **5 493 €** |

Déplacement du central : médiane **−506 €** (−12 % du prix médian), 68,5 % des annonces
baissent. L'effet croît avec le prix : −169 € en entrée de gamme, −3 406 € entre 10 et
20 k€, **−9 260 € (−34 %)** au-delà de 20 k€.

**B — hypothèse MNAR confirmée dans les données** : prix médian sans vs avec le champ —
Crit'Air −2 400 € (renseigné dans 33 % des annonces seulement), couleur −1 250 €,
puissance DIN −1 150 €, portes/places −900 €, CT −620 €.

**C — Clio de référence** (RENAULT Clio 2015, 120 000 km, Diesel) : central 7 479 € accordéon
vide contre 8 066 € rempli (+587 €) ; fourchette 5 478 € → 2 742 € de large.

## Conclusion

Le constat utilisateur est confirmé et son mécanisme identifié : le **central** est biaisé
vers le bas quand les champs manquent (il hérite du signal « annonce bâclée = voiture pas
chère » appris des données), surtout sur les véhicules chers. La **fourchette reste honnête** :
elle s'élargit comme l'adaptativité le veut et la garantie de couverture tient (82 %).

Remèdes candidats pour la marche suivante (décision à prendre, ADR à l'appui) : mise en avant
de la fourchette + avertissement chiffré dans l'interface (le moins risqué) ; imputation par
médiane marque/modèle au service (réintroduit un décalage — à mesurer) ; ré-entraînement avec
masquage aléatoire des champs (traite la cause, coûte re-calibration).
