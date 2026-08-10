# ML 0007 — Contrat d'entrée v2 : tous les champs obligatoires, huit variables

- **Date** : 2026-08-10
- **Statut** : Accepté
- **Pilier** : Prix
- **Remplace partiellement** : ML 0005 (le jeu de champs et leur caractère facultatif)

## Contexte — le biais des champs vides, mesuré

Le contrat 0005 laissait sept champs facultatifs, encodés en valeurs manquantes natives :
`HistGradientBoosting` « apprend de quel côté d'une coupe envoyer les trous », donc un champ
vide semblait n'être qu'une information de moins. C'était vrai à l'entraînement, faux au
service — et le carnet `05_champs_manquants.ipynb` l'a chiffré :

- accordéon facultatif vide, l'estimation **centrale baisse pour 68,5 % des annonces**
  (médiane **−506 €**, −12 % du prix médian), jusqu'à **−9 260 € (−34 %)** au-delà de
  20 000 € ; MAE du central 1 448 € → 2 544 € ;
- la cause est dans les données : les annonces qui omettent un champ sont des annonces
  bâclées de véhicules bon marché (sans Crit'Air : prix médian 3 600 € contre 6 000 € avec).
  Le modèle a appris « manquant = pas cher » — un motif de manquance **MNAR** (*Missing Not
  At Random*) qui n'a aucun sens quand c'est un vendeur qui ignore une valeur, et non un
  annonceur négligent ;
- la fourchette, elle, restait honnête (largeur +62 %, couverture 82 %) : seul le central —
  le chiffre le plus visible de la page — était biaisé.

## Décision

**Supprimer la cause plutôt que corriger l'effet** : plus aucun champ facultatif, et un
contrat réduit aux variables qui portent le prix.

- **NUM** : `age, kilometrage, puissance_din, boite_auto, niveau, modele_freq`
- **CAT** : `energie_grp, marque, etat` (4 crans)
- **Formulaire : 8 champs, tous obligatoires** — marque, modèle, année, kilométrage,
  énergie, boîte, état, puissance DIN (`niveau` et `modele_freq` restent dérivés côté
  service, jamais saisis).

### Chaque écart chiffré (carnet 06, validation croisée 5 plis sur le jeu d'ajustement seul)

| écart | importance (carnet 04) | coût MAE mesuré |
|---|---|---|
| − `critair`, `ct_valide_jusqu_a` | 0,0002 / −0,0005 | **−4 €** (retirer améliore) |
| − `couleur` | 0,0001 | **−15 €** (retirer améliore) |
| − `puissance_fisc` | 0,045 (la DIN : 0,208) | +11 € |
| − `portes`, `places` | 0,011 / 0,014 | +14 € |
| états 5 → 4 crans | — | +17 € |
| **cumul v2** | | **+43 €** (± 16) |

Les états fusionnent « hors service » et « à réparer » (prix médians 1 300–2 000 €,
proches) ; « bon » (9 000 €) et « excellent » (16 000 €) restent séparés.

### Le modèle v2, ré-entraîné et re-calibré (même test que les carnets 03-06)

| | v1 (0006) | v2 |
|---|---|---|
| MAE du central | 1 448 € | **1 482 €** (+34 €, cohérent avec la CV) |
| R² | 0,875 | 0,871 |
| **Couverture mesurée** (cible 80 %) | 79,48 % | **79,98 %** |
| Largeur médiane | 3 399 € | 3 605 € |
| Correction conforme `Q` | +145 € | +144 € |
| Croisements de quantiles | 64 (1,6 %) | 50 (1,3 %) |

Sur la Clio de référence (2015, 120 000 km, Diesel, 90 ch) : **5 877 — 8 275 €** (largeur
2 398 €), là où la même Clio à l'accordéon vide recevait 4 716 — 10 195 € en v1.

## Le compromis, dit clairement

Le v2 coûte **+34 € de MAE** sur un formulaire *parfaitement rempli* — c'est le prix payé
pour qu'un formulaire *réellement rempli* ne puisse plus être biaisé de −500 à −9 000 €.
L'ancien contrat était meilleur sur le papier et pire dans les mains d'un utilisateur.

## Rupture d'API assumée

`extra="forbid"` est conservé : un client resté au contrat 0005 qui envoie `critair` ou
`couleur` reçoit **422**, et les huit champs sont requis (plus de `| None`). Les crans
d'état changent de codes (`2_usure` remplace `3_usure`, etc.) : une valeur de l'ancienne
échelle vaut 422 aussi. Un refus explicite plutôt qu'un prix calculé sur un autre contrat
que celui que le client croit utiliser.

## Alternatives écartées

- **Masquage aléatoire à l'entraînement** (décorréler « manquant » du prix) : traite le
  symptôme en gardant les champs facultatifs ; inutile dès lors que le service n'envoie plus
  jamais de manquant, et coûte une complexité d'entraînement permanente.
- **Imputation au service** (médiane par marque/modèle) : réintroduit un décalage
  entraînement/service — le modèle n'a jamais vu de valeurs imputées.
- **Avertissement d'interface seul** : n'aurait rien corrigé, seulement prévenu que le
  chiffre était faux.

## Conséquences

- `app/src/preparation.py` porte le contrat v2 et ses écarts chiffrés en tête de module ;
  formulaire et API dans `app/src/page_prix.py` (garde-fou Gradio : champs manquants nommés,
  pas d'estimation partielle) ; `app/src/prix.py` ne dérive plus que `age`, `boite_auto`,
  `niveau`, `modele_freq`, `energie_grp`.
- Artefacts ré-entraînés (`app/models/prix.joblib` + `prix.json`).
- La couverture par tranche reste non uniforme (85,4 % en entrée de gamme, 72,8 % au-delà de
  20 000 €) : la limite « garantie marginale, pas conditionnelle » de la fiche 0006 tient
  toujours et reste affichée.
