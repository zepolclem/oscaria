# Contrat d'entrée v2 — champs obligatoires, jeu de variables réduit

- **Date** : 2026-08-10
- **Pilier** : Prix
- **Statut** : livré
- **Amont** : `2026-08-10-mesure-champs-manquants.md` (la mesure qui motive cette marche)

## Pourquoi

Le carnet 05 a montré que les champs facultatifs vides tirent l'estimation centrale vers le
bas (médiane −506 €, jusqu'à −34 % au-delà de 20 k€) : le modèle a appris « champ manquant =
annonce bâclée = pas cher » (MNAR), signal sans aucun sens pour un vendeur qui ne connaît
pas la valeur. Remède retenu : **supprimer la possibilité même du champ vide** — tous les
champs deviennent obligatoires — et **réduire le contrat** aux variables qui portent le prix.

Plus simple que le masquage aléatoire à l'entraînement, envisagé puis rendu inutile : si le
service n'envoie plus jamais de manquant, il n'y a plus de motif de manquance à décorréler.

## Décisions (appuyées sur l'importance par permutation du carnet 04)

- **Retirés** : `critair` (importance 0,0002), `ct_valide_jusqu_a` (−0,0005 — du bruit),
  `puissance_fisc` (0,045, redondante avec la DIN à 0,208), `couleur` (0,0001), `portes`
  (0,011) et `places` (0,014). Les deux dernières ne sont pas nulles : leur coût est mesuré
  avant adoption (carnet 06) et signalé s'il dépasse ~30 € de MAE.
- **États : 5 → 4 crans** — fusion « hors service » + « à réparer » (prix médians proches) ;
  « bon » et « excellent » restent séparés (7 000 € d'écart de prix médian).
- **Tout obligatoire**, y compris `modele` (encodage par fréquence : −35 € de MAE).

Contrat v2 : NUM = `age, kilometrage, puissance_din, boite_auto, niveau, modele_freq` ;
CAT = `energie_grp, marque, etat`. Formulaire : **8 champs, tous obligatoires** — marque,
modèle, année, kilométrage, énergie, boîte, état, puissance DIN (`niveau` et `modele_freq`
restent dérivés côté service).

## Marches

1. **Carnet `06_contrat_v2.ipynb`** : chaque écart chiffré en validation croisée 5 plis sur
   le jeu d'ajustement seul (11 928 lignes, splits seed 42 — calibration et test intouchés),
   modèle central quantile 0,5, MAE. Variantes : v1 ; −critair−ct ; −fisc ; −couleur ;
   −portes−places ; états 4 crans ; v2 cumulé.
2. **Contrat partagé + ré-entraînement** : `app/src/preparation.py` (NUM/CAT, états 4 crans,
   en-tête chiffré), puis `uv run --package oscaria-ml python ml/src/entrainement.py` →
   nouveaux artefacts, métriques comparées à v1 (MAE 1 448 €, couverture 79,48 %, largeur
   3 399 €, Q +145 €).
3. **API + formulaire** : `app/src/page_prix.py` (contrat `Vehicule` sans `| None`, Literal
   4 crans, accordéon supprimé, garde-fou champs vides côté Gradio), `app/src/prix.py`
   (champs retirés, export `COULEURS` supprimé). Rupture d'API assumée : un ancien client
   envoyant `critair` reçoit 422 (`extra="forbid"`).
4. **Dossier** : ADR `docs/decisions/ml/0007-contrat-entree-v2.md`, index mis à jour,
   résultats reportés ici.

## Résultats

**Carnet 06 (validation croisée 5 plis, jeu d'ajustement)** — coût de chaque écart en MAE :
−critair−ct **−4 €** ; −couleur **−15 €** ; −puissance_fisc +11 € ; −portes−places +14 € ;
états 4 crans +17 € ; **v2 cumulé +43 € (± 16)**. Aucun écart isolé au-dessus du seuil de
30 € ; le cumul est assumé en regard du biais supprimé (−506 € médian, carnet 05).

**Modèle v2 ré-entraîné** (même test que v1) :

| | v1 | v2 |
|---|---|---|
| MAE | 1 448 € | 1 482 € |
| Couverture (cible 80 %) | 79,48 % | **79,98 %** |
| Largeur médiane | 3 399 € | 3 605 € |
| `Q` | +145 € | +144 € |
| Croisements | 64 (1,6 %) | 50 (1,3 %) |

**Vérifications** : API testée — Clio complète 200 (5 877 — 8 275 €, largeur 2 398 € contre
5 478 € pour la même Clio à l'accordéon vide en v1) ; sans `puissance_din` → 422 ; ancien
client avec `critair` → 422 ; ancien cran `3_usure` → 422 ; `GET /prix/contrat` expose les
nouvelles listes. Garde-fou Gradio : champs manquants nommés, pas d'estimation partielle.

Décision et compromis documentés dans l'ADR `docs/decisions/ml/0007-contrat-entree-v2.md`.
