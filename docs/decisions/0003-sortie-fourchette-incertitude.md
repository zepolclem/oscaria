# ADR 0003 — Sortie en fourchette de prix avec incertitude

- **Statut** : Accepté
- **Date** : 2026-07-17 (chiffres mis à jour le 2026-07-17 après correctif de parsing, cf. ADR 0004)

## Contexte

Exigence produit d'OscarIA ([AGENTS.md](../../AGENTS.md)) : restituer une **fourchette de prix
avec incertitude**, pas un prix ponctuel. Un chiffre unique (« 18 400 € ») affiche une fausse
précision : il est faux d'environ 2 900 € en moyenne, et le vendeur ne le sait pas.

On veut donc un intervalle [prix bas, prix haut] dont la largeur exprime l'incertitude, et une
garantie mesurable que le vrai prix y tombe assez souvent (carnet
`04_fourchette_incertitude.ipynb`).

## Décision

1. **Régression quantile** (*quantile regression*) : trois modèles qui visent chacun un quantile
   du prix — le quantile 10 % (borne basse), le quantile 50 % (estimation médiane), le quantile
   90 % (borne haute). La fourchette est [quantile 10 %, quantile 90 %], visant **80 % de
   couverture**.
2. **Calibration conforme** (*Conformalized Quantile Regression*, CQR) : la fourchette brute ne
   couvre que ~62 % des cas (trop optimiste). Un jeu de calibration dédié mesure un
   « coussin » qu'on ajoute de chaque côté pour ramener la couverture vers la cible. Résultat
   mesuré : **76,7 %** de couverture sur des données jamais vues (largeur médiane 9 094 €,
   soit 39 % du prix médian ; estimation médiane : erreur absolue moyenne 2 931 €).
3. **Variables retenues** : les 13 variables du pipeline + la **fréquence du modèle exact**
   (`modele_freq`, apprise sur le jeu d'ajustement uniquement — anti-fuite).

## Alternatives écartées

- **Prix ponctuel** : rejeté par l'exigence produit (fausse précision, malhonnête).
- **Fourchette quantile sans calibration** : couverture réelle ~65 %, donc une fourchette qui
  ment sur sa fiabilité.
- **Niveau de confiance 90 %** (quantiles 5 % / 95 %) : fourchette plus large, moins actionnable
  pour fixer un prix. On a retenu 80 % comme compromis.

## Conséquences

- Sortie directement exploitable produit : « votre voiture vaut entre X et Y € », avec un prix
  de départ (borne haute) et un plancher de négociation (borne basse).
- **La couverture est mesurée, pas proclamée** — on affiche le vrai chiffre (76,7 %), preuve
  d'honnêteté attendue en BC04. L'écart à la cible de 80 % est compatible avec le bruit
  d'échantillonnage (~400 voitures de calibration et de test) ; la garantie de la méthode
  conforme vaut en moyenne, pas sur chaque tirage.
- La largeur de la fourchette **varie par voiture** : large quand le modèle doute (véhicules
  atypiques), serrée quand il est confiant (modèles courants). C'est de l'explicabilité.
- **Limites** : la couverture est globale (elle peut varier par segment) ; le coussin de
  calibration est symétrique. Raffinements possibles : calibration conforme asymétrique,
  calibration par segment.
