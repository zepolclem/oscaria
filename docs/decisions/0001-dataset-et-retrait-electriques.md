# ADR 0001 — Dataset candidat et retrait des véhicules électriques purs

- **Statut** : Accepté
- **Date** : 2026-07-16

## Contexte

Le pilier Prix a besoin d'un jeu de données de voitures d'occasion avec leur prix. Premier
candidat évalué : `French_second_hand_cars.csv` (données collectées ~2023). Après l'analyse
exploratoire des données (*Exploratory Data Analysis*, phase d'inspection avant tout traitement,
carnet `01_eda_inspection.ipynb`) : 2 441 lignes, 40 colonnes, en-têtes en français, données
sales (prix en texte avec devise, colonnes multi-lignes).

Constat clé : 9 colonnes sont vides à plus de 90 %, **toutes liées à la batterie**
(autonomie, voltage, prix incluant la batterie…). Elles ne concernent que les véhicules
électriques (40) et hybrides (146) sur 2 441. La colonne « prix incluant la batterie » révèle
en plus que quelques électriques ont un **prix non comparable** (batterie en location, donc
prix affiché artificiellement bas).

## Décision

1. **Garder** ce dataset comme premier candidat de travail.
2. **Retirer les véhicules électriques purs** (40 lignes, colonne `énergie == "Electrique"`).
3. **Conserver les hybrides** : leur batterie est toujours incluse dans le prix (pas de
   location), donc leur prix reste comparable.

## Alternatives écartées

- **Retirer aussi les hybrides** : perte de 146 lignes supplémentaires sans justification —
  leur prix est comparable, contrairement aux électriques en location.
- **Garder les électriques et imputer les colonnes batterie** : trop de valeurs manquantes
  (>90 %), et le problème de prix non comparable (batterie en location) resterait.

## Conséquences

- Jeu de travail ramené à 2 340 lignes exploitables.
- Périmètre implicite = véhicules thermiques et hybrides. Les électriques sont hors du champ
  du modèle — **limite à documenter** (attendu BC04).
- Les colonnes batterie deviennent quasi inutiles ; la colonne « prix incluant la batterie »
  est abandonnée (constante après retrait des électriques).
