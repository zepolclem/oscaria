# ADR 0002 — Périmètre produit : marché ≤ 50 000 €

- **Statut** : Accepté
- **Date** : 2026-07-17

## Contexte

Le jeu de données contient une petite queue de voitures très chères (78 voitures > 100 000 €,
soit 3,3 %, majoritairement de luxe : Ferrari, Porsche). Ces valeurs extrêmes (*outliers*)
gonflent l'erreur du modèle et ont peu de rapport avec la cible d'OscarIA : le **vendeur
particulier grand public**.

On a comparé trois périmètres (*scope*) sur le meilleur modèle, en évaluant chacun sur le
marché courant (carnet `04_outliers.ipynb`, seuil réglable). Rappel des termes :
**erreur absolue moyenne** = écart moyen en euros entre prix prédit et prix réel ;
**coefficient de détermination** (R²) = part des variations de prix expliquée (0 à 1) ;
**couverture calibrée** = pourcentage de vrais prix tombant dans la fourchette.

| Périmètre | % du marché | Erreur absolue moyenne | Coefficient de détermination | Couverture calibrée | Largeur / prix médian |
|-----------|-------------|------------------------|------------------------------|---------------------|-----------------------|
| ≤ 100 000 € | 97 % | 4 151 € | 0,869 | 76,4 % | 41 % |
| ≤ 60 000 € | 91 % | 3 576 € | 0,795 | 85,2 % | 51 % |
| **≤ 50 000 €** | **87 %** | **3 284 €** | 0,803 | **80,3 %** | 44 % |

## Décision

Adopter **≤ 50 000 €** comme périmètre produit par défaut (paramètre `SEUIL` dans les carnets
04 et 05).

Motifs : erreur absolue moyenne la plus basse (3 284 €), couverture calibrée pile sur la cible
de 80 %, fourchette la plus serrée en euros, tout en couvrant encore 87 % du marché réel.

## Alternatives écartées

- **≤ 100 000 €** : coefficient de détermination le plus flatteur (0,869) mais **trompeur** —
  c'est en réalité le pire choix produit : erreur en euros la plus élevée **et** fourchette qui
  ne tient pas sa promesse (couverture 76 % au lieu de 80 %).
- **≤ 60 000 €** : bon aussi, mais fourchette proportionnellement plus large (51 % du prix) et
  couverture qui dépasse la cible (85 %, donc intervalles inutilement larges).

## Conséquences

- Le modèle ne prétend **pas** estimer les véhicules de luxe (> 50 000 €). Limite assumée et
  documentée — cohérent avec la cible grand public.
- **Point de méthode** : on ne pilote pas au coefficient de détermination (il baisse
  mécaniquement quand on réduit la plage de prix, sans que le modèle soit moins bon). Les juges
  retenus sont l'**erreur absolue moyenne** (ressentie par le vendeur) et la **couverture
  calibrée** (honnêteté de la fourchette).
- Le seuil reste **réglable** en tête de carnet pour ré-évaluer si les données évoluent.
