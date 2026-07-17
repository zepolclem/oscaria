# ADR 0002 — Périmètre produit : marché ≤ 50 000 €

- **Statut** : Accepté
- **Date** : 2026-07-17 (chiffres mis à jour le 2026-07-17 après correctif de parsing, cf. ADR 0004)

## Contexte

Le jeu de données contient une queue de voitures chères, marginale pour la cible d'OscarIA :
le **vendeur particulier grand public**. Ces valeurs extrêmes (*outliers*) gonflent l'erreur
du modèle.

On a comparé trois périmètres (*scope*) sur le pipeline final (modèle quantile + calibration,
avec fréquence du modèle exact), en évaluant chacun sur son propre marché (carnet
`04_perimetre_fourchette.ipynb`, seuil réglable). Rappel des termes :
**erreur absolue moyenne** = écart moyen en euros entre prix prédit et prix réel ;
**coefficient de détermination** (R²) = part des variations de prix expliquée (0 à 1) ;
**couverture calibrée** = pourcentage de vrais prix tombant dans la fourchette (visée 80 %).

| Périmètre | % du marché | Erreur absolue moyenne (q50) | Coefficient de détermination | Couverture calibrée | Largeur / prix médian |
|-----------|-------------|------------------------------|------------------------------|---------------------|-----------------------|
| ≤ 100 000 € | 97 % | 3 494 € | 0,895 | 81,6 % | 44 % |
| ≤ 60 000 € | 91 % | 3 136 € | 0,844 | 78,8 % | 40 % |
| **≤ 50 000 €** | **87 %** | **2 931 €** | 0,839 | 76,7 % | **39 %** |

## Décision

Adopter **≤ 50 000 €** comme périmètre produit par défaut. **Depuis le 2026-07-17, le
périmètre est appliqué dès le préprocessing** (`clean_cars`, paramètre `max_price`, défaut
50 000 ; `max_price=None` pour les données complètes) — toute la chaîne travaille sur le
marché cible. Le carnet de justification a été retiré ; les chiffres de comparaison des
seuils restent consignés dans cette fiche.

Motifs : erreur absolue moyenne la plus basse (2 931 €) et fourchette la plus serrée en
proportion (39 % du prix médian), sur le vrai marché de volume (87 % des annonces). La
couverture (76,7 %) est légèrement sous la cible de 80 % — écart compatible avec le bruit
d'échantillonnage (~400 voitures de calibration) et suivi comme point d'attention.

## Alternatives écartées

- **≤ 100 000 €** : meilleur coefficient de détermination (0,895) et couverture (81,6 %), mais
  erreur en euros nettement plus élevée (3 494 €) et fourchette plus large — le R² plus haut
  est en partie un artefact de la plage de prix plus étendue, pas une meilleure précision pour
  le vendeur type.
- **≤ 60 000 €** : intermédiaire sur tous les critères, sans avantage décisif sur l'un d'eux.

## Conséquences

- Le modèle ne prétend **pas** estimer les véhicules de luxe (> 50 000 €). Limite assumée et
  documentée — cohérent avec la cible grand public.
- **Point de méthode** : on ne pilote pas au coefficient de détermination (il baisse
  mécaniquement quand on réduit la plage de prix, sans que le modèle soit moins bon). Les juges
  retenus sont l'**erreur absolue moyenne** (ressentie par le vendeur) et la **couverture
  calibrée** (honnêteté de la fourchette).
- Le seuil reste **réglable** en tête de carnet pour ré-évaluer si les données évoluent.
