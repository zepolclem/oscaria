# ML 0001 — Dataset `leboncoin-private` retenu pour le pilier Prix

- **Date** : 2026-08-08
- **Statut** : Accepté
- **Pilier** : Prix

## Contexte

`ml/AGENTS.md` impose un processus par dataset candidat : EDA d'inspection standardisée →
**STOP, validation utilisateur** → verdict *garder / écarter* consigné. L'inspection a été
menée (`ml/notebooks/leboncoin-private/01_eda_inspection.ipynb`) et le verdict prononcé, mais
jamais écrit — les fiches de l'arc précédent ont été supprimées à la remise à zéro du
2026-08-06. Cette fiche comble le manque a posteriori, sur la base des chiffres mesurés.

Le dataset provient d'une collecte propre (`collecte/scraper/`) vers un Postgres, rapatriée
en Parquet par `collecte/scraper/dataset.py`.

## Décision

**Dataset gardé**, et retenu comme unique candidat du pilier Prix à ce stade.

| Sujet | Constat mesuré | Conséquence |
|---|---|---|
| Volume | 20 915 annonces | suffisant pour un modèle à arbres sur ~15 variables |
| Vendeurs | `owner_type` = `private` sur **20 915 / 20 915** | correspond exactement à la cible produit (vendeur particulier) |
| Complétude | marque, modèle, année, kilométrage, énergie, boîte, état : renseignés sur la quasi-totalité | pas d'imputation massive nécessaire |
| Attributs riches | bloc JSON `attributes` : puissance, portes, places, Crit'Air, contrôle technique, date de mise en circulation | permet des variables que les colonnes « chaudes » du Parquet ne portent pas |
| Colonnes à variance nulle | `is_import` = `false` sur 20 915 / 20 915 ; `body` vide sur 20 915 / 20 915 | écartées (cf. fiche 0003) |
| Photos | 22 204 images associées | alimentent le pilier plaques, pas le pilier Prix |

> **Erratum (2026-08-09)** — la ligne « Photos » ci-dessus est périmée : les photos
> n'alimentent plus le pilier plaques (plaques floutées à la source par leboncoin, dossier
> `raw/` constaté vide). Voir la fiche
> [ADR DL 0003](../dl/0003-cloture-arc-leboncoin-uc3m.md). Sans effet sur la présente
> décision : le pilier Prix n'utilise que les annonces, pas les photos.

## Alternatives écartées

- **`french-second-hand-cars`** (dataset public, collecte ~février 2023) : conservé comme
  point de comparaison, code de nettoyage dédié dans `ml/src/features.py`. Écarté comme
  dataset principal — données de 2023, marché décalé, et il mélange professionnels et
  particuliers alors que la cible produit est le particulier.
- **`la-centrale-fr`** : arborescence créée, jamais alimentée. Le scraping concurrentiel est
  juridiquement risqué en France (jurisprudence leboncoin / La Centrale), et la collecte
  propre rendait ce candidat inutile.

## Limites de ce dataset — ce qu'il ne permet pas

Ces limites ne sont pas des défauts à corriger, ce sont les bornes de ce que le produit peut
honnêtement affirmer.

1. **Aucun professionnel.** Le modèle ne dit rien du marché pro, ni de l'écart de prix entre
   une vente entre particuliers et une reprise en concession. Le périmètre produit s'arrête
   au particulier.
2. **Aucun historique d'annonce.** Une seule photographie du marché, à une date. Impossible
   d'en tirer un délai de vente ou une saisonnalité — le pilier Date reste hors d'atteinte
   avec ces données seules.
3. **Aucun prix de vente conclu.** Cf. fiche 0002 : la cible est un prix demandé.
4. **État déclaratif.** `vehicle_damage` est saisi par le vendeur, jamais vérifié. Un vendeur
   optimiste décrit sa voiture en « bon état » et la met en vente au prix correspondant : le
   modèle apprend la cohérence entre déclaration et prix demandé, pas la réalité mécanique.
5. **Biais de plateforme.** leboncoin n'est pas tout le marché de l'occasion. Ses prix ne se
   transposent pas tels quels à une reprise, une vente à un mandataire ou une enchère.
6. **Date figée.** La collecte s'arrête au 2026-07-27 (`date_reference` de l'artefact). Le
   modèle vieillit à partir de là ; c'est ce repère qui dit quand ré-entraîner.

## Conséquences

- Le nettoyage et les variables dérivées sont propres à ce dataset :
  `ml/src/leboncoin.py`, distinct de `ml/src/features.py`.
- Les limites 1 à 5 doivent apparaître dans l'interface, pas seulement ici. Appliqué : la
  mention affichée au vendeur cite le prix demandé, l'état déclaratif et la couverture
  inégale.
- Un second dataset reste souhaitable pour mesurer la robustesse hors leboncoin. Non fait, et
  assumé comme tel.
