# 0004 — Colonnes écartées des variables du modèle

- **Date** : 2026-08-08
- **Statut** : Accepté
- **Pilier** : Prix

## Contexte

Le dataset porte des colonnes qui améliorent mécaniquement les scores mais qui, pour des
raisons de fond, n'ont pas leur place dans un estimateur de prix. Le motif de l'exclusion
compte ici autant que l'exclusion elle-même : deux de ces colonnes sont écartées pour des
raisons de **positionnement produit**, pas de statistique.

Les exclusions sont déclarées dans `ml/src/leboncoin.py` (constante `COLONNES_ECARTEES`),
avec leur justification attachée au code.

## Décision

| Colonne | Présence | Motif de l'exclusion |
|---|---|---|
| `car_price_min` | 45 % des annonces | **C'est l'estimation de prix de leboncoin elle-même.** L'utiliser ferait recopier un estimateur tiers au lieu d'apprendre le marché — et le modèle s'effondrerait sur les 55 % d'annonces qui ne la portent pas. Réservée comme **comparateur externe** à l'évaluation. |
| `car_price_max` | 45 % | idem `car_price_min`. |
| `old_price` | 9 % | Prix précédent de la **même** annonce. Ancre quasi circulaire pour la cible : le modèle apprendrait « le prix vaut à peu près l'ancien prix », ce qui est vrai et inutile. Réservé au pilier Date, où une baisse de prix est un signal de délai de vente. |
| `is_import` | 100 % | `false` sur 20 915 / 20 915 — variance nulle, zéro information. |
| `body` | 100 % | Vide sur 20 915 / 20 915 : la description n'est pas rendue sur les pages de liste de leboncoin. |

### Le cas `car_price_*`, qui est une décision de positionnement

OscarIA se présente comme un **tiers neutre** face aux plateformes qui ont un intérêt dans la
transaction. Bâtir son estimation sur l'estimation de leboncoin viderait cette promesse de
sens : le produit ne dirait plus « voici ce que dit le marché », mais « voici ce que dit
leboncoin ». La performance en aurait probablement bénéficié — d'où la nécessité de tracer ce
refus, sans quoi il ressemblerait à un oubli.

Ces colonnes restent utiles **à l'évaluation** : comparer nos fourchettes à celles de
leboncoin sur les 45 % d'annonces qui les portent est une mesure externe légitime. Non fait à
ce stade.

## Alternatives écartées

- **Utiliser `car_price_*` avec un indicateur de présence** (une variable « estimation
  disponible : oui / non ») : techniquement propre, et écarté quand même pour le motif de
  positionnement ci-dessus.
- **Utiliser `old_price` comme variable du pilier Prix** : écarté pour circularité. Conservé
  pour le pilier Date.
- **Supprimer ces colonnes du `raw/`** : refusé. Le `raw/` n'est jamais modifié
  (`ml/AGENTS.md`). Elles sont retirées du jeu nettoyé et restent récupérables à tout moment
  depuis le Parquet, par jointure sur `ad_id`.

## Conséquences

- `clean_leboncoin` retire `raw` et `images` du jeu nettoyé après les avoir consommés :
  les garder ferait voyager les colonnes écartées jusqu'au jeu d'entraînement, où elles
  n'ont rien à faire.
- Une évaluation comparative contre l'estimateur de leboncoin reste possible et souhaitable.
  Elle constituerait une mesure externe utile au dossier ; elle n'a pas été faite.
