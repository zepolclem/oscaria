# 0006 — Tri des vues : filtrer les photos non exploitables avant toute notation

- **Date** : 2026-08-03
- **Statut** : Acceptée
- **Répond à** : l'exigence posée par [0003](0003-etiquette-declaree-asymetrique.md) — « le tri des
  vues est un prérequis de la chaîne produit, pas une option »

## Contexte

Le désordre des photos d'annonce, chiffré sur trois lots successifs :

| mesure | valeur |
|---|---|
| photos jugées non exploitables à l'annotation (intérieur, document, montage, capture d'écran, flou) | **27 %** (177 / 648) |
| photos écartées en amont faute de véhicule visible (`detect.py`) | **19 %** (152 / 800) |
| détourages ne contenant **aucune** case de tôle de carrosserie (second lot, critère plus strict) | **49 %** (65 / 132) |

Soit environ quatre photos sur dix qui ne portent aucune information sur l'état de la carrosserie.
Inspection des cas : ce sont des habitacles, des banquettes arrière et des compartiments moteur —
sur lesquels `detect.py` déclenche légitimement, puisqu'il y voit pare-brise, montants et portières.

Un modèle d'état appliqué sans filtre note donc des tableaux de bord et des cartes grises.

## Décision

Un classifieur binaire **exploitable / non exploitable**, en tête de la chaîne État.

- **Étiquettes : aucune annotation nouvelle** — le verdict `jeter` (177) contre `intact` / `abime`
  (471), déjà présents dans `verdicts.csv`.
- **Entrée à 224 px, pas 384.** Reconnaître un document ou un habitacle est une décision globale
  sur l'image ; la résolution n'avait apporté un gain que pour la détection de dégât, qui exige un
  détail fin ([0004](0004-binaire-domaine-annonce.md)). Le réglage gagnant d'une tâche n'est pas
  transposé à une autre sans être retesté.
- **Compromis asymétrique, décidé et inscrit dans le code** : laisser passer une photo inexploitable
  pollue le score de l'annonce entière ; en écarter une bonne à tort ne coûte presque rien, une
  annonce en comptant plusieurs. Le seuil vise donc un **rappel de 0,80** sur la classe « à jeter ».

Code : `dl/src/tri_vues.py`. Checkpoint : `dl/models/tri_vues.pt`. Rejouable par
`python dl/src/experiences_annonce.py tri`.

## Mesures

Validation croisée à 5 blocs sur les 648 photos annotées :

| mesure | valeur |
|---|---|
| aire ROC | **0,972** (blocs de 0,951 à 0,994) |
| précision moyenne | **0,961** — plancher 0,273 |
| au seuil visant 0,80 de rappel | **précision 1,000**, rappel 0,819 |
| part de photos écartées | 22,4 % |

**Précision 1,00** : sur les photos que le modèle écarte, aucune n'était exploitable. Il en rate
18 %, mais n'en jette jamais une bonne.

**Transfert vérifié sur une question voisine** : appliqué à des détourages pour prédire « cette
photo contient-elle de la tôle cliquable ? » — question différente, photos différentes — il obtient
0,882 d'aire ROC, écarte 34 % des photos dont **98 % étaient effectivement inutilisables**, pour
**une seule** photo utile perdue sur 132.

## Alternatives écartées

| alternative | raison du rejet |
|---|---|
| se contenter de `detect.py` | il déclenche sur les habitacles — c'est bien une voiture, mais pas de la carrosserie. 49 % des détourages du second lot en témoignent |
| prédire le sous-type (intérieur / document / montage / flou) | 177 exemples répartis sur cinq causes laisseraient ~35 exemples par classe. On promet « exploitable / non exploitable », rien de plus |
| annoter davantage avant d'entraîner | inutile : 0,972 avec les étiquettes déjà là. Mesurer avant d'investir, comme la courbe d'apprentissage plate de la fiche 0004 |
| entrée à 384 px | non retesté par réflexe : la nature de la décision ne l'exige pas, et 224 px suffit largement au vu du résultat |

## Conséquences

- **Effet mesuré sur la chaîne** : le filtre fait passer le score au niveau annonce de 0,793 à
  **0,807** d'aire ROC ([0007](0007-pilier-etat-coherence-annonce.md)). Gain modeste mais gratuit.
- **Bénéfice inattendu** : le modèle a servi à filtrer une file d'annotation en cours, réduisant de
  161 à 113 les photos restant à trier à la main. Première brique du projet qui **rend** du temps.
- **Cas à traiter en production** : 7 annonces sur 600 perdent toutes leurs photos après filtrage.
  Le produit doit alors répondre « photos insuffisantes pour se prononcer », jamais un score par
  défaut.

## Limites

- La classe `jeter` **agrège des causes hétérogènes** (un document et une photo floue n'ont rien en
  commun visuellement) et intègre un critère subjectif : la consigne d'annotation disait « dans le
  doute, jeter ». Le modèle apprend donc aussi à imiter l'hésitation d'**un seul** annotateur, non
  expert, sans accord inter-annotateurs mesuré.
- Mesuré sur 648 photos d'un seul segment (annonces de particuliers, une plateforme, juillet 2026).
- Le seuil retenu privilégie délibérément la précision : 18 % des photos inexploitables passent
  encore. En production, elles diluent le score de l'annonce sans le fausser gravement — c'est
  l'arbitrage assumé.
