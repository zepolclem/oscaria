# 0003 — Ce que le modèle prédit : un prix **demandé**, dans un périmètre borné

- **Date** : 2026-08-08
- **Statut** : Accepté
- **Pilier** : Prix

## Contexte

Deux questions se posent avant toute modélisation, et elles n'ont pas de réponse technique :
**quelle grandeur on prédit**, et **sur quel segment de marché**. Elles sont réunies ici parce
qu'elles répondent toutes deux à « de quoi parle-t-on ».

La première a failli passer inaperçue. L'interface annonçait *« cette fourchette contient le
prix de vente »* jusqu'à ce que la rédaction de cette fiche impose de vérifier la nature
exacte de `prix_eur` — c'est un prix **affiché sur une annonce en ligne**. Aucune colonne, ni
aucun attribut du bloc JSON, ne porte un prix de transaction conclue.

## Décision

### 1. La cible est le prix demandé

Le modèle apprend `prix_eur` : le montant qu'un particulier **affiche** sur son annonce.

La question à laquelle il répond est donc :

> « À combien des particuliers affichent-ils une voiture comparable à la mienne ? »

et **non** :

> « Combien vais-je en tirer ? »

L'écart entre les deux est la négociation, et il est systématiquement dans le même sens : le
prix conclu est en général **inférieur** au prix affiché. Le modèle est donc structurellement
**optimiste** pour un vendeur qui l'interpréterait comme un prix de vente.

Ce biais n'est pas mesurable avec ces données : il faudrait des prix de transaction, que
leboncoin ne publie pas. Il est donc **déclaré, pas corrigé**.

### 2. Le périmètre est [500 €, 50 000 €]

| Borne | Annonces écartées | Motif |
|---|---|---|
| `prix_eur >= 500` | **340** / 20 915 | en dessous, ce sont des épaves, des pièces détachées ou des erreurs de saisie (le minimum brut est à 1 €) — pas un marché d'occasion |
| `prix_eur <= 50 000` | **147** / 20 915 | au-dessus, véhicules de luxe et de collection, dont la cote suit la rareté et non les caractéristiques mécaniques ; 147 exemples ne suffisent pas à l'apprendre |
| `annee >= 1980` | **475** / 20 915 | avant 1980, c'est de la collection : l'âge fait *monter* le prix au lieu de le faire baisser, relation inverse de celle du reste du jeu |

S'y ajoutent 71 quasi-doublons (même véhicule republié) retirés avant tout découpage, pour
éviter qu'une même voiture se retrouve à la fois à l'entraînement et au test.

**Total : 1 033 annonces écartées sur 20 915, soit 4,9 %. Il en reste 19 882.**

Chaque règle est journalisée par `clean_leboncoin`, qui renvoie un tableau chiffrant ligne à
ligne le coût de chaque filtre. Aucune suppression silencieuse.

## Alternatives écartées

- **Prédire un prix de vente** : impossible, la donnée n'existe pas. Y parvenir supposerait un
  partenariat avec une plateforme disposant des transactions.
- **Appliquer une décote forfaitaire** (« le prix conclu vaut 90 % du prix affiché ») pour
  convertir la prédiction : ce chiffre serait inventé. Une décote non mesurée maquillerait le
  biais au lieu de le déclarer, et donnerait une fausse impression de rigueur.
- **Ne pas borner le prix** : mesuré au carnet 03 — la distribution est très asymétrique
  (médiane 4 350 €, maximum brut 999 999 €), et les extrêmes tirent l'apprentissage vers le
  haut de gamme, qui n'est pas la cible produit.
- **Écarter les véhicules non roulants** : refusé. Ils représentent 5 % du jeu et constituent
  un vrai segment du marché entre particuliers. Le modèle doit savoir les estimer.

## Conséquences

- **L'interface doit dire « prix affiché », pas « prix de vente ».** Corrigé dans
  `app/src/page_prix.py`, où la mention précise en outre que le prix réellement obtenu est
  généralement plus bas après négociation.
- Un véhicule hors périmètre reçoit quand même une estimation (l'API ne le refuse pas), mais
  celle-ci est extrapolée : le modèle rabat sur sa dernière coupe apprise. Cas non traité à ce
  stade, à consigner si le produit s'ouvre au haut de gamme.
- La borne haute est le seul paramètre du périmètre exposé en argument
  (`clean_leboncoin(max_price=…)`), pour pouvoir remesurer un autre périmètre sans toucher au
  `raw/`.
