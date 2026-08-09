# Décisions — socle (transverse)

Fiches de décision qui ne relèvent d'aucun pilier de modélisation : déploiement, packaging,
choix d'infrastructure, structure de l'espace de travail.

Format et règles communes : voir le [journal général](../README.md). Numérotation propre à cet
espace ; une fiche se cite « ADR Socle 0001 ».

## Index

*Aucune fiche pour l'instant.*

Décisions déjà prises et appliquées, mais pas encore consignées ici — à rédiger :

- routage de `torch`/`torchvision` vers l'index `pytorch-cpu` sous marker
  `sys_platform == 'linux'` (image de 5,53 Go ramenée à 1,5 Go ; macOS conservé sur PyPI pour
  garder le backend MPS) — et le rejet de la forme « deux extras cpu/cu128 », qu'uv n'applique
  pas dans cet espace de travail ;
- déploiement continu sur Coolify (build pack `dockercompose`, domaine et certificat générés
  par `SERVICE_FQDN_APP_8000`, webhook sur `main`) ;
- modèles embarqués dans l'image plutôt que téléchargés au démarrage — démarrer, c'est savoir
  répondre.
