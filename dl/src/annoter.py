"""Préparation d'un lot d'annotation **aveugle** + interface HTML autonome.

Pourquoi aveugle : les photos leboncoin sont rangées par état **déclaré par le vendeur**. Montrer
cet état à l'annotateur, même indirectement par le chemin du fichier, aligne son jugement dessus
(biais d'ancrage) — alors que l'objectif est justement de **mesurer l'écart** entre la déclaration
et ce que la photo montre. Les photos sont donc copiées sous des noms neutres, dans un ordre
mélangé, et l'état déclaré reste dans un manifeste que la page HTML ne lit pas.

Chaîne : échantillonnage par annonce -> copie redimensionnée -> filtre « véhicule présent »
(`detect.py`, qui ne voit jamais l'étiquette) -> page HTML -> export CSV des verdicts.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd
from PIL import Image

# Les deux extrémités de l'échelle d'état déclaré. Les états intermédiaires
# (`minor_repairs_needed`, etc.) sont écartés : « réparations mineures » désigne aussi bien une
# révision qu'une aile froissée, l'étiquette y est trop ambiguë pour servir de cible.
DOSSIERS_INTACT = ["undamaged", "excellent_condition"]
DOSSIERS_ABIME = ["damaged", "not_drivable"]


def echantillonner(racine_images, n_par_dossier: int = 200, par_annonce: int = 1,
                   seed: int = 0) -> pd.DataFrame:
    """Tire `n_par_dossier` annonces dans chacun des 4 dossiers retenus, puis des photos dedans.

    Une photo par annonce par défaut : à temps d'annotation égal, ça maximise le nombre de
    voitures différentes et supprime la question de la fuite à l'intérieur du lot (deux photos
    d'une même voiture ne peuvent pas se retrouver de part et d'autre d'un découpage).

    La photo est tirée au hasard et non prise en première position : la photo 1 d'une annonce est
    presque toujours le plan de trois-quarts avant valorisant, ce qui biaiserait le cadrage.
    """
    rng = random.Random(seed)
    racine = Path(racine_images)
    lignes = []
    for dossier in DOSSIERS_INTACT + DOSSIERS_ABIME:
        annonces = sorted(p for p in (racine / dossier).iterdir() if p.is_dir())
        for annonce in rng.sample(annonces, min(n_par_dossier, len(annonces))):
            photos = sorted(annonce.glob("*.jpg"))
            for photo in rng.sample(photos, min(par_annonce, len(photos))):
                lignes.append({
                    "chemin_origine": str(photo),
                    "ad_id": annonce.name,
                    "etat_declare": dossier,
                    "classe_declaree": "intact" if dossier in DOSSIERS_INTACT else "abime",
                })
    df = pd.DataFrame(lignes).sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "id", [f"{i:04d}" for i in range(1, len(df) + 1)])
    return df


def copier_anonyme(df: pd.DataFrame, dossier_img, cote_max: int = 1024,
                   qualite: int = 88) -> pd.DataFrame:
    """Copie chaque photo sous un nom neutre (`0001.jpg`), redimensionnée.

    Deux effets : le chemin ne révèle plus l'état déclaré, et la page se charge instantanément.
    Les originaux dans `raw/` ne sont jamais touchés.
    """
    dossier_img = Path(dossier_img)
    dossier_img.mkdir(parents=True, exist_ok=True)
    fichiers = []
    for ligne in df.itertuples():
        img = Image.open(ligne.chemin_origine).convert("RGB")
        if max(img.size) > cote_max:
            ratio = cote_max / max(img.size)
            img = img.resize((round(img.width * ratio), round(img.height * ratio)),
                             Image.LANCZOS)
        nom = f"{ligne.id}.jpg"
        img.save(dossier_img / nom, quality=qualite)
        fichiers.append(nom)
    return df.assign(fichier=fichiers)


def filtrer_vehicule(df: pd.DataFrame, dossier_img, device, fraction_min: float = 0.3,
                     score_min: float = 0.5) -> pd.DataFrame:
    """Marque les photos où aucun véhicule n'occupe une part suffisante du cadre.

    Écarte automatiquement habitacles, compteurs, cartes grises et pièces détachées : aucun
    véhicule détecté, donc rien à juger. Le détecteur ne voit **jamais** l'étiquette déclarée —
    un filtre qui l'utiliserait fabriquerait le raccourci qu'on cherche à éviter.
    """
    from detect import detecter, load_detector

    dossier_img = Path(dossier_img)
    detecteur, categories = load_detector(device)
    res = detecter(detecteur, categories, device,
                   [str(dossier_img / f) for f in df["fichier"]], score_min=score_min)
    return df.assign(
        vehicule=[r["boite"] is not None for r in res],
        fraction_cadre=[r["fraction_cadre"] for r in res],
        garde=[r["fraction_cadre"] is not None and r["fraction_cadre"] >= fraction_min
               for r in res],
    )


_PAGE = """<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><title>Tri des photos — OscarIA</title>
<style>
 :root{color-scheme:dark}
 body{margin:0;font:15px/1.4 system-ui,sans-serif;background:#111;color:#eee;
      display:flex;flex-direction:column;height:100vh}
 header{padding:.6rem 1rem;background:#1b1b1b;display:flex;gap:1.2rem;align-items:center;
        flex-wrap:wrap}
 .barre{flex:1;height:8px;background:#333;border-radius:4px;overflow:hidden;min-width:160px}
 .barre>div{height:100%;background:#4ade80;width:0}
 main{flex:1;display:flex;align-items:center;justify-content:center;overflow:hidden;padding:.5rem}
 img{max-width:100%;max-height:100%;object-fit:contain}
 footer{padding:.7rem 1rem;background:#1b1b1b;display:flex;gap:.6rem;align-items:center;
        flex-wrap:wrap}
 button{font:inherit;padding:.5rem .9rem;border:0;border-radius:6px;cursor:pointer;color:#111}
 .i{background:#4ade80}.a{background:#f87171}.j{background:#9ca3af}.n{background:#374151;color:#eee}
 kbd{background:#333;border-radius:4px;padding:0 .3rem;font-size:.85em}
 #fin{display:none;padding:2rem;text-align:center}
 textarea{width:100%;height:9rem;background:#000;color:#0f0;font-family:ui-monospace}
 #actuel{padding:.35rem .7rem;border-radius:6px;font-weight:600}
 main{position:relative}
 #actuel{position:absolute;top:.6rem;left:.6rem}
</style></head><body>
<header>
  <strong id="compteur">0 / 0</strong>
  <div class="barre"><div id="progres"></div></div>
  <span id="stats"></span>
  <button class="n" onclick="exporter()">exporter CSV</button>
</header>
<main><span id="actuel"></span><img id="photo" alt=""><div id="fin"><h2>Terminé</h2>
  <p>Clique « exporter CSV », le fichier arrive dans tes téléchargements.</p>
  <textarea id="apercu" readonly></textarea></div></main>
<footer>
  <button class="i" onclick="juger('intact')">1 — intact</button>
  <button class="a" onclick="juger('abime')">2 — abîmé</button>
  <button class="j" onclick="juger('jeter')">3 — à jeter</button>
  <button class="n" onclick="naviguer(-1)">← précédente</button>
  <button class="n" onclick="naviguer(1)">suivante →</button>
  <button class="n" onclick="positionner()">⤓ reprendre au premier non trié</button>
  <span>raccourcis <kbd>1</kbd> <kbd>2</kbd> <kbd>3</kbd> <kbd>&larr;</kbd> <kbd>&rarr;</kbd>
        — reculer ne supprime rien, appuie sur 1/2/3 pour corriger. Progression sauvegardée,
        tu peux fermer et reprendre.</span>
</footer>
<script>
const FICHIERS = __FICHIERS__;
const CLE = "oscaria-tri-__SIGNATURE__";
let verdicts = JSON.parse(localStorage.getItem(CLE) || "{}");
let i = 0;

const ETIQUETTE = {intact:["intact","#4ade80"], abime:["abîmé","#f87171"],
                   jeter:["à jeter","#9ca3af"]};

function positionner(){          // saute au premier non trié
  i = 0;
  while (i < FICHIERS.length && verdicts[FICHIERS[i]]) i++;
  afficher();
}
function afficher(){
  const total = FICHIERS.length, faits = Object.keys(verdicts).length;
  document.getElementById("compteur").textContent = `${Math.min(i+1,total)} / ${total}`;
  document.getElementById("progres").style.width = (faits/total*100)+"%";
  const c = {intact:0, abime:0, jeter:0};
  Object.values(verdicts).forEach(v => c[v]++);
  document.getElementById("stats").textContent =
    `intact ${c.intact} · abîmé ${c.abime} · jetées ${c.jeter} · reste ${total-faits}`;

  const img = document.getElementById("photo"), fin = document.getElementById("fin"),
        badge = document.getElementById("actuel");
  if (i >= total){
    img.style.display="none"; badge.style.display="none"; fin.style.display="block";
    document.getElementById("apercu").value = csv();
    return;
  }
  img.style.display=""; fin.style.display="none"; img.src = "img/" + FICHIERS[i];
  if (i+1 < total) new Image().src = "img/" + FICHIERS[i+1];   // préchargement
  const v = verdicts[FICHIERS[i]];                             // verdict déjà posé ?
  if (v){ badge.style.display=""; badge.textContent = "déjà noté : " + ETIQUETTE[v][0];
          badge.style.background = ETIQUETTE[v][1]; badge.style.color = "#111"; }
  else  { badge.style.display="none"; }
}
function juger(v){
  if (i >= FICHIERS.length) return;
  verdicts[FICHIERS[i]] = v;
  localStorage.setItem(CLE, JSON.stringify(verdicts));
  i++; afficher();
}
function naviguer(pas){          // déplacement seul : ne supprime AUCUN verdict
  i = Math.min(FICHIERS.length, Math.max(0, i + pas));
  afficher();
}
function csv(){
  return "fichier,verdict\\n" + FICHIERS.filter(f => verdicts[f])
         .map(f => `${f},${verdicts[f]}`).join("\\n") + "\\n";
}
function exporter(){
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv()], {type:"text/csv"}));
  a.download = "verdicts.csv"; a.click();
}
addEventListener("keydown", e => {
  if (e.key === "1") juger("intact");
  else if (e.key === "2") juger("abime");
  else if (e.key === "3") juger("jeter");
  else if (e.key === "ArrowLeft") naviguer(-1);
  else if (e.key === "ArrowRight") naviguer(1);
});
positionner();
</script></body></html>
"""


def generer_html(df: pd.DataFrame, chemin_html, signature: str = "v1") -> Path:
    """Écrit la page d'annotation. Ne reçoit que les noms de fichiers neutres.

    `signature` sert de clé de sauvegarde dans le navigateur : la changer repart de zéro,
    la garder permet de fermer l'onglet et de reprendre où on s'était arrêté.
    """
    chemin_html = Path(chemin_html)
    chemin_html.parent.mkdir(parents=True, exist_ok=True)
    page = (_PAGE
            .replace("__FICHIERS__", json.dumps(list(df["fichier"])))
            .replace("__SIGNATURE__", signature))
    chemin_html.write_text(page, encoding="utf-8")
    return chemin_html


_PAGE_GRILLE = """<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><title>Grille — tôle saine — OscarIA</title>
<style>
 :root{color-scheme:dark}
 body{margin:0;font:15px/1.4 system-ui,sans-serif;background:#111;color:#eee;
      display:flex;flex-direction:column;height:100vh}
 header{padding:.6rem 1rem;background:#1b1b1b;display:flex;gap:1.2rem;align-items:center;
        flex-wrap:wrap}
 .barre{flex:1;height:8px;background:#333;border-radius:4px;overflow:hidden;min-width:160px}
 .barre>div{height:100%;background:#4ade80;width:0}
 main{flex:1;display:flex;align-items:center;justify-content:center;overflow:hidden;padding:.5rem}
 #cadre{position:relative;max-width:100%;max-height:100%}
 #cadre img{display:block;max-width:100%;max-height:calc(100vh - 150px);object-fit:contain}
 #grille{position:absolute;inset:0;display:grid}
 .case{border:1px solid rgba(255,255,255,.35);cursor:pointer;transition:background .08s}
 .case:hover{background:rgba(255,255,255,.12)}
 .case.on{background:rgba(74,222,128,.42);border-color:#4ade80;border-width:2px}
 footer{padding:.7rem 1rem;background:#1b1b1b;display:flex;gap:.6rem;align-items:center;
        flex-wrap:wrap}
 button{font:inherit;padding:.5rem .9rem;border:0;border-radius:6px;cursor:pointer;color:#111;
        background:#374151;color:#eee}
 button.v{background:#4ade80;color:#111}
 kbd{background:#333;border-radius:4px;padding:0 .3rem;font-size:.85em}
 #fin{display:none;padding:2rem;text-align:center}
 textarea{width:100%;height:9rem;background:#000;color:#0f0;font-family:ui-monospace}
</style></head><body>
<header>
  <strong id="compteur">0 / 0</strong>
  <div class="barre"><div id="progres"></div></div>
  <span id="stats"></span>
  <button onclick="exporter()">exporter CSV</button>
</header>
<main>
  <div id="cadre"><img id="photo" alt=""><div id="grille"></div></div>
  <div id="fin"><h2>Terminé</h2>
    <p>Clique « exporter CSV », le fichier arrive dans tes téléchargements.</p>
    <textarea id="apercu" readonly></textarea></div>
</main>
<footer>
  <button class="v" onclick="valider()">Entrée — valider et suivante</button>
  <button onclick="tout()">A — tout sélectionner</button>
  <button onclick="rien()">R — tout désélectionner</button>
  <button onclick="naviguer(-1)">← précédente</button>
  <button onclick="positionner()">⤓ reprendre au premier non fait</button>
  <span>clique les cases montrant de la <strong>tôle saine</strong> — pas le fond, pas les
        vitres, pas les roues. <kbd>Entrée</kbd> <kbd>A</kbd> <kbd>R</kbd> <kbd>&larr;</kbd>
        <kbd>&rarr;</kbd>. Progression sauvegardée.</span>
</footer>
<script>
const FICHIERS = __FICHIERS__, COLS = __COLS__, RANGS = __RANGS__;
const CLE = "oscaria-grille-__SIGNATURE__";
// selection[fichier] = [indices cochés] ; une entrée existe dès que la photo est validée,
// même vide : « aucune case utilisable » est une réponse, pas une absence de réponse.
let selection = JSON.parse(localStorage.getItem(CLE) || "{}");
let courant = new Set();
let i = 0;

const grille = document.getElementById("grille");
grille.style.gridTemplateColumns = `repeat(${COLS}, 1fr)`;
grille.style.gridTemplateRows = `repeat(${RANGS}, 1fr)`;
for (let k = 0; k < COLS * RANGS; k++){
  const d = document.createElement("div");
  d.className = "case"; d.dataset.k = k;
  d.onclick = () => basculer(k);
  grille.appendChild(d);
}
const cases = () => [...grille.children];

function basculer(k){
  courant.has(k) ? courant.delete(k) : courant.add(k);
  cases()[k].classList.toggle("on", courant.has(k));
}
function tout(){ for (let k = 0; k < COLS*RANGS; k++) if (!courant.has(k)) basculer(k); }
function rien(){ [...courant].forEach(basculer); }

function positionner(){
  i = 0;
  while (i < FICHIERS.length && selection[FICHIERS[i]] !== undefined) i++;
  afficher();
}
function afficher(){
  const total = FICHIERS.length, faits = Object.keys(selection).length;
  const tuiles = Object.values(selection).reduce((s, v) => s + v.length, 0);
  document.getElementById("compteur").textContent = `${Math.min(i+1,total)} / ${total}`;
  document.getElementById("progres").style.width = (faits/total*100)+"%";
  document.getElementById("stats").textContent =
    `${faits} photos faites · ${tuiles} tuiles récoltées · reste ${total-faits}`;

  const cadre = document.getElementById("cadre"), fin = document.getElementById("fin");
  if (i >= total){
    cadre.style.display = "none"; fin.style.display = "block";
    document.getElementById("apercu").value = csv(); return;
  }
  cadre.style.display = ""; fin.style.display = "none";
  document.getElementById("photo").src = "detourages/" + FICHIERS[i];
  if (i+1 < total) new Image().src = "detourages/" + FICHIERS[i+1];
  courant = new Set(selection[FICHIERS[i]] || []);
  cases().forEach((c, k) => c.classList.toggle("on", courant.has(k)));
}
function valider(){
  if (i >= FICHIERS.length) return;
  selection[FICHIERS[i]] = [...courant].sort((a,b) => a-b);
  localStorage.setItem(CLE, JSON.stringify(selection));
  i++; afficher();
}
function naviguer(pas){            // déplacement seul : ne supprime aucune sélection
  i = Math.min(FICHIERS.length, Math.max(0, i + pas));
  afficher();
}
function csv(){
  // case = -1 : photo examinée, aucune case utilisable (intérieur, moteur, document…).
  // C'est une réponse, et elle doit être traçable — sinon « rien d'utilisable » devient
  // indistinguable de « pas encore vue », et le taux de rejet n'est plus mesurable.
  const l = ["fichier,case"];
  for (const [f, ks] of Object.entries(selection)) {
    if (ks.length === 0) l.push(`${f},-1`);
    else ks.forEach(k => l.push(`${f},${k}`));
  }
  return l.join("\\n") + "\\n";
}
function exporter(){
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv()], {type:"text/csv"}));
  a.download = "cases_saines.csv"; a.click();
}
addEventListener("keydown", e => {
  if (e.key === "Enter" || e.key === "ArrowRight") valider();
  else if (e.key === "ArrowLeft") naviguer(-1);
  else if (e.key.toLowerCase() === "a") tout();
  else if (e.key.toLowerCase() === "r") rien();
  else if (/^[1-9]$/.test(e.key)) basculer(parseInt(e.key) - 1);
});
positionner();
</script></body></html>
"""


def generer_html_grille(df: pd.DataFrame, chemin_html, cols: int = 3, rangs: int = 2,
                        signature: str = "grille1") -> Path:
    """Page d'annotation par grille : cliquer les cases montrant de la tôle saine.

    Plus rapide qu'un tri de tuiles isolées : la photo entière donne le contexte, et un clic
    étiquette une case. Les coordonnées ne sont pas figées ici — seul l'index de case est exporté,
    ce qui permet de changer la grille sans réannoter (`negatifs.decouper_cases`).

    **Une photo validée sans aucune case cochée est une réponse** (« rien d'utilisable ici »), pas
    une absence de réponse : elle compte comme faite et n'est pas reproposée.
    """
    chemin_html = Path(chemin_html)
    chemin_html.parent.mkdir(parents=True, exist_ok=True)
    page = (_PAGE_GRILLE
            .replace("__FICHIERS__", json.dumps(list(df["fichier"])))
            .replace("__COLS__", str(cols))
            .replace("__RANGS__", str(rangs))
            .replace("__SIGNATURE__", signature))
    chemin_html.write_text(page, encoding="utf-8")
    return chemin_html


def fusionner_verdicts(manifeste: pd.DataFrame, chemin_csv) -> pd.DataFrame:
    """Recolle les verdicts annotés au manifeste (état déclaré, ad_id, chemin d'origine)."""
    verdicts = pd.read_csv(chemin_csv)
    return manifeste.merge(verdicts, on="fichier", how="inner")
