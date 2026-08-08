"""Lot d'annotation **aveugle** de plaques + page HTML de tracé de boîtes.

Mécanique héritée de `annoter.py` (arc dégâts, cf. commit tombeau) : copies sous noms
neutres mélangés, manifeste séparé que la page ne lit pas, page HTML autonome avec
sauvegarde locale et export CSV. Adaptations plaques :

- l'annotateur **trace des rectangles à la souris** (plusieurs par photo possibles) ;
- bouton « aucune plaque visible » → ligne `-1,-1,-1,-1` ;
- aveugle vis-à-vis du **modèle** : la page n'affiche aucune prédiction — l'annotation
  humaine ne doit pas être influencée par ce que le détecteur trouve ;
- **EXIF supprimés** dans les copies (photos personnelles : GPS embarqué = donnée
  personnelle ; `exif_transpose` d'abord, sinon les coordonnées tracées dans le
  navigateur ne correspondraient pas aux pixels vus par le modèle).

Export attendu : `plaques_verite.csv` (`fichier,x1,y1,x2,y2`, coordonnées en pixels
de la copie). Dès l'export : `git add -f` du CSV + manifeste (convention du plan).

CLI : uv run --project dl python dl/src/annoter_plaques.py \
          --source dl/data/plaques-perso/raw --sortie dl/data/plaques-perso/annot
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

EXTENSIONS = {".jpg", ".jpeg", ".png"}


def preparer_lot(source, sortie, cote_max: int = 1280, qualite: int = 90,
                 seed: int = 0) -> pd.DataFrame:
    """Copies neutres mélangées (EXIF retirés, orientation appliquée) + manifeste."""
    source, sortie = Path(source), Path(sortie)
    dossier_img = sortie / "img"
    dossier_img.mkdir(parents=True, exist_ok=True)

    photos = sorted(p for p in source.iterdir() if p.suffix.lower() in EXTENSIONS)
    random.Random(seed).shuffle(photos)

    lignes = []
    for i, chemin in enumerate(photos, 1):
        img = ImageOps.exif_transpose(Image.open(chemin)).convert("RGB")
        if max(img.size) > cote_max:
            ratio = cote_max / max(img.size)
            img = img.resize((round(img.width * ratio), round(img.height * ratio)),
                             Image.LANCZOS)
        nom = f"{i:04d}.jpg"
        img.save(dossier_img / nom, quality=qualite)  # save() ne copie pas les EXIF
        lignes.append({"id": f"{i:04d}", "fichier": nom, "chemin_origine": str(chemin),
                       "largeur": img.width, "hauteur": img.height})
    df = pd.DataFrame(lignes)
    df.to_csv(sortie / "manifeste.csv", index=False)
    return df


_PAGE = """<!doctype html>
<html lang="fr"><head><meta charset="utf-8"><title>Annotation plaques — OscarIA</title>
<style>
 :root{color-scheme:dark}
 body{margin:0;font:15px/1.4 system-ui,sans-serif;background:#111;color:#eee;
      display:flex;flex-direction:column;height:100vh}
 header{padding:.6rem 1rem;background:#1b1b1b;display:flex;gap:1.2rem;align-items:center;flex-wrap:wrap}
 .barre{flex:1;height:8px;background:#333;border-radius:4px;overflow:hidden;min-width:160px}
 .barre>div{height:100%;background:#4ade80;width:0}
 main{flex:1;display:flex;align-items:center;justify-content:center;overflow:hidden;
      padding:.5rem;position:relative}
 #scene{position:relative}
 #photo{max-width:100%;max-height:78vh;display:block;user-select:none;-webkit-user-drag:none}
 #calque{position:absolute;inset:0;cursor:crosshair}
 .boite{position:absolute;border:2px solid #f87171;background:rgba(248,113,113,.15)}
 footer{padding:.7rem 1rem;background:#1b1b1b;display:flex;gap:.6rem;align-items:center;flex-wrap:wrap}
 button{font:inherit;padding:.5rem .9rem;border:0;border-radius:6px;cursor:pointer;color:#111}
 .v{background:#4ade80}.z{background:#fbbf24}.n{background:#374151;color:#eee}.r{background:#9ca3af}
 kbd{background:#333;border-radius:4px;padding:0 .3rem;font-size:.85em}
 #fin{display:none;padding:2rem;text-align:center;max-width:640px}
 textarea{width:100%;height:9rem;background:#000;color:#0f0;font-family:ui-monospace}
 #badge{position:absolute;top:.6rem;left:.6rem;padding:.35rem .7rem;border-radius:6px;
        font-weight:600;background:#374151}
</style></head><body>
<header>
  <strong id="compteur">0 / 0</strong>
  <div class="barre"><div id="progres"></div></div>
  <span id="stats"></span>
  <button class="n" onclick="exporter()">exporter CSV</button>
</header>
<main>
  <div id="scene"><img id="photo" alt=""><div id="calque"></div><span id="badge"></span></div>
  <div id="fin"><h2>Terminé 🎉</h2>
    <p>Clique « exporter CSV » : dépose <code>plaques_verite.csv</code> dans le dossier
       d'annotation.</p><textarea id="apercu" readonly></textarea></div>
</main>
<footer>
  <button class="v" onclick="valider()">✔ valider la photo (Entrée)</button>
  <button class="z" onclick="aucune()">0 — aucune plaque visible</button>
  <button class="r" onclick="annulerBoite()">⌫ retirer la dernière boîte</button>
  <button class="n" onclick="naviguer(-1)">← précédente</button>
  <button class="n" onclick="naviguer(1)">suivante →</button>
  <button class="n" onclick="positionner()">⤓ reprendre à la première non faite</button>
  <span>Trace un rectangle à la souris sur <em>chaque</em> plaque (même partielles/floues),
        puis valide. <kbd>Entrée</kbd> valider · <kbd>0</kbd> aucune · <kbd>⌫</kbd> retirer ·
        <kbd>←</kbd><kbd>→</kbd>. Progression sauvegardée, tu peux fermer et reprendre.</span>
</footer>
<script>
const FICHIERS = __FICHIERS__;      // [{f:"0001.jpg", w:960, h:1280}, ...]
const CLE = "oscaria-plaques-__SIGNATURE__";
let fait = JSON.parse(localStorage.getItem(CLE) || "{}");  // f -> [[x1,y1,x2,y2],...] ou []
let i = 0, encours = [], depart = null, rect = null;

const img = document.getElementById("photo"), calque = document.getElementById("calque"),
      badge = document.getElementById("badge");

function positionner(){ i = 0; while (i < FICHIERS.length && fait[FICHIERS[i].f]) i++; afficher(); }

function echelle(){ return img.clientWidth / FICHIERS[i].w; }

function dessiner(){
  calque.querySelectorAll(".boite").forEach(b => b.remove());
  const e = echelle();
  for (const [x1,y1,x2,y2] of encours){
    const d = document.createElement("div");
    d.className = "boite";
    d.style.left = x1*e+"px"; d.style.top = y1*e+"px";
    d.style.width = (x2-x1)*e+"px"; d.style.height = (y2-y1)*e+"px";
    calque.appendChild(d);
  }
  badge.textContent = encours.length ? encours.length + " boîte(s)" :
                      (fait[FICHIERS[i]?.f]?.length === 0 ? "aucune plaque (validé)" : "0 boîte");
}

function afficher(){
  const total = FICHIERS.length, n = Object.keys(fait).length;
  document.getElementById("compteur").textContent = `${Math.min(i+1,total)} / ${total}`;
  document.getElementById("progres").style.width = (n/total*100)+"%";
  const boites = Object.values(fait).reduce((s,b)=>s+b.length,0);
  document.getElementById("stats").textContent =
    `faites ${n} · boîtes ${boites} · reste ${total-n}`;
  const fin = document.getElementById("fin"), scene = document.getElementById("scene");
  if (i >= total){ scene.style.display="none"; fin.style.display="block";
                   document.getElementById("apercu").value = csv(); return; }
  scene.style.display=""; fin.style.display="none";
  img.src = "img/" + FICHIERS[i].f;
  encours = (fait[FICHIERS[i].f] || []).map(b => b.slice());
  img.onload = dessiner;
  if (i+1 < total) new Image().src = "img/" + FICHIERS[i+1].f;
}

calque.addEventListener("mousedown", ev => {
  const r = calque.getBoundingClientRect(), e = echelle();
  depart = [(ev.clientX-r.left)/e, (ev.clientY-r.top)/e];
  rect = document.createElement("div"); rect.className = "boite"; calque.appendChild(rect);
});
calque.addEventListener("mousemove", ev => {
  if (!depart) return;
  const r = calque.getBoundingClientRect(), e = echelle();
  const x = (ev.clientX-r.left)/e, y = (ev.clientY-r.top)/e;
  const x1 = Math.min(depart[0],x), y1 = Math.min(depart[1],y),
        x2 = Math.max(depart[0],x), y2 = Math.max(depart[1],y);
  rect.style.left = x1*e+"px"; rect.style.top = y1*e+"px";
  rect.style.width = (x2-x1)*e+"px"; rect.style.height = (y2-y1)*e+"px";
});
window.addEventListener("mouseup", ev => {
  if (!depart) return;
  const r = calque.getBoundingClientRect(), e = echelle(), F = FICHIERS[i];
  const x = Math.max(0, Math.min(F.w, (ev.clientX-r.left)/e)),
        y = Math.max(0, Math.min(F.h, (ev.clientY-r.top)/e));
  const b = [Math.min(depart[0],x), Math.min(depart[1],y),
             Math.max(depart[0],x), Math.max(depart[1],y)].map(Math.round);
  depart = null; rect.remove(); rect = null;
  if (b[2]-b[0] >= 4 && b[3]-b[1] >= 4) encours.push(b);   // ignore les clics accidentels
  dessiner();
});

function sauver(v){ fait[FICHIERS[i].f] = v; localStorage.setItem(CLE, JSON.stringify(fait)); }
function valider(){ if (i>=FICHIERS.length || !encours.length) return; sauver(encours); i++; afficher(); }
function aucune(){ if (i>=FICHIERS.length) return; sauver([]); i++; afficher(); }
function annulerBoite(){ encours.pop(); dessiner(); }
function naviguer(p){ i = Math.min(FICHIERS.length, Math.max(0, i+p)); afficher(); }

document.addEventListener("keydown", ev => {
  if (ev.key === "Enter") valider();
  else if (ev.key === "0") aucune();
  else if (ev.key === "Backspace"){ ev.preventDefault(); annulerBoite(); }
  else if (ev.key === "ArrowLeft") naviguer(-1);
  else if (ev.key === "ArrowRight") naviguer(1);
});

function csv(){
  const lignes = ["fichier,x1,y1,x2,y2"];
  for (const F of FICHIERS){
    const v = fait[F.f]; if (!v) continue;
    if (!v.length) lignes.push(`${F.f},-1,-1,-1,-1`);
    else v.forEach(b => lignes.push(`${F.f},${b.join(",")}`));
  }
  return lignes.join("\\n") + "\\n";
}
function exporter(){
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([csv()], {type:"text/csv"}));
  a.download = "plaques_verite.csv"; a.click();
}
positionner();
</script></body></html>
"""


def generer_page(df: pd.DataFrame, sortie) -> Path:
    fichiers = [{"f": l.fichier, "w": int(l.largeur), "h": int(l.hauteur)}
                for l in df.itertuples()]
    signature = f"{len(df)}-{df['fichier'].iloc[0]}-{df['fichier'].iloc[-1]}"
    html = (_PAGE.replace("__FICHIERS__", pd.io.json.ujson_dumps(fichiers))
            .replace("__SIGNATURE__", signature))
    chemin = Path(sortie) / "annoter.html"
    chemin.write_text(html, encoding="utf-8")
    return chemin


if __name__ == "__main__":
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument("--source", default="dl/data/plaques-perso/raw")
    parseur.add_argument("--sortie", default="dl/data/plaques-perso/annot")
    args = parseur.parse_args()
    df = preparer_lot(args.source, args.sortie)
    page = generer_page(df, args.sortie)
    print(f"{len(df)} photos préparées (EXIF retirés) → ouvre {page}")
