'use strict';
// Extraction d'une annonce occasion.largus.fr — cœur réutilisable (Phase 1).
//
// `pageExtractor` s'exécute DANS le contexte navigateur (page.evaluate) : aucune
// dépendance Node, aucune référence à la portée module. Copiable tel quel dans un
// Code node n8n (ou passé à page.evaluate par scrape-one.js / le node n8n-playwright).
//
// Deux sources combinées pour la robustesse :
//   - JSON-LD schema.org/Car + BreadcrumbList : backbone stable typé (prix, km, marque, localisation)
//   - DOM section "Informations générales" (.grid label/valeur) : champs bruts largus (boîte, portes,
//     puissance, mise en circulation...) absents du JSON-LD.
// `informations_generales` conserve les libellés largus TELS QUELS (schéma brut ; mapping vers
// ml/src/features.py différé — voir README).

function pageExtractor() {
  const clean = (s) => (s == null ? null : String(s).replace(/\s+/g, ' ').trim());

  // --- JSON-LD : Car + BreadcrumbList ---
  let car = null;
  let breadcrumb = null;
  for (const s of document.querySelectorAll('script[type="application/ld+json"]')) {
    let data;
    try { data = JSON.parse(s.textContent); } catch { continue; }
    for (const node of Array.isArray(data) ? data : [data]) {
      if (node && node['@type'] === 'Car') car = node;
      if (node && node['@type'] === 'BreadcrumbList') breadcrumb = node;
    }
  }

  // --- Section "Informations générales" : toutes les paires label/valeur brutes ---
  const infos = {};
  const h2Info = [...document.querySelectorAll('h2, h3')]
    .find((h) => /informations? g[ée]n[ée]rales?/i.test(h.textContent));
  if (h2Info) {
    // le conteneur (grid de lignes) suit le titre ; on saute les nœuds vides
    let grid = h2Info.nextElementSibling;
    while (grid && !grid.querySelector('span')) grid = grid.nextElementSibling;
    for (const row of grid ? grid.querySelectorAll(':scope > div') : []) {
      const spans = row.querySelectorAll('span');
      if (spans.length >= 2) {
        const k = clean(spans[0].textContent);
        const v = clean(spans[spans.length - 1].textContent);
        if (k) infos[k] = v;
      }
    }
  }

  // --- Description (élément suivant le titre "Description") ---
  let description = null;
  const h2Desc = [...document.querySelectorAll('h2, h3')]
    .find((h) => /^description\b/i.test(h.textContent.trim()));
  if (h2Desc) {
    const sib = h2Desc.nextElementSibling;
    description = clean(sib ? sib.textContent
      : h2Desc.parentElement.textContent.replace(/^\s*Description/i, ''));
  }

  // --- Vendeur : SIRET + type pro/particulier ---
  const bodyText = document.body.innerText || '';
  const siretMatch = bodyText.match(/SIRET\D*(\d(?:[\d\s]{12,18}\d))/i);
  const siret = siretMatch ? siretMatch[1].replace(/\s+/g, '') : null;
  // SIRET = signal pro fiable (un particulier n'en a pas) ; sinon on tombe sur les libellés.
  const vendeurType = siret ? 'pro'
    : /particulier/i.test(bodyText) ? 'particulier'
    : /professionnel|vendeur\s*pro/i.test(bodyText) ? 'pro'
    : null;

  // --- Localisation depuis le breadcrumb (Accueil > Voiture occasion > Région > Dept > Ville > ...) ---
  let localisation = null;
  if (breadcrumb && Array.isArray(breadcrumb.itemListElement)) {
    const names = breadcrumb.itemListElement.map((x) => x.name);
    localisation = { region: names[2] || null, departement: names[3] || null, ville: names[4] || null };
  }
  const cpMatch = bodyText.match(/\b(\d{5})\b\s*\)?\s*-\s*[A-Za-zÀ-ÿ' -]+/);
  if (cpMatch) {
    localisation = localisation || {};
    localisation.code_postal = cpMatch[1];
  }

  // --- Prix : JSON-LD (typé) sinon repli DOM sur un montant en € ---
  let prix_eur = car && car.offers ? Number(car.offers.price) : null;
  if (!prix_eur) {
    const m = bodyText.match(/(\d[\d\s ]{2,})\s*€/);
    if (m) prix_eur = Number(m[1].replace(/[\s ]/g, ''));
  }

  // --- Images : assets largus, thumb -> large (HD), dédup + tri par index ---
  const urls = new Set();
  for (const el of document.querySelectorAll('img, source')) {
    const cands = [
      el.currentSrc, el.getAttribute('src'), el.getAttribute('data-src'),
      ...((el.getAttribute('srcset') || '').split(',').map((s) => s.trim().split(/\s+/)[0])),
    ];
    for (const u of cands) {
      if (u && /assets\d*\.largus\.fr/.test(u) && /\/(?:large|thumb)-/.test(u)) {
        urls.add(u.replace('/thumb-', '/large-'));
      }
    }
  }
  const idxOf = (u) => { const m = u.match(/-(\d+)\.\w+(?:\?|$)/); return m ? Number(m[1]) : 1e9; };
  const images = [...urls].sort((a, b) => idxOf(a) - idxOf(b));

  return {
    titre: clean(document.querySelector('h1') && document.querySelector('h1').textContent)
      || (car && car.name) || null,
    prix_eur,
    marque: car && car.brand ? car.brand.name : infos['Marque'] || null,
    modele: car ? car.model : infos['Modèle'] || null,
    informations_generales: infos, // schéma largus BRUT
    description,
    vendeur: { type: vendeurType, siret },
    localisation,
    images,
    jsonld_car: car, // backbone brut conservé pour recoupement
  };
}

module.exports = { pageExtractor };
