#!/usr/bin/env node
'use strict';
// Runner standalone — scrape UNE annonce largus et imprime le JSON sur stdout.
//   node scrape-one.js <url-annonce-largus>
//   HEADED=1 node scrape-one.js <url>   # navigateur visible (debug)
//
// Sert au test hors n8n et valide la logique d'extraction (extract.js) avant intégration.

const { chromium } = require('playwright');
const { pageExtractor } = require('./extract');

// UA de navigateur réaliste (largus sert du HTML complet, pas d'anti-bot observé —
// on reste néanmoins poli et identifiable comme un vrai Chrome).
const USER_AGENT =
  'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 ' +
  '(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36';

const UUID_RE =
  /annonce-([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/i;

function uuidFromUrl(url) {
  const m = url.match(UUID_RE);
  return m ? m[1] : null;
}

async function scrapeAnnonce(url, { headless = true } = {}) {
  const browser = await chromium.launch({ headless });
  try {
    const ctx = await browser.newContext({
      userAgent: USER_AGENT,
      locale: 'fr-FR',
      viewport: { width: 1366, height: 900 },
    });
    const page = await ctx.newPage();
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 45_000 });

    // Consentement cookies (best-effort, non bloquant s'il n'apparaît pas).
    for (const sel of [
      '#didomi-notice-agree-button',
      'button:has-text("Tout accepter")',
      'button:has-text("Accepter")',
    ]) {
      try { await page.click(sel, { timeout: 1500 }); break; } catch { /* absent */ }
    }

    // Déclenche le lazy-load de la galerie (les vignettes n'apparaissent qu'au scroll ;
    // les URLs HD sont ensuite dérivées thumb -> large par pageExtractor).
    await page.evaluate(async () => {
      await new Promise((resolve) => {
        let y = 0;
        const timer = setInterval(() => {
          window.scrollBy(0, 800);
          y += 800;
          if (y >= document.body.scrollHeight) { clearInterval(timer); resolve(); }
        }, 100);
      });
    });
    await page.waitForTimeout(500);

    const raw = await page.evaluate(pageExtractor);
    return {
      source: 'occasion.largus.fr',
      url,
      id: uuidFromUrl(url),
      scraped_at: new Date().toISOString(),
      ...raw,
    };
  } finally {
    await browser.close();
  }
}

module.exports = { scrapeAnnonce, uuidFromUrl };

if (require.main === module) {
  const url = process.argv[2];
  if (!url) {
    console.error('Usage: node scrape-one.js <url-annonce-largus>');
    process.exit(1);
  }
  scrapeAnnonce(url, { headless: !process.env.HEADED })
    .then((a) => console.log(JSON.stringify(a, null, 2)))
    .catch((e) => { console.error('ERREUR:', e.message); process.exit(1); });
}
