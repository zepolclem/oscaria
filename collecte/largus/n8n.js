#!/usr/bin/env node
'use strict';
// CLI de pilotage du n8n-playwright (VPS) pour la collecte largus.
//
//   node n8n.js deploy               # (ré)installe + active le workflow "largus-collect" via l'API REST
//   node n8n.js collect --pages 1-2  # appelle le webhook page par page, écrit le NDJSON + les images
//   node n8n.js status               # liste le workflow et l'URL du webhook
//
// Config lue depuis .env (gitignored) : N8N_BASE_URL, N8N_API_KEY, N8N_WEBHOOK_PATH.
// `extract.js` reste la source de vérité de l'extraction : son pageExtractor est INLINÉ dans le
// scriptCode du node Playwright au moment du deploy (pas de copie divergente).

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { pageExtractor } = require('./extract');

const ROOT = __dirname;
const REPO = path.resolve(ROOT, '../..');
const DATA_DIR = path.join(REPO, 'ml/data/largus-occasion/raw');
const WF_NAME = 'largus-collect';

// --- .env ---------------------------------------------------------------
function loadEnv() {
  const p = path.join(ROOT, '.env');
  const env = {};
  if (fs.existsSync(p)) {
    for (const line of fs.readFileSync(p, 'utf8').split('\n')) {
      const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$/);
      if (m) env[m[1]] = m[2];
    }
  }
  const base = (env.N8N_BASE_URL || '').replace(/\/$/, '');
  const key = env.N8N_API_KEY || '';
  const hook = env.N8N_WEBHOOK_PATH || 'largus-collect';
  if (!base || !key) { console.error('ERREUR: N8N_BASE_URL / N8N_API_KEY manquants dans .env'); process.exit(1); }
  return { base, key, hook };
}

// --- API REST n8n -------------------------------------------------------
async function api(env, method, route, body) {
  const res = await fetch(`${env.base}/api/v1${route}`, {
    method,
    headers: { 'X-N8N-API-KEY': env.key, 'content-type': 'application/json', accept: 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  let json; try { json = JSON.parse(text); } catch { json = text; }
  if (!res.ok) throw new Error(`API ${method} ${route} -> ${res.status}: ${text.slice(0, 300)}`);
  return json;
}

// --- Construction du workflow ------------------------------------------
// Script exécuté DANS le node Playwright (contexte Node avec $browser / $input).
// 1 appel webhook = 1 page listing -> URLs annonces -> scrape chacune (poli) -> renvoie le tableau.
function buildScriptCode() {
  const UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
    + '(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36';
  const extractorSrc = pageExtractor.toString(); // inliné -> source unique (extract.js)
  return `
const UA = ${JSON.stringify(UA)};
const input = ($input.first() && $input.first().json) || {};
const body = input.body || input;
const pageN = Number(body.page || 1);
const LISTING = 'https://occasion.largus.fr/auto/?currentpage=' + pageN;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const uuidOf = (u) => { const m = u.match(/annonce-([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})/i); return m ? m[1] : null; };
const PAGE_EXTRACTOR = ${extractorSrc};

const ctx = await $browser.newContext({ userAgent: UA, locale: 'fr-FR', timezoneId: 'Europe/Paris', viewport: { width: 1366, height: 900 } });
await ctx.addInitScript(() => { Object.defineProperty(navigator, 'webdriver', { get: () => undefined }); });
const page = await ctx.newPage();

await page.goto(LISTING, { waitUntil: 'domcontentloaded', timeout: 45000 });
const urls = await page.evaluate(() => [...new Set([...document.querySelectorAll('a[href*="/auto/annonce-"]')].map((a) => a.href))]);

const annonces = [];
for (const url of urls) {
  try {
    await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 45000 });
    for (const sel of ['#didomi-notice-agree-button', 'button:has-text("Tout accepter")']) { try { await page.click(sel, { timeout: 1200 }); break; } catch (e) {} }
    await page.evaluate(async () => { await new Promise((res) => { let y = 0; const t = setInterval(() => { window.scrollBy(0, 800); y += 800; if (y >= document.body.scrollHeight) { clearInterval(t); res(); } }, 80); }); });
    const data = await page.evaluate(PAGE_EXTRACTOR);
    annonces.push(Object.assign({ source: 'occasion.largus.fr', url, id: uuidOf(url), scraped_at: new Date().toISOString() }, data));
  } catch (e) {
    annonces.push({ source: 'occasion.largus.fr', url, id: uuidOf(url), error: String((e && e.message) || e) });
  }
  await sleep(2000 + Math.floor(Math.random() * 3000)); // politesse 2-5 s
}
await ctx.close();
return [{ json: { page: pageN, listing: LISTING, count: annonces.length, annonces } }];
`.trim();
}

function buildWorkflow(env, webhookId) {
  return {
    name: WF_NAME,
    settings: { executionOrder: 'v1' },
    nodes: [
      {
        // webhookId REQUIS : sans lui, un webhook créé via l'API REST n'est jamais enregistré -> 404.
        parameters: { httpMethod: 'POST', path: env.hook, responseMode: 'responseNode', authentication: 'none', options: {} },
        name: 'Webhook', type: 'n8n-nodes-base.webhook', typeVersion: 2, position: [260, 300],
        webhookId,
      },
      {
        parameters: { operation: 'runCustomScript', scriptCode: buildScriptCode(), browserOptions: {} },
        name: 'Playwright', type: 'n8n-nodes-playwright.playwright', typeVersion: 1, position: [560, 300],
      },
      {
        parameters: { respondWith: 'json', responseBody: '={{ $json }}', options: {} },
        name: 'Respond', type: 'n8n-nodes-base.respondToWebhook', typeVersion: 1, position: [860, 300],
      },
    ],
    connections: {
      Webhook: { main: [[{ node: 'Playwright', type: 'main', index: 0 }]] },
      Playwright: { main: [[{ node: 'Respond', type: 'main', index: 0 }]] },
    },
  };
}

// --- Commandes ----------------------------------------------------------
async function findWorkflow(env) {
  const list = await api(env, 'GET', '/workflows?limit=100');
  return (list.data || list).find((w) => w.name === WF_NAME) || null;
}

async function deploy(env) {
  const existing = await findWorkflow(env);
  // Réutiliser le webhookId existant (URL stable) sinon en générer un — INDISPENSABLE à l'enregistrement.
  let webhookId = null;
  if (existing) {
    const full = await api(env, 'GET', `/workflows/${existing.id}`);
    const wh = (full.nodes || []).find((n) => n.type === 'n8n-nodes-base.webhook');
    webhookId = (wh && wh.webhookId) || null;
  }
  if (!webhookId) webhookId = crypto.randomUUID();
  const wf = buildWorkflow(env, webhookId);

  let id;
  if (existing) {
    id = existing.id;
    // désactiver avant modif puis réactiver -> force le (ré)enregistrement du webhook
    try { await api(env, 'POST', `/workflows/${id}/deactivate`); } catch (e) {}
    await api(env, 'PUT', `/workflows/${id}`, wf);
    console.log(`Workflow mis à jour (id=${id})`);
  } else {
    const created = await api(env, 'POST', '/workflows', wf);
    id = created.id;
    console.log(`Workflow créé (id=${id})`);
  }
  try { await api(env, 'POST', `/workflows/${id}/activate`); } catch (e) { console.warn('activate:', e.message); }
  console.log(`Webhook : ${env.base}/webhook/${env.hook}  (POST {"page":N})  webhookId=${webhookId}`);
  return id;
}

async function status(env) {
  const w = await findWorkflow(env);
  if (!w) { console.log('Aucun workflow "largus-collect".'); return; }
  console.log(`${w.name}  id=${w.id}  active=${w.active}`);
  console.log(`Webhook : ${env.base}/webhook/${env.hook}`);
}

function readSeenIds(ndjsonPath) {
  const seen = new Set();
  if (fs.existsSync(ndjsonPath)) {
    for (const line of fs.readFileSync(ndjsonPath, 'utf8').split('\n')) {
      if (!line.trim()) continue;
      try { const o = JSON.parse(line); if (o.id) seen.add(o.id); } catch (e) {}
    }
  }
  return seen;
}

async function downloadImages(imgDir, id, images) {
  if (!id || !images || !images.length) return 0;
  const dir = path.join(imgDir, id);
  fs.mkdirSync(dir, { recursive: true });
  let n = 0;
  for (let i = 0; i < images.length; i++) {
    const dest = path.join(dir, String(i).padStart(2, '0') + '.webp');
    if (fs.existsSync(dest)) { n++; continue; }
    try {
      const res = await fetch(images[i]);
      if (!res.ok) continue;
      fs.writeFileSync(dest, Buffer.from(await res.arrayBuffer()));
      n++;
    } catch (e) { /* image manquante, on continue */ }
  }
  return n;
}

async function collect(env, from, to) {
  fs.mkdirSync(DATA_DIR, { recursive: true });
  const ndjsonPath = path.join(DATA_DIR, 'annonces.ndjson');
  const imgDir = path.join(DATA_DIR, 'images');
  const seen = readSeenIds(ndjsonPath);
  const webhook = `${env.base}/webhook/${env.hook}`;
  let added = 0;

  for (let p = from; p <= to; p++) {
    process.stdout.write(`Page ${p}… `);
    let res;
    try {
      const r = await fetch(webhook, {
        method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ page: p }),
      });
      if (!r.ok) { console.log(`webhook ${r.status} — skip`); continue; }
      res = await r.json();
    } catch (e) { console.log(`erreur webhook: ${e.message} — skip`); continue; }

    const annonces = (res && res.annonces) || [];
    let pageAdded = 0, imgs = 0;
    for (const a of annonces) {
      if (a.error || !a.id || seen.has(a.id)) continue;
      fs.appendFileSync(ndjsonPath, JSON.stringify(a) + '\n');
      seen.add(a.id);
      imgs += await downloadImages(imgDir, a.id, a.images);
      pageAdded++;
    }
    added += pageAdded;
    console.log(`${annonces.length} annonces, +${pageAdded} nouvelles, ${imgs} images`);
    if (p < to) await new Promise((r) => setTimeout(r, 3000)); // politesse entre pages
  }
  console.log(`\nTerminé : +${added} annonces -> ${ndjsonPath}`);
}

// --- main ---------------------------------------------------------------
(async () => {
  const env = loadEnv();
  const [cmd, ...rest] = process.argv.slice(2);
  if (cmd === 'deploy') return deploy(env);
  if (cmd === 'status') return status(env);
  if (cmd === 'collect') {
    const arg = (rest.find((a) => a.startsWith('--pages')) || '').split('=')[1]
      || rest[rest.indexOf('--pages') + 1] || '1-1';
    const [from, to] = arg.split('-').map(Number);
    return collect(env, from || 1, to || from || 1);
  }
  console.log('Usage: node n8n.js <deploy|status|collect --pages A-B>');
})().catch((e) => { console.error(e.message); process.exit(1); });
