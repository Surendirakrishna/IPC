// Ground-truth collector for IP India Public Search (benchmark data).
//
// A visible Edge window opens with the search pre-filled; a HUMAN types the CAPTCHA and clicks
// Search. The CAPTCHA is never solved automatically. In that same browser session the script then
// samples result pages (seeded random), opens a bounded number of application detail pages through
// the site's own form endpoints, and saves the raw HTML. Requests are spaced to stay polite.
//
// Usage: node scripts/ipindia-collect.mjs <outDir> <fromDate mm/dd/yyyy> <toDate mm/dd/yyyy>
//                                         [pages=26] [perPage=5] [seed=20261010]
import { chromium } from "@playwright/test";
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";

const [outDir, fromDate, toDate, pagesArg = "26", perPageArg = "5", seedArg = "20261010"] = process.argv.slice(2);
if (!outDir || !fromDate || !toDate) throw new Error("usage: outDir fromDate toDate [pages] [perPage] [seed]");
const nPages = Number(pagesArg), perPage = Number(perPageArg);
// EXCLUDE_DIRS: comma-separated "details" folders of other samples (e.g. the dev set). Applications saved
// there are never re-sampled, so a test sample stays disjoint from the dev sample.
const excluded = new Set((process.env.EXCLUDE_DIRS || "").split(",").filter(Boolean).flatMap((d) =>
  existsSync(d) ? readdirSync(d).filter((f) => f.endsWith(".html")).map((f) => f.replace(/\.html$/, "")) : []));
const BASE = "https://iprsearch.ipindia.gov.in/PublicSearch";
mkdirSync(path.join(outDir, "pages"), { recursive: true });
mkdirSync(path.join(outDir, "details"), { recursive: true });

// Deterministic PRNG (mulberry32) so the sample is reproducible from the seed.
let s = Number(seedArg) >>> 0;
const rand = () => { s = (s + 0x6d2b79f5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const politeDelay = () => sleep(1500 + Math.floor(rand() * 1500));

const browser = await chromium.launch({ channel: "msedge", headless: false });
const context = await browser.newContext();
const page = await context.newPage();

async function fillForm() {
  await page.check("#Published").catch(() => {});
  await page.selectOption("#DateField", "PD").catch(() => {});
  await page.fill("#FromDate", fromDate).catch(() => {});
  await page.fill("#ToDate", toDate).catch(() => {});
  await page.keyboard.press("Escape").catch(() => {}); // close datepicker popups
  await page.focus("#CaptchaText").catch(() => {});
}

await page.goto(`${BASE}/`, { waitUntil: "domcontentloaded" });
await fillForm();
console.log("READY: type the CAPTCHA in the Edge window and click Search");
const deadline = Date.now() + 15 * 60_000;
for (;;) {
  if (Date.now() > deadline) throw new Error("timed out waiting for search results");
  await page.waitForEvent("load", { timeout: 15 * 60_000 });
  await page.waitForLoadState("networkidle").catch(() => {});
  if ((await page.$("#CaptchaText")) === null && (await page.$("#tableData")) !== null) break;
  if (/Invalid captcha/i.test(await page.content())) {
    console.log("CAPTCHA rejected: type the new CAPTCHA (case-sensitive) and click Search again");
    await fillForm();
  }
}

// Pager form fields (session-bound QueryString token etc.).
const pagerFields = await page.$eval('form[action$="PatentSearchResult"]', (f) =>
  Object.fromEntries([...f.querySelectorAll("input[type=hidden]")].map((i) => [i.name, i.value])));
const detailFields = {
  IP: await page.$eval("#IP", (e) => e.value).catch(() => ""),
  ConnectionName: pagerFields.ConnectionName || "PublicationConnection",
};
const totalPages = Number(pagerFields.TotalPages);
const totalResults = Number(pagerFields.TotalResult);
console.log(`SEARCH OK: ${totalResults} documents in ${totalPages} pages`);

// Server HTML quotes attribute values with single quotes; the live DOM normalises to double quotes.
const parseRows = (html) =>
  [...html.matchAll(/name="ApplicationNumber" value=['"]\s*([0-9A-Za-z/]+)\s*['"]/g)].map((m) => m[1]);

async function fetchPage(p) {
  // Reuse a result page saved by an earlier run of the same query (avoids re-requesting it).
  const saved = path.join(outDir, "pages", `page-${p}.html`);
  if (existsSync(saved)) return readFileSync(saved, "utf8");
  if (p === 1) return page.content();
  const res = await page.request.post(`${BASE}/PublicationSearch/PatentSearchResult`, {
    form: { ...pagerFields, page: String(p) },
  });
  return res.text();
}

async function fetchDetail(appNo) {
  const res = await page.request.post(`${BASE}/PublicationSearch/PatentDetails`, {
    form: { ApplicationNumber: appNo, ...detailFields },
  });
  return { status: res.status(), html: await res.text() };
}

// Seeded, distinct page sample across the whole result set.
const pageSet = new Set();
while (pageSet.size < Math.min(nPages, totalPages)) pageSet.add(1 + Math.floor(rand() * totalPages));
const pages = [...pageSet].sort((a, b) => a - b);

const index = { source: `${BASE}/`, query: { published: true, dateField: "PD", fromDate, toDate },
  totalResults, totalPages, seed: Number(seedArg), sampledPages: pages, excludedApplications: excluded.size,
  records: [],
  collectedAt: new Date().toISOString() };
let checkedDetailFormat = false;

for (const p of pages) {
  const html = await fetchPage(p);
  writeFileSync(path.join(outDir, "pages", `page-${p}.html`), html, "utf8");
  const rows = parseRows(html).filter((a) => !excluded.has(a));
  if (rows.length === 0) { console.log(`WARN page ${p}: no rows (session may have expired)`); continue; }
  // Seeded choice of rows on this page.
  const chosen = [...rows].sort(() => rand() - 0.5).slice(0, perPage);
  for (const appNo of chosen) {
    const file = path.join(outDir, "details", `${appNo}.html`);
    if (existsSync(file)) continue;
    await politeDelay();
    const { status, html: dh } = await fetchDetail(appNo);
    writeFileSync(file, dh, "utf8");
    index.records.push({ applicationNumber: appNo, page: p, httpStatus: status, bytes: dh.length });
    if (!checkedDetailFormat) {
      checkedDetailFormat = true;
      const ok = status === 200 && /abstract/i.test(dh);
      console.log(`DETAIL CHECK ${appNo}: http=${status} bytes=${dh.length} hasAbstract=${ok}`);
      if (!ok) {
        writeFileSync(path.join(outDir, "index.json"), JSON.stringify(index, null, 2));
        console.log("STOP: detail page format not as expected; saved for inspection");
        await browser.close();
        process.exit(3);
      }
    }
  }
  console.log(`PAGE ${p}: ${rows.length} rows, collected ${index.records.length} so far`);
  writeFileSync(path.join(outDir, "index.json"), JSON.stringify(index, null, 2));
  await politeDelay();
}

writeFileSync(path.join(outDir, "index.json"), JSON.stringify(index, null, 2));
console.log(`DONE: ${index.records.length} detail pages saved to ${outDir}`);
await browser.close();
