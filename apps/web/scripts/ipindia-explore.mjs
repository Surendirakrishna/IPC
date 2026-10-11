// Exploration helper: opens IP India Public Search in a visible Edge window with the query
// pre-filled. A human types the CAPTCHA and clicks Search. The CAPTCHA is never solved automatically.
// Saves the results page HTML and a screenshot so a parser can be written.
// Usage: node scripts/ipindia-explore.mjs <outDir> <fromDate mm/dd/yyyy> <toDate mm/dd/yyyy>
import { chromium } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";

const [outDir, fromDate = "09/01/2026", toDate = "09/30/2026"] = process.argv.slice(2);
mkdirSync(outDir, { recursive: true });

const browser = await chromium.launch({ channel: "msedge", headless: false });
const page = await (await browser.newContext()).newPage();
await page.goto("https://iprsearch.ipindia.gov.in/PublicSearch/", { waitUntil: "domcontentloaded" });
await page.check("#Published");
await page.selectOption("#DateField", "PD");
await page.fill("#FromDate", fromDate);
await page.fill("#ToDate", toDate);
await page.focus("#CaptchaText");
console.log("READY: type the CAPTCHA in the Edge window and click Search (waiting up to 15 min)");

// A rejected CAPTCHA re-renders the form at the same /Search URL with "Invalid captcha".
// Keep waiting (and re-applying the filters) until a page with a results table appears.
const deadline = Date.now() + 15 * 60_000;
let n = 0;
for (;;) {
  if (Date.now() > deadline) throw new Error("timed out waiting for search results");
  await page.waitForEvent("load", { timeout: 15 * 60_000 });
  await page.waitForLoadState("networkidle").catch(() => {});
  const html = await page.content();
  const hasCaptchaBox = (await page.$("#CaptchaText")) !== null;
  const messages = await page.$$eval(".text-danger, .field-validation-error, .validation-summary-errors, .alert",
    (els) => els.map((e) => e.textContent.trim()).filter(Boolean)).catch(() => []);
  n += 1;
  writeFileSync(path.join(outDir, `load-${n}.html`), html, "utf8");
  console.log(`LOAD ${n}: url=${page.url()} captchaBox=${hasCaptchaBox} messages=${JSON.stringify(messages)}`);
  // Result pages do not contain the search form's CAPTCHA box.
  if (!hasCaptchaBox) break;
  if (/Invalid captcha/i.test(html)) {
    console.log("CAPTCHA rejected: type the new CAPTCHA (case-sensitive) and click Search again");
    await page.check("#Published").catch(() => {});
    await page.selectOption("#DateField", "PD").catch(() => {});
    await page.fill("#FromDate", fromDate).catch(() => {});
    await page.fill("#ToDate", toDate).catch(() => {});
    await page.focus("#CaptchaText").catch(() => {});
  }
}
writeFileSync(path.join(outDir, "results.html"), await page.content(), "utf8");
await page.screenshot({ path: path.join(outDir, "results.png"), fullPage: true });
console.log("SAVED results page:", page.url());

// Describe clickable elements in the results so the parser can target detail views.
const controls = await page.$$eval("a, button, input[type=submit], input[type=button]", (els) =>
  els.slice(0, 200).map((e) => ({ tag: e.tagName, text: (e.textContent || e.value || "").trim().slice(0, 60),
    href: e.getAttribute("href"), onclick: e.getAttribute("onclick"), name: e.getAttribute("name"),
    formaction: e.getAttribute("formaction"), cls: e.className })));
writeFileSync(path.join(outDir, "controls.json"), JSON.stringify(controls, null, 2), "utf8");
console.log("SAVED controls.json; closing in 5 s");
await page.waitForTimeout(5000);
await browser.close();
