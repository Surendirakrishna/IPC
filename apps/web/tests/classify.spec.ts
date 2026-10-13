import { expect, test, type Page } from "@playwright/test";
import {
  ABSTRACT,
  FIXTURE_PATENT_CORPUS_UNAVAILABLE,
  fixtureFallbackResponse,
  fixtureResponse,
  fixtureVoteResponse,
  fixtureWeakVoteResponse,
} from "./fixtures";
import { MAX_NEIGHBOURS, type ClassifyResponse } from "../lib/types";

/** Collects console errors and uncaught page errors; the happy paths must produce none. */
function trackErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  page.on("pageerror", (e) => errors.push(e.message));
  return errors;
}

async function classifyWith(page: Page, body: ClassifyResponse, abstract = ABSTRACT) {
  await page.route("**/api/classify", (route) => route.fulfill({ json: body }));
  await page.goto("/");
  await page.getByLabel(/^Abstract/).fill(abstract);
  await page.getByRole("button", { name: "Classify" }).click();
  await expect(page.getByTestId("results")).toBeVisible();
}

test("abstract is required; optional documents are not", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Classify" }).click();
  await expect(page.locator("#abstract-error")).toHaveText("An abstract is required.");
  await expect(page.getByLabel(/^Abstract/)).toHaveAttribute("aria-invalid", "true");
});

test("vote primary: first code with its vote share, per-candidate shares, similar patents (mocked contract)", async ({ page }) => {
  const errors = trackErrors(page);
  const r = fixtureVoteResponse();
  await classifyWith(page, r);

  // Two-row summary: exact headers, server-rendered confidence_text shown verbatim.
  const table = page.getByTestId("summary-table");
  await expect(table.locator("th")).toHaveText(["Field Of Invention", "Classification (IPC)"]);
  await expect(page.getByTestId("field-of-invention")).toHaveText("COMMUNICATION");
  const lines = page.getByTestId("classification-cell").locator("li");
  await expect(lines).toHaveText(["H04L 9/08 — vote share: 79.1%", "H04L 9/32 — vote share: 47.9%"]);
  // An uncalibrated vote share is never labelled a probability or a confidence.
  await expect(page.getByTestId("classification-cell")).not.toContainText(/probability|confidence/i);
  await expect(page.getByTestId("classification-cell")).not.toContainText("H02J 13/00"); // alternative stays out
  await expect(page.getByTestId("vote-share-note")).toContainText("not a calibrated probability");

  // No fallback notice when the vote chose the codes.
  await expect(page.getByTestId("fallback-notice")).toHaveCount(0);

  // Each candidate shows its vote share on the collapsed card, worded as a vote share.
  const cards = page.getByTestId("candidate-card");
  await expect(cards).toHaveCount(2);
  await expect(cards.getByTestId("candidate-score")).toHaveText(["vote share 79.1%", "vote share 47.9%"]);
  await expect(cards.locator(":scope > summary")).toHaveCount(2); // the card's own summary, not nested ones
  await expect(cards.locator(":scope > summary").filter({ hasText: /probability|confidence/i })).toHaveCount(0);

  // Expanded: uncalibrated wording plus the per-level breakdown.
  const first = cards.nth(0);
  await first.locator("summary").first().click();
  await expect(first).toContainText("Vote share: 79.1%");
  await expect(first).toContainText("(uncalibrated)");
  await expect(first.getByTestId("vote-share-breakdown")).toHaveText(
    "Vote share by level: H04L 95.5% · H04L 9/00 88.6% · H04L 9/08 79.1%",
  );
  await expect(first).not.toContainText("offset mismatch");

  // Second candidate has zero supporting passages (allowed with the vote primary): renders, no crash.
  const second = cards.nth(1);
  await second.locator("summary").first().click();
  await expect(second).toContainText("No supporting passage.");
  await expect(second).toContainText("Vote share: 47.9%");

  // Alternatives carry their vote share too and stay separate from the summary.
  await expect(page.getByTestId("alternatives")).toContainText("H02J 13/00");
  await expect(page.getByTestId("alternative-score")).toHaveText("vote share 14.5%"); // server: round(100 * 0.1455, 1) = 14.5

  // Similar patents: title, id, similarity, IPC codes, at most MAX_NEIGHBOURS rows.
  const similar = page.getByTestId("similar-patents");
  await expect(similar.getByRole("heading", { name: "Similar patents" })).toBeVisible();
  await expect(similar).toContainText("The 10 nearest of the 100 patents that voted (corpus hupd-2017, 293,711 documents)");
  await expect(similar.locator("th")).toHaveText(["#", "Title", "Patent id", "Similarity", "Vote weight", "IPC codes (main first)"]);
  const rows = similar.getByTestId("similar-patent");
  await expect(rows).toHaveCount(r.vote!.neighbours.length);
  await expect(rows.nth(0).locator("td")).toHaveText([
    "1",
    "Quantum key distribution among nodes of a power distribution network",
    "15300001",
    "0.9812",
    "34.4%",
    "H04L 9/08, H04L 9/32",
  ]);
  // Null title is shown as such, not as an empty cell.
  await expect(rows.nth(9).locator("td").nth(1)).toHaveText("Untitled");
  await expect(rows.nth(9).locator("td").nth(5)).toHaveText("H04W 12/04");
  // IPC codes are keyboard-reachable links with their code as the accessible name.
  await expect(rows.nth(0).getByRole("link", { name: "H04L 9/32" })).toHaveAttribute("href", "/taxonomy?code=H04L%209%2F32");

  expect(errors).toEqual([]);
});

test("similar patents never exceed the contract maximum", async ({ page }) => {
  const r = fixtureVoteResponse();
  const base = r.vote!.neighbours;
  r.vote!.neighbours = [...base, ...base.slice(0, 3).map((n, i) => ({ ...n, document_id: `extra-${i}` }))];
  expect(r.vote!.neighbours.length).toBeGreaterThan(MAX_NEIGHBOURS);
  await classifyWith(page, r);
  await expect(page.getByTestId("similar-patent")).toHaveCount(MAX_NEIGHBOURS);
});

test("class-level vote candidate and calibrated probability are worded by status", async ({ page }) => {
  const r = fixtureVoteResponse();
  // Class candidate: all-null vote_share, class-level share in confidence.score.
  r.candidates[0] = {
    ...r.candidates[0],
    code: "H04",
    display_code: "H04",
    title: "ELECTRIC COMMUNICATION TECHNIQUE",
    level: "class",
    confidence: { ...r.candidates[0].confidence, score: 0.97 },
    vote_share: { subclass: null, maingroup: null, exact: null },
    parent_path: r.candidates[0].parent_path.slice(0, 1),
  };
  // A calibrated candidate is the only one called a probability.
  r.candidates[1] = {
    ...r.candidates[1],
    confidence: { score: 0.893, status: "calibrated", calibrator_id: "vote-cal-1", interpretation: null, detail: null },
  };
  await classifyWith(page, r);
  const cards = page.getByTestId("candidate-card");
  await expect(cards.getByTestId("candidate-score")).toHaveText(["vote share 97.0%", "Probability 89.3%"]);
  await cards.nth(0).locator("summary").first().click();
  await expect(cards.nth(0)).toContainText("Vote share: 97.0%");
  await expect(cards.nth(0).getByTestId("vote-share-breakdown")).toHaveCount(0);
  await cards.nth(1).locator("summary").first().click();
  await expect(cards.nth(1)).toContainText("Probability 89.3%");
  await expect(cards.nth(1)).toContainText("calibrator vote-cal-1");
});

test("definition fallback: v1 summary, visible notice, no similar patents (mocked contract)", async ({ page }) => {
  const errors = trackErrors(page);
  let sent: Record<string, unknown> | null = null;
  await page.route("**/api/classify", async (route) => {
    sent = route.request().postDataJSON();
    await route.fulfill({ json: fixtureFallbackResponse() });
  });
  await page.goto("/");
  await page.getByLabel(/^Abstract/).fill(ABSTRACT);
  await page.getByRole("button", { name: "Classify" }).click();

  const table = page.getByTestId("summary-table");
  await expect(table.locator("th").nth(0)).toHaveText("Field Of Invention");
  await expect(page.getByTestId("field-of-invention")).toHaveText("COMMUNICATION");
  await expect(table.locator("th").nth(1)).toHaveText("Classification (IPC)");
  await expect(page.getByTestId("classification-cell")).toHaveText("H04L 9/08 — confidence: not calibrated");
  await expect(page.getByTestId("classification-cell")).not.toContainText("%");
  await expect(page.getByTestId("vote-share-note")).toHaveCount(0);
  // Alternatives are not mixed into the compact cell.
  await expect(page.getByTestId("classification-cell")).not.toContainText("H04L 9/32");
  await expect(page.getByTestId("alternatives")).toContainText("H04L 9/32");

  // classifier === "definition_fallback" -> visible notice; no similar-patents list.
  const notice = page.getByTestId("fallback-notice");
  await expect(notice).toBeVisible();
  await expect(notice).toContainText("Codes from definition matching (fallback).");
  await expect(notice).toContainText("No patent corpus is serving");
  await expect(page.getByTestId("similar-patents")).toHaveCount(0);
  await expect(page.getByTestId("candidate-score")).toHaveCount(0); // no number without a score

  // Optional fields omitted -> sent as null; the request is still accepted.
  expect(sent).toMatchObject({ abstract: ABSTRACT, specification: null, claims: null, title: null });

  // Evidence highlighting converts code-point offsets to UTF-16 (emoji precedes the quote).
  const card = page.getByTestId("candidate-card");
  await card.locator("summary").first().click();
  await expect(card).toContainText("Confidence: not calibrated");
  await expect(card).not.toContainText("offset mismatch");
  await card.getByText("Show abstract with evidence highlighted").click();
  await expect(card.locator("mark")).toHaveText("Quantum keys are distributed among smart grid nodes.");

  expect(errors).toEqual([]);
});

test("weak vote fallback: notice gives the share; the weak vote's neighbours are not shown", async ({ page }) => {
  await classifyWith(page, fixtureWeakVoteResponse());
  const notice = page.getByTestId("fallback-notice");
  await expect(notice).toContainText("Neighbour vote too weak (20.2%, below the 30.0% minimum).");
  await expect(notice).toContainText("not shown");
  await expect(page.getByTestId("similar-patents")).toHaveCount(0);
  await expect(page.getByTestId("results")).not.toContainText("15400001");
  await expect(page.getByTestId("classification-cell")).toHaveText("H04L 9/08 — confidence: not calibrated");
});

test("vote disabled still shows a fallback notice", async ({ page }) => {
  const r = fixtureResponse();
  r.fallback_reason = "neighbour_vote_disabled";
  await classifyWith(page, r);
  await expect(page.getByTestId("fallback-notice")).toContainText("The neighbour vote is turned off.");
});

test("shows explicit insufficient evidence instead of an invented code", async ({ page }) => {
  const r = fixtureResponse();
  r.status = "insufficient_evidence";
  r.candidates = [];
  r.alternatives = [];
  r.summary = {
    field_of_invention: "UNDETERMINED",
    classification_lines: [],
    classification_cell_text: "Insufficient evidence",
    markdown: "| Field Of Invention | UNDETERMINED |\n| :--- | :--- |\n| Classification (IPC) | Insufficient evidence |\n",
  };
  r.field_of_invention = { ...r.field_of_invention, label: "UNDETERMINED", mapping_key: null, rationale: "No support." };
  await classifyWith(page, r, "Too vague.");
  await expect(page.getByTestId("classification-cell")).toHaveText("Insufficient evidence");
  await expect(page.getByTestId("field-of-invention")).toHaveText("UNDETERMINED");
  await expect(page.getByTestId("fallback-notice")).toBeVisible();
});

test("503 patent_corpus_unavailable: service unavailable, nothing to show", async ({ page }) => {
  await page.route("**/api/classify", (route) =>
    route.fulfill({ status: FIXTURE_PATENT_CORPUS_UNAVAILABLE.status, json: FIXTURE_PATENT_CORPUS_UNAVAILABLE.body }),
  );
  await page.goto("/");
  await page.getByLabel(/^Abstract/).fill("Something.");
  await page.getByRole("button", { name: "Classify" }).click();
  const error = page.getByTestId("error");
  await expect(error).toContainText("Service unavailable (patent_corpus_unavailable)");
  await expect(error).toContainText("No serving patent corpus, and the definition fallback is unavailable.");
  await expect(error).toContainText("try again later");
  await expect(page.getByTestId("results")).toHaveCount(0);
  // The form stays usable for a retry.
  await expect(page.getByRole("button", { name: "Classify" })).toBeEnabled();
});

test("shows unavailable service state (1.x 503 codes unchanged)", async ({ page }) => {
  await page.route("**/api/classify", (route) =>
    route.fulfill({ status: 503, json: { error: "embeddings_unavailable", message: "No serving embedding run." } }),
  );
  await page.goto("/");
  await page.getByLabel(/^Abstract/).fill("Something.");
  await page.getByRole("button", { name: "Classify" }).click();
  await expect(page.getByTestId("error")).toContainText("Service unavailable");
});
