import { expect, test } from "@playwright/test";
import { ABSTRACT, fixtureResponse } from "./fixtures";

test("abstract is required; optional documents are not", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Classify" }).click();
  await expect(page.locator("#abstract-error")).toHaveText("An abstract is required.");
  await expect(page.getByLabel(/^Abstract/)).toHaveAttribute("aria-invalid", "true");
});

test("renders the two-row summary, details, and separate alternatives (mocked contract)", async ({ page }) => {
  let sent: Record<string, unknown> | null = null;
  await page.route("**/api/classify", async (route) => {
    sent = route.request().postDataJSON();
    await route.fulfill({ json: fixtureResponse() });
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
  // Alternatives are not mixed into the compact cell.
  await expect(page.getByTestId("classification-cell")).not.toContainText("H04L 9/32");
  await expect(page.getByTestId("alternatives")).toContainText("H04L 9/32");

  // Optional fields omitted -> sent as null; the request is still accepted.
  expect(sent).toMatchObject({ abstract: ABSTRACT, specification: null, claims: null, title: null });

  // Evidence highlighting converts code-point offsets to UTF-16 (emoji precedes the quote).
  const card = page.getByTestId("candidate-card");
  await card.locator("summary").first().click();
  await expect(card).not.toContainText("offset mismatch");
  await card.getByText("Show abstract with evidence highlighted").click();
  await expect(card.locator("mark")).toHaveText("Quantum keys are distributed among smart grid nodes.");
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
  await page.route("**/api/classify", (route) => route.fulfill({ json: r }));
  await page.goto("/");
  await page.getByLabel(/^Abstract/).fill("Too vague.");
  await page.getByRole("button", { name: "Classify" }).click();
  await expect(page.getByTestId("classification-cell")).toHaveText("Insufficient evidence");
  await expect(page.getByTestId("field-of-invention")).toHaveText("UNDETERMINED");
});

test("shows unavailable service state", async ({ page }) => {
  await page.route("**/api/classify", (route) =>
    route.fulfill({ status: 503, json: { error: "embeddings_unavailable", message: "No serving embedding run." } }),
  );
  await page.goto("/");
  await page.getByLabel(/^Abstract/).fill("Something.");
  await page.getByRole("button", { name: "Classify" }).click();
  await expect(page.getByTestId("error")).toContainText("Service unavailable");
});
