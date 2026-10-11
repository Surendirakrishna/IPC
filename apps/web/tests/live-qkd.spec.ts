import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";
import path from "node:path";

// Live end-to-end: real API, real models, real database. Enabled with IPC_E2E_LIVE=1.
test.skip(process.env.IPC_E2E_LIVE !== "1", "set IPC_E2E_LIVE=1 with the API running");

const QKD = JSON.parse(
  readFileSync(path.join(__dirname, "..", "..", "..", "data", "evaluation", "qkd_contract_request.json"), "utf8"),
);

test("QKD abstract-only request returns the summary layout from the live pipeline", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel(/^Abstract/).fill(QKD.abstract);
  await page.getByRole("button", { name: "Classify" }).click();
  await expect(page.getByTestId("results")).toBeVisible({ timeout: 200_000 });
  await expect(page.getByTestId("summary-table").locator("th").nth(0)).toHaveText("Field Of Invention");
  const cell = page.getByTestId("classification-cell");
  await expect(cell).toBeVisible();
  // Without an active calibrator no percentage may appear.
  await expect(cell).not.toContainText("%");
});
