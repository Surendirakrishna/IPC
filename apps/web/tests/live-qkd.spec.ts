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
  await expect(page.getByTestId("summary-table").locator("th").nth(1)).toHaveText("Classification (IPC)");
  const cell = page.getByTestId("classification-cell");
  await expect(cell).toBeVisible();
  // Contract 2.0: each line is "<code> — <confidence_text>"; a percentage appears only as a calibrated
  // probability ("89.3%") or an explicitly named vote share ("vote share: 79.1%"), never otherwise.
  const lineTexts = await cell.locator("li").allTextContents();
  if (lineTexts.length === 0) await expect(cell).toHaveText("Insufficient evidence"); // fallback only
  for (const t of lineTexts) {
    expect(t).toMatch(/^\S.* — (\d{1,3}\.\d%|vote share: \d{1,3}\.\d%|confidence: .+)$/);
  }
  await expect(cell).not.toContainText(/probability/i);
  // The similar-patents list appears exactly when the vote chose the codes (no fallback notice).
  const fallback = await page.getByTestId("fallback-notice").count();
  await expect(page.getByTestId("similar-patents")).toHaveCount(fallback === 0 ? 1 : 0);
});
