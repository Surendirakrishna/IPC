import { defineConfig, devices } from "@playwright/test";

// Mocked-contract tests run against `next dev` only. Live tests (QKD end-to-end) additionally
// need the API at IPC_API_BASE_URL and run only when IPC_E2E_LIVE=1.
export default defineConfig({
  testDir: "./tests",
  timeout: 240_000,
  // PW_CHANNEL=msedge|chrome uses an installed, signed browser — needed where Windows Application
  // Control blocks Playwright's downloaded browser binaries.
  use: { baseURL: "http://localhost:3000", trace: "retain-on-failure", channel: process.env.PW_CHANNEL || undefined },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: "npm run dev",
    url: "http://localhost:3000",
    reuseExistingServer: true,
    timeout: 120_000,
  },
});
