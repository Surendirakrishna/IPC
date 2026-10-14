# ipc-classifier web app

The Next.js 16 (App Router, TypeScript, Tailwind) user interface for the IPC classifier. The
project overview is in the [repository README](../../README.md), and the HTTP contract is in
[docs/api.md](../../docs/api.md).

## Pages

| Route | What it shows |
| --- | --- |
| `/` | Paste an abstract (plus optional title, claims and specification) and classify it. It shows the two-row summary ("Field Of Invention" / "Classification (IPC)"), a card per code with its vote share or confidence status, the "Similar patents" table when the neighbour vote answered, and a notice when the definition fallback answered. |
| `/taxonomy` | Browse and search the IPC 2026.01 catalogue. |
| `/status` | API readiness: models, active release, embedding run, calibration and neighbour vote state. |

## How it talks to the API

The browser never calls the backend directly. Server-side route handlers in `app/api/*`
(`classify`, `ipc/search`, `ipc/lookup`, `status`) forward requests through `lib/backend.ts` to
`IPC_API_BASE_URL`, which defaults to `http://127.0.0.1:8000`.

- **Timeouts and outages.** Requests time out after 180 s with `504 timeout`. If the API is down the
  handler returns `503 backend_unreachable`.
- **Response types.** `lib/types.ts` mirrors the backend contract (`apps/api/app/schemas/classify.py`).
  `apps/api/tests/test_contract_classify_v2.py` fails if the two drift apart.

## Commands

```bash
npm ci
npm run dev          # http://localhost:3000 (start the API on :8000 first)
npm run build        # production build
npm run start        # serve the production build
npm run typecheck    # tsc --noEmit
npm run lint         # eslint
```

## Tests (Playwright)

```bash
PW_CHANNEL=msedge npx playwright test                                       # mocked contract (tests/fixtures.ts)
IPC_E2E_LIVE=1 PW_CHANNEL=msedge npx playwright test tests/live-qkd.spec.ts  # live API, QKD example
```

- **`tests/classify.spec.ts`** covers the vote, fallback and weak-vote responses and the error
  states, using the contract fixtures in `tests/fixtures.ts`.
- **`PW_CHANNEL=msedge`** uses the installed Edge, because Windows Application Control on the
  development machine blocks Playwright's downloaded browsers.
- **An existing dev server is reused.** Playwright reuses whatever is already on port 3000. To test
  another checkout, stop that server first or use a separate port.

## Scripts

`scripts/ipindia-explore.mjs` and `scripts/ipindia-collect.mjs` collect the IP India benchmark data.

- They open IP India Public Search in a visible Edge window.
- A **human types the CAPTCHA**; it is never solved automatically.
- They then sample result pages at a polite rate.
- Usage, sampling and the dev/test split are described in
  [docs/benchmark-ipindia.md](../../docs/benchmark-ipindia.md#reproduce).

## Note on `AGENTS.md`

`next dev` regenerates `AGENTS.md`. Leave it in place.
