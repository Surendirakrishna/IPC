import { backendFetch } from "@/lib/backend";

export async function GET() {
  const [status, ready] = await Promise.all([
    backendFetch("/v1/taxonomy/status", { timeoutMs: 20_000 }),
    backendFetch("/ready", { timeoutMs: 20_000 }),
  ]);
  const parse = async (r: Response) => { try { return await r.json(); } catch { return null; } };
  return Response.json({ status: await parse(status), ready: await parse(ready), ready_http: ready.status });
}
