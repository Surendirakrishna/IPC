// Server-only helper for the route-handler proxy. The backend URL stays server-side.
import "server-only";

const BASE = process.env.IPC_API_BASE_URL ?? "http://127.0.0.1:8000";

export async function backendFetch(path: string, init?: RequestInit & { timeoutMs?: number }): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), init?.timeoutMs ?? 180_000);
  try {
    const res = await fetch(`${BASE}${path}`, { ...init, signal: controller.signal, cache: "no-store" });
    const body = await res.text();
    return new Response(body, {
      status: res.status,
      headers: { "content-type": res.headers.get("content-type") ?? "application/json" },
    });
  } catch (err) {
    const aborted = err instanceof Error && err.name === "AbortError";
    return Response.json(
      {
        error: aborted ? "timeout" : "backend_unreachable",
        message: aborted ? "The classification service did not respond in time." : "The classification service is unreachable.",
      },
      { status: aborted ? 504 : 503 },
    );
  } finally {
    clearTimeout(timer);
  }
}
