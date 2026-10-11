import type { NextRequest } from "next/server";
import { backendFetch } from "@/lib/backend";

export async function GET(request: NextRequest) {
  const code = request.nextUrl.searchParams.get("code")?.trim();
  if (!code) return Response.json({ error: "invalid_request", message: "code is required" }, { status: 422 });
  return backendFetch(`/v1/ipc/${encodeURIComponent(code)}`, { timeoutMs: 20_000 });
}
