import type { NextRequest } from "next/server";
import { backendFetch } from "@/lib/backend";

const MAX_BYTES = 1_500_000;

export async function POST(request: NextRequest) {
  const body = await request.text();
  if (new TextEncoder().encode(body).length > MAX_BYTES) {
    return Response.json({ error: "request_too_large", message: "Request body is too large." }, { status: 413 });
  }
  return backendFetch("/v1/classify", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body,
  });
}
