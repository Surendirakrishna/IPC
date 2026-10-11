import type { NextRequest } from "next/server";
import { backendFetch } from "@/lib/backend";

const ALLOWED = ["q", "record_type", "section", "subclass", "limit", "offset", "taxonomy_version"];

export async function GET(request: NextRequest) {
  const params = new URLSearchParams();
  for (const key of ALLOWED) {
    const v = request.nextUrl.searchParams.get(key);
    if (v) params.set(key, v);
  }
  return backendFetch(`/v1/ipc/search?${params.toString()}`, { timeoutMs: 20_000 });
}
