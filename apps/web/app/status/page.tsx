"use client";

import { useEffect, useState } from "react";

interface StatusBody {
  corpus?: { active_release?: string | null; records_total?: number; serving_embedding_run?: string | null };
  models?: { state?: string; embedding_model?: string; embedding_revision?: string | null; reranker_model?: string };
  calibration?: { state?: string };
  keyword_mode?: string;
  source_warnings?: string[];
}

export default function StatusPage() {
  const [data, setData] = useState<Record<string, unknown> | null>(null);
  useEffect(() => {
    fetch("/api/status").then((r) => r.json()).then(setData).catch(() => setData({ error: "unreachable" }));
  }, []);
  const status = (data?.status ?? {}) as StatusBody;
  const ready = (data?.ready ?? {}) as { ready?: boolean };
  const corpus = status.corpus ?? {};
  return (
    <main className="mx-auto w-full max-w-5xl space-y-4 p-6">
      <h1 className="text-2xl font-semibold">Source and service status</h1>
      {!data && <p>Loading…</p>}
      {data && (
        <>
          <section className="rounded border border-zinc-300 p-3 text-sm dark:border-zinc-700">
            <p><b>Ready:</b> {String(ready.ready ?? false)}</p>
            <p><b>Models:</b> {status.models?.state ?? "unknown"} — {status.models?.embedding_model} @ {String(status.models?.embedding_revision ?? "")}; reranker {status.models?.reranker_model}</p>
            <p><b>Active release:</b> {corpus.active_release ?? "none"} ({corpus.records_total ?? 0} records)</p>
            <p><b>Serving embedding run:</b> {corpus.serving_embedding_run ?? "none"}</p>
            <p><b>Calibration:</b> {status.calibration?.state ?? "not_calibrated"}</p>
            <p><b>Keyword mode:</b> {status.keyword_mode}</p>
          </section>
          {Array.isArray(status.source_warnings) && status.source_warnings.length > 0 && (
            <section className="rounded border border-amber-500 p-3 text-sm">
              <h2 className="font-semibold">Source-quality warnings</h2>
              <ul className="list-disc pl-5">{status.source_warnings.map((w: string, i: number) => <li key={i}>{w}</li>)}</ul>
            </section>
          )}
          <details><summary className="cursor-pointer text-sm">Raw status</summary>
            <pre className="overflow-x-auto text-xs">{JSON.stringify(data, null, 2)}</pre></details>
        </>
      )}
    </main>
  );
}
