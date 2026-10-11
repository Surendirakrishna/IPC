"use client";

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";

interface SearchItem { code: string; display_code: string; record_type: string; title: string; match_mode: string }
interface Detail {
  code: string; display_code: string; record_type: string; title: string; title_raw: string | null;
  original_title: string | null; semantic_text: string;
  parent_path: { code: string; display_code: string; title: string; level: string }[];
  children: { code: string; display_code: string; title: string }[];
  duplicate_context: { is_duplicate_symbol: boolean | null; titles: string[] | null; canonical_title: string | null; context_heading: string | null };
  source_occurrences: { occurrence_id: number; source_path: string[]; original_title: string }[];
  references: { target_code: string; target_display_code: string | null; target_title: string | null; kind: string; original_wording: string | null; target_exists: boolean }[];
  referenced_by: { source_code: string; source_display_code: string; source_title: string; kind: string }[];
}

const codeHref = (c: string) => `/taxonomy?code=${encodeURIComponent(c)}`;

export function TaxonomyBrowser() {
  const params = useSearchParams();
  const router = useRouter();
  const code = params.get("code");
  const [q, setQ] = useState(params.get("q") ?? "");
  const [items, setItems] = useState<SearchItem[] | null>(null);
  const [total, setTotal] = useState(0);
  const [loaded, setLoaded] = useState<{ code: string; detail: Detail | null; error: string | null } | null>(null);
  const [searchError, setError] = useState<string | null>(null);
  // Derived: only show the detail that matches the current ?code= (no setState during render/effect body).
  const detail = code && loaded?.code === code ? loaded.detail : null;
  const error = searchError ?? (code && loaded?.code === code ? loaded.error : null);

  useEffect(() => {
    if (!code) return;
    let cancelled = false;
    fetch(`/api/ipc/lookup?code=${encodeURIComponent(code)}`).then(async (r) => {
      const body = await r.json();
      if (cancelled) return;
      setLoaded(r.ok ? { code, detail: body, error: null } : { code, detail: null, error: body.message ?? "Lookup failed" });
    });
    return () => { cancelled = true; };
  }, [code]);

  async function search(e: React.FormEvent) {
    e.preventDefault();
    const r = await fetch(`/api/ipc/search?q=${encodeURIComponent(q)}&limit=50`);
    const body = await r.json();
    if (!r.ok) { setError(body.message ?? "Search failed"); return; }
    setError(null); setItems(body.items); setTotal(body.total);
  }

  return (
    <div className="space-y-4">
      <form onSubmit={search} className="flex gap-2">
        <input value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search codes or keywords"
          placeholder="e.g. H04L 9/08 or quantum key distribution"
          className="flex-1 rounded border border-zinc-300 p-2 dark:border-zinc-700 dark:bg-zinc-900" />
        <button className="rounded bg-zinc-900 px-3 text-white dark:bg-zinc-100 dark:text-zinc-900">Search</button>
      </form>
      <p className="text-xs text-zinc-500">Keyword ranking is PostgreSQL full-text search (ts_rank_cd), not BM25. Code lookup is exact/prefix.</p>
      {error && <p role="alert" className="text-sm text-red-600">{error}</p>}

      {items && (
        <div>
          <p className="text-sm text-zinc-600">{total} matches</p>
          <ul className="divide-y divide-zinc-200 text-sm dark:divide-zinc-800">
            {items.map((it) => (
              <li key={it.code} className="py-1">
                <button className="text-left" onClick={() => router.push(codeHref(it.display_code))}>
                  <span className="font-mono font-semibold">{it.display_code}</span>{" "}
                  <span className="text-xs uppercase text-zinc-500">{it.record_type}</span> {it.title}
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      {detail && (
        <article className="space-y-3 rounded border border-zinc-300 p-4 dark:border-zinc-700" data-testid="code-detail">
          <h2 className="text-xl"><span className="font-mono">{detail.display_code}</span> {detail.title}</h2>
          <p className="text-xs text-zinc-500">Internal code <span className="font-mono">{detail.code}</span> · {detail.record_type}</p>
          <ol className="text-sm">
            {detail.parent_path.map((p) => (
              <li key={p.code}><span className="text-xs uppercase text-zinc-500">{p.level}</span>{" "}
                <Link className="font-mono underline" href={codeHref(p.display_code)}>{p.display_code}</Link> {p.title}</li>
            ))}
          </ol>
          {detail.original_title && <p className="text-sm"><b>Original wording:</b> {detail.original_title}</p>}
          {detail.duplicate_context.is_duplicate_symbol && (
            <p className="text-sm"><b>Duplicate symbol contexts:</b> {(detail.duplicate_context.titles ?? []).join(" | ")}
              {" "}(context heading: {detail.duplicate_context.context_heading})</p>
          )}
          <details><summary className="cursor-pointer text-sm">semantic_text (retrieval text)</summary>
            <pre className="whitespace-pre-wrap text-xs">{detail.semantic_text}</pre></details>
          <details><summary className="cursor-pointer text-sm">Source tree occurrences ({detail.source_occurrences.length})</summary>
            <ul className="text-xs">{detail.source_occurrences.map((o) => (
              <li key={o.occurrence_id}>#{o.occurrence_id} {o.source_path.join(" › ")} — “{o.original_title}”</li>))}</ul></details>
          {detail.references.length > 0 && (
            <div><h3 className="font-semibold">References</h3><ul className="text-sm">{detail.references.map((r) => (
              <li key={`${r.kind}-${r.target_code}`}>{r.kind.replace("_", " ")}:{" "}
                {r.target_exists ? <Link className="font-mono underline" href={codeHref(r.target_display_code ?? r.target_code)}>{r.target_display_code ?? r.target_code}</Link>
                  : <span className="font-mono">{r.target_code} (not in catalogue)</span>} {r.target_title}
                {r.original_wording && <span className="block text-xs text-zinc-500">“{r.original_wording}”</span>}</li>))}</ul></div>
          )}
          {detail.referenced_by.length > 0 && (
            <details><summary className="cursor-pointer text-sm">Referenced by ({detail.referenced_by.length})</summary>
              <ul className="text-xs">{detail.referenced_by.map((r) => (
                <li key={`${r.kind}-${r.source_code}`}><Link className="font-mono underline" href={codeHref(r.source_display_code)}>{r.source_display_code}</Link> {r.source_title}</li>))}</ul></details>
          )}
          {detail.children.length > 0 && (
            <details open={detail.children.length <= 30}><summary className="cursor-pointer text-sm">Children ({detail.children.length})</summary>
              <ul className="text-sm">{detail.children.map((c) => (
                <li key={c.code}><Link className="font-mono underline" href={codeHref(c.display_code)}>{c.display_code}</Link> {c.title}</li>))}</ul></details>
          )}
        </article>
      )}
    </div>
  );
}
