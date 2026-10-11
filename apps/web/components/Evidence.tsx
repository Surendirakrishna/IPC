"use client";

import { highlightSegments, sliceByCodePoints } from "@/lib/text";
import type { SupportingPassage } from "@/lib/types";

const SOURCE_LABEL: Record<string, string> = {
  abstract: "Abstract",
  title: "Title",
  specification: "Specification",
  claims: "Claims",
};

export function PassageList({ passages, fields }: { passages: SupportingPassage[]; fields: Record<string, string> }) {
  if (passages.length === 0) return <p className="text-sm text-zinc-500">No supporting passage.</p>;
  return (
    <ul className="space-y-2">
      {passages.map((p, i) => {
        const field = fields[p.source];
        // Verify client-side too: the quote must equal the code-point slice of the original field.
        const verified = field !== undefined && sliceByCodePoints(field, p.start, p.end) === p.quote;
        return (
          <li key={i} className="rounded border-l-4 border-sky-500 bg-sky-50 p-2 text-sm dark:bg-sky-950/40">
            <div className="mb-1 flex flex-wrap gap-2 text-xs text-zinc-600 dark:text-zinc-400">
              <span className="font-semibold">{SOURCE_LABEL[p.source] ?? p.source}</span>
              {p.claim_number !== null && <span>claim {p.claim_number}</span>}
              <span>chars {p.start}–{p.end}</span>
              <span>{p.evidence_type === "llm_cited_verified" ? "LLM-cited, quote verified" : "retrieval evidence — requires review"}</span>
              {!verified && <span className="font-semibold text-red-600">offset mismatch</span>}
            </div>
            <blockquote className="italic">“{p.quote}”</blockquote>
          </li>
        );
      })}
    </ul>
  );
}

export function HighlightedField({ label, text, spans }: { label: string; text: string; spans: { start: number; end: number }[] }) {
  const segments = highlightSegments(text, spans);
  return (
    <details className="text-sm">
      <summary className="cursor-pointer text-zinc-600 dark:text-zinc-400">Show {label} with evidence highlighted</summary>
      <p className="mt-2 whitespace-pre-wrap rounded bg-zinc-50 p-2 dark:bg-zinc-900">
        {segments.map((s, i) =>
          s.highlighted ? (
            <mark key={i} className="bg-yellow-200 dark:bg-yellow-700/60">{s.text}</mark>
          ) : (
            <span key={i}>{s.text}</span>
          ),
        )}
      </p>
    </details>
  );
}
