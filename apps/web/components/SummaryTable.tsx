"use client";

import { useState } from "react";
import type { ClassifyResponse } from "@/lib/types";

// Renders the canonical two-row summary semantically from validated fields (never injected Markdown).
export function SummaryTable({ result }: { result: ClassifyResponse }) {
  const [copied, setCopied] = useState(false);
  const lines = result.summary.classification_lines;
  return (
    <section aria-labelledby="summary-heading" className="space-y-2">
      <h2 id="summary-heading" className="sr-only">Classification summary</h2>
      <table className="w-full border-collapse text-left text-sm" data-testid="summary-table">
        <tbody>
          <tr className="border border-zinc-300 dark:border-zinc-700">
            <th scope="row" className="w-56 bg-zinc-100 px-3 py-2 font-semibold dark:bg-zinc-800">Field Of Invention</th>
            <td className="px-3 py-2 font-semibold tracking-wide" data-testid="field-of-invention">
              {result.summary.field_of_invention}
            </td>
          </tr>
          <tr className="border border-zinc-300 dark:border-zinc-700">
            <th scope="row" className="bg-zinc-100 px-3 py-2 align-top font-semibold dark:bg-zinc-800">Classification (IPC)</th>
            <td className="px-3 py-2" data-testid="classification-cell">
              {lines.length === 0 ? (
                <span className="font-medium text-amber-700 dark:text-amber-400">Insufficient evidence</span>
              ) : (
                <ul className="space-y-1">
                  {lines.map((ln) => (
                    <li key={ln.display_code}>
                      <span className="font-mono font-semibold">{ln.display_code}</span>
                      <span className="text-zinc-500"> — </span>
                      <span className={ln.confidence_text.startsWith("confidence:") ? "italic text-zinc-600 dark:text-zinc-400" : ""}>
                        {ln.confidence_text}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </td>
          </tr>
        </tbody>
      </table>
      <div className="flex items-center gap-3 text-xs">
        <button
          type="button"
          className="rounded border border-zinc-300 px-2 py-1 hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-800"
          onClick={async () => {
            await navigator.clipboard.writeText(result.summary.markdown);
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
          }}
        >
          {copied ? "Copied" : "Copy Markdown"}
        </button>
        <a
          className="underline"
          download={`ipc-classification-${result.taxonomy_version}.md`}
          href={`data:text/markdown;charset=utf-8,${encodeURIComponent(result.summary.markdown)}`}
        >
          Export .md
        </a>
        <span className="text-zinc-500">Status: <span data-testid="status">{result.status}</span></span>
      </div>
    </section>
  );
}
