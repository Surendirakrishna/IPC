"use client";

import Link from "next/link";
import { MAX_NEIGHBOURS, type ClassifyResponse } from "@/lib/types";
import { pct } from "./CandidateCard";

// The nearest corpus patents behind the neighbour vote (contract 2.0, vote.neighbours).
// Shown only when the vote chose the codes: in every fallback, including vote_below_minimum where `vote`
// is still present, the neighbours do not support the shown codes and must not be presented as if they did.
export function SimilarPatents({ result }: { result: ClassifyResponse }) {
  if (result.classifier !== "neighbour_vote" || result.vote === null) return null;
  const vote = result.vote;
  const shown = vote.neighbours.slice(0, MAX_NEIGHBOURS); // server caps at 10 too; never render more
  const corpus = result.provenance.patent_corpus;
  return (
    <section aria-labelledby="similar-heading" className="space-y-2" data-testid="similar-patents">
      <h3 id="similar-heading" className="font-semibold">Similar patents</h3>
      <p className="text-xs text-zinc-600 dark:text-zinc-400">
        The {shown.length} nearest of the {vote.neighbour_count} patents that voted
        {corpus ? ` (corpus ${corpus.corpus_id}, ${corpus.document_count.toLocaleString("en")} documents)` : ""}.
        {" "}Similarity is cosine similarity, not a probability. Vote weight is the patent&apos;s share of the vote.
      </p>
      {shown.length === 0 ? (
        <p className="text-sm text-zinc-500">No similar patent was returned.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-zinc-300 text-xs text-zinc-600 dark:border-zinc-700 dark:text-zinc-400">
                <th scope="col" className="py-1 pr-3 font-semibold">#</th>
                <th scope="col" className="py-1 pr-3 font-semibold">Title</th>
                <th scope="col" className="py-1 pr-3 font-semibold">Patent id</th>
                <th scope="col" className="py-1 pr-3 font-semibold">Similarity</th>
                <th scope="col" className="py-1 pr-3 font-semibold">Vote weight</th>
                <th scope="col" className="py-1 font-semibold">IPC codes (main first)</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((n, i) => (
                <tr key={`${i}-${n.document_id}`} className="border-b border-zinc-200 align-top dark:border-zinc-800" data-testid="similar-patent">
                  <td className="py-1 pr-3 text-zinc-500">{i + 1}</td>
                  <td className="py-1 pr-3">{n.title ?? <span className="italic text-zinc-500">Untitled</span>}</td>
                  <td className="py-1 pr-3 font-mono">{n.document_id}</td>
                  <td className="py-1 pr-3 font-mono">{n.similarity.toFixed(4)}</td>
                  <td className="py-1 pr-3">{pct(n.vote_weight)}</td>
                  <td className="py-1">
                    {n.ipc_codes.length === 0 ? (
                      <span className="text-zinc-500">—</span>
                    ) : (
                      n.ipc_codes.map((code, j) => (
                        <span key={`${j}-${code}`}>
                          {j > 0 && ", "}
                          <Link className="whitespace-nowrap font-mono underline" href={`/taxonomy?code=${encodeURIComponent(code)}`}>{code}</Link>
                        </span>
                      ))
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
