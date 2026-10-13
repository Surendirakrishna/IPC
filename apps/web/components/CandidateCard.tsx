"use client";

import Link from "next/link";
import type { Alternative, Candidate, Confidence, ConfidenceStatus, PathEntry, TechnicalScores, VoteShare } from "@/lib/types";
import { HighlightedField, PassageList } from "./Evidence";

/**
 * Share in [0, 1] -> "79.1%", mirroring the server's confidence_text `f"{round(100 * score, 1)}%"`.
 * Compute 100 * v first, as Python does: Math.round(v * 1000) / 10 differs for some 4-decimal shares
 * (0.8865 -> 88.7 there, 88.6 on the server), so card and summary would disagree. toFixed rounds the exact
 * binary value like Python's round(); they differ only on exact binary ties (Python: half-to-even).
 */
export const pct = (v: number) => `${(100 * v).toFixed(1)}%`;

/**
 * Short score label for a collapsed card, worded by confidence.status (contract 2.0): only `calibrated`
 * is a probability; `vote_share_uncalibrated` is a raw vote share and is never called a probability.
 */
export function scoreLabel(c: Confidence): string | null {
  if (c.score === null) return null;
  if (c.status === "calibrated") return `Probability ${pct(c.score)}`;
  if (c.status === "vote_share_uncalibrated") return `vote share ${pct(c.score)}`;
  return null;
}

function ConfidenceView({ c }: { c: Confidence }) {
  if (c.status === "calibrated" && c.score !== null) {
    return (
      <div className="text-sm">
        <span className="font-semibold">Probability {pct(c.score)}</span>
        <span className="ml-2 text-xs text-zinc-500">calibrator {c.calibrator_id}</span>
        {c.interpretation && <p className="mt-1 text-xs text-zinc-600 dark:text-zinc-400">{c.interpretation}</p>}
      </div>
    );
  }
  if (c.status === "vote_share_uncalibrated" && c.score !== null) {
    return (
      <div className="text-sm">
        <span className="font-semibold">Vote share: {pct(c.score)}</span>
        <span className="ml-2 text-xs text-zinc-500">(uncalibrated)</span>
        {c.interpretation && <p className="mt-1 text-xs text-zinc-600 dark:text-zinc-400">{c.interpretation}</p>}
        {c.detail && <p className="text-xs text-zinc-500">{c.detail}</p>}
      </div>
    );
  }
  // Exhaustive over ConfidenceStatus so a new status fails the type check instead of rendering undefined.
  const labels: Record<ConfidenceStatus, string> = {
    not_calibrated: "Confidence: not calibrated",
    incompatible_artifact: "Confidence: calibrator incompatible with this pipeline",
    unsupported_input_mode: "Confidence: not calibrated for this input mode",
    calibrated: "Confidence: not available",
    vote_share_uncalibrated: "Vote share: not available",
  };
  const label = labels[c.status];
  return (
    <div className="text-sm">
      <span className="italic">{label}</span>
      {c.detail && <p className="text-xs text-zinc-500">{c.detail}</p>}
    </div>
  );
}

/**
 * Per-level vote shares, e.g. "H04L 95.5% · H04L 9/00 88.6% · H04L 9/08 79.1%". Null levels are not
 * applicable (fallback, subclass or class candidate) and are omitted; 0 is a real share and is shown.
 * Ancestor codes come from parent_path when present; otherwise from the display code's IPC structure
 * (subclass = first four characters, main group = "<subclass> <n>/00").
 */
function VoteShareBreakdown({ v, displayCode, level, path }: { v: VoteShare; displayCode: string; level: string; path?: PathEntry[] }) {
  const fromPath = (lvl: string) => path?.find((p) => p.level === lvl)?.display_code;
  const subclass = fromPath("subclass") ?? displayCode.slice(0, 4);
  const slash = displayCode.indexOf("/");
  const maingroup = fromPath("maingroup") ?? (slash > 0 ? `${displayCode.slice(0, slash)}/00` : displayCode);
  const parts: { key: string; label: string; share: number }[] = [];
  if (v.subclass !== null) parts.push({ key: "subclass", label: subclass, share: v.subclass });
  if (v.maingroup !== null) {
    parts.push({ key: "maingroup", label: level === "maingroup" ? `${maingroup} (any code)` : maingroup, share: v.maingroup });
  }
  if (v.exact !== null) {
    parts.push({ key: "exact", label: level === "maingroup" ? `${displayCode} exactly` : displayCode, share: v.exact });
  }
  if (parts.length < 2) return null; // a single level repeats the score shown above
  return (
    <p className="text-xs text-zinc-600 dark:text-zinc-400" data-testid="vote-share-breakdown">
      Vote share by level:{" "}
      {parts.map((p, i) => (
        <span key={p.key}>
          {i > 0 && " · "}
          <span className="font-mono">{p.label}</span> {pct(p.share)}
        </span>
      ))}
    </p>
  );
}

export function ScoresPanel({ scores, origin }: { scores: TechnicalScores; origin?: string[] }) {
  const fmt = (v: number | null, d = 4) => (v === null ? "—" : v.toFixed(d));
  return (
    <details className="text-xs">
      <summary className="cursor-pointer text-zinc-500">Technical retrieval scores (relevance, not probability)</summary>
      <table className="mt-1 font-mono">
        <tbody>
          <tr><td className="pr-4">cosine_similarity</td><td>{fmt(scores.cosine_similarity)}</td></tr>
          <tr><td className="pr-4">vector_rank</td><td>{scores.vector_rank ?? "—"}</td></tr>
          <tr><td className="pr-4">keyword ts_rank_cd</td><td>{fmt(scores.keyword_score)}</td></tr>
          <tr><td className="pr-4">keyword_rank</td><td>{scores.keyword_rank ?? "—"}</td></tr>
          <tr><td className="pr-4">rrf_score</td><td>{fmt(scores.rrf_score, 5)}</td></tr>
          <tr><td className="pr-4">fused_rank</td><td>{scores.fused_rank ?? "—"}</td></tr>
          <tr><td className="pr-4">reranker_score (raw)</td><td>{fmt(scores.reranker_score, 3)} ({scores.reranker_source ?? "—"})</td></tr>
          {origin && <tr><td className="pr-4 align-top">origin</td><td>{origin.join(", ")}</td></tr>}
        </tbody>
      </table>
    </details>
  );
}

export function CandidateCard({ c, fields }: { c: Candidate; fields: Record<string, string> }) {
  // With the vote primary a candidate may have no passage at all (contract 2.0); never index [0].
  const abstractSpans = c.supporting_passages.filter((p) => p.source === "abstract");
  const score = scoreLabel(c.confidence);
  return (
    <details className="rounded border border-zinc-300 p-3 dark:border-zinc-700" data-testid="candidate-card">
      <summary className="cursor-pointer">
        <span className="font-mono font-semibold">{c.display_code}</span>
        <span className="ml-2">{c.title}</span>
        <span className="ml-2 text-xs uppercase text-zinc-500">{c.level}</span>
        {score && <span className="ml-2 whitespace-nowrap text-sm font-medium" data-testid="candidate-score">{score}</span>}
      </summary>
      <div className="mt-3 space-y-3">
        <div className="text-xs text-zinc-500">Internal code <span className="font-mono">{c.code}</span></div>
        {c.original_title && c.original_title !== c.title && (
          <div className="text-sm"><span className="font-semibold">Original wording: </span>{c.original_title}</div>
        )}
        <div className="space-y-1">
          <ConfidenceView c={c.confidence} />
          <VoteShareBreakdown v={c.vote_share} displayCode={c.display_code} level={c.level} path={c.parent_path} />
        </div>
        <div>
          <h4 className="text-sm font-semibold">Supporting passages</h4>
          <PassageList passages={c.supporting_passages} fields={fields} />
          {c.depends_on_optional_documents && (
            <p className="mt-1 text-xs text-amber-700 dark:text-amber-400">Depends on an optional-document passage, not the abstract alone.</p>
          )}
          {abstractSpans.length > 0 && <HighlightedField label="abstract" text={fields.abstract} spans={abstractSpans} />}
        </div>
        <div>
          <h4 className="text-sm font-semibold">Rationale</h4>
          <p className="text-sm">{c.rationale}</p>
        </div>
        <div>
          <h4 className="text-sm font-semibold">Parent scope</h4>
          <ol className="text-sm">
            {c.parent_path.map((p) => (
              <li key={p.code}>
                <span className="text-xs uppercase text-zinc-500">{p.level}</span>{" "}
                <Link className="font-mono underline" href={`/taxonomy?code=${encodeURIComponent(p.display_code)}`}>{p.display_code}</Link>{" "}
                {p.title}
              </li>
            ))}
          </ol>
        </div>
        {c.unresolved_ambiguity.length > 0 && (
          <div>
            <h4 className="text-sm font-semibold">Unresolved ambiguity</h4>
            <ul className="list-disc pl-5 text-sm text-amber-800 dark:text-amber-300">
              {c.unresolved_ambiguity.map((a, i) => <li key={i}>{a}</li>)}
            </ul>
          </div>
        )}
        {c.references.length > 0 && (
          <div>
            <h4 className="text-sm font-semibold">References from this place</h4>
            <ul className="text-sm">
              {c.references.map((r) => (
                <li key={`${r.kind}-${r.target_code}`}>
                  <span className="text-xs text-zinc-500">{r.kind.replace("_", " ")}</span>{" "}
                  <Link className="font-mono underline" href={`/taxonomy?code=${encodeURIComponent(r.target_display_code ?? r.target_code)}`}>
                    {r.target_display_code ?? r.target_code}
                  </Link>{" "}
                  {r.target_title}
                  {r.original_wording && <span className="block text-xs text-zinc-500">wording: “{r.original_wording}”</span>}
                  <span className="block text-xs text-amber-700 dark:text-amber-400">{r.scope_check}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
        <ScoresPanel scores={c.scores} origin={c.origin} />
      </div>
    </details>
  );
}

export function AlternativesPanel({ alts, fields }: { alts: Alternative[]; fields: Record<string, string> }) {
  if (alts.length === 0) return null;
  return (
    <section aria-labelledby="alt-heading" className="rounded border border-dashed border-amber-500 p-3" data-testid="alternatives">
      <h3 id="alt-heading" className="font-semibold">Alternatives for review (not selected)</h3>
      <p className="mb-2 text-xs text-zinc-600 dark:text-zinc-400">
        Tentative codes with unresolved scope. They are not part of the Classification (IPC) result.
      </p>
      <ul className="space-y-2">
        {alts.map((a) => {
          const score = scoreLabel(a.confidence);
          return (
            <li key={a.code}>
              <details>
                <summary className="cursor-pointer text-sm">
                  <span className="font-mono">{a.display_code}</span> {a.title}
                  {score && <span className="ml-2 whitespace-nowrap font-medium" data-testid="alternative-score">{score}</span>}
                  <span className="ml-2 text-xs text-zinc-500">— {a.reason}</span>
                </summary>
                <div className="ml-4 mt-2 space-y-2">
                  {score && (
                    <div className="space-y-1">
                      <ConfidenceView c={a.confidence} />
                      <VoteShareBreakdown v={a.vote_share} displayCode={a.display_code} level={a.level} />
                    </div>
                  )}
                  <PassageList passages={a.supporting_passages} fields={fields} />
                  {a.unresolved_ambiguity.length > 0 && (
                    <ul className="list-disc pl-5 text-xs">{a.unresolved_ambiguity.map((u, i) => <li key={i}>{u}</li>)}</ul>
                  )}
                  <ScoresPanel scores={a.scores} />
                </div>
              </details>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
