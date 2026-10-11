"use client";

import { useState } from "react";
import { AlternativesPanel, CandidateCard } from "@/components/CandidateCard";
import { SummaryTable } from "@/components/SummaryTable";
import type { ApiErrorBody, ClassifyRequest, ClassifyResponse } from "@/lib/types";

type UiState =
  | { kind: "idle" }
  | { kind: "loading" }
  | { kind: "error"; status: number; body: ApiErrorBody }
  | { kind: "done"; result: ClassifyResponse; fields: Record<string, string> };

const blankToNull = (s: string) => (s.trim() === "" ? null : s);

export default function ClassifyPage() {
  const [abstract, setAbstract] = useState("");
  const [title, setTitle] = useState("");
  const [specification, setSpecification] = useState("");
  const [claims, setClaims] = useState("");
  const [maxResults, setMaxResults] = useState(5);
  const [useLlm, setUseLlm] = useState(false);
  const [touched, setTouched] = useState(false);
  const [state, setState] = useState<UiState>({ kind: "idle" });

  const abstractMissing = abstract.trim() === "";

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setTouched(true);
    if (abstractMissing) return;
    const req: ClassifyRequest = {
      abstract,
      title: blankToNull(title),
      specification: blankToNull(specification),
      claims: blankToNull(claims),
      target_level: "group",
      max_results: maxResults,
      use_llm: useLlm,
    };
    setState({ kind: "loading" });
    try {
      const res = await fetch("/api/classify", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(req),
      });
      const body = await res.json();
      if (!res.ok) {
        setState({ kind: "error", status: res.status, body });
        return;
      }
      const fields: Record<string, string> = { abstract };
      if (req.title) fields.title = req.title;
      if (req.specification) fields.specification = req.specification;
      if (req.claims) fields.claims = req.claims;
      setState({ kind: "done", result: body as ClassifyResponse, fields });
    } catch {
      setState({ kind: "error", status: 0, body: { error: "network_error", message: "Could not reach the server." } });
    }
  }

  return (
    <main className="mx-auto w-full max-w-5xl space-y-6 p-6">
      <header>
        <h1 className="text-2xl font-semibold">Abstract → IPC classification</h1>
        <p className="text-sm text-zinc-600 dark:text-zinc-400">
          Candidate review aid. Results are not an official classification; evidence and scope require expert review.
        </p>
      </header>

      <form onSubmit={submit} className="space-y-4" noValidate>
        <div>
          <label htmlFor="abstract" className="block font-medium">
            Abstract <span className="text-red-600">*</span>
          </label>
          <textarea
            id="abstract"
            name="abstract"
            required
            rows={12}
            value={abstract}
            onChange={(e) => setAbstract(e.target.value)}
            aria-invalid={touched && abstractMissing}
            aria-describedby="abstract-error"
            className="mt-1 w-full rounded border border-zinc-300 p-2 font-sans dark:border-zinc-700 dark:bg-zinc-900"
            placeholder="Paste the patent abstract (plain text)."
          />
          {touched && abstractMissing && (
            <p id="abstract-error" role="alert" className="text-sm text-red-600">An abstract is required.</p>
          )}
        </div>

        <div>
          <label htmlFor="title" className="block font-medium">Title (optional)</label>
          <input
            id="title"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            className="mt-1 w-full rounded border border-zinc-300 p-2 dark:border-zinc-700 dark:bg-zinc-900"
          />
        </div>

        <details className="rounded border border-zinc-300 p-3 dark:border-zinc-700">
          <summary className="cursor-pointer font-medium">Additional documents (optional)</summary>
          <div className="mt-3 space-y-3">
            <div>
              <label htmlFor="specification" className="block text-sm font-medium">Specification</label>
              <textarea id="specification" rows={10} value={specification} onChange={(e) => setSpecification(e.target.value)}
                className="mt-1 w-full rounded border border-zinc-300 p-2 font-mono text-sm dark:border-zinc-700 dark:bg-zinc-900" />
            </div>
            <div>
              <label htmlFor="claims" className="block text-sm font-medium">Claims (keep the original numbering)</label>
              <textarea id="claims" rows={10} value={claims} onChange={(e) => setClaims(e.target.value)}
                className="mt-1 w-full rounded border border-zinc-300 p-2 font-mono text-sm dark:border-zinc-700 dark:bg-zinc-900" />
            </div>
          </div>
        </details>

        <div className="flex flex-wrap items-center gap-4 text-sm">
          <label className="flex items-center gap-2">
            Max results
            <input type="number" min={1} max={20} value={maxResults}
              onChange={(e) => setMaxResults(Math.max(1, Math.min(20, Number(e.target.value) || 5)))}
              className="w-16 rounded border border-zinc-300 p-1 dark:border-zinc-700 dark:bg-zinc-900" />
          </label>
          <label className="flex items-center gap-2">
            <input type="checkbox" checked={useLlm} onChange={(e) => setUseLlm(e.target.checked)} />
            Use constrained LLM comparison (only if enabled by the operator)
          </label>
          <button type="submit" disabled={state.kind === "loading"}
            className="rounded bg-zinc-900 px-4 py-2 font-medium text-white disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900">
            {state.kind === "loading" ? "Classifying…" : "Classify"}
          </button>
        </div>
      </form>

      {state.kind === "loading" && (
        <p role="status" className="text-sm text-zinc-600">Retrieving, reranking and checking evidence…</p>
      )}

      {state.kind === "error" && (
        <div role="alert" className="rounded border border-red-400 bg-red-50 p-3 text-sm dark:bg-red-950/40" data-testid="error">
          <p className="font-semibold">
            {state.status === 503 ? "Service unavailable" : state.status === 422 || state.status === 413 ? "Input rejected" : "Request failed"}
            {" "}({state.body.error})
          </p>
          <p>{state.body.message}</p>
        </div>
      )}

      {state.kind === "done" && <Results result={state.result} fields={state.fields} />}
    </main>
  );
}

function Results({ result, fields }: { result: ClassifyResponse; fields: Record<string, string> }) {
  const cov = result.input_coverage;
  const partial = cov.specification === "partial" || cov.claims === "partial";
  return (
    <div className="space-y-4" data-testid="results">
      <SummaryTable result={result} />
      {result.field_of_invention.additional_labels.length > 0 && (
        <p className="text-sm">
          Additional fields: {result.field_of_invention.additional_labels.map((a) => `${a.label} (${a.from_code})`).join(", ")}
        </p>
      )}
      {partial && (
        <div role="alert" className="rounded border border-amber-500 bg-amber-50 p-2 text-sm dark:bg-amber-950/40" data-testid="partial-warning">
          Partial input coverage: only part of the optional documents was reviewed within the query budget. Omitted text was not used as evidence.
        </div>
      )}
      {result.degraded_mode && (
        <div role="alert" className="rounded border border-amber-500 p-2 text-sm">Degraded mode: {result.degraded_mode}</div>
      )}
      {result.status === "insufficient_evidence" && (
        <div className="rounded border border-amber-500 p-2 text-sm" data-testid="no-evidence">
          No supported classification: {result.field_of_invention.rationale}
        </div>
      )}

      {result.candidates.length > 0 && (
        <section className="space-y-2">
          <h3 className="font-semibold">Candidate details</h3>
          {result.candidates.map((c) => <CandidateCard key={c.code} c={c} fields={fields} />)}
        </section>
      )}

      <AlternativesPanel alts={result.alternatives} fields={fields} />

      <details className="text-sm">
        <summary className="cursor-pointer text-zinc-600 dark:text-zinc-400">Warnings, coverage and provenance</summary>
        <ul className="mt-2 list-disc pl-5">{result.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>
        <p className="mt-2">Coverage — abstract: {cov.abstract}; specification: {cov.specification}; claims: {cov.claims}</p>
        <p className="mt-2">Field rationale: {result.field_of_invention.rationale} (mapping v{result.field_of_invention.mapping_version})</p>
        <details className="mt-2">
          <summary className="cursor-pointer">Extracted keyword terms</summary>
          <p className="font-mono text-xs">{result.extracted_terms.map((t) => `${t.term} [${t.source}]`).join(" · ")}</p>
        </details>
        <pre className="mt-2 overflow-x-auto rounded bg-zinc-50 p-2 text-xs dark:bg-zinc-900">
          {JSON.stringify({ provenance: result.provenance, timings_ms: result.timings_ms }, null, 2)}
        </pre>
      </details>
    </div>
  );
}
