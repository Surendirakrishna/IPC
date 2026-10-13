"use client";

import type { ClassifyResponse, FallbackReason } from "@/lib/types";
import { pct } from "./CandidateCard";

const REASON_TEXT: Record<Exclude<FallbackReason, "vote_below_minimum">, string> = {
  no_serving_corpus: "No patent corpus is serving, so the neighbour vote did not run.",
  patent_encoder_unavailable: "The patent encoder is unavailable, so the neighbour vote did not run.",
  vote_empty: "No similar patent with a usable IPC label was found, so the neighbour vote had nothing to count.",
  vote_failed: "The neighbour vote failed (details under Warnings).",
  neighbour_vote_disabled: "The neighbour vote is turned off.",
};

// Contract 2.0: the client reads `classifier` (never infers it from `vote` or `origin`) and shows a
// notice whenever the definition fallback chose the codes.
export function FallbackNotice({ result }: { result: ClassifyResponse }) {
  if (result.classifier !== "definition_fallback") return null;
  const reason = result.fallback_reason;
  let why: string;
  if (reason === "vote_below_minimum") {
    const share = result.vote?.first_code_share.subclass ?? null;
    const min = result.provenance.vote_settings?.min_vote_share ?? null;
    why =
      share === null
        ? "Neighbour vote too weak."
        : `Neighbour vote too weak (${pct(share)}${min === null ? "" : `, below the ${pct(min)} minimum`}).`;
  } else if (reason === null) {
    why = "The neighbour vote did not choose these codes."; // inconsistent response; still say what we know
  } else {
    why = REASON_TEXT[reason];
  }
  return (
    <div role="status" className="rounded border border-amber-500 bg-amber-50 p-2 text-sm dark:bg-amber-950/40" data-testid="fallback-notice">
      <p>
        <span className="font-semibold">Codes from definition matching (fallback).</span> {why}
      </p>
      <p className="text-xs text-zinc-600 dark:text-zinc-400">
        {reason === "vote_below_minimum"
          ? "The weak vote's nearest patents are not shown, because they do not support these codes."
          : "No similar-patents list is shown, because these codes were not chosen by the neighbour vote."}
      </p>
    </div>
  );
}
