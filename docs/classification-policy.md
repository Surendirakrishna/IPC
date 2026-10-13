# Classification policy (before calibration)

Two classifiers share one pipeline ([pipeline.py](../apps/api/app/services/classification/pipeline.py)):

- **The neighbour vote** (primary, when `NEIGHBOUR_VOTE_ENABLED=true`): the IPC codes of the most
  similar labelled patents pick the codes and their order.
- **Definition matching** (below): the abstract against IPC scheme text. It explains the vote's codes,
  and it is the classifier whenever the vote is off, cannot run or is too weak.

The response's `classifier` field says which one chose the codes, and `fallback_reason` says why the
vote did not ([api.md](api.md)). The decision is ADR-0005 ("A PaECTER neighbour vote is the primary
classifier").

There is no fitted calibrator, so no numeric confidence threshold can exist yet. Vote codes carry
their raw vote share, and the definition path's selection depends on a documented **relevance and
evidence policy**. Its thresholds are heuristics in raw reranker-logit units. Neither is a
probability, and expert-labelled data must replace them.

## Neighbour vote (primary classifier)

### Selection rule

1. **Preconditions.** The vote runs only if it is enabled, a validated corpus is serving, PaECTER is
   loaded, and its revision equals the one the corpus was embedded with. Otherwise the definition
   pipeline answers, with the reason.
2. **Search.** PaECTER encodes "title. abstract". The `NEIGHBOUR_K` (100) nearest corpus patents are
   found through the serving corpus's partial HNSW index (`hnsw.ef_search` = `PATENT_HNSW_EF_SEARCH`,
   200). Fewer than k rows is a failure (`vote_failed`), not a smaller vote.
3. **Vote** ([neighbour_vote.py](../apps/api/app/services/classification/neighbour_vote.py)). Neighbour
   n weighs `exp((s_n − s₁)/T)`, with s₁ the highest similarity. Its main label counts 1 and each
   secondary label `NEIGHBOUR_BETA`, and it counts once per code at each level (subclass, main group,
   exact code). A code's **vote share** is its weighted count over the total weight of all neighbours.
   Patents carry several codes, so shares of different codes do not sum to 1.
4. **First code,** always at the deepest level, chosen top-down: the best subclass, then the best main
   group inside it, then the best exact code inside that main group. The exact code is a subgroup, or a
   main group when patents are labelled with the main group itself.
5. **Minimum share (the gate).** If the first code's **subclass** share, rounded to 4 decimals, is
   below `NEIGHBOUR_MIN_VOTE_SHARE` (0.2), the definition pipeline answers instead
   (`fallback_reason: vote_below_minimum`; the weak vote is still returned in `vote`). The 0.2 is
   **provisional**: it has not been tuned on the served vote.
6. **Codes reported,** in vote order:
   - the first code, reported as its ancestor when `target_level` is coarser;
   - then codes from the vote ranking at the requested level whose rounded own-level share is ≥ the
     same minimum share, skipping any code that is an ancestor or descendant of one already chosen;
   - `max_results` is a ceiling; up to as many again go to alternatives ("Beyond max_results ceiling.").

   | `target_level` | Ranking used for further codes |
   | --- | --- |
   | `class` | class shares (any code in the class) |
   | `subclass` | subclass shares |
   | `maingroup` | main-group shares |
   | `group` | the neighbours' exact label codes: subgroups, and main groups labelled as such |
   | `subgroup` | the exact label codes that are subgroups |

7. **Explanation, never veto.** The definition path then runs on the vote's codes only: hydrate,
   rerank (scores only; it never reorders the vote), and evidence passages. A missing or weak passage
   and the lexical scope checks become review notes. They never remove a vote code.
8. **Rules arbitrate only.** WIPO precedence rules (`APPLY_PRECEDENCE_RULES`) are applied between vote
   codes as on the definition path. An unconditional demotion moves a code to alternatives with the
   rule quoted. A rule never adds a code: the preferred place must itself be a vote code. If the first
   code yields, the code it yields to leads. If a cycle of rules would demote every code, the first
   code stays with a note.
9. **Status** is `needs_review` while vote shares are uncalibrated. `insufficient_evidence` cannot
   happen with the vote: there is always at least the first code.

Also:
- `section` / `subclass` request filters apply to definition matching only. The vote ignores them and
  says so in `warnings`.
- `use_llm` is not applied to vote codes (warning).
- A vote code missing from the taxonomy release is dropped with a warning. If the first code is
  missing, the vote counts as failed (`vote_failed`) and the definition pipeline answers.
- If the vote could not run at all (`no_serving_corpus`, `patent_encoder_unavailable`, `vote_empty`,
  `vote_failed`) and the definition pipeline cannot serve either, the API returns 503
  `patent_corpus_unavailable`.

### Edge cases to know when reading a response

- **The first candidate can be coarser than requested.** With `target_level` `subgroup` or
  `maingroup`, `candidates[0]` is the vote's first code at its own level when that code is coarser
  than requested, for example a main group for a `subgroup` request, or a subclass when no main group
  was voted. The definition path never does this: it only returns codes of the requested level.
- **Main groups at `target_level=group`.** The ranking holds the neighbours' exact label codes, so a
  main group appears when patents are labelled with the main group itself. It is gated and scored by
  its **main-group share** (any code in that main group), not by the share labelled with exactly that
  main group.
- **Place-rule demotions don't reorder.** With `APPLY_PLACE_RULES=true` (off by default), a place rule
  can demote the first code, but the code it yields to is not moved to the front. The winner is found
  by matching the precedence wording "(X takes precedence" in the demotion reason, which a place rule
  doesn't have. The next code in vote order leads instead. In the QKD example, H04L 9/08 carries
  "WIPO place rule (not applied)" notes naming the last-place rule for H04L 9/06–9/32.
  **Don't enable `APPLY_PLACE_RULES` together with the vote** until that combination has been checked:
  on the QKD example the last-place rule would demote H04L 9/08 (87.9%) and H04L 9/14 and put
  H04L 9/32 (22.1%) first (worked out from the rule logic in rules.py, not run).

## Definition pipeline (explanation and fallback)

The rest of this document describes definition matching. With the vote primary, only steps 5 and 6
(reranking and evidence) and the WIPO rules run, on the vote's codes.

1. **Query views** ([query_views.py](../apps/api/app/services/retrieval/query_views.py)):
   - An abstract (+title) view is always built. It is the only view for an abstract-only request.
   - Claims are split at reliable numbering ("1.", "2."…), keeping each claim's dependencies.
   - The specification is split at detected headings, then into overlapping chunks bounded by the
     BGE-M3 tokenizer (`QUERY_CHUNK_TOKENS`, overlap `QUERY_CHUNK_OVERLAP_TOKENS`).
   - BACKGROUND / PRIOR ART / RELATED ART sections are marked. They get fewer keyword terms, are
     reviewed last, and are never used as evidence.
   - The total number of views is capped at `MAX_QUERY_VIEWS`, chosen in a fixed order:
     1. independent claims;
     2. dependent claims;
     3. non-background specification chunks;
     4. background chunks.

     Any omission sets `input_coverage=partial` and adds a warning.
2. **Per-view retrieval.** Claims are evidence for review and do not vote here by default
   (`CLAIMS_RETRIEVAL_MODE=off`). With `independent_weighted`, each independent claim votes with
   `INDEPENDENT_CLAIM_WEIGHT`, each dependent claim with `DEPENDENT_CLAIM_WEIGHT` (0 = left out), and the
   claims source as a whole with `CLAIM_SOURCE_WEIGHT` against the abstract's 1.0.
   For group-level requests retrieval is **hierarchical**. Subclasses are voted from a deep group search
   (`ROUTING_GROUP_POOL`=300 keyword + 300 vector; each subclass scores the summed RRF of its groups).
   The top `HIERARCHICAL_SUBCLASS_LIMIT`=30 subclasses are kept, and main groups and subgroups are then
   searched again (100 + 100) and ranked only inside them. Matching subclass headings alone routed
   poorly: on the dev set it kept the recorded subclass in the top 20 for 32% of documents, against 61%
   for the group vote.
   - Keyword: PostgreSQL full-text search with `websearch_to_tsquery`. The query is an OR of up to 24
     deterministically extracted phrases, ranked with `ts_rank_cd`. This is **not BM25**.
   - Vector: pgvector HNSW cosine search over BGE-M3 vectors, with `hnsw.ef_search` set
     transaction-locally.
   - Pool sizes are 100 + 100.
3. **Fusion.** RRF with k=60 is applied at two levels. Within each view, the keyword and vector
   lists are fused. Within each source (abstract, claims, specification), the view lists are fused
   (weighted by view weight) and the result capped. Across sources, each source contributes equally
   unless a source weight is set (only the claims source has one).
4. **Expansion.** The top 20 fused candidates add:
   - their parent main groups;
   - up to 4 siblings each, ranked by exact cosine to the abstract.

   A cross-reference never creates a candidate: a cited code is context, not an assignment.

   Up to 15 strong keyword-only candidates are reserved. The pool is capped at 100.
5. **Reranking.** BGE-reranker-v2-m3 scores (scope document, abstract view) pairs, and also scores up
   to two optional views per candidate. The *shorter* text goes in the query slot, matching how the
   reranker was trained. The other way round compressed every score into 3.1–3.7 and could not
   separate H04L 9/08 from E05B 19/06 "Key bits". A scope document is built deterministically by `scope-v3`:
   1. the code and its WIPO title;
   2. the original wording;
   3. the full WIPO parent path, including every dotted subgroup parent;
   4. the context heading;
   5. the titles of referenced targets, labelled as not an assignment instruction;
   6. the version-matched WIPO notes, title references and definitions that apply to the code or an
      ancestor (including range notes such as "A01D 46/02-46/28"), each labelled with its rule kind and
      edition. Precedence and place rules come first. **Off by default** (`RERANK_OFFICIAL_CONTEXT=false`):
   with the notes, 81% of scopes exceeded the reranker's query cap, and dev subclass top-1 fell from
   42.3% to 31.5%. The notes still feed the rules below and the review output.
6. **Evidence.** The abstract, title and selected passages are split into sentences or clauses with
   exact code-point offsets. Each (sentence, scope document) pair is scored, again with the shorter
   text as the query. The policy keeps
   the best abstract passage, and adds the best optional-document passage only when it scores higher.
   Every quote is checked so that `field[start:end] == quote`.

## Selection rules on the definition path, policy v2 ([pipeline.py `_apply_policy`](../apps/api/app/services/classification/pipeline.py))

Policy v1 rejected candidates on lexical heuristics and absolute logit floors. On the IP India
benchmark it abstained on 87% of real applications and rejected the correct candidate almost every
time (see [benchmark-ipindia.md](benchmark-ipindia.md)). Policy v2 was chosen on the
**dev** set only; the held-out results are in the same report.

| Rule (pass 1) | Default (v2) | Otherwise |
| --- | --- | --- |
| `reranker_score >= MIN_RERANKER_SCORE` | **off** (`None`) | dropped |
| a passage scoring `>= MIN_EVIDENCE_SCORE` | **−5.0** | alternative ("no supporting passage") |
| at least one extracted supporting passage | always required | alternative |
| application-context, subclass-scope and defining-terms lexical checks | **reported as review notes** (`ENFORCE_LEXICAL_CHECKS=false`) | when set to `true`: alternative with the reason |

| Rule (pass 2, relevance) | Default (v2) | Otherwise |
| --- | --- | --- |
| within `MAX_RERANKER_GAP` of the best candidate that passed pass 1 | **0.5** | alternative ("well below the best supported candidate") |

The lexical checks are still computed. Every failed check appears in the candidate's
`unresolved_ambiguity` as "Scope check (lexical heuristic, needs review): …". For example, H04W's
"WIRELESS" qualifier is flagged when the input never mentions wireless.

After that:
- A main group whose own subgroup is supported moves to alternatives as the broader parent.
- **WIPO priority rules** ([rules.py](../apps/api/app/services/classification/rules.py)) are applied
  between candidates that both passed the rules above. The rule text comes from the 2026.01 master
  files, read at import by what it says (see [architecture.md](architecture.md)):
  - *precedence* (`APPLY_PRECEDENCE_RULES`): if X and Y are both supported and a rule covering X says Y
    takes precedence, X moves to alternatives with the rule quoted. A rule never adds Y; if Y was only
    ranked, X stays and gets a review note. Rules with a stated condition ("For apparatus used in
    drying…", "If…", "e.g.") only add a review note.
  - *last/first place* (`APPLY_PLACE_RULES`, **off by default**): when enabled, between two supported
    candidates in the rule's scope (neither an ancestor of the other) the later/earlier place in scheme
    order is kept. A place rule decides between places for the *same* subject, and that is not checked,
    so by default it only adds a review note. On dev it would have demoted a recorded code. Notes that
    say a part must *also* be classified somewhere are additional-classification instructions, not
    place rules.
  - *limiting references*: if another supported candidate is the cited place, both stay with a note.
  - Indexing notes, definitions, scope statements and informative references are context only.
- Supported candidates that share a main group are both kept, with a note to check the main group's
  scope and precedence notes.
- `max_results` is a ceiling. The overflow goes to alternatives.
- No supported candidate gives `status=insufficient_evidence`, field `UNDETERMINED` and the
  classification cell "Insufficient evidence".
- `status=candidates_found` requires every candidate to have calibrated confidence and no unresolved
  ambiguity. Otherwise the status is `needs_review`.

## Optional LLM (off by default)

When `LLM_ENABLED=true` and the request sets `use_llm=true`:
- The model sees the top `LLM_MAX_CANDIDATES` candidates.
- A JSON schema restricts `code` to the per-request candidate list.
- The backend rejects out-of-set codes and quotes that are not verbatim substrings of the named field.
- One correction attempt is allowed. If it still fails, the response falls back to the reranker policy
  with `degraded_mode` set.
- No confidence value is ever taken from the LLM.

## Why these defaults

These are the definition path's thresholds. The values were chosen by grid search on the 130-application IP India dev set:
- floor ∈ {off, −6 … −1};
- gap ∈ {0.25 … 3};
- evidence ∈ {off, −6 … −2};
- lexical enforcement on or off.

The objective was main-group top-1 agreement over all documents, with abstention counted as a miss
and ties broken by set-F1. Effect of the lexical checks on dev, with the other settings fixed:

| Lexical checks enforced | Abstains | Subclass top-1 | Main-group top-1 |
| --- | --- | --- | --- |
| none (v2) | 0% | 37.7% | 22.3% |
| application-context only | 0% | 35.4% | 20.8% |
| defining-terms only | 9.2% | 33.1% | 17.7% |
| subclass-scope only | 17.7% | 28.5% | 14.6% |
| all three | 30.8% | 21.5% | 11.5% |

**Known trade-off (definition path only):**
- With the vote off, the QKD contract example selects H04W 12/0433 "Key management protocols". It
  carries a review note that the H04W "wireless" scope isn't evidenced. With the vote on it returns
  H04L 9/08 at 87.9% vote share ([README](../README.md)).
- Under v1, the subclass-scope check had demoted it in favour of H04L 9/08, which is now an
  alternative.
- The example was not used to tune the policy.

The ground truth is applicant-declared IPC, not examiner-verified, so these remain heuristics. Refit
them once examiner-labelled data exists, and replace them with the calibrator.
