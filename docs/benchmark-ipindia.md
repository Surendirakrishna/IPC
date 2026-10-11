# Benchmark: IP India published applications

**Summary.** On 123 held-out real Indian patent applications (title + abstract only), the current
pipeline (policy v2, field mapping v2) answers 98.4% of the time. Its first code matches a recorded
IPC symbol at:
- **subclass:** 28.5% (95% CI 21.2–37.0);
- **main group:** 17.1% (11.4–24.7);
- **exact symbol:** 8.1% (4.5–14.3).

Its field of invention matches the office's label **41.6%** (33.3–50.4) of the time.

The previous policy (v1) abstained on 87.8% of the same documents, with subclass top-1 of 7.3% and
field agreement of 0%.

Treat these as **agreement with the IPC recorded on publication**, not examiner-verified accuracy.
They're useful for comparing versions of this system, not as a claim of official classification
accuracy.

> **2026-10-11: the pipeline has changed since these held-out figures.** It now uses the WIPO
> 2026.01 hierarchy, subclass-first retrieval and a reranker without WIPO notes. On the dev set it
> beats the version above at every level (see
> [Dev-set results after the WIPO 2026.01 rebuild](#dev-set-results-after-the-wipo-202601-rebuild-2026-10-11)).
> The held-out test set has not been re-scored yet, so the figures above still describe the
> previous pipeline.

## Data

| | Dev set (design and tuning) | Test set (held out, scored once) |
| --- | --- | --- |
| File | `data/evaluation/ipindia_2026-09.jsonl` | `data/evaluation/ipindia_test_pool_2026-10.jsonl` |
| Applications | 130 | 125 (123 scorable) |
| Sampling | 26 seeded-random result pages × 5 seeded rows (seed 20261010) | 30 seeded pages × 5 rows (seed 20261011), **excluding every dev application** (overlap 0). 6 pages lost to an expired session, so 125 were collected |
| Journal issues | 04/09 – 02/10/2026 | same five issues |
| With claims | 101 | 72 |

### Source

IP India Public Search, *Published* applications. Both samples come from the same 14,417-application
result pool. The site ignored the publication-date range entered in the form, so the pool is the
journal issues dated 4 September to 2 October 2026. It is not a single calendar month, and the dev
and test sets are disjoint draws from the same pool.

### Collection

- A human typed one CAPTCHA per session in a visible Edge window; CAPTCHAs were never solved
  automatically.
- Requests were spaced 1.5–3 s apart.
- Script: `apps/web/scripts/ipindia-collect.mjs`. Setting `EXCLUDE_DIRS` keeps samples disjoint.

### Ground truth

- **IPC:** the recorded *Classification (IPC)* of each application, on average about 4.5 symbols. It
  is usually declared by the applicant at filing.
- **Field:** the recorded *Field Of Invention*.
- **Missing codes:** 19 dev symbols and some test symbols come from older IPC editions and are absent
  from the 2026.01 catalogue. They are scored at the deepest level that exists. Two test applications
  had no recorded symbol in the catalogue at all and are excluded (125 → 123).

### Scoring levels

- **subclass:** e.g. `G06N`.
- **main group:** e.g. `G06N 3/00`.
- **exact:** the recorded symbol itself.

A prediction is correct at a level if it matches **any** recorded symbol at that level. "Top-1" uses
the first supported code, and abstention counts as wrong.

## What changed between v1 and v2 (decided on the dev set only)

1. **Selection policy v2.** The lexical scope heuristics became review notes instead of rejections,
   the absolute reranker floor was removed, and the evidence threshold changed from −3 to −5. The
   relative gap from the best candidate changed from 1.5 to 0.5. Grid search and per-check ablation
   are in [classification-policy.md](classification-policy.md).
2. **Field mapping v2.** It maps IPC class → IP India *Field Of Invention* vocabulary, with subclass
   overrides and section fallbacks, learned only from the dev set. `H04 → COMMUNICATION` is kept, and
   the dev data also supports it. File: `apps/api/app/services/fields/field_mapping.json`, which
   records its provenance.

## Results on the held-out test set (123 applications)

### Abstract-only (title + abstract)

| Variant | Abstains | Subclass top-1 | Main-group top-1 | Exact top-1 | Subclass, any returned code | Field label |
| --- | --- | --- | --- | --- | --- | --- |
| v1 policy (replayed on the same rankings) | 87.8% | 7.3% (3.9–13.3) | 2.4% (0.8–6.9) | 1.6% (0.4–5.7) | 7.3% | 0% |
| **v2 policy (live API)** | **1.6%** | **28.5% (21.2–37.0)** | **17.1% (11.4–24.7)** | **8.1% (4.5–14.3)** | **35.0%** | **41.6% (33.3–50.4)** |
| Baseline: always the top-ranked candidate | 0% | 28.5% | 16.3% | 6.5% | 28.5% | – |
| Baseline: always the top 3 | 0% | 28.5% | 16.3% | 6.5% | 49.6% | – |

Candidate ranking, before selection:

| Correct code within the top *k* | Subclass | Main group | Exact |
| --- | --- | --- | --- |
| hit@1 | 28.5% | 16.3% | 6.5% |
| hit@3 | 54.5% | 29.3% | 13.8% |
| hit@5 | 63.4% | 37.4% | 17.1% |
| hit@10 | 71.5% | 48.0% | 20.3% |

### Abstract + claims

The same 125 test applications were run with the recorded claims added. 72 have claims; the other 53
behave as abstract-only. The policy settings are the abstract-tuned v2 values.

| Variant (all 123 scorable) | Abstains | Subclass top-1 | Main-group top-1 | Exact top-1 | Field label | Median time per document |
| --- | --- | --- | --- | --- | --- | --- |
| Abstract-only, v2 | 1.6% | 28.5% | 17.1% | 8.1% | 41.6% | 7.4 s |
| Abstract + claims, v2 | 0.8% | 27.6% (20.5–36.1) | 10.6% (6.3–17.2) | 2.4% (0.8–6.9) | 37.6% | 15.2 s |

Restricted to the 72 applications that have claims:

| Mode | Subclass top-1 | Main-group top-1 | Exact top-1 | Subclass hit@10 | Field label |
| --- | --- | --- | --- | --- | --- |
| Title + abstract | 34.7% (24.8–46.2) | 23.6% (15.3–34.6) | 13.9% (7.7–23.7) | 68.1% | 31 / 72 |
| Title + abstract + claims | 33.3% (23.5–44.8) | 12.5% (6.7–22.1) | 4.2% (1.4–11.5) | 69.4% | 26 / 72 |

**Adding claims does not help in the current pipeline.** Subclass agreement is unchanged, but main-group
and exact agreement roughly halve, and requests take about twice as long. Likely causes:
- Claim chunks get the same retrieval vote as the abstract (equal per-source caps).
- Claim language is generic ("a system comprising…"), which pulls candidates towards broad or
  neighbouring groups.
- The thresholds were tuned for abstract-only scores.

Until claims-mode weighting is tuned on a dev set, title + abstract is the recommended input for
classification. Claims stay useful as evidence text for review.

## Reading the numbers

- **What v2 fixed.** v1's abstention problem is gone. v2's first code is about as good as the
  reranker's top-1 candidate. It adds extra codes only when they score within 0.5 of the best, which
  raises "any returned code correct" from 28.5% to 35.0%.
- **Dev versus test.** Dev subclass top-1 was 37.7% and test is 28.5%. The confidence intervals
  overlap, so this is consistent with sampling variation on about 125 documents. Because the
  thresholds were chosen on dev, the test figure is the one to quote.
- **Where errors come from.** The ranking is the remaining bottleneck. The correct subclass is in the
  top 10 for 71.5% of documents but first for only 28.5%, and exact-symbol agreement is low.
  - Recorded codes are applicant-declared, often broad and sometimes from old IPC editions.
  - Abstracts alone often omit the defining feature of the recorded subgroup.
  - The catalogue's `semantic_text` is dominated by long ancestor headings.
- **Field labels.** Field agreement went from 0% to 41.6%. The most frequent confusions are
  ELECTRONICS ↔ COMPUTER SCIENCE / MECHANICAL ENGINEERING and MECHANICAL ↔ BIO-MEDICAL. These
  categories genuinely overlap, and the office's label is chosen by the applicant.
- **QKD contract example.** The example now returns H04W 12/0433 "Key management protocols", with a
  review note that the H04W "wireless" scope isn't evidenced. Under v1 the subclass-scope check had
  rejected it in favour of H04L 9/08, which is now listed as an alternative. The dev data showed that
  enforcing that check costs about 9 points of subclass top-1 overall.

## Dev-set results after the WIPO 2026.01 rebuild (2026-10-11)

All figures in this section are from the **130-application dev set**, title + abstract only. They
were used to make design decisions, so they overstate what a held-out test will show. What changed:

- **Catalogue:** the WIPO 2026.01 master files replace the flattened catalogue. Subgroups now sit
  under their real dotted parents, notes keep their owning symbol, type and edition, and indexing
  codes (`entry_type` `I`) are never returned. New BGE-M3 vectors: `emb-bge_m3-61c53c85a23d`.
- **Retrieval:** subclass-first. A deep group search over 300 candidates votes for up to 30 subclasses,
  and groups are then searched and fused only inside them.
- **Claims:** excluded from retrieval. They are used only as review evidence.
- **Reranker input:** candidate title and WIPO path, **without** the WIPO notes
  (`RERANK_OFFICIAL_CONTEXT=false`). The notes still drive review notes and the rules engine.
- **WIPO rules:** precedence rules apply; place rules are off (`APPLY_PLACE_RULES=false`, see below).
- **Bug fix:** after the re-import, a main group's `maingroup_code` is its own code, and the policy
  read that as "a subgroup of this main group is selected". Every eligible main group was demoted,
  which caused every dev abstention (13.8%). Fixed in `pipeline._apply_policy` and in the offline
  replay; covered by `test_simulate_policy_main_group_is_not_its_own_subgroup`.

### First-code agreement and candidate recall

Runs were made with the live API; the selection policy was then replayed offline with the
hierarchy fix, using `_score_docs` and rules off.

| Pipeline | Subclass top-1 | Main-group top-1 | Exact top-1 | Median time per document |
| --- | --- | --- | --- | --- |
| Old flat v2 (`ipindia_2026-09__abstract`) | 37.7% | 22.3% | 10.8% | 8.1 s |
| WIPO 2026.01 + routing, reranker **with** WIPO notes (devA, `wipo2026_dev__abstract`) | 31.5% | 19.2% | 6.2% | 23.8 s |
| WIPO 2026.01 + routing, reranker **without** notes (devB, `wipo2026_dev__abstract__no_official_context`) | **42.3%** | **24.6%** | **11.5%** | 10.1 s |

With the recorded WIPO-rule demotions also replayed, devB gives 43.1 / 23.8 / 10.8%. That
difference is the rules' whole effect on this set: one document at each level.

The correct code appears somewhere in the top 100 candidates more often than before (hit@100):

| hit@100 | Subclass | Main group | Exact |
| --- | --- | --- | --- |
| Old flat v2 | 76.2% | 54.6% | 39.2% |
| WIPO 2026.01 + routing | 83.1% | 66.2% | 50.0% |

### Per-stage recall

The benchmark now reports where the correct code drops out (`stage_recall` in `report.json`). Each
cell is the share of documents with at least one recorded code that survives the stage, followed in
brackets by the mean share of recorded codes that survive:

| Stage (subclass level) | Heading routing (earlier attempt) | Group-vote routing (current) |
| --- | --- | --- |
| Subclass route, top 10 | 45.4% (0.247) | 78.5% (0.519) |
| Subclass route, full list | 55.4% (0.323) | 89.2% (0.642) |
| Group candidates, top 100 | 51.5% | 86.9% |
| After reranking, top 100 | 50.8% | 83.1% |

- **Heading routing** matched the abstract against subclass headings. It lost almost half the
  documents at the first stage, and its live subclass top-1 was 20.0%.
- **Group-vote routing** scores each subclass by the summed fused score of its own groups in a deep
  group search.
- **Recall before reranking** (fused top 100, all 130 documents):

  | Candidate generation | Subclass | Main group | Exact |
  | --- | --- | --- | --- |
  | Flat retrieval, new vectors | 85.4% | 70.0% | 48.5% |
  | Flat retrieval, old vectors | 84.6% | 69.2% | 50.8% |
  | Routed, top 30 subclasses | 85.4% | 71.5% | 50.0% |

  **Neither routing nor the new vectors loses recall.** The remaining gap is in ranking.

### Reranker input A/B: why the WIPO notes were removed

Each document's same 100 candidates were reranked offline with four versions of the candidate text;
nothing else changed. Data: `data/manifests/benchmarks/wipo2026_dev__scope_ab/`.

| Candidate text | Subclass top-1 | Main-group top-1 | Exact top-1 | Subclass hit@10 | Median rerank time |
| --- | --- | --- | --- | --- | --- |
| Title + path + all WIPO notes and references | 30.8% | 18.5% | 6.2% | 66.2% | 9.4 s |
| Title + path + references, **no notes (adopted)** | 41.5% | 22.3% | 10.0% | 75.4% | 4.4 s |
| Title + path only | 42.3% | 21.5% | 13.1% | 74.6% | 3.9 s |
| Title + path + the code's own notes, ≤ 600 characters | 26.9% | 14.6% | 6.2% | 70.0% | 4.5 s |

- With all notes included, the median candidate text is 1,576 tokens. 81% of candidates exceed the
  reranker's 768-token limit for that side of the pair, so the reranker mostly read notes cut off
  mid-text.
- Truncation is not the whole cause. The short, untruncated own-notes version is the worst of the
  four, so the note wording itself (cited symbols, exclusions, "e.g." lists) misleads this
  general-purpose reranker.
- "Title + path only" looks better at the exact level, but by 17 documents against 13, which is
  within noise. It may be worth an end-to-end run.
- Caveats:
  - Top-1 here reuses devA's evidence scores. For 36–41 documents per variant the new first code had
    no recorded evidence score and was assumed to pass.
  - This A/B scores against all recorded symbols, not only those in the catalogue, so it can differ
    from the CLI reports by about one document.

### WIPO rules on the dev set

5 of 130 documents had a candidate demoted by a WIPO rule:

- **Precedence was correct:** for example A61B 8/08 gave way to A61B 8/02.
- **Place rules were wrong:**
  - in one case they demoted a recorded code (C01B 21/26);
  - in another they read an "also classify" note as an instruction to pick a place.

Place rules are therefore off by default until they can be checked against reviewed labels.

## Claims weighting (dev)

The 101 dev applications that have claims (`ipindia_2026-09_with_claims.jsonl`). The pipeline is the
shipped configuration (no WIPO notes in the reranker, precedence rules on, place rules as notes only).
Only claim handling changes. Live results, `_score_docs`:

| Claims in the request | Subclass top-1 | Main group top-1 | Exact top-1 | Subclass routing recall (30) | Median s |
| --- | --- | --- | --- | --- | --- |
| off: evidence only (default) | 40.6% | 22.8% | 7.9% | 0.625 | 11.6 |
| independent claims vote, claims source weight **0.35** | **42.6%** | **25.7%** | **8.9%** | 0.672 | 13.6 |
| independent claims vote, claims source weight 0.70 | 38.6% | 23.8% | 8.9% | 0.679 | 13.7 |
| *(same 101 documents, abstract only, devB replay)* | *43.6%* | *25.7%* | *11.9%* | *0.625* | *10.0* |

- Dependent claims never vote (`DEPENDENT_CLAIM_WEIGHT=0`). Independent claims raise subclass routing
  recall (0.625 → 0.67), and the 0.35 weight is best at every level. The gain over evidence-only claims
  is 2–3 documents of 101, within noise. At 0.70, claim wording starts to pull subclass top-1 down.
- Supplying claims as evidence alone, with no retrieval vote, does not beat abstract-only. Claim passages
  change which candidates pass the evidence check, without adding better candidates.
- **Decision:** the default stays `CLAIMS_RETRIEVAL_MODE=off`, as the review asked ("keep claims out of
  retrieval for now"). `independent_weighted` at 0.35 is the tested alternative, and it is reported on
  the held-out set below. Runs: `wipo2026_devclaims__abstract_claims__{off,indep035,indep070}`.
- Precedence rules demoted a candidate in 4 of these 101 documents. Each demotion follows the scheme
  text, e.g. A63B 23/00 "A63B 22/00 takes precedence". In one case the applicant had recorded the
  demoted code (A63B 23/12). Applicant-declared codes cannot say which is right; reviewed labels can.

## Next improvements, by expected value

1. **Re-score the held-out test set** with the current pipeline, once, and quote that figure instead
   of the dev numbers above.
2. **Train a patent-specific reranker** on expert-reviewed examples, with hard negatives from nearby
   parts of the scheme. The ranking gap is large: on dev, the correct subclass is in the top 100 for
   83% of documents but ranked first for about 42%.
   - The tooling is ready: `ipc review-queue`, `review-finalize` and `mine-negatives`, described in
     [model-operation.md](model-operation.md#review-family-splits-and-reranker-training-data).
   - It needs reviewed labels; applicant-declared codes are too noisy to train on.
3. **Build a larger expert-reviewed evaluation set**, keeping patent families together across
   splits (`ipc families`), and use it to fit the confidence calibrator. About 500 documents halves
   the width of the intervals. In the current samples, two families have two members each; both
   pairs sit within a single split.
4. **Have a domain expert review field mapping v2**: 12 subclass overrides were learned from small
   counts.

## Reproduce

```bash
# collect (a human types one CAPTCHA per session)
cd apps/web
node scripts/ipindia-collect.mjs ../../data/evaluation/ipindia_raw/2026-09 09/01/2026 09/30/2026 26 5 20261010
EXCLUDE_DIRS=../../data/evaluation/ipindia_raw/2026-09/details \
  node scripts/ipindia-collect.mjs ../../data/evaluation/ipindia_raw/test_pool_2026-10 08/01/2026 08/31/2026 30 5 20261011

# parse, run (API on :8000), compare
cd ../api
uv run ipc ipindia-parse ../../data/evaluation/ipindia_raw/test_pool_2026-10/details --out ../../data/evaluation/ipindia_test_pool_2026-10.jsonl
uv run ipc benchmark ../../data/evaluation/ipindia_test_pool_2026-10.jsonl --mode abstract
uv run ipc benchmark ../../data/evaluation/ipindia_test_pool_2026-10.jsonl --mode abstract+claims
uv run ipc benchmark-compare ../../data/evaluation/ipindia_test_pool_2026-10.jsonl \
  ../../data/manifests/benchmarks/ipindia_test_pool_2026-10__abstract/results.jsonl

# dev-only tuning (as used to choose v2)
uv run ipc benchmark-tune ../../data/evaluation/ipindia_2026-09.jsonl \
  ../../data/manifests/benchmarks/ipindia_2026-09__abstract/results.jsonl --dev-fraction 1.0 --objective-level maingroup
```

The output folders `data/manifests/benchmarks/<docs>__<mode>/` each contain `results.jsonl` (full
reranked lists with flags), `report.json` and `comparison.json`.
