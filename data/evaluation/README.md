# Evaluation data

| File | Kind | May be used for calibration? |
| --- | --- | --- |
| `qkd_contract_request.json` | The exact QKD abstract-only request from the build spec. A **contract/acceptance fixture**, not labelled data. | **No.** Never reuse as calibration evidence. |
| `*.jsonl` (you supply) | Expert-labelled evaluation documents (schema below). | Yes, once its provenance has been reviewed. |

No expert-labelled dataset ships with this repository. The calibration harness runs, but no
calibrator has been fitted. The API therefore returns `confidence.score = null` with
`status = "not_calibrated"`.

## Expert-labelled JSONL schema (one document per line)

```json
{
  "document_id": "EP1234567A1",
  "patent_family_id": "F-000123",
  "abstract": "…",
  "title": null,
  "specification": null,
  "claims": null,
  "taxonomy_version": "20260101",
  "requested_level": "group",
  "labels": [
    {"code": "H04L0009080000", "judgement": "accept", "reviewer": "examiner-A"},
    {"code": "H04L0009320000", "judgement": "reject", "reviewer": "examiner-A"}
  ],
  "judgement_completeness": "partial",
  "provenance": {"source": "…", "protocol": "docs/model-operation.md#label-protocol", "reviewed_at": "2026-10-01"},
  "input_mode": "abstract_only",
  "split": "fit",
  "synthetic_contract_fixture": false
}
```

Rules enforced by `ipc generate-pairs`:
- `split` must be one of `dev`, `fit` or `test`. A `patent_family_id` must not appear in more than one split.
- `synthetic_contract_fixture: true` is rejected.
- A code the expert did not list counts as a negative only when `judgement_completeness` is `"complete"`.
- Use internal codes (for example `H04L0009080000`) exactly as they appear in the catalogue.

## Producing expert labels (run from `apps/api`)

1. **Families and splits.** `ipc families dev.jsonl=dev test.jsonl=test [new.jsonl] --raw-dir <details dir> --out families.json`.
   Applications sharing a priority claim, a parent PCT application, or a near-identical title and
   abstract form one family. Pinned files keep their split, other families get a seeded split, and the
   command exits non-zero if any family spans two splits. `families_ipindia.json` is the current
   dev/test result: 255 applications, 253 families, no cross-split family.
2. **Review queue.** `ipc review-queue docs.jsonl --results-file <benchmark results.jsonl> --families-file families.json --out queue.jsonl --key-out key.jsonl`.
   Each item lists the pipeline's top 10 plus the declared codes found in the catalogue, shuffled,
   without rank or source. Claims are included as reading evidence. **Give reviewers `queue.jsonl`
   only.** `key.jsonl` records where each candidate came from.
3. **Review.** For each candidate, set `review.judgements[code]` to `accept`, `reject` or `unsure`.
   Put any missing correct code in `added_codes`. Fill `reviewer` and `reviewed_at`. Set
   `accepted_codes_are_exhaustive: true` only when no other code in the whole catalogue applies.
4. **Finalize.** `ipc review-finalize queue.jsonl --source "<who, when, what>" --out labelled.jsonl`
   validates every reviewed item and writes this schema. `unsure` and unjudged codes are left out.
5. **Reranker training data.** `ipc mine-negatives labelled.jsonl --results-file <results.jsonl> --out groups.jsonl --flag-out rows.jsonl`
   uses only the `fit` split by default. Negatives are explicit rejects, high-ranked wrong candidates
   (complete reviews only), and nearby codes in the scheme (siblings, other main groups of the
   subclass, other subclasses of the class), never an ancestor or descendant of an accepted code.
   Nearby codes under a partial review are flagged `assumed: true`.
