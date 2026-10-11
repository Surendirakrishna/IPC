"""Expert review queue: benchmark candidates -> blinded review items -> expert-labelled JSONL.

A review item lists, for one document, the pipeline's top-k candidates plus the applicant-declared
codes that exist in the catalogue. The list is shuffled per document and carries no rank and no
"declared" marker, so neither the model's ordering nor the applicant's choice anchors the reviewer.
Where each candidate came from is written to a separate key file for later analysis.

`finalize_reviews` turns completed items into the schema read by
`app.services.confidence.evaluation.load_eval_jsonl` (see data/evaluation/README.md):
- "accept"/"reject" judgements become labels; "unsure" and unjudged candidates are left out;
- codes the reviewer adds are accepts;
- judgement_completeness is "complete" only when the reviewer explicitly says the accepted codes are
  exhaustive for the whole catalogue (unlisted codes then become negatives), otherwise "partial".
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable
from typing import Any

QUEUE_VERSION = 1
JUDGEMENTS = ("accept", "reject", "unsure")
LABEL_PROTOCOL = "docs/model-operation.md#label-protocol"


class ReviewDataError(ValueError):
    pass


def build_review_item(
    doc: dict[str, Any],
    ranked_codes: list[str],
    declared_codes: list[str],
    describe: Callable[[list[str]], dict[str, dict[str, Any]]],
    *,
    split: str,
    family_id: str | None,
    taxonomy_version: str,
    top_k: int = 10,
    seed: str = "review-v1",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (review item, key entry). Codes are internal catalogue codes; `describe` returns
    {code: {"display_code", "title", "path"}} for the codes that exist in the catalogue."""
    doc_id = str(doc.get("document_id") or doc["application_number"])
    model = list(dict.fromkeys(ranked_codes))[:top_k]
    pool = list(dict.fromkeys([*model, *declared_codes]))
    info = describe(pool)
    present = [c for c in pool if c in info]
    rng = random.Random(int(hashlib.sha256(f"{seed}:{doc_id}".encode()).hexdigest()[:16], 16))
    shuffled = present[:]
    rng.shuffle(shuffled)
    item = {
        "queue_version": QUEUE_VERSION,
        "document_id": doc_id,
        "patent_family_id": family_id,
        "split": split,
        "taxonomy_version": taxonomy_version,
        "requested_level": "group",
        "title": doc.get("title"),
        "abstract": doc["abstract"],
        # Claims are review evidence only; they are not used for candidate retrieval.
        "claims": doc.get("claims"),
        "candidates": [{"code": c, **info[c]} for c in shuffled],
        "review": {"reviewer": None, "reviewed_at": None,
                   "judgements": {c: None for c in shuffled},
                   "added_codes": [], "accepted_codes_are_exhaustive": False, "notes": ""},
    }
    key = {
        "document_id": doc_id,
        "sources": {c: {"model_rank": model.index(c) + 1 if c in model else None,
                        "declared": c in declared_codes} for c in present},
        "declared_missing_from_catalogue": [c for c in declared_codes if c not in info],
    }
    return item, key


def finalize_reviews(
    items: list[dict[str, Any]],
    *,
    source: str,
    code_exists: Callable[[str], bool] | None = None,
    protocol: str = LABEL_PROTOCOL,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (expert-labelled documents, skipped items with reasons). Raises ReviewDataError when a
    reviewed item is malformed, so a bad review file never silently produces labels."""
    out, skipped, errors = [], [], []
    for i, it in enumerate(items, start=1):
        rv = it.get("review") or {}
        judged = {c: j for c, j in (rv.get("judgements") or {}).items() if j is not None}
        added = list(dict.fromkeys(rv.get("added_codes") or []))
        if not judged and not added:
            skipped.append({"document_id": it.get("document_id"), "reason": "not reviewed"})
            continue
        where = f"item {i} ({it.get('document_id')})"
        if not rv.get("reviewer"):
            errors.append(f"{where}: reviewer is required")
        if not rv.get("reviewed_at"):
            errors.append(f"{where}: reviewed_at is required")
        candidates = {c["code"] for c in it.get("candidates", [])}
        for c, j in judged.items():
            if j not in JUDGEMENTS:
                errors.append(f"{where}: judgement for {c} must be one of {JUDGEMENTS}, got {j!r}")
            if c not in candidates:
                errors.append(f"{where}: judged code {c} is not one of the item's candidates")
        if code_exists is not None:
            errors += [f"{where}: added code {c} is not in the catalogue" for c in added if not code_exists(c)]
        exhaustive = bool(rv.get("accepted_codes_are_exhaustive"))
        accepts = [c for c, j in judged.items() if j == "accept"] + [c for c in added if c not in judged]
        if exhaustive and not accepts:
            errors.append(f"{where}: accepted_codes_are_exhaustive is set but no code was accepted")
        labels = [{"code": c, "judgement": "accept", "reviewer": rv.get("reviewer")} for c in accepts]
        labels += [{"code": c, "judgement": "reject", "reviewer": rv.get("reviewer")}
                   for c, j in judged.items() if j == "reject"]
        out.append({
            "document_id": it["document_id"],
            "patent_family_id": it.get("patent_family_id"),
            "abstract": it["abstract"],
            "title": it.get("title"),
            "specification": None,
            "claims": it.get("claims"),
            "taxonomy_version": it["taxonomy_version"],
            "requested_level": it.get("requested_level", "group"),
            "labels": labels,
            "judgement_completeness": "complete" if exhaustive else "partial",
            "provenance": {"source": source, "protocol": protocol, "reviewed_at": rv.get("reviewed_at"),
                           "review_queue_version": it.get("queue_version")},
            # Calibration and reranker scores use the abstract-only pipeline input.
            "input_mode": "abstract_only",
            "split": it["split"],
            "synthetic_contract_fixture": False,
        })
    if errors:
        raise ReviewDataError("; ".join(errors))
    return out, skipped
