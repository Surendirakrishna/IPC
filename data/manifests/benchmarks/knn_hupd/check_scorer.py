"""Check that offline scoring reproduces the live report for wipo2026_test__abstract."""

import json

from knn_common import REPO, TEST_DOCS, catalogue_codes, headline, load_jsonl, score_rows

bench = REPO / "data" / "manifests" / "benchmarks" / "wipo2026_test__abstract"
results = {r["application_number"]: r for r in load_jsonl(bench / "results.jsonl") if r.get("status") != "error"}
rows, _, errors = score_rows(load_jsonl(TEST_DOCS), results, catalogue_codes())
live = json.loads((bench / "report.json").read_text(encoding="utf-8"))["metrics"]
print("offline:", json.dumps(headline(rows)))
print("live:   ", {lvl: live[lvl]["top1_all_documents"]["rate"] for lvl in ("subclass", "maingroup", "exact")},
      "n =", live["documents"])
print("skipped:", errors)
