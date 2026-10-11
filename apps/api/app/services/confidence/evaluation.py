"""Calibration harness: trusted evaluation data -> candidate pairs -> fit -> held-out evaluation.

Evaluation JSONL schema (one document per line), see data/evaluation/README.md:
{
  "document_id": str, "patent_family_id": str|null,
  "abstract": str, "title": str|null, "specification": str|null, "claims": str|null,
  "taxonomy_version": str, "requested_level": "group",
  "labels": [{"code": str, "judgement": "accept"|"reject", "reviewer": str|null}],
  "judgement_completeness": "complete"|"partial",
  "provenance": {"source": str, "protocol": str, "reviewed_at": str},
  "input_mode": "abstract_only"|"abstract_with_optional_documents",
  "split": "dev"|"fit"|"test",
  "synthetic_contract_fixture": bool   # must be false for calibration
}

Pair labels: accept -> 1; explicit reject -> 0; unlisted code -> 0 only when
judgement_completeness == "complete", otherwise excluded (never an implicit negative).
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED = ("document_id", "abstract", "taxonomy_version", "requested_level", "labels",
            "judgement_completeness", "provenance", "input_mode", "split")
SPLITS = ("dev", "fit", "test")


class EvaluationDataError(ValueError):
    pass


def load_eval_jsonl(path: Path) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    errors: list[str] = []
    families: dict[str, set[str]] = defaultdict(set)
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        d = json.loads(line)
        missing = [k for k in REQUIRED if k not in d]
        if missing:
            errors.append(f"line {i}: missing {missing}")
            continue
        if d.get("synthetic_contract_fixture"):
            errors.append(f"line {i}: synthetic contract fixtures cannot be used as calibration evidence")
        if d["split"] not in SPLITS:
            errors.append(f"line {i}: split must be one of {SPLITS}")
        if not str(d["abstract"]).strip():
            errors.append(f"line {i}: blank abstract")
        if d.get("patent_family_id"):
            families[d["patent_family_id"]].add(d["split"])
        docs.append(d)
    leaks = {f: sorted(s) for f, s in families.items() if len(s) > 1}
    if leaks:
        errors.append(f"patent families span multiple splits (leakage): {dict(list(leaks.items())[:5])}")
    if errors:
        raise EvaluationDataError("; ".join(errors[:20]))
    return docs


@dataclass
class Pair:
    document_id: str
    split: str
    input_mode: str
    requested_level: str
    code: str
    reranker_score: float
    label: int


def label_pairs(doc: dict[str, Any], scored: list[tuple[str, float]]) -> list[Pair]:
    accepted = {lab["code"] for lab in doc["labels"] if lab["judgement"] == "accept"}
    rejected = {lab["code"] for lab in doc["labels"] if lab["judgement"] == "reject"}
    complete = doc["judgement_completeness"] == "complete"
    out: list[Pair] = []
    for code, score in scored:
        if code in accepted:
            y = 1
        elif code in rejected or complete:
            y = 0
        else:
            continue  # unlisted in an incompletely labelled document: not a negative
        out.append(Pair(doc["document_id"], doc["split"], doc["input_mode"], doc["requested_level"], code,
                        float(score), y))
    return out


def fit_platt(pairs: list[Pair]) -> dict[str, float]:
    from sklearn.linear_model import LogisticRegression

    x = np.array([[p.reranker_score] for p in pairs])
    y = np.array([p.label for p in pairs])
    if len(set(y.tolist())) < 2:
        raise EvaluationDataError("fit split needs both accepted and rejected pairs")
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)  # ~unregularized Platt scaling
    model.fit(x, y)
    return {"coef": float(model.coef_[0][0]), "intercept": float(model.intercept_[0])}


def fit_isotonic(pairs: list[Pair]) -> dict[str, list[float]]:
    from sklearn.isotonic import IsotonicRegression

    x = np.array([p.reranker_score for p in pairs])
    y = np.array([p.label for p in pairs])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(x, y)
    return {"x_thresholds": [float(v) for v in iso.X_thresholds_], "y_thresholds": [float(v) for v in iso.y_thresholds_]}


def reliability(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    """Equal-width bins on [0,1]; ECE = sum_b (n_b/N) |acc_b - conf_b|. Empty bins are reported with n=0."""
    eps = 1e-12
    n = len(probs)
    bins = []
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & ((probs < hi) if i < n_bins - 1 else (probs <= hi))
        k = int(mask.sum())
        if k:
            conf, acc = float(probs[mask].mean()), float(labels[mask].mean())
            ece += (k / n) * abs(acc - conf)
            bins.append({"lo": round(lo, 2), "hi": round(hi, 2), "n": k, "mean_confidence": conf, "accept_rate": acc})
        else:
            bins.append({"lo": round(lo, 2), "hi": round(hi, 2), "n": 0})
    p = np.clip(probs, eps, 1 - eps)
    return {
        "n": n,
        "positives": int(labels.sum()),
        "brier": float(np.mean((probs - labels) ** 2)),
        "log_loss": float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))),
        "ece": float(ece),
        "binning": f"{n_bins} equal-width bins on [0,1]",
        "bins": bins,
        "note": "Brier and log loss also reflect discrimination; they are not standalone proof of calibration.",
    }


def predict(method: str, params: dict[str, Any], scores: np.ndarray) -> np.ndarray:
    if method == "platt":
        z = params["coef"] * scores + params["intercept"]
        return 1.0 / (1.0 + np.exp(-z))
    xs, ys = np.array(params["x_thresholds"]), np.array(params["y_thresholds"])
    return np.interp(scores, xs, ys)


def evaluate(method: str, params: dict[str, Any], pairs: list[Pair]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not pairs:
        return {"overall": None, "note": "no pairs"}
    scores = np.array([p.reranker_score for p in pairs])
    labels = np.array([p.label for p in pairs], dtype=float)
    out["overall"] = reliability(predict(method, params, scores), labels)
    slices: dict[str, list[Pair]] = defaultdict(list)
    for p in pairs:
        slices[f"input_mode={p.input_mode}"].append(p)
        slices[f"level={p.requested_level}"].append(p)
    out["slices"] = {
        name: reliability(predict(method, params, np.array([p.reranker_score for p in ps])),
                          np.array([p.label for p in ps], dtype=float))
        for name, ps in slices.items()
    }
    return out


def safe_log_loss(p: float, y: int) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))
