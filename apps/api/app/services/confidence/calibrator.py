"""Per-code confidence from a fitted, held-out-evaluated calibrator.

Calibrated event: an expert accepts this code as a supported candidate at the requested level
under the selected taxonomy, based on the supplied input fields and documented rule checks.

Artifacts are plain JSON (method + numeric parameters + compatibility manifest + metrics); no
pickles are ever deserialized. A score is produced only when an active artifact is compatible with
the running pipeline fingerprint and supports the request's input mode and level. Otherwise
score=None with status not_calibrated | incompatible_artifact | unsupported_input_mode.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

INTERPRETATION = (
    "Estimated probability that an expert accepts this code as a supported candidate at the requested level, "
    "measured on comparable held-out requests under the documented expert-label protocol. Not an official "
    "examiner acceptance probability, and not a measure of novelty or patentability."
)
ACTIVE_POINTER = "ACTIVE.json"


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def pipeline_fingerprint(components: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """components: embedding model/revision/run policy, reranker model/revision, scope formatter
    version, retrieval settings, taxonomy version + source hashes, field mapping version."""
    return components, sha256_obj(components)


@dataclass
class ConfidenceResult:
    score: float | None
    status: str
    calibrator_id: str | None
    interpretation: str | None = None
    detail: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "status": self.status,
            "calibrator_id": self.calibrator_id,
            "interpretation": self.interpretation,
            "detail": self.detail,
        }


@dataclass
class CalibratorArtifact:
    manifest: dict[str, Any]

    @property
    def calibrator_id(self) -> str:
        return self.manifest["calibrator_id"]

    def params_hash(self) -> str:
        return sha256_obj({"method": self.manifest["method"], "parameters": self.manifest["parameters"]})

    def predict(self, raw_score: float) -> float:
        method = self.manifest["method"]
        p = self.manifest["parameters"]
        if method == "platt":
            z = float(p["coef"]) * float(raw_score) + float(p["intercept"])
            return 1.0 / (1.0 + math.exp(-z)) if z >= 0 else math.exp(z) / (1.0 + math.exp(z))
        if method == "isotonic":
            xs, ys = p["x_thresholds"], p["y_thresholds"]
            if raw_score <= xs[0]:
                return float(ys[0])
            if raw_score >= xs[-1]:
                return float(ys[-1])
            for i in range(1, len(xs)):
                if raw_score <= xs[i]:
                    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
                    return float(y0 + (y1 - y0) * (raw_score - x0) / (x1 - x0)) if x1 > x0 else float(y1)
        raise ValueError(f"unsupported calibration method {method!r}")


class ConfidenceService:
    def __init__(self, calibration_dir: Path, fingerprint_hash: str) -> None:
        self.dir = calibration_dir
        self.fingerprint_hash = fingerprint_hash
        self.artifact: CalibratorArtifact | None = None
        self.load_error: str | None = None
        self._load_active()

    def _load_active(self) -> None:
        pointer = self.dir / ACTIVE_POINTER
        if not pointer.exists():
            return
        try:
            ref = json.loads(pointer.read_text(encoding="utf-8"))
            manifest = json.loads((self.dir / ref["manifest"]).read_text(encoding="utf-8"))
            art = CalibratorArtifact(manifest)
            if art.params_hash() != manifest.get("artifact_sha256") or ref.get("artifact_sha256") != art.params_hash():
                self.load_error = "artifact hash mismatch"
                return
            if manifest.get("status") != "active":
                self.load_error = f"artifact status is {manifest.get('status')!r}"
                return
            self.artifact = art
        except Exception as exc:  # noqa: BLE001
            self.load_error = f"could not load active calibrator: {exc}"

    def status(self) -> dict[str, Any]:
        if self.artifact is None:
            return {"state": "not_calibrated", "detail": self.load_error, "calibrator_id": None}
        compatible = self.artifact.manifest["compatibility"]["fingerprint_hash"] == self.fingerprint_hash
        return {
            "state": "calibrated" if compatible else "incompatible_artifact",
            "calibrator_id": self.artifact.calibrator_id,
            "input_modes": self.artifact.manifest.get("input_modes"),
            "requested_levels": self.artifact.manifest.get("requested_levels"),
        }

    def confidence(self, raw_score: float | None, *, input_mode: str, level: str) -> ConfidenceResult:
        art = self.artifact
        if art is None:
            return ConfidenceResult(None, "not_calibrated", None, None,
                                    self.load_error or "No validated calibrator artifact is active.")
        if art.manifest["compatibility"]["fingerprint_hash"] != self.fingerprint_hash:
            return ConfidenceResult(None, "incompatible_artifact", art.calibrator_id, None,
                                    "Active calibrator was fitted for a different pipeline/taxonomy fingerprint.")
        if input_mode not in art.manifest.get("input_modes", []) or level not in art.manifest.get("requested_levels", []):
            return ConfidenceResult(None, "unsupported_input_mode", art.calibrator_id, None,
                                    f"Calibrator was not evaluated for input_mode={input_mode}, level={level}.")
        if raw_score is None:
            return ConfidenceResult(None, "not_calibrated", art.calibrator_id, None, "No reranker score for candidate.")
        return ConfidenceResult(round(art.predict(raw_score), 4), "calibrated", art.calibrator_id, INTERPRETATION)
