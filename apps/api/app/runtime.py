"""Process-wide runtime: loads models once (in a background thread at startup) and owns the
bounded inference queue. One model-owning API process; scale later via a dedicated inference service.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from app.config import Settings, get_settings

log = logging.getLogger(__name__)


@dataclass
class Runtime:
    settings: Settings
    state: str = "not_started"  # not_started | loading | ready | failed
    error: str | None = None
    provider: Any = None
    reranker: Any = None
    pipeline: Any = None
    confidence: Any = None
    loaded_at: float | None = None
    fingerprint: dict[str, Any] = field(default_factory=dict)
    fingerprint_hash: str | None = None
    _semaphore: asyncio.Semaphore | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.settings.inference_queue_size)
        return self._semaphore

    def start_loading(self) -> None:
        with self._lock:
            if self.state in ("loading", "ready"):
                return
            self.state = "loading"
        threading.Thread(target=self._load, name="model-loader", daemon=True).start()

    def _load(self) -> None:
        try:
            from app.services.classification import llm as llm_mod
            from app.services.classification.pipeline import ClassificationPipeline
            from app.services.classification.reranker import get_reranker
            from app.services.embeddings.providers import get_provider
            from app.services.fields.mapping import load_mapping

            s = self.settings
            provider = get_provider(s)
            reranker = get_reranker(s)
            mapping = load_mapping(s.field_mapping_path)
            self.provider, self.reranker = provider, reranker
            self.refresh_confidence()
            client = llm_mod.OpenAICompatibleClient(s) if s.llm_enabled else None
            self.pipeline = ClassificationPipeline(s, provider, reranker, self.confidence, mapping, client)
            self.state, self.loaded_at = "ready", time.time()
            log.info("models ready")
        except Exception as exc:  # noqa: BLE001
            log.exception("model loading failed")
            self.state, self.error = "failed", f"{type(exc).__name__}: {exc}"[:500]

    def refresh_confidence(self) -> None:
        """Recompute the pipeline fingerprint (needs DB for release/run) and reload the calibrator."""
        from app.services.confidence.calibrator import ConfidenceService
        from app.services.confidence.fingerprint import current_fingerprint

        fp, fp_hash = current_fingerprint(self.settings, self.provider, self.reranker)
        self.fingerprint, self.fingerprint_hash = fp, fp_hash
        self.confidence = ConfidenceService(self.settings.calibration_dir, fp_hash)
        if self.pipeline is not None:
            self.pipeline.confidence = self.confidence


_runtime: Runtime | None = None


def get_runtime() -> Runtime:
    global _runtime
    if _runtime is None:
        _runtime = Runtime(get_settings())
    return _runtime
