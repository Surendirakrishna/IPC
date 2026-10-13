"""Process-wide runtime: loads models once (in a background thread at startup) and owns the
bounded inference queue. One model-owning API process; scale later via a dedicated inference service.

With ``NEIGHBOUR_VOTE_ENABLED=true`` the PaECTER patent encoder loads here too, once, after BGE-M3 and the
reranker. Its failure does not fail the runtime: the definition pipeline still serves, every response says
``fallback_reason: "patent_encoder_unavailable"``, and ``/ready`` reports why (``patent_encoder_error``).
The serving patent corpus is not bound here: the pipeline's store reads it per request (activation needs no
restart), and the pipeline itself refuses a corpus embedded with another encoder revision, with a warning.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings

log = logging.getLogger(__name__)

# A local copy of the pinned PaECTER revision (the Hugging Face downloader stalls on this network). Read from
# the process environment, because ``Settings`` has no field for it yet; a ``patent_encoder_local_path``
# settings field, once added to config.py, takes precedence.
PATENT_ENCODER_LOCAL_PATH_ENV = "PATENT_ENCODER_LOCAL_PATH"


def patent_encoder_local_path(settings: Settings) -> Path | None:
    value = getattr(settings, "patent_encoder_local_path", None) or os.environ.get(PATENT_ENCODER_LOCAL_PATH_ENV)
    return Path(value) if value else None


def load_patent_encoder(settings: Settings) -> Any:
    """PaECTER at the configured model id and revision, on ``EMBEDDING_DEVICE``. A local copy is checked
    against the revision's known weights hash (``paecter.check_local_copy``)."""
    from app.services.embeddings.paecter import PaecterEncoder

    return PaecterEncoder.load(
        local_path=patent_encoder_local_path(settings),
        model_id=settings.patent_encoder_model_id,
        revision=settings.patent_encoder_model_revision,
        device=settings.embedding_device,
    )


@dataclass
class Runtime:
    settings: Settings
    # True: build the pipeline with the neighbour vote switched off and load no patent encoder, while the
    # fingerprint still describes the configured (served) pipeline. For the calibration CLI, whose pairs must
    # come from the definition path (reranker scores) and whose artifact must match the API's fingerprint.
    definition_only: bool = False
    state: str = "not_started"  # not_started | loading | ready | failed
    error: str | None = None
    provider: Any = None
    reranker: Any = None
    patent_encoder: Any = None  # PaecterEncoder when NEIGHBOUR_VOTE_ENABLED and it loaded
    patent_encoder_error: str | None = None  # why it did not load (vote enabled only)
    pipeline: Any = None
    confidence: Any = None
    loaded_at: float | None = None
    fingerprint: dict[str, Any] = field(default_factory=dict)
    fingerprint_hash: str | None = None
    _semaphore: asyncio.Semaphore | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def neighbour_vote_enabled(self) -> bool:
        return self.settings.neighbour_vote_enabled and not self.definition_only

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
            if self.neighbour_vote_enabled:
                self._load_patent_encoder()
            self.refresh_confidence()
            client = llm_mod.OpenAICompatibleClient(s) if s.llm_enabled else None
            pipeline_settings = s.model_copy(update={"neighbour_vote_enabled": False}) if self.definition_only else s
            self.pipeline = ClassificationPipeline(pipeline_settings, provider, reranker, self.confidence, mapping,
                                                   client, patent_encoder=self.patent_encoder)
            self.state, self.loaded_at = "ready", time.time()
            log.info("models ready")
        except Exception as exc:  # noqa: BLE001
            log.exception("model loading failed")
            self.state, self.error = "failed", f"{type(exc).__name__}: {exc}"[:500]

    def _load_patent_encoder(self) -> None:
        """Load PaECTER once. A failure is recorded, not raised: the definition fallback still serves."""
        try:
            enc = load_patent_encoder(self.settings)
        except Exception as exc:  # noqa: BLE001 - CUDA, out of memory, missing or mismatched weights
            log.exception("patent encoder (PaECTER) failed to load; the neighbour vote is unavailable")
            self.patent_encoder, self.patent_encoder_error = None, f"{type(exc).__name__}: {exc}"[:500]
            return
        self.patent_encoder, self.patent_encoder_error = enc, None
        log.info("patent encoder ready: %s@%s on %s (%s)", enc.model_id, enc.revision, enc.device, enc.source)
        self._warn_on_corpus_mismatch()

    def _warn_on_corpus_mismatch(self) -> None:
        """Log (once, at startup) when the serving corpus was embedded with another encoder. The pipeline
        refuses such a corpus on every request (fallback_reason patent_encoder_unavailable, with a warning)."""
        try:
            from app.db import patents as patents_db
            from app.db.session import api_session, database_configured

            if not database_configured():
                return
            with api_session() as session:
                corpus = patents_db.get_serving_corpus(session)
        except Exception:  # noqa: BLE001 - informational only; /ready reports the live state
            log.warning("could not read the serving patent corpus at startup", exc_info=True)
            return
        enc = self.patent_encoder
        if corpus is not None and (corpus.encoder_model_id, corpus.encoder_model_revision) != (
                enc.model_id, enc.revision):  # fmt: skip
            log.warning("serving patent corpus %r was embedded with %s@%s, but the loaded encoder is %s@%s: the "
                        "neighbour vote is disabled until they match", corpus.corpus_id, corpus.encoder_model_id,
                        corpus.encoder_model_revision, enc.model_id, enc.revision)  # fmt: skip

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
