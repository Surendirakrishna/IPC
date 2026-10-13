"""Process-wide runtime: loads models once (in a background thread at startup) and owns the
bounded inference queue. One model-owning API process; scale later via a dedicated inference service.

With ``NEIGHBOUR_VOTE_ENABLED=true`` the PaECTER patent encoder loads too, once, but only **after** the pipeline
is built and the runtime is ``ready``, in its own background thread. Until it is attached to the pipeline, and
for good if it fails, the vote reports ``patent_encoder_unavailable`` and the definition fallback serves; a slow
or stalled model download can never hold the API in ``loading``. ``/ready`` shows ``patent_encoder_state`` and
the error. The serving patent corpus is not bound here: the pipeline's store reads it per request (activation
needs no restart), and the pipeline itself refuses a corpus embedded with another encoder revision, with a
warning.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import Settings, get_settings
from app.services.embeddings.paecter import PaecterError

log = logging.getLogger(__name__)


class PatentEncoderUnavailable(PaecterError):
    """The PaECTER weights could not be found or fetched in time (no local copy, not cached, download too slow)."""


def patent_encoder_local_path(settings: Settings) -> Path | None:
    """``PATENT_ENCODER_LOCAL_PATH`` (a ``Settings`` field, so ``.env`` works too)."""
    return settings.patent_encoder_local_path


def _cached_snapshot(settings: Settings) -> Path | None:
    """The pinned PaECTER snapshot in the Hugging Face cache, without any network access, or None."""
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(settings.patent_encoder_model_id, revision=settings.patent_encoder_model_revision,
                                      cache_dir=str(settings.model_cache_dir) if settings.model_cache_dir else None,
                                      local_files_only=True))  # fmt: skip
    except Exception:  # noqa: BLE001 - LocalEntryNotFoundError and friends: not cached
        return None


def _download_snapshot(settings: Settings, timeout_s: float) -> Path:
    """Download the pinned snapshot, waiting at most ``timeout_s``. The download runs in a daemon thread, which is
    abandoned (not killed) on timeout; the caller gets ``PatentEncoderUnavailable`` at once."""
    from huggingface_hub import snapshot_download

    result: dict[str, Any] = {}

    def fetch() -> None:
        try:
            result["path"] = snapshot_download(
                settings.patent_encoder_model_id, revision=settings.patent_encoder_model_revision,
                cache_dir=str(settings.model_cache_dir) if settings.model_cache_dir else None)  # fmt: skip
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller's thread
            result["error"] = exc

    worker = threading.Thread(target=fetch, name="paecter-download", daemon=True)
    worker.start()
    worker.join(timeout_s)
    if worker.is_alive():
        raise PatentEncoderUnavailable(
            f"downloading {settings.patent_encoder_model_id}@{settings.patent_encoder_model_revision} from the Hugging "
            f"Face hub did not finish within {timeout_s:.0f} s; set PATENT_ENCODER_LOCAL_PATH to a local copy "
            "(e.g. D:/ipc_knn/models/paecter) or raise PATENT_ENCODER_DOWNLOAD_TIMEOUT_S")  # fmt: skip
    if "error" in result:
        exc = result["error"]
        raise PatentEncoderUnavailable(f"downloading PaECTER failed ({type(exc).__name__}: {exc}); set "
                                       "PATENT_ENCODER_LOCAL_PATH to a local copy") from exc  # fmt: skip
    return Path(result["path"])


def resolve_patent_encoder_path(settings: Settings, local_path: Path | None = None) -> Path:
    """Where the PaECTER weights come from: ``local_path``, else ``PATENT_ENCODER_LOCAL_PATH``, else the pinned
    snapshot in the HF cache, else a download bounded by ``PATENT_ENCODER_DOWNLOAD_TIMEOUT_S``. Only the last one
    touches the network. The model is then always loaded from a directory, so ``PaecterEncoder.load`` never
    reaches the hub itself."""
    path = local_path or patent_encoder_local_path(settings)
    if path is not None:
        return Path(path)
    cached = _cached_snapshot(settings)
    if cached is not None:
        return cached
    return _download_snapshot(settings, settings.patent_encoder_download_timeout_s)


def load_patent_encoder(settings: Settings, local_path: Path | None = None) -> Any:
    """PaECTER at the configured model id and revision, on ``EMBEDDING_DEVICE``. The directory is checked against
    the revision (``paecter.check_local_copy``: snapshot name, weights SHA-256)."""
    from app.services.embeddings.paecter import PaecterEncoder

    return PaecterEncoder.load(
        local_path=resolve_patent_encoder_path(settings, local_path),
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
    patent_encoder: Any = None  # PaecterEncoder once loaded (vote enabled only)
    patent_encoder_state: str = "disabled"  # disabled | not_started | loading | ready | failed
    patent_encoder_error: str | None = None  # why it did not load
    pipeline: Any = None
    confidence: Any = None
    loaded_at: float | None = None
    fingerprint: dict[str, Any] = field(default_factory=dict)
    fingerprint_hash: str | None = None
    _semaphore: asyncio.Semaphore | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _encoder_thread: threading.Thread | None = None

    def __post_init__(self) -> None:
        if self.neighbour_vote_enabled:
            self.patent_encoder_state = "not_started"

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
            self.refresh_confidence()
            client = llm_mod.OpenAICompatibleClient(s) if s.llm_enabled else None
            pipeline_settings = s.model_copy(update={"neighbour_vote_enabled": False}) if self.definition_only else s
            # No patent encoder yet: the vote answers patent_encoder_unavailable until _load_patent_encoder attaches it.
            self.pipeline = ClassificationPipeline(pipeline_settings, provider, reranker, self.confidence, mapping,
                                                   client)  # fmt: skip
            self.state, self.loaded_at = "ready", time.time()
            log.info("models ready")
        except Exception as exc:  # noqa: BLE001
            log.exception("model loading failed")
            self.state, self.error = "failed", f"{type(exc).__name__}: {exc}"[:500]
            return
        if self.neighbour_vote_enabled:
            self.patent_encoder_state = "loading"
            self._encoder_thread = threading.Thread(target=self._load_patent_encoder, name="patent-encoder-loader",
                                                    daemon=True)  # fmt: skip
            self._encoder_thread.start()

    def wait_for_patent_encoder(self, timeout: float | None = None) -> bool:
        """Block until the background encoder load has finished (tests, CLI); True if it finished."""
        t = self._encoder_thread
        if t is None:
            return True
        t.join(timeout)
        return not t.is_alive()

    def _load_patent_encoder(self) -> None:
        """Load PaECTER once, then attach it to the pipeline. A failure is recorded, not raised.

        Attaching is a single reference assignment of a fully built encoder (atomic in CPython), and the pipeline
        reads ``self.patent_encoder`` once per request (``ClassificationPipeline._attempt_vote``), so a request
        sees either no encoder or the complete one, never a half-initialised object."""
        try:
            enc = load_patent_encoder(self.settings)
        except Exception as exc:  # noqa: BLE001 - CUDA, out of memory, missing/mismatched weights, slow download
            log.exception("patent encoder (PaECTER) failed to load; the neighbour vote is unavailable")
            self.patent_encoder_error = f"{type(exc).__name__}: {exc}"[:500]
            self.patent_encoder_state = "failed"
            return
        self.patent_encoder, self.patent_encoder_error = enc, None
        if self.pipeline is not None:
            self.pipeline.patent_encoder = enc
        self.patent_encoder_state = "ready"
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
