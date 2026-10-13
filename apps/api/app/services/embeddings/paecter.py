"""PaECTER patent encoder (ADR-0005): one pinned model for corpus documents and user queries.

``mpi-inno-comp/paecter`` is a BERT-large SentenceTransformer: mean pooling, 1024 dimensions, inputs
truncated at 512 tokens. The input text is "title. abstract" (the paper's input; the abstract alone
when there is no title), exactly ``knn_common.query_text`` in the offline experiment, so stored
vectors, imported offline vectors and query vectors are comparable. fp16 on CUDA, vectors returned
as L2-normalised float32.

Kept apart from ``providers.py`` on purpose: the main pipeline keys its single provider on
``embedding_provider`` and must keep working unchanged.

The model loads from a local directory (the Hugging Face downloader stalled on this network; a copy
fetched at the pinned commit lives at ``D:/ipc_knn/models/paecter``) or from the hub by id + revision.
A plain local directory does not record its revision, so the caller states it and the weights' SHA-256 is
checked against ``KNOWN_WEIGHTS_SHA256[revision]``. A revision with no known hash is refused unless the
caller passes ``verify_weights=False`` explicitly (logged as a warning). A Hugging Face cache snapshot path
(``.../snapshots/<commit>``) must name the stated revision.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

PAECTER_MODEL_ID = "mpi-inno-comp/paecter"
PAECTER_REVISION = "1f355d9e9a34dc4859e22d05ce27e970eeacfbd2"
DIMENSION = 1024
MAX_TOKENS = 512
TEXT_TEMPLATE = "title. abstract"  # the name the offline run recorded in meta.json "text"
DEFAULT_BATCH_SIZE = 32
# LFS SHA-256 of model.safetensors per revision (Hugging Face API paths-info, checked 2026-10-13).
KNOWN_WEIGHTS_SHA256 = {
    PAECTER_REVISION: "b29a686d0af879defdb1ad50f17ed506bc3d95cde12ec52122681ae15d0058bd",
}
NORM_TOLERANCE = 1e-3


class PaecterError(RuntimeError):
    pass


class EncoderMismatch(PaecterError):
    """The encoder's model id or revision differs from the one a corpus was embedded with."""


def patent_text(title: str | None, abstract: str) -> str:
    """The text PaECTER embeds: "title. abstract", or the abstract alone without a title."""
    t = (title or "").strip()
    a = (abstract or "").strip()
    return f"{t}. {a}" if t else a


@dataclass(frozen=True)
class PatentEncodingPolicy:
    """Everything that decides what a vector means. Two vectors are comparable iff their policies match."""

    model_id: str
    model_revision: str
    dimension: int = DIMENSION
    max_tokens: int = MAX_TOKENS
    pooling: str = "mean"
    normalized: bool = True
    text_template: str = TEXT_TEMPLATE
    dtype: str = "float16"  # inference precision; stored as halfvec either way

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def identity(self) -> dict[str, Any]:
        """The fields that change a vector's meaning (dtype is precision, not meaning)."""
        return {k: v for k, v in self.as_dict().items() if k != "dtype"}

    def identity_json(self) -> str:
        return json.dumps(self.identity(), sort_keys=True, separators=(",", ":"))


def require_same_encoder(model_id: str, revision: str, *, corpus_id: str, corpus_model_id: str,
                         corpus_revision: str) -> None:  # fmt: skip
    if (model_id, revision) != (corpus_model_id, corpus_revision):
        raise EncoderMismatch(
            f"corpus {corpus_id!r} was embedded with {corpus_model_id}@{corpus_revision}, "
            f"but the encoder is {model_id}@{revision}; its vectors are not comparable"
        )


def sha256_file(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def check_local_copy(path: Path, revision: str, *, verify_weights: bool = True) -> None:
    """Refuse a local model directory that is not the stated revision, as far as can be checked."""
    if not (path / "modules.json").is_file():
        raise PaecterError(f"{path} is not a SentenceTransformer directory (no modules.json)")
    if path.parent.name == "snapshots" and path.name != revision:
        raise EncoderMismatch(f"{path} is the Hugging Face snapshot of revision {path.name}, not {revision}")
    if not verify_weights:
        log.warning("PaECTER local copy %s: weights NOT verified for revision %s (verify_weights=False)", path,
                    revision)  # fmt: skip
        return
    want = KNOWN_WEIGHTS_SHA256.get(revision)
    if want is None:
        raise EncoderMismatch(
            f"no known weights SHA-256 for revision {revision}, so {path} cannot be checked against it; add the "
            "hash to KNOWN_WEIGHTS_SHA256 or pass verify_weights=False explicitly"
        )
    weights = path / "model.safetensors"
    if not weights.is_file():
        raise PaecterError(f"{weights} not found")
    got = sha256_file(weights)
    if got != want:
        raise EncoderMismatch(f"{weights} has SHA-256 {got}, expected {want} for revision {revision}")


def _is_oom(exc: BaseException) -> bool:
    # torch.cuda.OutOfMemoryError subclasses RuntimeError and says "CUDA out of memory"; a host-side
    # allocation failure raises MemoryError. Neither needs torch imported to recognise.
    return isinstance(exc, MemoryError) or (isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower())


def _release_cuda_cache() -> None:
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_device(pref: str) -> str:
    if pref != "auto":
        return pref
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class PaecterEncoder:
    """Pinned PaECTER. ``encode`` is thread-safe (one lock around the model).

    Load with ``PaecterEncoder.load(...)``. The constructor takes an already loaded SentenceTransformer-like
    ``model`` (anything with ``encode(texts, batch_size=, normalize_embeddings=, convert_to_numpy=,
    show_progress_bar=)``), which is how tests inject a fake.
    """

    def __init__(
        self,
        model: Any,
        *,
        model_id: str = PAECTER_MODEL_ID,
        revision: str = PAECTER_REVISION,
        source: str = "injected",
        device: str = "cpu",
        dtype: str = "float32",
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.model = model
        self.model_id = model_id
        self.revision = revision
        self.source = source
        self.device = device
        self.batch_size = batch_size
        self.dimension = DIMENSION
        self.policy = PatentEncodingPolicy(model_id=model_id, model_revision=revision, dtype=dtype)
        self._lock = threading.Lock()

    @classmethod
    def load(
        cls,
        *,
        local_path: str | Path | None = None,
        model_id: str = PAECTER_MODEL_ID,
        revision: str = PAECTER_REVISION,
        device: str = "auto",
        batch_size: int = DEFAULT_BATCH_SIZE,
        cache_dir: str | Path | None = None,
        verify_weights: bool = True,
    ) -> PaecterEncoder:
        """Load from ``local_path`` (a copy of ``model_id`` at ``revision``) or from the hub at ``revision``."""
        import torch
        from sentence_transformers import SentenceTransformer

        if not revision:
            raise PaecterError("a pinned revision is required")
        dev = _resolve_device(device)
        dtype = torch.float16 if dev == "cuda" else torch.float32
        kwargs: dict[str, Any] = {"device": dev, "model_kwargs": {"torch_dtype": dtype}}
        if local_path is not None:
            path = Path(local_path)
            check_local_copy(path, revision, verify_weights=verify_weights)
            model = SentenceTransformer(str(path), **kwargs)
            source = f"local:{path.as_posix()}"
        else:
            model = SentenceTransformer(model_id, revision=revision,
                                        cache_folder=str(cache_dir) if cache_dir else None, **kwargs)  # fmt: skip
            source = f"hub:{model_id}@{revision}"
        model.max_seq_length = MAX_TOKENS
        model.eval()
        # sentence-transformers 6 renamed the method (the old name warns); older versions only have the old one.
        get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        dim = get_dim()
        if dim != DIMENSION:
            raise PaecterError(f"{source} reports dimension {dim}, expected {DIMENSION}")
        return cls(model, model_id=model_id, revision=revision, source=source, device=dev,
                   dtype=str(dtype).replace("torch.", ""), batch_size=batch_size)  # fmt: skip

    def require_compatible(self, corpus: Any) -> None:
        """Refuse to serve or fill a corpus embedded with another model or revision.

        ``corpus`` is a ``db.patents.Corpus`` (or anything with corpus_id / encoder_model_id /
        encoder_model_revision).
        """
        require_same_encoder(self.model_id, self.revision, corpus_id=corpus.corpus_id,
                             corpus_model_id=corpus.encoder_model_id, corpus_revision=corpus.encoder_model_revision)  # fmt: skip

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        """One forward pass: the chunk is the batch."""
        kwargs = {"batch_size": len(texts), "normalize_embeddings": True, "convert_to_numpy": True,
                  "show_progress_bar": False}  # fmt: skip
        torch = sys.modules.get("torch")
        if torch is not None:
            with torch.inference_mode():
                return self.model.encode(texts, **kwargs)
        return self.model.encode(texts, **kwargs)

    def encode(self, texts: Sequence[str], *, on_batch: Callable[[int], None] | None = None) -> np.ndarray:
        """L2-normalised float32 vectors, shape (len(texts), 1024), in input order.

        Texts are encoded longest first (less padding, and an out-of-memory error shows up on the first
        batch); the model truncates each at 512 tokens. On out-of-memory the failed batch is retried at
        half the batch size, down to 1, and the smaller size is kept for the rest of the call.
        """
        texts = list(texts)
        if not texts:
            return np.zeros((0, DIMENSION), dtype=np.float32)
        for i, t in enumerate(texts):
            if not isinstance(t, str) or not t.strip():
                raise ValueError(f"text {i} is empty")
        order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))  # stable: equal lengths keep input order
        out = np.empty((len(texts), DIMENSION), dtype=np.float32)
        with self._lock:
            batch = self.batch_size
            start = 0
            while start < len(order):
                idx = order[start:start + batch]
                try:
                    vecs = self._encode_batch([texts[i] for i in idx])
                except BaseException as exc:
                    if not _is_oom(exc) or batch == 1:
                        raise
                    _release_cuda_cache()
                    batch = max(1, batch // 2)
                    log.warning("PaECTER out of memory; retrying with batch size %d", batch)
                    continue
                out[idx] = _normalised(vecs, len(idx))
                start += len(idx)
                if on_batch is not None:
                    on_batch(len(idx))
            self.batch_size = batch
        return out


def _normalised(vectors: Any, n: int) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.shape != (n, DIMENSION):
        raise PaecterError(f"expected ({n}, {DIMENSION}) vectors, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise PaecterError("non-finite values in PaECTER output")
    norms = np.linalg.norm(arr, axis=1)
    if (norms == 0).any():
        raise PaecterError("zero vector in PaECTER output")
    return arr / norms[:, None]  # fp16 inference leaves norms ~1e-3 off; renormalise in float32


def check_normalised(vectors: np.ndarray, *, tolerance: float = NORM_TOLERANCE) -> None:
    norms = np.linalg.norm(np.asarray(vectors, dtype=np.float32), axis=1)
    worst = float(np.max(np.abs(norms - 1.0))) if len(norms) else 0.0
    if not math.isfinite(worst) or worst > tolerance:
        raise PaecterError(f"vectors are not L2-normalised (max |norm - 1| = {worst:.2e})")
