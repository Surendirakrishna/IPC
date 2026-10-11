"""Embedding providers behind one interface.

- BgeM3Provider: BAAI/bge-m3 via SentenceTransformer, pinned revision, normalized dense vectors,
  1024 dims, no query instruction prefix. Primary served index (table ipc_embeddings).
- GemmaApiProvider: self-hosted EmbeddingGemma over an OpenAI-compatible endpoint, 768 dims,
  stored separately (table ipc_embeddings_gemma) for comparison. Never mixed with BGE-M3.

Every vector is validated: correct dimension, all finite, nonzero norm.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from app.config import Settings

log = logging.getLogger(__name__)


class EmbeddingValidationError(ValueError):
    pass


@dataclass(frozen=True)
class EncodingPolicy:
    provider: str
    model_id: str
    model_revision: str
    dimension: int
    storage_table: str
    normalized: bool
    document_template: str
    query_template: str
    taxonomy_max_tokens: int
    query_chunk_tokens: int
    dtype: str
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class EmbeddingProvider(Protocol):
    policy: EncodingPolicy

    def encode_documents(self, texts: list[str]) -> np.ndarray: ...

    def encode_queries(self, texts: list[str], max_tokens: int | None = None) -> np.ndarray: ...


def validate_vectors(vectors: np.ndarray, dim: int, *, renormalize: bool = False) -> np.ndarray:
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != dim:
        raise EmbeddingValidationError(f"expected (n, {dim}) vectors, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise EmbeddingValidationError("non-finite values in embedding batch")
    norms = np.linalg.norm(arr, axis=1)
    if (norms == 0).any() or not np.isfinite(norms).all():
        raise EmbeddingValidationError("zero-norm embedding in batch")
    if renormalize:
        arr = arr / norms[:, None]
    return arr


def resolve_device(pref: str) -> str:
    if pref != "auto":
        return pref
    try:
        import torch
    except ImportError:  # pragma: no cover
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.version.cuda is not None:
        # A CUDA build that cannot initialise (e.g. Windows "paging file is too small", WinError 1455)
        # silently degrades to fp32 CPU inference, which needs several GB more RAM. Make it visible.
        log.warning(
            "EMBEDDING_DEVICE=auto: CUDA build of torch %s but CUDA is unavailable; falling back to CPU. "
            "Set EMBEDDING_DEVICE=cuda to fail instead.", torch.__version__,
        )
    return "cpu"


class BgeM3Provider:
    """BAAI/bge-m3 dense embeddings. Loaded once; thread-safe encode via a lock."""

    DIMENSION = 1024
    TABLE = "ipc_embeddings"

    def __init__(self, settings: Settings) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self.settings = settings
        self.device = resolve_device(settings.embedding_device)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = SentenceTransformer(
            settings.embedding_model_id,
            revision=settings.embedding_model_revision,
            device=self.device,
            cache_folder=str(settings.model_cache_dir) if settings.model_cache_dir else None,
            model_kwargs={"torch_dtype": dtype},
        )
        self.model.eval()
        self.tokenizer = self.model.tokenizer
        model_max = int(getattr(self.tokenizer, "model_max_length", 8192) or 8192)
        self.model_max_tokens = min(model_max, 8192)
        dim = self.model.get_sentence_embedding_dimension()
        if dim != self.DIMENSION:
            raise EmbeddingValidationError(f"bge-m3 reported dimension {dim}, expected {self.DIMENSION}")
        self._lock = threading.Lock()
        self.batch_size = settings.embedding_batch_size
        self.policy = EncodingPolicy(
            provider="bge_m3",
            model_id=settings.embedding_model_id,
            model_revision=settings.embedding_model_revision,
            dimension=self.DIMENSION,
            storage_table=self.TABLE,
            normalized=True,
            document_template="{semantic_text}",
            query_template="{text}",
            taxonomy_max_tokens=settings.taxonomy_max_tokens,
            query_chunk_tokens=min(settings.query_chunk_tokens, self.model_max_tokens - 2),
            dtype=str(dtype).replace("torch.", ""),
            extra={"device": self.device, "model_max_tokens": self.model_max_tokens,
                   "library": "sentence-transformers"},
        )

    def token_count(self, text: str) -> int:
        """Token count including special tokens."""
        return len(self.tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])

    def _encode(self, texts: list[str], max_tokens: int) -> np.ndarray:
        import torch

        for t in texts:
            n = self.token_count(t)
            if n > max_tokens:
                raise EmbeddingValidationError(
                    f"input of {n} tokens exceeds configured limit {max_tokens}; chunk it first"
                )
        batch = self.batch_size
        attempts = 0
        while True:
            try:
                with self._lock, torch.inference_mode():
                    self.model.max_seq_length = max_tokens
                    out = self.model.encode(
                        texts,
                        batch_size=batch,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                    )
                return validate_vectors(out, self.DIMENSION)
            except torch.cuda.OutOfMemoryError:
                attempts += 1
                torch.cuda.empty_cache()
                if batch == 1 or attempts > 4:
                    raise
                batch = max(1, batch // 2)
                log.warning("CUDA OOM; retrying with batch_size=%d", batch)
                self.batch_size = batch

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, self.settings.taxonomy_max_tokens)

    def encode_queries(self, texts: list[str], max_tokens: int | None = None) -> np.ndarray:
        limit = min(max_tokens or self.policy.query_chunk_tokens, self.model_max_tokens)
        return self._encode(texts, limit)

    def token_offsets(self, text: str) -> list[tuple[int, int]]:
        """Token (start, end) code-point offsets from the fast tokenizer, special tokens excluded."""
        enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
        return [(a, b) for a, b in enc["offset_mapping"] if b > a]


class GemmaApiProvider:
    """EmbeddingGemma over an OpenAI-compatible /v1/embeddings endpoint.

    Uses the EmbeddingGemma model-card prompt templates for retrieval
    (query: "task: search result | query: ...", document: "title: none | text: ...").
    Vectors are L2-normalized client-side and validated.
    """

    DIMENSION = 768
    TABLE = "ipc_embeddings_gemma"
    DOC_TEMPLATE = "title: none | text: {text}"
    QUERY_TEMPLATE = "task: search result | query: {text}"

    def __init__(self, settings: Settings) -> None:
        import httpx

        if not settings.gemma_embedding_api_base or settings.gemma_embedding_api_key is None:
            raise RuntimeError("GEMMA_EMBEDDING_API_BASE / GEMMA_EMBEDDING_API_KEY are not configured")
        self.settings = settings
        self.url = settings.gemma_embedding_api_base
        self.client = httpx.Client(
            timeout=settings.gemma_embedding_timeout_s,
            headers={"Authorization": f"Bearer {settings.gemma_embedding_api_key.get_secret_value()}"},
        )
        self.batch_size = settings.embedding_batch_size
        self.policy = EncodingPolicy(
            provider="gemma_api",
            model_id=settings.gemma_embedding_model,
            model_revision=self._served_revision(),
            dimension=self.DIMENSION,
            storage_table=self.TABLE,
            normalized=True,
            document_template=self.DOC_TEMPLATE,
            query_template=self.QUERY_TEMPLATE,
            taxonomy_max_tokens=settings.taxonomy_max_tokens,
            query_chunk_tokens=min(settings.query_chunk_tokens, 2000),
            dtype="float32",
            extra={"endpoint_host": self.url.split("/")[2], "normalization": "client_l2"},
        )

    def _served_revision(self) -> str:
        """The endpoint does not expose a weight hash; record served model id + created stamp."""
        base = self.url.rsplit("/embeddings", 1)[0]
        try:
            data = self.client.get(f"{base}/models").json()["data"]
            m = next(d for d in data if d["id"] == self.settings.gemma_embedding_model)
            return f"{m.get('root', m['id'])}@created:{m.get('created')}"
        except Exception:  # noqa: BLE001
            return "unknown"

    def _post(self, inputs: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        for start in range(0, len(inputs), self.batch_size):
            chunk = inputs[start : start + self.batch_size]
            for attempt in range(3):
                try:
                    resp = self.client.post(
                        self.url, json={"model": self.settings.gemma_embedding_model, "input": chunk}
                    )
                    resp.raise_for_status()
                    data = sorted(resp.json()["data"], key=lambda d: d["index"])
                    out.extend(d["embedding"] for d in data)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    import time

                    time.sleep(2**attempt)
        return validate_vectors(np.asarray(out, dtype=np.float32), self.DIMENSION, renormalize=True)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._post([self.DOC_TEMPLATE.format(text=t) for t in texts])

    def encode_queries(self, texts: list[str], max_tokens: int | None = None) -> np.ndarray:
        return self._post([self.QUERY_TEMPLATE.format(text=t) for t in texts])


_providers: dict[str, EmbeddingProvider] = {}
_lock = threading.Lock()


def get_provider(settings: Settings, name: str | None = None) -> EmbeddingProvider:
    """Load a provider once per process."""
    key = name or settings.embedding_provider
    with _lock:
        if key not in _providers:
            if key == "bge_m3":
                _providers[key] = BgeM3Provider(settings)
            elif key == "gemma_api":
                _providers[key] = GemmaApiProvider(settings)
            else:
                raise ValueError(f"unknown embedding provider {key!r}")
        return _providers[key]


def cosine_from_distance(distance: float | None) -> float | None:
    if distance is None or math.isnan(distance):
        return None
    return 1.0 - float(distance)
