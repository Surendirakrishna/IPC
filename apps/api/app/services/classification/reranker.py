"""BAAI/bge-reranker-v2-m3 cross-encoder via FlagEmbedding.FlagReranker.

reranker_score is the raw model relevance logit. A sigmoid of it is still a ranking score,
not a calibrated probability of classification correctness; confidence comes only from a
fitted calibrator (app/services/confidence).
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from app.config import Settings
from app.services.embeddings.providers import resolve_device

log = logging.getLogger(__name__)


@dataclass
class PairScore:
    score: float
    query_truncated: bool
    passage_truncated: bool


class Reranker:
    def __init__(self, settings: Settings) -> None:
        from FlagEmbedding import FlagReranker
        from huggingface_hub import snapshot_download

        self.settings = settings
        self.device = resolve_device(settings.embedding_device)
        # FlagReranker has no revision argument: resolve the pinned snapshot first.
        local_path = snapshot_download(
            settings.reranker_model_id,
            revision=settings.reranker_model_revision,
            cache_dir=str(settings.model_cache_dir) if settings.model_cache_dir else None,
        )
        self.model = FlagReranker(
            local_path,
            use_fp16=self.device == "cuda",
            devices=[f"{self.device}:0"] if self.device == "cuda" else ["cpu"],
            cache_dir=str(settings.model_cache_dir) if settings.model_cache_dir else None,
            batch_size=settings.reranker_batch_size,
            max_length=settings.reranker_max_length,
            normalize=False,
        )
        self.tokenizer = self.model.tokenizer
        self.max_length = settings.reranker_max_length
        self._lock = threading.Lock()
        self.model_id = settings.reranker_model_id
        self.revision = settings.reranker_model_revision

    def _count(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])

    def score(self, pairs: list[tuple[str, str]]) -> list[PairScore]:
        """Score (query, passage) pairs. Truncation (if any) is reported per pair."""
        if not pairs:
            return []
        flags = []
        for q, p in pairs:
            qn, pn = self._count(q), self._count(p)
            over = qn + pn + 4 > self.max_length
            flags.append((over and qn > self.max_length // 2, over))
        with self._lock:
            raw = self.model.compute_score(
                [[q, p] for q, p in pairs],
                batch_size=self.settings.reranker_batch_size,
                max_length=self.max_length,
                normalize=False,
            )
        if isinstance(raw, float):
            raw = [raw]
        return [PairScore(float(s), qt, pt) for s, (qt, pt) in zip(raw, flags, strict=True)]


_instance: Reranker | None = None
_lock = threading.Lock()


def get_reranker(settings: Settings) -> Reranker:
    global _instance
    with _lock:
        if _instance is None:
            _instance = Reranker(settings)
        return _instance
