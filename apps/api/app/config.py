"""Typed environment configuration with secret redaction.

Values come from the process environment and the repository-root .env file.
Secrets are typed as SecretStr so they never appear in reprs, logs or errors.
"""

from __future__ import annotations

import logging
import urllib.parse
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

API_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = API_DIR.parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", API_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Database -------------------------------------------------------
    local_database_url: SecretStr | None = Field(default=None, description="Local docker pgvector")
    # sakthi: self-hosted ShaktiDB/PostgreSQL 18 on the LAN (primary serving database since 2026-10-10).
    sakthi_db_host: str | None = None
    sakthi_db_port: int = 5432
    sakthi_db_name: str = "ipc_db"
    sakthi_db_user: str | None = None
    sakthi_db_password: SecretStr | None = None
    # The sakthi server has ssl=off; TLS cannot be negotiated. "disable" is an explicit operator choice
    # for this LAN host only (remote hosts default to sslmode=require).
    sakthi_db_sslmode: Literal["disable", "prefer", "require", "verify-full"] = "disable"
    ipc_db_target: Literal["sakthi", "local"] = "sakthi"
    db_connect_timeout_s: int = 10
    db_pool_size: int = 5
    db_pool_max_overflow: int = 2
    db_statement_timeout_ms: int = 30_000

    # --- Source data ----------------------------------------------------
    source_dir: Path = REPO_ROOT / "data" / "source"
    source_index_file: str = "ipc_classification_index.jsonl"
    source_tree_file: str = "ipc_schema_tree.json"
    source_duplicates_file: str = "ipc_maingroup_duplicates.json"
    source_wipo_scheme_file: str = "wipo_ipc_scheme_20260101.zip"
    source_wipo_definitions_file: str = "wipo_ipc_definitions_20260101.zip"
    source_wipo_catchwords_file: str = "wipo_ipc_catchwordindex_20260101.zip"
    manifests_dir: Path = REPO_ROOT / "data" / "manifests"
    import_batch_size: int = 2_000

    # --- Embeddings -----------------------------------------------------
    embedding_provider: Literal["bge_m3", "gemma_api"] = "bge_m3"
    embedding_model_id: str = "BAAI/bge-m3"
    embedding_model_revision: str = "5617a9f61b028005a4858fdac845db406aefb181"
    embedding_device: Literal["auto", "cpu", "cuda"] = "auto"
    embedding_batch_size: int = 32
    taxonomy_max_tokens: int = 512
    abstract_max_tokens: int = 2048
    query_chunk_tokens: int = 1024
    query_chunk_overlap_tokens: int = 128
    model_cache_dir: Path | None = None

    gemma_embedding_api_base: str | None = None
    gemma_embedding_api_key: SecretStr | None = None
    gemma_embedding_model: str = "embeddinggemma-2"
    gemma_embedding_timeout_s: float = 60.0

    # --- Reranker -------------------------------------------------------
    reranker_model_id: str = "BAAI/bge-reranker-v2-m3"
    reranker_model_revision: str = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    reranker_max_length: int = 1024
    reranker_batch_size: int = 16

    # --- Retrieval ------------------------------------------------------
    vector_pool: int = 100
    keyword_pool: int = 100
    rerank_pool: int = 100
    rrf_k: int = 60
    # Benchmarked on this catalogue (docs/retrieval-evaluation.md): m=32/efc=200/ef=200 gave recall@100 0.62.
    hnsw_ef_search: int = 500
    hnsw_m: int = 64
    hnsw_ef_construction: int = 800
    # hnsw (default) | exact: sequential exact cosine search (full recall, ~155 ms/query locally).
    vector_search_mode: Literal["hnsw", "exact"] = "hnsw"
    hnsw_ef_search_max: int = 1000
    keyword_max_terms: int = 24
    expansion_top_n: int = 20
    expansion_max_siblings: int = 4
    max_query_views: int = 24
    per_source_candidate_cap: int = 100
    subclass_pool: int = 50
    # Group-level requests route first: subclasses are voted from a ROUTING_GROUP_POOL-deep group search,
    # and the top HIERARCHICAL_SUBCLASS_LIMIT are searched again in depth. Chosen on the IP India dev set.
    routing_group_pool: int = 300
    hierarchical_subclass_limit: int = 30
    # Claims are evidence for review by default. "independent_weighted" lets claim views vote in
    # candidate retrieval: each claim list counts with its claim weight inside the claims source, and the
    # claims source counts claim_source_weight against the abstract (1.0). A weight of 0 drops those claims.
    claims_retrieval_mode: Literal["off", "independent_weighted"] = "off"
    independent_claim_weight: float = 1.0
    dependent_claim_weight: float = 0.0
    claim_source_weight: float = 0.35
    # Version-matched WIPO notes/definitions in the candidate scope text given to the reranker. Off: on the
    # IP India dev set they pushed 81% of scopes past the reranker's query cap and lowered subclass top-1
    # from 42.3% to 31.5% (median 10 s -> 24 s). They still feed the rules and the review output.
    rerank_official_context: bool = False

    # --- Classification policy (pre-calibration relevance policy) -------
    max_results_default: int = 5
    max_results_limit: int = 20
    # Pre-calibration selection policy in raw reranker-logit units (NOT probabilities). Values chosen on the
    # IP India Sep-2026 dev set (docs/benchmark-ipindia.md); None disables an absolute floor.
    min_reranker_score: float | None = None
    max_reranker_gap: float = 0.5
    min_evidence_score: float | None = -5.0
    enforce_lexical_checks: bool = False
    # WIPO priority rules between candidates that both passed the evidence policy. A rule never adds a
    # code: the preferred place must itself be supported. Conditional rules only produce review notes.
    # Place (first/last) rules decide between places for the *same* subject; whether two candidates are
    # the same subject is not checked, so by default they only add review notes (dev: they demoted a
    # recorded code and misread an "also classify" note).
    apply_precedence_rules: bool = True
    apply_place_rules: bool = False

    # --- Request limits -------------------------------------------------
    max_abstract_chars: int = 10_000
    max_title_chars: int = 1_000
    max_specification_chars: int = 400_000
    max_claims_chars: int = 200_000
    max_request_bytes: int = 1_500_000
    classify_timeout_s: float = 120.0
    inference_queue_size: int = 4

    # --- Optional LLM ---------------------------------------------------
    llm_enabled: bool = False
    llm_api_base: str | None = None
    llm_api_key: SecretStr | None = None
    llm_model: str | None = None
    llm_timeout_s: float = 90.0
    llm_max_candidates: int = 20

    # --- Confidence / fields --------------------------------------------
    calibration_dir: Path = REPO_ROOT / "data" / "calibration"
    field_mapping_path: Path = API_DIR / "app" / "services" / "fields" / "field_mapping.json"

    # --- Retention / admin ----------------------------------------------
    store_abstracts: bool = False
    store_input_documents: bool = False
    admin_token: SecretStr | None = None
    cors_origins: list[str] = ["http://localhost:3000"]

    def sakthi_url(self) -> SecretStr | None:
        if not (self.sakthi_db_host and self.sakthi_db_user and self.sakthi_db_password):
            return None
        pw = urllib.parse.quote(self.sakthi_db_password.get_secret_value(), safe="")
        user = urllib.parse.quote(self.sakthi_db_user, safe="")
        return SecretStr(
            f"postgresql://{user}:{pw}@{self.sakthi_db_host}:{self.sakthi_db_port}/{self.sakthi_db_name}"
            f"?sslmode={self.sakthi_db_sslmode}"
        )

    @field_validator("ipc_db_target", mode="before")
    @classmethod
    def _legacy_target(cls, v: object) -> object:
        # Neon support was removed on 2026-10-10. A leftover IPC_DB_TARGET=neon in .env must not stop startup.
        if isinstance(v, str) and v.strip().lower() == "neon":
            logging.getLogger(__name__).warning(
                "IPC_DB_TARGET=neon is no longer supported (Neon was removed); using sakthi. Update .env."
            )
            return "sakthi"
        return v

    def pooled_url(self) -> SecretStr | None:
        """URL for the API's bounded SQLAlchemy pool (no external pooler)."""
        return self.local_database_url if self.ipc_db_target == "local" else self.sakthi_url()

    def direct_url(self) -> SecretStr | None:
        """URL for migrations, bulk ingestion, embedding and index jobs."""
        return self.pooled_url()

    @property
    def retain_inputs(self) -> bool:
        return self.store_input_documents or self.store_abstracts

    def source_path(
        self,
        kind: Literal["index", "tree", "duplicates", "wipo_scheme", "wipo_definitions", "wipo_catchwords"],
    ) -> Path:
        name = {
            "index": self.source_index_file,
            "tree": self.source_tree_file,
            "duplicates": self.source_duplicates_file,
            "wipo_scheme": self.source_wipo_scheme_file,
            "wipo_definitions": self.source_wipo_definitions_file,
            "wipo_catchwords": self.source_wipo_catchwords_file,
        }[kind]
        return self.source_dir / name


def redact_url(url: str | SecretStr | None) -> str:
    """Return a connection URL with any password replaced by ***."""
    if url is None:
        return "<unset>"
    raw = url.get_secret_value() if isinstance(url, SecretStr) else url
    parts = urllib.parse.urlsplit(raw)
    netloc = parts.netloc
    if "@" in netloc:
        creds, host = netloc.rsplit("@", 1)
        netloc = f"{creds.split(':', 1)[0]}:***@{host}"
    return urllib.parse.urlunsplit((parts.scheme, netloc, parts.path, "", ""))


@lru_cache
def get_settings() -> Settings:
    return Settings()
