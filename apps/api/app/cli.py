"""Operator CLI: repeatable import, embedding, indexing, validation, evaluation, calibration.

Run from apps/api:  uv run ipc --help
Select the database with IPC_DB_TARGET=sakthi|local (default sakthi) or --target.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Annotated, Any

import typer

from app.config import get_settings, redact_url

app = typer.Typer(no_args_is_help=True, add_completion=False, help=__doc__)
log = logging.getLogger("ipc")


@app.callback()
def _main(
    target: Annotated[str | None, typer.Option(help="Database target: sakthi or local")] = None,
    verbose: Annotated[bool, typer.Option("-v", "--verbose")] = False,
) -> None:
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(message)s")
    if target:
        if target not in ("sakthi", "local"):
            raise typer.BadParameter("target must be sakthi or local")
        os.environ["IPC_DB_TARGET"] = target
        get_settings.cache_clear()


def _echo(payload: object) -> None:
    typer.echo(json.dumps(payload, indent=2, default=str))


@app.command("db-info")
def db_info() -> None:
    """Show which database is targeted (password redacted) and server/extension versions."""
    from app.db.session import connect_direct

    s = get_settings()
    with connect_direct() as conn:
        version = conn.execute("SHOW server_version").fetchone()[0]
        exts = dict(conn.execute("SELECT extname, extversion FROM pg_extension").fetchall())
        size = conn.execute("SELECT pg_size_pretty(pg_database_size(current_database()))").fetchone()[0]
    _echo({"target": s.ipc_db_target, "url": redact_url(s.direct_url()), "server_version": version,
           "extensions": exts, "database_size": size})  # fmt: skip


@app.command()
def migrate(revision: str = "head") -> None:
    """Apply Alembic migrations through the direct connection."""
    from alembic.config import Config

    from alembic import command

    api_dir = Path(__file__).resolve().parents[1]
    cfg = Config(str(api_dir / "alembic.ini"))
    cfg.set_main_option("script_location", str(api_dir / "alembic"))
    command.upgrade(cfg, revision)
    typer.echo(f"migrated to {revision} on {get_settings().ipc_db_target}")


@app.command("validate-sources")
def validate_sources_cmd() -> None:
    """Hash and validate the source files without touching the database."""
    from app.services.ingestion.importer import load_and_validate, write_manifest

    s = get_settings()
    src = load_and_validate(s)
    out = {
        "taxonomy_version": src.taxonomy_version,
        "source_files": src.files,
        "source_hashes": src.hashes,
        "source_root": src.root_meta,
        "expected_manifest": src.expected_manifest,
        "validation": src.report.as_dict(),
    }
    path = write_manifest(s.manifests_dir, f"validation_{src.taxonomy_version}", out)
    _echo({**out, "manifest": str(path)})
    raise typer.Exit(0 if src.report.ok else 1)


@app.command("import")
def import_cmd(
    activate: Annotated[bool, typer.Option(help="Activate the release if fully validated")] = False,
) -> None:
    """Import (or resume importing) the validated snapshot. Idempotent for unchanged sources."""
    from app.db.session import connect_direct
    from app.services.ingestion.importer import load_and_validate, run_import

    s = get_settings()
    src = load_and_validate(s)
    with connect_direct() as conn:
        manifest = run_import(conn, s, src, activate=activate)
    _echo({k: manifest[k] for k in ("job_id", "taxonomy_version", "release", "rows_changed",
                                    "database_counts", "complete", "activated")})  # fmt: skip


@app.command("activate-release")
def activate_release_cmd(version: str) -> None:
    """Activate a fully validated release (only one release can be active)."""
    from app.db.session import connect_direct
    from app.services.ingestion.importer import activate_release

    with connect_direct() as conn:
        activate_release(conn, version)
    typer.echo(f"activated {version}")


def _active_version(conn) -> str:
    row = conn.execute("SELECT taxonomy_version FROM taxonomy_releases WHERE is_active").fetchone()
    if not row:
        raise typer.BadParameter("no active release; run `ipc import --activate` first")
    return row[0]


@app.command("check-token-lengths")
def check_token_lengths(provider: str = "bge_m3") -> None:
    """Measure actual tokenizer lengths of every semantic_text against TAXONOMY_MAX_TOKENS."""
    from app.db.session import connect_direct
    from app.services.embeddings.providers import get_provider

    s = get_settings()
    p = get_provider(s, provider)
    count = getattr(p, "token_count", None)
    if count is None:
        raise typer.BadParameter("provider has no local tokenizer")
    with connect_direct() as conn:
        version = _active_version(conn)
        rows = conn.execute("SELECT code, semantic_text FROM ipc_records WHERE taxonomy_version = %s", [version]).fetchall()
    lengths = sorted((count(t), c) for c, t in rows)
    n = len(lengths)
    out = {"records": n, "limit": s.taxonomy_max_tokens, "p50": lengths[n // 2][0], "p99": lengths[int(n * 0.99)][0],
           "max": lengths[-1][0], "max_code": lengths[-1][1],
           "over_limit": sum(1 for x, _ in lengths if x > s.taxonomy_max_tokens)}
    write_manifest_safe(f"token_lengths_{version}_{provider}", out)
    _echo(out)
    raise typer.Exit(0 if out["over_limit"] == 0 else 1)


def write_manifest_safe(stem: str, payload: dict) -> None:
    from app.services.ingestion.importer import write_manifest

    write_manifest(get_settings().manifests_dir, stem, payload)


@app.command()
def embed(
    provider: Annotated[str, typer.Option(help="bge_m3 or gemma_api")] = "bge_m3",
    max_records: Annotated[int | None, typer.Option(help="Bound a partial/smoke run")] = None,
    batch_records: int = 512,
) -> None:
    """Embed (or resume embedding) every record's semantic_text for the active release."""
    from app.db.session import connect_direct
    from app.services.embeddings.providers import get_provider
    from app.services.embeddings.runner import run_embeddings

    s = get_settings()
    p = get_provider(s, provider)
    with connect_direct() as conn:
        version = _active_version(conn)
        result = run_embeddings(conn, version, p, batch_records=batch_records, max_records=max_records)
    write_manifest_safe(f"embedding_{version}_{provider}", {**result, "encoding_policy": p.policy.as_dict()})
    _echo(result)


@app.command("build-index")
def build_index(
    table: str = "ipc_embeddings",
    m: Annotated[int | None, typer.Option(help="default HNSW_M (64)")] = None,
    ef_construction: Annotated[int | None, typer.Option(help="default HNSW_EF_CONSTRUCTION (800)")] = None,
    maintenance_work_mem: str | None = None,
) -> None:
    """Build the HNSW cosine index after embeddings are loaded (separate from row import)."""
    from app.db.session import connect_direct
    from app.services.retrieval.index import build_hnsw

    s = get_settings()
    m = m or s.hnsw_m
    ef_construction = ef_construction or s.hnsw_ef_construction
    with connect_direct() as conn:
        _echo(build_hnsw(conn, table, m=m, ef_construction=ef_construction,
                         maintenance_work_mem=maintenance_work_mem))


@app.command("activate-embeddings")
def activate_embeddings(run_id: str) -> None:
    """Serve a validated embedding run (requires its HNSW index)."""
    from app.db.session import connect_direct
    from app.services.embeddings.runner import activate_run

    with connect_direct() as conn:
        activate_run(conn, run_id)
    typer.echo(f"serving {run_id}")


@app.command("explain-vector")
def explain_vector(table: str = "ipc_embeddings", ef_search: int = 200) -> None:
    """Print a representative EXPLAIN ANALYZE for the vector query."""
    from app.db.session import connect_direct
    from app.services.retrieval.index import explain_vector_query

    with connect_direct() as conn:
        typer.echo(explain_vector_query(conn, table, _active_version(conn), 1024 if table == "ipc_embeddings" else 768,
                                        ef_search))


@app.command("estimate-storage")
def estimate_storage(
    measured_json: Annotated[Path | None, typer.Option(help="storage probe results (JSON lines)")] = None,
) -> None:
    """Combine current table sizes with measured vector/HNSW sizes (storage-probe output) into projections."""
    from app.db.session import connect_direct

    with connect_direct() as conn:
        db_bytes = conn.execute("SELECT pg_database_size(current_database())").fetchone()[0]
        emb = conn.execute("SELECT coalesce(sum(pg_total_relation_size(c.oid)),0) FROM pg_class c"
                           " WHERE relname IN ('ipc_embeddings','ipc_embeddings_gemma')").fetchone()[0]
    probes = []
    if measured_json and measured_json.exists():
        probes = [json.loads(line) for line in measured_json.read_text().splitlines() if line.strip()]
    base = db_bytes - emb
    out = {"target": get_settings().ipc_db_target, "database_bytes_now": db_bytes,
           "non_vector_bytes": base,
           "projections": [{"variant": f"{p['kind']}({p['dim']}) m={p['m']}", "vector_total_mib": p["total_mib"],
                            "projected_total_mib": round((base + p["total_bytes"]) / 2**20, 1)} for p in probes]}
    write_manifest_safe(f"storage_estimate_{out['target']}", out)
    _echo(out)


# --- Patent corpus (ADR-0005, TASK-0009) ---------------------------------------------------------------------
# Wiring only: the work is in services.patents (ingest, embed, index) and db.patents. Each lifecycle command holds
# a session-level advisory lock on its corpus for its whole run (activation and retirement also a global one), so
# a second command on the same corpus fails fast instead of interleaving (wave 3 review B5).

PATENT_SOURCES = ("hupd", "uspto-odp")
DEFAULT_PATENT_SEED = 20261013
LOCK_NAMESPACE_CORPUS = 727_001  # pg_try_advisory_lock(727001, hashtext(corpus_id))
LOCK_NAMESPACE_GLOBAL = 727_002  # pg_try_advisory_lock(727002, 0): serving changes
LOAD_STEPS = ("ingest", "embed", "index", "validate")
USPTO_UNVERIFIED_WARNING = (
    "WARNING: the USPTO Open Data Portal reader is UNVERIFIED: its endpoint, query and field names have never been "
    "checked against a live response (no USPTO_ODP_API_KEY yet), and its year paging has an open review finding "
    "(TASK-0004). Inspect the first pages and the ingest manifest before trusting a large load."
)


class PatentLockBusy(RuntimeError):
    """Another ipc command holds the advisory lock for this corpus (or for serving changes)."""


class PatentResumeRefused(RuntimeError):
    """load-patents would build on a corpus loaded with other ingest parameters."""


class PrecomputedTextMismatch(RuntimeError):
    """Too many precomputed vectors were computed from text that differs from the stored documents."""


class IngestConcordanceMismatch(RuntimeError):
    """The concordance lists loaded are not the files the ingest parameters recorded (should not happen)."""


def _patent_errors() -> tuple[type[BaseException], ...]:
    """Expected failures of the patent commands: reported as one clear line and exit code 1. Anything else is a
    bug and keeps its traceback."""
    import psycopg

    from app.db.patents import PatentCorpusError
    from app.db.session import DatabaseNotConfigured
    from app.services.embeddings.paecter import PaecterError
    from app.services.patents.concordance import ConcordanceError
    from app.services.patents.embed import PatentEmbedError
    from app.services.patents.index import PatentIndexError
    from app.services.patents.ingest import IngestError
    from app.services.patents.sources.uspto_odp import UsptoReaderError

    return (PatentLockBusy, PatentResumeRefused, PrecomputedTextMismatch, IngestConcordanceMismatch, PatentCorpusError, DatabaseNotConfigured, PaecterError, ConcordanceError,
            PatentEmbedError, PatentIndexError, IngestError, UsptoReaderError, psycopg.Error, FileNotFoundError,
            ValueError, typer.BadParameter)  # fmt: skip


def _describe(exc: BaseException) -> str:
    import psycopg

    if isinstance(exc, (psycopg.errors.LockNotAvailable, psycopg.errors.QueryCanceled)):
        return (f"{type(exc).__name__}: a lock on patent_embeddings was not granted within the lock timeout (vote "
                f"searches or another job hold it); nothing was dropped. Retry later. ({str(exc).strip()})")
    from app.services.patents.ingest import HoldoutLeak

    if isinstance(exc, typer.BadParameter):
        return str(exc)
    if isinstance(exc, HoldoutLeak):
        return (f"HoldoutLeak: {str(exc).strip()} (the ingest manifest lists them; --allow-holdout-leak stores "
                "anyway, which is only right if the hold-out is not used for evaluation)")
    return f"{type(exc).__name__}: {str(exc).strip()}"


def _fail(message: str) -> None:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(1)


@contextmanager
def _patent_errors_reported() -> Iterator[None]:
    try:
        yield
    except _patent_errors() as exc:
        _fail(_describe(exc))


@contextmanager
def _advisory_lock(conn: Any, namespace: int, corpus_id: str | None, what: str) -> Iterator[None]:
    """Hold a session-level advisory lock for the block; refuse at once if another session holds it.

    Session-level locks survive the rollback that ends the lock statement's transaction, so the connection is idle
    again for the services (index work needs an idle connection) while the lock stays held until unlocked or the
    connection closes."""
    key = "0" if corpus_id is None else "hashtext(%s)"
    params = [namespace] if corpus_id is None else [namespace, corpus_id]
    got = conn.execute(f"SELECT pg_try_advisory_lock(%s, {key})", params).fetchone()[0]  # key is one of two literals
    conn.rollback()
    if not got:
        raise PatentLockBusy(f"another ipc command holds the {what} lock; wait for it to finish (`ipc patent-status` "
                             "shows the corpus state) and re-run. Nothing was changed.")  # fmt: skip
    try:
        yield
    finally:
        try:
            conn.rollback()
            conn.execute(f"SELECT pg_advisory_unlock(%s, {key})", params)
            conn.rollback()
        except Exception:  # noqa: BLE001 - a broken connection releases its locks when it closes
            log.debug("advisory unlock failed; the lock goes with the connection", exc_info=True)


@contextmanager
def _patent_job(corpus_id: str, *, serving_change: bool = False) -> Iterator[Any]:
    """A direct connection holding the corpus lock (and the global serving lock when ``serving_change``)."""
    from app.db.session import connect_direct

    with connect_direct() as conn, ExitStack() as locks:
        if serving_change:
            locks.enter_context(_advisory_lock(conn, LOCK_NAMESPACE_GLOBAL, None, "patent serving (activation)"))
        locks.enter_context(_advisory_lock(conn, LOCK_NAMESPACE_CORPUS, corpus_id, f"patent corpus {corpus_id!r}"))
        yield conn


def _manifest_name(corpus_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", corpus_id)  # ':' is allowed in corpus ids but not in Windows file names


def _write_patent_manifest(step: str, corpus_id: str, payload: dict) -> Path:
    from app.services.ingestion.importer import write_manifest

    return write_manifest(get_settings().manifests_dir / "patents", f"{step}_{_manifest_name(corpus_id)}", payload)


def _read_patent_manifest(step: str, corpus_id: str) -> dict | None:
    path = get_settings().manifests_dir / "patents" / f"{step}_{_manifest_name(corpus_id)}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _parse_years(value: str | None) -> list[int]:
    """"2017", "2021-2025" or "2017,2019"."""
    if not value or not value.strip():
        raise typer.BadParameter("--years is required, e.g. 2017 or 2021-2025")
    years: list[int] = []
    for part in value.split(","):
        part = part.strip()
        m = re.fullmatch(r"(\d{4})(?:-(\d{4}))?", part)
        if not m:
            raise typer.BadParameter(f"--years: {part!r} is not YYYY or YYYY-YYYY")
        lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
        if hi < lo:
            raise typer.BadParameter(f"--years: {part!r} ends before it starts")
        years += [y for y in range(lo, hi + 1) if y not in years]
    return years


def _check_source(source: str) -> None:
    if source not in PATENT_SOURCES:
        raise typer.BadParameter(f"--source must be one of {', '.join(PATENT_SOURCES)}")


class LimitedReader:
    """The first ``limit`` records of a re-iterable reader, itself re-iterable (ingest samples in two passes)."""

    def __init__(self, reader: Any, limit: int) -> None:
        self.reader, self.limit = reader, limit

    @property
    def stats(self) -> Any:
        return getattr(self.reader, "stats", None)

    def __iter__(self) -> Iterator[Any]:
        import itertools

        return itertools.islice(iter(self.reader), self.limit)


def _patent_reader(source: str, years: list[int], limit: int | None, data_dir: Path | None) -> tuple[Any, dict]:
    s = get_settings()
    if source == "hupd":
        from app.services.patents.sources.hupd import HupdReader

        directory = data_dir or s.hupd_data_dir
        reader: Any = HupdReader.for_years(directory, years)
        if limit is not None:
            reader = LimitedReader(reader, limit)
        return reader, {"source": source, "years": years, "limit": limit, "data_dir": str(directory)}
    from app.services.patents.sources.uspto_odp import UsptoOdpReader

    return UsptoOdpReader(years, s.uspto_odp_api_key, limit=limit), {"source": source, "years": years, "limit": limit}


def _catalogue(conn: Any, version: str) -> set[str]:
    """Display codes of the taxonomy release the labels are resolved against."""
    rows = conn.execute("SELECT display_code FROM ipc_records WHERE taxonomy_version = %s", [version]).fetchall()
    conn.rollback()
    return {r[0] for r in rows}


def _load_concordance_if_present(enabled: bool, catalogue: set[str]) -> tuple[Any, str]:
    """(concordance or None, why). Loaded from data/source/ when its WIPO lists are there (TASK-0015)."""
    from app.services.patents.concordance import FILE_GLOB, load_concordance

    if not enabled:
        return None, "disabled (--no-concordance)"
    directory = get_settings().source_dir
    if not any(directory.glob(FILE_GLOB)):
        return None, f"not applied: no {FILE_GLOB} in {directory}"
    conc = load_concordance(directory, catalogue=catalogue)
    return conc, f"applied: {len(getattr(conc, 'sources', []))} WIPO concordance list(s) from {directory}"


def _corpus_row(conn: Any, corpus_id: str) -> Any:
    from app.db import patents

    try:
        return patents.get_corpus(conn, corpus_id)
    finally:
        conn.rollback()


def _concordance_plan(enabled: bool) -> tuple[list[str], str]:
    """(WIPO list file names an ingest would apply, note), from the file names alone: no loading. Same names, in
    the same order, as ``load_concordance(directory).sources``."""
    from app.services.patents.concordance import FILE_GLOB

    if not enabled:
        return [], "disabled (--no-concordance)"
    directory = get_settings().source_dir
    names = [p.name for p in sorted(directory.glob(FILE_GLOB))]
    if not names:
        return [], f"not applied: no {FILE_GLOB} in {directory}"
    return names, f"applied: {len(names)} WIPO concordance list(s) from {directory}"


def _ingest_parameters(source: str, years: list[int], limit: int | None, sample: int | None, seed: int,
                       holdout: int, *, concordance: bool, data_dir: Path | None,
                       label_ipc_edition: str | None) -> dict:  # fmt: skip
    """Everything a completed ingest is compared on before ``load-patents`` resumes past it: the source selection,
    the sampling, the labels' edition and which concordance lists would be applied."""
    sources, _ = _concordance_plan(concordance)
    hupd_dir = str(Path(data_dir or get_settings().hupd_data_dir)) if source == "hupd" else None
    return {"source": source, "years": years, "limit": limit, "sample": sample, "seed": seed, "holdout": holdout,
            "data_dir": hupd_dir, "label_ipc_edition": label_ipc_edition, "concordance_applied": bool(sources),
            "concordance_sources": sources}  # fmt: skip


def _live_document_count(conn: Any, corpus_id: str) -> int:
    n = conn.execute("SELECT count(*) FROM patent_documents WHERE corpus_id = %s", [corpus_id]).fetchone()[0]
    conn.rollback()
    return int(n)


def _parameter_diff(last: dict | None, parameters: dict) -> str:
    if not last or not last.get("completed"):
        return "no completed ingest is recorded for this corpus"
    old = last.get("parameters") or {}
    keys = sorted(k for k in set(old) | set(parameters) if old.get(k) != parameters.get(k))
    return "; ".join(f"{k}: {old.get(k)!r} -> {parameters.get(k)!r}" for k in keys)


def _ingest_step(conn: Any, corpus_id: str, *, source: str, years: list[int], limit: int | None, sample: int | None,
                 seed: int, holdout: int, allow_holdout_leak: bool, concordance: bool, data_dir: Path | None,
                 label_ipc_edition: str | None, description: str | None, batch: int) -> dict:  # fmt: skip
    from app.services.patents.ingest import HoldoutLeak, ingest

    s = get_settings()
    parameters = _ingest_parameters(source, years, limit, sample, seed, holdout, concordance=concordance,
                                    data_dir=data_dir, label_ipc_edition=label_ipc_edition)  # fmt: skip
    # Written before any document: a run that dies mid-ingest leaves "completed": false, never the previous run's
    # completed manifest, so a resume cannot mistake a mixed corpus for a clean one.
    _write_patent_manifest("ingest", corpus_id, {"corpus_id": corpus_id, "completed": False, "parameters": parameters,
                                                 "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})  # fmt: skip
    version = _active_version(conn)
    conn.rollback()
    catalogue = _catalogue(conn, version)
    conc, conc_note = _load_concordance_if_present(concordance, catalogue)
    reader, source_parameters = _patent_reader(source, years, limit, data_dir)
    if list(getattr(conc, "sources", None) or []) != parameters["concordance_sources"]:
        raise IngestConcordanceMismatch(f"loaded concordance lists {getattr(conc, 'sources', None)} differ from the "
                                        f"files found {parameters['concordance_sources']}")  # fmt: skip
    extra = {"concordance_applied": conc is not None, "concordance_note": conc_note, "parameters": parameters}
    try:
        manifest = ingest(
            reader, corpus_id, catalogue, seed=seed, sample_size=sample, holdout_size=holdout, concordance=conc,
            repository=conn, encoder_model_id=s.patent_encoder_model_id,
            encoder_model_revision=s.patent_encoder_model_revision, description=description,
            label_ipc_edition=label_ipc_edition, labels_taxonomy_version=version, source_parameters=source_parameters,
            batch_size=batch, allow_holdout_leak=allow_holdout_leak,
        )  # fmt: skip
    except HoldoutLeak as exc:
        _write_patent_manifest("ingest", corpus_id, {**exc.manifest, **extra, "completed": False, "error": str(exc)})
        raise
    manifest = {**manifest, **extra, "completed": True}
    _write_patent_manifest("ingest", corpus_id, manifest)
    return manifest


def _patent_encoder(model_path: Path | None) -> Any:
    """PaECTER from --model-path, else PATENT_ENCODER_LOCAL_PATH, else the HF cache, else a download bounded by
    PATENT_ENCODER_DOWNLOAD_TIMEOUT_S (``PatentEncoderUnavailable``, a PaecterError: one clear line, exit 1)."""
    from app.runtime import load_patent_encoder

    return load_patent_encoder(get_settings(), model_path)


def _embed_step(conn: Any, corpus_id: str, *, batch: int, from_npy: Path | None, ids: Path | None,
                model_path: Path | None, max_text_mismatch: float = 0.01) -> dict:  # fmt: skip
    """Import precomputed vectors (if given), then embed whatever still lacks a vector with PaECTER, then finalize.
    Both services skip documents that already have a vector, so a re-run resumes.

    If more than ``max_text_mismatch`` of the corpus's documents were skipped by the import because their text
    differs from the text the vectors were computed from, stop (``PrecomputedTextMismatch``) instead of quietly
    re-embedding them on the GPU: usually the id list belongs to another corpus or sample."""
    from app.db.patents import CorpusNotFound
    from app.services.embeddings.paecter import PatentEncodingPolicy
    from app.services.patents.embed import (
        DbEmbeddingStore,
        PrecomputedSource,
        embed_corpus,
        finalize_corpus,
        import_precomputed,
    )

    if (from_npy is None) != (ids is None):
        raise typer.BadParameter("--from-npy and --ids go together (the shards and their id list)")
    t0 = time.perf_counter()
    store = DbEmbeddingStore(conn)
    out: dict[str, Any] = {"corpus_id": corpus_id, "from_npy": str(from_npy) if from_npy else None,
                           "ids": str(ids) if ids else None, "batch": batch, "imported": None,
                           "skipped_text_mismatch": None, "text_checked": None}  # fmt: skip
    if from_npy is not None:
        src = PrecomputedSource.from_directory(from_npy, ids, with_texts=True)
        report = import_precomputed(store, corpus_id, src, batch_size=batch).as_dict()
        out["import"] = report
        out.update(imported=report.get("imported"), skipped_text_mismatch=report.get("skipped_text_mismatch"),
                   text_checked=report.get("text_checked"))  # fmt: skip
        documents, embedded = store.counts(corpus_id)
        mismatched = int(report.get("skipped_text_mismatch") or 0)
        if documents and mismatched > max_text_mismatch * documents:
            out.update(documents=documents, embedded=embedded, state=getattr(store.corpus(corpus_id), "state", None),
                       error="text mismatch above threshold", seconds=round(time.perf_counter() - t0, 1))
            _write_patent_manifest("embed", corpus_id, out)
            raise PrecomputedTextMismatch(
                f"{mismatched} of {documents} documents ({100 * mismatched / documents:.2f}%) were skipped because their "
                f"text differs from the text the precomputed vectors were computed from (threshold "
                f"{100 * max_text_mismatch:g}%). Not re-embedding them on the GPU: check that --ids belongs to this "
                "corpus (same source, seed and sample), or pass --max-text-mismatch 1 to embed them with PaECTER.")
    corpus = store.corpus(corpus_id)
    if corpus is None:
        raise CorpusNotFound(f"corpus {corpus_id!r} not found (run ingest-patents first)")
    documents, embedded = store.counts(corpus_id)
    if documents > embedded:
        out["embed"] = embed_corpus(store, corpus_id, _patent_encoder(model_path), page_size=batch).as_dict()
    elif corpus.state != "embedded":  # nothing left to encode: record the input hash without loading PaECTER
        policy = PatentEncodingPolicy(model_id=corpus.encoder_model_id, model_revision=corpus.encoder_model_revision)
        out["finalize"] = finalize_corpus(store, corpus_id, policy).as_dict()
    documents, embedded = store.counts(corpus_id)
    final = store.corpus(corpus_id)
    out.update(documents=documents, embedded=embedded, state=final.state if final else None,
               input_hash=final.input_hash if final else None, seconds=round(time.perf_counter() - t0, 1))
    _write_patent_manifest("embed", corpus_id, out)
    return out


def _index_step(conn: Any, corpus_id: str, *, m: int, ef_construction: int, maintenance_work_mem: str | None,
                replace: bool = False, concurrently: bool = False) -> dict:  # fmt: skip
    from app.services.patents.index import build_patent_index

    manifest = build_patent_index(conn, corpus_id, m=m, ef_construction=ef_construction,
                                  maintenance_work_mem=maintenance_work_mem, replace=replace,
                                  concurrently=concurrently)  # fmt: skip
    _write_patent_manifest("index", corpus_id, manifest)
    return manifest


def _validate_step(conn: Any, corpus_id: str, *, probes: int, min_recall: float, k: int | None) -> dict:
    from app.services.patents.index import validate_corpus

    manifest = validate_corpus(conn, corpus_id, probes=probes, k=k or get_settings().neighbour_k, threshold=min_recall)
    _write_patent_manifest("validate", corpus_id, manifest)
    return manifest


def _ingest_decision(conn: Any, corpus_id: str, parameters: dict) -> tuple[str, str]:
    """("run" | "skip" | "refuse", why) for the ingest step of a ``load-patents`` re-run.

    - no corpus: run;
    - embedded / validated: skip only if the last ingest manifest is completed with exactly these parameters;
      otherwise refuse, so a smoke load (or one with another sample, source or concordance) never passes for the
      requested load;
    - loading (or failed): skip only if the last manifest is completed with these parameters **and** the live
      document count equals the count it recorded; otherwise run (the upsert is idempotent).
    """
    corpus = _corpus_row(conn, corpus_id)
    if corpus is None:
        return "run", "new corpus"
    last = _read_patent_manifest("ingest", corpus_id)
    same = bool(last and last.get("completed") and last.get("parameters") == parameters)
    if corpus.state in ("embedded", "validated"):
        if same:
            return "skip", f"corpus is {corpus.state} and its last completed ingest had these parameters"
        return "refuse", (f"corpus {corpus_id!r} is {corpus.state}, but these parameters differ from the last "
                          f"completed ingest ({_parameter_diff(last, parameters)}); use --reingest or a new corpus id")
    if same:
        live = _live_document_count(conn, corpus_id)
        if last.get("corpus_document_count") == live:  # type: ignore[union-attr]
            return "skip", f"a completed ingest with these parameters left {live} documents"
        return "run", (f"the corpus holds {live} documents, the last completed ingest left "
                       f"{last.get('corpus_document_count')}")  # type: ignore[union-attr]
    return "run", _parameter_diff(last, parameters)


# Shared options.
CorpusOpt = Annotated[str, typer.Option("--corpus", help="Corpus id, e.g. hupd-2017 (letters, digits, . _ : -)")]
SourceOpt = Annotated[str, typer.Option("--source", help="hupd or uspto-odp")]
YearsOpt = Annotated[str, typer.Option("--years", help="Filing (HUPD) or grant (USPTO) years: 2017, 2021-2025, 2017,2019")]
LimitOpt = Annotated[int | None, typer.Option("--limit", min=1, help="Read at most N source records")]
SampleOpt = Annotated[int | None, typer.Option("--sample", min=1, help="Seeded sample of N eligible records")]
SeedOpt = Annotated[int, typer.Option("--seed", help="Sampling seed")]
HoldoutOpt = Annotated[int, typer.Option("--holdout", min=0, help="Seeded hold-out of N records, never stored")]
LeakOpt = Annotated[bool, typer.Option("--allow-holdout-leak", help="Store even if hold-out ids are already in the corpus")]
ConcordanceOpt = Annotated[bool, typer.Option("--concordance/--no-concordance",
                                              help="Apply the WIPO concordance from data/source/ when present")]
DataDirOpt = Annotated[Path | None, typer.Option("--data-dir", help="HUPD tarball directory (default HUPD_DATA_DIR)")]
EditionOpt = Annotated[str | None, typer.Option("--label-ipc-edition", help="IPC edition of the source labels, e.g. 2017.01")]
DescriptionOpt = Annotated[str | None, typer.Option("--description")]
IngestBatchOpt = Annotated[int, typer.Option("--ingest-batch", min=1, help="Documents per committed ingest batch")]
BatchOpt = Annotated[int, typer.Option("--batch", min=1, help="Documents per committed embedding batch")]
FromNpyOpt = Annotated[Path | None, typer.Option("--from-npy", help="Directory of precomputed PaECTER .npy shards + meta.json")]
IdsOpt = Annotated[Path | None, typer.Option("--ids", help="Id list for --from-npy (JSONL with title/abstract: text-checked)")]
ModelPathOpt = Annotated[Path | None, typer.Option("--model-path", help="Local PaECTER copy (default PATENT_ENCODER_LOCAL_PATH)")]
MOpt = Annotated[int, typer.Option("--m", help="HNSW m (required: measure it for the corpus, TASK-0007)")]
EfcOpt = Annotated[int, typer.Option("--ef-construction", help="HNSW ef_construction (required, TASK-0007)")]
MwmOpt = Annotated[str | None, typer.Option("--maintenance-work-mem", help="e.g. 2GB, this session only")]
ProbesOpt = Annotated[int, typer.Option("--probes", min=1, help="Recall probes")]
RecallOpt = Annotated[float, typer.Option("--min-recall", min=0.0, max=1.0, help="Minimum mean recall@k")]
KOpt = Annotated[int | None, typer.Option("--k", min=1, help="Neighbours per probe (default NEIGHBOUR_K)")]
MismatchOpt = Annotated[float, typer.Option(
    "--max-text-mismatch", min=0.0, max=1.0,
    help="Largest fraction of documents --from-npy may skip for a text mismatch before stopping (default 0.01)")]


@app.command("ingest-patents")
def ingest_patents(
    source: SourceOpt,
    corpus: CorpusOpt,
    years: YearsOpt,
    limit: LimitOpt = None,
    sample: SampleOpt = None,
    seed: SeedOpt = DEFAULT_PATENT_SEED,
    holdout: HoldoutOpt = 0,
    allow_holdout_leak: LeakOpt = False,
    concordance: ConcordanceOpt = True,
    data_dir: DataDirOpt = None,
    label_ipc_edition: EditionOpt = None,
    description: DescriptionOpt = None,
    ingest_batch: IngestBatchOpt = 2_000,
) -> None:
    """Load labelled patents from a source into a corpus (state loading). Manifest: data/manifests/patents/."""
    _check_source(source)
    year_list = _parse_years(years)
    if source == "uspto-odp":
        typer.echo(USPTO_UNVERIFIED_WARNING, err=True)
    with _patent_errors_reported(), _patent_job(corpus) as conn:
        result = _ingest_step(conn, corpus, source=source, years=year_list, limit=limit, sample=sample, seed=seed,
                              holdout=holdout, allow_holdout_leak=allow_holdout_leak, concordance=concordance,
                              data_dir=data_dir, label_ipc_edition=label_ipc_edition, description=description,
                              batch=ingest_batch)  # fmt: skip
    _echo({k: v for k, v in result.items() if k != "holdout_ids"} | {"holdout_ids": len(result.get("holdout_ids", []))})


@app.command("embed-patents")
def embed_patents(
    corpus: CorpusOpt,
    batch: BatchOpt = 1_000,
    from_npy: FromNpyOpt = None,
    ids: IdsOpt = None,
    model_path: ModelPathOpt = None,
    max_text_mismatch: MismatchOpt = 0.01,
) -> None:
    """Fill the corpus's vectors (resumable): import precomputed shards if given, embed the rest with PaECTER."""
    with _patent_errors_reported(), _patent_job(corpus) as conn:
        result = _embed_step(conn, corpus, batch=batch, from_npy=from_npy, ids=ids, model_path=model_path,
                             max_text_mismatch=max_text_mismatch)  # fmt: skip
    _echo(result)
    if result["state"] != "embedded":
        _fail(f"corpus {corpus!r} is {result['state']!r}: {result['embedded']} of {result['documents']} documents "
              "have a vector; re-run to continue")  # fmt: skip


@app.command("build-patent-index")
def build_patent_index_cmd(
    corpus: CorpusOpt,
    m: MOpt,
    ef_construction: EfcOpt,
    maintenance_work_mem: MwmOpt = None,
    replace: Annotated[bool, typer.Option("--replace", help="Rebuild an existing valid index")] = False,
    concurrently: Annotated[bool, typer.Option("--concurrently", help="CREATE INDEX CONCURRENTLY")] = False,
) -> None:
    """Build the corpus's partial HNSW index (refused while the corpus is serving)."""
    with _patent_errors_reported(), _patent_job(corpus) as conn:
        _echo(_index_step(conn, corpus, m=m, ef_construction=ef_construction,
                          maintenance_work_mem=maintenance_work_mem, replace=replace, concurrently=concurrently))


@app.command("validate-patent-corpus")
def validate_patent_corpus(
    corpus: CorpusOpt,
    probes: ProbesOpt = 200,
    min_recall: RecallOpt = 0.95,
    k: KOpt = None,
) -> None:
    """Measure recall@k of the HNSW search against exact search; mark the corpus validated if it passes."""
    with _patent_errors_reported(), _patent_job(corpus) as conn:
        result = _validate_step(conn, corpus, probes=probes, min_recall=min_recall, k=k)
    _echo(result)
    if not result.get("passed"):
        _fail(f"corpus {corpus!r} failed validation: " + "; ".join(result.get("reasons", [])))


@app.command("activate-patent-corpus")
def activate_patent_corpus(corpus: CorpusOpt) -> None:
    """Serve a validated corpus with a usable index (any other corpus stops serving). No API restart needed."""
    from app.services.patents.index import activate_corpus

    with _patent_errors_reported(), _patent_job(corpus, serving_change=True) as conn:
        c = activate_corpus(conn, corpus)
    out = {"corpus_id": c.corpus_id, "is_serving": c.is_serving, "state": c.state, "document_count": c.document_count,
           "served_at": c.served_at,
           "note": "The API reads the serving corpus per request. Run POST /v1/admin/reload-confidence so the "
                   "confidence fingerprint names this corpus."}  # fmt: skip
    _write_patent_manifest("activate", corpus, out)
    _echo(out)


@app.command("retire-patent-corpus")
def retire_patent_corpus(
    corpus: CorpusOpt,
    stop_serving: Annotated[bool, typer.Option("--stop-serving", help="Also stop serving it (vote falls back)")] = False,
) -> None:
    """Drop the corpus's index and set it back to embedded."""
    from app.services.patents.index import retire_corpus

    with _patent_errors_reported(), _patent_job(corpus, serving_change=True) as conn:
        out = retire_corpus(conn, corpus, stop_serving=stop_serving)
    _write_patent_manifest("retire", corpus, out)
    _echo(out)


@app.command("load-patents")
def load_patents(
    source: SourceOpt,
    corpus: CorpusOpt,
    years: YearsOpt,
    m: MOpt,
    ef_construction: EfcOpt,
    limit: LimitOpt = None,
    sample: SampleOpt = None,
    seed: SeedOpt = DEFAULT_PATENT_SEED,
    holdout: HoldoutOpt = 0,
    allow_holdout_leak: LeakOpt = False,
    concordance: ConcordanceOpt = True,
    data_dir: DataDirOpt = None,
    label_ipc_edition: EditionOpt = None,
    description: DescriptionOpt = None,
    ingest_batch: IngestBatchOpt = 2_000,
    batch: BatchOpt = 1_000,
    from_npy: FromNpyOpt = None,
    ids: IdsOpt = None,
    model_path: ModelPathOpt = None,
    maintenance_work_mem: MwmOpt = None,
    probes: ProbesOpt = 200,
    min_recall: RecallOpt = 0.95,
    max_text_mismatch: MismatchOpt = 0.01,
    reingest: Annotated[bool, typer.Option(
        "--reingest", help="Run ingest even if a completed one could be skipped, or its parameters differ")] = False,
) -> None:
    """Ingest, embed, index and validate one corpus, in order; stop at the first failure. Re-run to resume.

    Activation stays a separate, deliberate step (activate-patent-corpus). Each step writes its manifest under
    data/manifests/patents/; the whole run holds the corpus's advisory lock.
    """
    _check_source(source)
    year_list = _parse_years(years)
    if (from_npy is None) != (ids is None):
        raise typer.BadParameter("--from-npy and --ids go together (the shards and their id list)")
    if source == "uspto-odp":
        typer.echo(USPTO_UNVERIFIED_WARNING, err=True)
    parameters = _ingest_parameters(source, year_list, limit, sample, seed, holdout, concordance=concordance,
                                    data_dir=data_dir, label_ipc_edition=label_ipc_edition)  # fmt: skip
    steps: list[dict[str, Any]] = []
    summary = {"corpus_id": corpus, "source": source, "years": year_list, "steps": steps}

    def run(name: str, fn: Callable[[], dict], brief: Callable[[dict], dict]) -> dict:
        t0 = time.perf_counter()
        try:
            result = fn()
        except _patent_errors() as exc:
            steps.append({"step": name, "status": "failed", "error": _describe(exc),
                          "seconds": round(time.perf_counter() - t0, 1)})  # fmt: skip
            _echo(summary)
            _fail(f"load-patents stopped at step '{name}': {_describe(exc)}. Completed steps are kept; fix the cause "
                  "and re-run the same command to resume.")  # fmt: skip
        steps.append({"step": name, "status": "done", "seconds": round(time.perf_counter() - t0, 1), **brief(result)})
        return result

    with _patent_errors_reported(), _patent_job(corpus) as conn:
        decision, why = ("run", "--reingest") if reingest else _ingest_decision(conn, corpus, parameters)
        if decision == "refuse":
            steps.append({"step": "ingest", "status": "refused", "why": why})
            _echo(summary)
            raise PatentResumeRefused(why)
        start = _corpus_row(conn, corpus)
        if decision == "skip" and start is not None and start.state == "validated":
            steps += [{"step": n, "status": "skipped", "why": f"corpus is already validated ({why})"} for n in LOAD_STEPS]
            summary["next"] = f"ipc activate-patent-corpus --corpus {corpus}"
            _echo(summary)
            return
        if decision == "skip":
            steps.append({"step": "ingest", "status": "skipped", "why": why})
        else:
            run("ingest", lambda: _ingest_step(
                conn, corpus, source=source, years=year_list, limit=limit, sample=sample, seed=seed, holdout=holdout,
                allow_holdout_leak=allow_holdout_leak, concordance=concordance, data_dir=data_dir,
                label_ipc_edition=label_ipc_edition, description=description, batch=ingest_batch),
                lambda r: {"stored": r["counts"]["stored"], "corpus_document_count": r["corpus_document_count"],
                           "concordance_applied": r["concordance_applied"]})  # fmt: skip
        current = _corpus_row(conn, corpus)
        if current is not None and current.state == "embedded":
            steps.append({"step": "embed", "status": "skipped", "why": "corpus is embedded"})
        else:
            embedded = run("embed", lambda: _embed_step(conn, corpus, batch=batch, from_npy=from_npy, ids=ids,
                                                        model_path=model_path, max_text_mismatch=max_text_mismatch),
                           lambda r: {"documents": r["documents"], "embedded": r["embedded"], "state": r["state"],
                                      "imported": r["imported"], "skipped_text_mismatch": r["skipped_text_mismatch"],
                                      "text_checked": r["text_checked"]})  # fmt: skip
            if embedded["state"] != "embedded":
                steps[-1]["status"] = "failed"
                _echo(summary)
                _fail(f"load-patents stopped at step 'embed': {embedded['embedded']} of {embedded['documents']} "
                      "documents have a vector (see the embed manifest). Re-run the same command to continue.")
        run("index", lambda: _index_step(conn, corpus, m=m, ef_construction=ef_construction,
                                         maintenance_work_mem=maintenance_work_mem),
            lambda r: {"index_status": r["status"], "bytes": r.get("bytes")})
        validated = run("validate", lambda: _validate_step(conn, corpus, probes=probes, min_recall=min_recall, k=None),
                        lambda r: {"passed": r["passed"], "recall_at_k": r.get("recall_at_k"), "state": r["state"]})
        if not validated["passed"]:
            steps[-1]["status"] = "failed"
            _echo(summary)
            _fail("load-patents stopped at step 'validate': " + "; ".join(validated["reasons"])
                  + ". Rebuild the index with larger --m / --ef-construction (build-patent-index --replace) or fix "
                    "the load, then re-run.")  # fmt: skip
    summary["next"] = f"ipc activate-patent-corpus --corpus {corpus}"
    _write_patent_manifest("load", corpus, summary)
    _echo(summary)


@app.command("patent-status")
def patent_status() -> None:
    """List patent corpora: state, serving flag, document / embedding counts, encoder, index and its size."""
    from app.db.patents import CORPUS_FIELDS, Corpus
    from app.db.session import connect_direct
    from app.services.patents.index import corpus_index

    with _patent_errors_reported(), connect_direct() as conn:
        if not conn.execute("SELECT to_regclass('public.patent_corpora') IS NOT NULL").fetchone()[0]:
            conn.rollback()
            _fail("the patent tables do not exist (run `ipc migrate`)")
        corpora = [Corpus(*r) for r in conn.execute(
            f"SELECT {CORPUS_FIELDS} FROM patent_corpora ORDER BY created_at, corpus_id").fetchall()]  # fixed fields
        docs = dict(conn.execute("SELECT corpus_id, count(*) FROM patent_documents GROUP BY corpus_id").fetchall())
        embs = dict(conn.execute("SELECT corpus_id, count(*) FROM patent_embeddings GROUP BY corpus_id").fetchall())
        conn.rollback()
        out = []
        for c in corpora:
            try:
                index = corpus_index(conn, c.corpus_id)
            except ValueError:  # an id that cannot have a partial index (should not exist)
                index = None
            out.append({
                "corpus_id": c.corpus_id, "source": c.source, "state": c.state, "is_serving": c.is_serving,
                "document_count": c.document_count, "documents": int(docs.get(c.corpus_id, 0)),
                "embeddings": int(embs.get(c.corpus_id, 0)),
                "encoder": {"model_id": c.encoder_model_id, "model_revision": c.encoder_model_revision},
                "input_hash": c.input_hash, "label_ipc_edition": c.label_ipc_edition,
                "labels_taxonomy_version": c.labels_taxonomy_version,
                "index": None if index is None else {"name": index["index"], "bytes": index["bytes"],
                                                     "mib": round(index["bytes"] / 2**20, 1), "valid": index["valid"]},
                "created_at": c.created_at, "updated_at": c.updated_at, "served_at": c.served_at,
            })  # fmt: skip
    _echo({"target": get_settings().ipc_db_target, "corpora": out})


def _load_pipeline():
    """Load models + pipeline in-process (same code path as the API), on the definition path only.

    Calibration pairs are reranker scores of definition-matched candidates. With NEIGHBOUR_VOTE_ENABLED the
    pipeline would answer with vote codes instead, so the runtime is built ``definition_only``: the vote is off in
    the pipeline and PaECTER is not loaded, while the fingerprint still describes the configured (served)
    pipeline, so a fitted calibrator is compatible with the API that serves it.
    """
    from app.runtime import Runtime

    rt = Runtime(get_settings(), definition_only=True)
    rt._load()  # synchronous for CLI use
    if rt.state != "ready":
        raise typer.Exit(f"model loading failed: {rt.error}")
    return rt


@app.command("generate-pairs")
def generate_pairs(
    eval_file: Annotated[Path, typer.Option(help="Expert-labelled evaluation JSONL")],
    out: Annotated[Path | None, typer.Option(help="Output pairs JSONL")] = None,
) -> None:
    """Score candidate pairs with the fixed real pipeline (frozen models) for calibration."""
    from app.db.session import api_session
    from app.schemas.classify import ClassifyRequest
    from app.services.confidence.evaluation import label_pairs, load_eval_jsonl

    docs = load_eval_jsonl(eval_file)
    rt = _load_pipeline()
    s = get_settings()
    out = out or s.calibration_dir / f"pairs_{eval_file.stem}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    n_pairs = 0
    with out.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"header": True, "fingerprint_hash": rt.fingerprint_hash, "fingerprint": rt.fingerprint,
                            "eval_file": str(eval_file)}) + "\n")
        for d in docs:
            req = ClassifyRequest(abstract=d["abstract"], title=d.get("title"), specification=d.get("specification"),
                                  claims=d.get("claims"), taxonomy_version=d["taxonomy_version"],
                                  target_level=d["requested_level"], max_results=s.max_results_limit)
            with api_session() as session, session.begin():
                res = rt.pipeline.classify(session, req)
            scored = [(c.code, c.scores.reranker_score) for c in res.candidates] + \
                     [(a.code, a.scores.reranker_score) for a in res.alternatives]
            for p in label_pairs(d, [(c, sc) for c, sc in scored if sc is not None]):
                f.write(json.dumps(p.__dict__) + "\n")
                n_pairs += 1
    _echo({"documents": len(docs), "pairs": n_pairs, "out": str(out), "fingerprint_hash": rt.fingerprint_hash})


def _read_pairs(path: Path):
    from app.services.confidence.evaluation import Pair

    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    header = lines[0] if lines and lines[0].get("header") else {}
    pairs = [Pair(**x) for x in lines if not x.get("header")]
    return header, pairs


@app.command("fit-calibrator")
def fit_calibrator(
    pairs_file: Path,
    method: Annotated[str, typer.Option(help="platt (default) or isotonic")] = "platt",
    label_provenance: Annotated[str, typer.Option(help="Who labelled, under which protocol")] = "unspecified",
) -> None:
    """Fit on split=fit, report dev-split metrics for method selection. Never touches split=test."""
    import hashlib
    import uuid

    from app.services.confidence.calibrator import sha256_obj
    from app.services.confidence.evaluation import evaluate, fit_isotonic, fit_platt

    header, pairs = _read_pairs(pairs_file)
    fit = [p for p in pairs if p.split == "fit"]
    dev = [p for p in pairs if p.split == "dev"]
    params = fit_platt(fit) if method == "platt" else fit_isotonic(fit)
    cal_id = f"cal-{method}-{uuid.uuid4().hex[:10]}"
    manifest = {
        "calibrator_id": cal_id, "method": method, "feature": "reranker_score (raw logit)", "parameters": params,
        "artifact_sha256": sha256_obj({"method": method, "parameters": params}),
        "compatibility": {"fingerprint_hash": header.get("fingerprint_hash"), "fingerprint": header.get("fingerprint")},
        "input_modes": sorted({p.input_mode for p in fit}),
        "requested_levels": sorted({p.requested_level for p in fit}),
        "label_provenance": label_provenance,
        "splits": {"pairs_file": str(pairs_file),
                   "pairs_sha256": hashlib.sha256(pairs_file.read_bytes()).hexdigest(),
                   "fit_pairs": len(fit), "dev_pairs": len(dev),
                   "fit_documents": len({p.document_id for p in fit}), "dev_documents": len({p.document_id for p in dev})},
        "metrics": {"dev": evaluate(method, params, dev)},
        "thresholds": {},
        "status": "fitted",
    }
    path = get_settings().calibration_dir / f"{cal_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _echo({"manifest": str(path), "parameters": params, "dev_overall": manifest["metrics"]["dev"].get("overall")})


@app.command("evaluate-calibration")
def evaluate_calibration(manifest_path: Path, pairs_file: Path) -> None:
    """Final held-out evaluation on split=test. Records metrics in the manifest."""
    from app.services.confidence.evaluation import evaluate

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    header, pairs = _read_pairs(pairs_file)
    if header.get("fingerprint_hash") != manifest["compatibility"]["fingerprint_hash"]:
        raise typer.BadParameter("pairs were generated by a different pipeline fingerprint than the calibrator")
    test = [p for p in pairs if p.split == "test"]
    manifest["metrics"]["test"] = evaluate(manifest["method"], manifest["parameters"], test)
    manifest["splits"]["test_pairs"] = len(test)
    manifest["splits"]["test_documents"] = len({p.document_id for p in test})
    manifest["tested_input_modes"] = sorted({p.input_mode for p in test})
    manifest["status"] = "evaluated"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _echo({"test_overall": manifest["metrics"]["test"].get("overall"), "slices": list(manifest["metrics"]["test"].get("slices", {}))})


@app.command("activate-calibrator")
def activate_calibrator(
    manifest_path: Path,
    max_ece: Annotated[float, typer.Option(help="Acceptance threshold on held-out ECE (record your justification)")],
    min_test_pairs: Annotated[int, typer.Option()] = 200,
    min_test_positives: Annotated[int, typer.Option()] = 30,
    justification: Annotated[str, typer.Option(help="Why these thresholds are acceptable")] = "",
) -> None:
    """Activate only an evaluated, compatible calibrator whose held-out metrics pass recorded thresholds."""
    from app.services.confidence.calibrator import ACTIVE_POINTER, sha256_obj

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    test = (manifest.get("metrics", {}).get("test") or {}).get("overall")
    problems = []
    if manifest.get("status") != "evaluated" or not test:
        problems.append("calibrator has no held-out test evaluation (run evaluate-calibration)")
    else:
        if test["n"] < min_test_pairs:
            problems.append(f"test pairs {test['n']} < {min_test_pairs}")
        if test["positives"] < min_test_positives:
            problems.append(f"test positives {test['positives']} < {min_test_positives}")
        if test["ece"] > max_ece:
            problems.append(f"test ECE {test['ece']:.4f} > {max_ece}")
    if sha256_obj({"method": manifest["method"], "parameters": manifest["parameters"]}) != manifest["artifact_sha256"]:
        problems.append("artifact hash mismatch")
    rt = _load_pipeline()
    if manifest["compatibility"]["fingerprint_hash"] != rt.fingerprint_hash:
        problems.append("calibrator fingerprint does not match the current pipeline")
    if not justification.strip():
        problems.append("--justification is required to record why the thresholds are acceptable")
    if problems:
        _echo({"activated": False, "problems": problems})
        raise typer.Exit(1)
    manifest["status"] = "active"
    manifest["thresholds"] = {"max_ece": max_ece, "min_test_pairs": min_test_pairs,
                              "min_test_positives": min_test_positives, "justification": justification,
                              "measured_test_ece": test["ece"]}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pointer = manifest_path.parent / ACTIVE_POINTER
    pointer.write_text(json.dumps({"manifest": manifest_path.name, "artifact_sha256": manifest["artifact_sha256"]}),
                       encoding="utf-8")
    _echo({"activated": True, "calibrator_id": manifest["calibrator_id"]})


@app.command("eval-retrieval")
def eval_retrieval(
    queries: Annotated[int, typer.Option(help="Number of catalogue-derived probe queries")] = 200,
    k: int = 100,
    ef_search: int = 500,
    table: str = "ipc_embeddings",
    seed: int = 13,
) -> None:
    """HNSW vs exact nearest-neighbour recall@k (index fidelity only, not classification accuracy).

    Probe queries are taxonomy titles re-encoded as queries; this measures ANN fidelity, never correctness.
    """
    import random

    from app.db.session import api_session, connect_direct
    from app.services.embeddings.providers import get_provider
    from app.services.retrieval.vector import vector_search

    s = get_settings()
    provider = get_provider(s, "bge_m3" if table == "ipc_embeddings" else "gemma_api")
    with connect_direct() as conn:
        version = _active_version(conn)
        titles = conn.execute("SELECT title FROM ipc_records WHERE taxonomy_version = %s AND record_type IN"
                              " ('maingroup','subgroup') ORDER BY code", [version]).fetchall()
    rng = random.Random(seed)
    sample = [t[0] for t in rng.sample(titles, min(queries, len(titles)))]
    vecs = provider.encode_queries(sample)
    recalls = []
    for v in vecs:
        with api_session() as session, session.begin():
            approx = vector_search(session, table=table, version=version, query_vector=v, limit=k, ef_search=ef_search)
        with api_session() as session, session.begin():
            exact = vector_search(session, table=table, version=version, query_vector=v, limit=k, ef_search=ef_search,
                                  exact=True)
        a, e = {h.code for h in approx}, {h.code for h in exact}
        recalls.append(len(a & e) / max(1, len(e)))
    out = {"table": table, "queries": len(recalls), "k": k, "ef_search": ef_search,
           "mean_recall_at_k": sum(recalls) / len(recalls), "min_recall_at_k": min(recalls),
           "note": "ANN fidelity vs exact search only; not a measure of correct IPC classification."}
    write_manifest_safe(f"retrieval_recall_{table}_k{k}_ef{ef_search}", out)
    _echo(out)


@app.command("storage-probe")
def storage_probe_cmd(
    rows: int = 80145,
    variants: Annotated[str, typer.Option(help="comma list of kind:dim")] = "vector:1024,halfvec:1024,vector:768",
    m: int = 32,
    ef_construction: int = 200,
    out: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Measure real table+HNSW storage with synthetic unit vectors in scratch tables (dropped afterwards).

    Use against a LOCAL database: it temporarily needs the full measured size.
    """
    from app.db.session import connect_direct
    from app.services.retrieval.storage_probe import probe

    s = get_settings()
    if s.ipc_db_target != "local":
        raise typer.BadParameter("storage-probe writes large scratch tables; run it with --target local")
    results = []
    with connect_direct() as conn:
        for spec in variants.split(","):
            kind, dim = spec.split(":")
            results.append(probe(conn, rows=rows, dim=int(dim), kind=kind, m=m, ef_construction=ef_construction))
            typer.echo(json.dumps(results[-1]))
    out = out or s.manifests_dir / "storage_probe_local.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in results), encoding="utf-8")


@app.command("compare-precision")
def compare_precision(queries: int = 100, k: int = 100, seed: int = 13,
                      extra_query_file: Annotated[Path | None, typer.Option()] = None) -> None:
    """Exact top-k agreement between float32 and halfvec(1024) cosine search (storage alternative test).

    Exact sequential scans on both sides, so differences come from precision alone, not ANN error.
    """
    import random

    from app.db.session import connect_direct
    from app.services.embeddings.providers import get_provider
    from app.services.retrieval.vector import to_vector_literal

    s = get_settings()
    provider = get_provider(s, "bge_m3")
    with connect_direct() as conn:
        version = _active_version(conn)
        titles = [t[0] for t in conn.execute(
            "SELECT title FROM ipc_records WHERE taxonomy_version = %s AND record_type IN ('maingroup','subgroup')"
            " ORDER BY code", [version]).fetchall()]
        rng = random.Random(seed)
        texts = rng.sample(titles, min(queries, len(titles)))
        if extra_query_file and extra_query_file.exists():
            texts.append(json.loads(extra_query_file.read_text(encoding="utf-8"))["abstract"])
        vecs = provider.encode_queries(texts, max_tokens=s.abstract_max_tokens)
        at10, atk = [], []
        for v in vecs:
            q = to_vector_literal(v)
            with conn.transaction():
                conn.execute("SET LOCAL enable_indexscan = off")
                f32 = [r[0] for r in conn.execute(
                    "SELECT code FROM ipc_embeddings WHERE taxonomy_version = %s"
                    " ORDER BY embedding <=> %s::vector(1024), code LIMIT %s", [version, q, k]).fetchall()]
                f16 = [r[0] for r in conn.execute(
                    "SELECT code FROM ipc_embeddings WHERE taxonomy_version = %s"
                    " ORDER BY embedding::halfvec(1024) <=> %s::halfvec(1024), code LIMIT %s", [version, q, k]).fetchall()]
            at10.append(len(set(f32[:10]) & set(f16[:10])) / 10)
            atk.append(len(set(f32) & set(f16)) / max(1, len(f32)))
    out = {"queries": len(at10), "k": k, "mean_overlap_at_10": sum(at10) / len(at10), "min_overlap_at_10": min(at10),
           f"mean_overlap_at_{k}": sum(atk) / len(atk), f"min_overlap_at_{k}": min(atk),
           "method": "exact float32 vs exact halfvec(1024) cosine top-k on the same stored BGE-M3 vectors",
           "note": "Measures precision-induced ranking change only; not classification accuracy."}
    write_manifest_safe(f"precision_comparison_halfvec_k{k}", out)
    _echo(out)


def _catalogue_presence(docs: list[dict]) -> set[str]:
    """Which recorded symbols (exact, main group, subclass) exist in the active catalogue."""
    from sqlalchemy import text as sqltext

    from app.db import repository as repo
    from app.db.session import api_session
    from app.services.confidence.ipc_codes import parse_symbols

    wanted: set[str] = set()
    for d in docs:
        for s in parse_symbols(", ".join(d["ground_truth"])):
            wanted |= {s.display, s.subclass} | ({s.maingroup_display} if s.maingroup_display else set())
    with api_session() as session:
        version = repo.active_release(session).taxonomy_version
        return {r[0] for r in session.execute(
            sqltext("SELECT display_code FROM ipc_records WHERE taxonomy_version = :v AND display_code = ANY(:c)"),
            {"v": version, "c": sorted(wanted)}).all()}


def _score_docs(docs: list[dict], done: dict, present: set[str], supported_override: dict | None = None):
    """Score each document; truth at each level uses only recorded symbols present at that level."""
    from app.services.confidence.ipc_codes import parse_symbols
    from app.services.evaluation import benchmark as bm

    rows, scored_docs, errors, missing, field_pairs = [], [], [], [], []
    for d in docs:
        r = done.get(d["application_number"])
        if not r or r["status"] == "error":
            errors.append({"application_number": d["application_number"], "error": (r or {}).get("error")})
            continue
        supported = (supported_override or {}).get(d["application_number"], r["supported"])
        syms = parse_symbols(", ".join(d["ground_truth"]))
        missing += [s.display for s in syms if s.display not in present]
        gt = bm.GroundTruth(symbols=[s for s in syms if s.subclass in present])
        if not gt.symbols:
            errors.append({"application_number": d["application_number"], "error": "no recorded symbol in catalogue"})
            continue
        row = bm.score_document(gt, supported, r["ranked"], (r.get("field_key") or "")[:3] or None,
                                retrieval_debug=r.get("retrieval_debug"))
        if supported_override is not None:
            row["field_class_match"] = None  # field label depends on the live primary candidate
        exact_syms = [s for s in syms if s.display in present]
        row["exact"] = (bm.score_document(bm.GroundTruth(symbols=exact_syms), supported, r["ranked"], None,
                                           retrieval_debug=r.get("retrieval_debug"))["exact"]
                        if exact_syms else None)
        mg_syms = [s for s in syms if s.maingroup_display in present]
        row["maingroup"] = (bm.score_document(bm.GroundTruth(symbols=mg_syms), supported, r["ranked"], None,
                                               retrieval_debug=r.get("retrieval_debug"))
                            ["maingroup"] if mg_syms else None)
        rows.append(row)
        scored_docs.append(d)
        field_pairs.append((d.get("field_of_invention"), r.get("field_label")))
    return rows, scored_docs, errors, missing, field_pairs


@app.command("ipindia-parse")
def ipindia_parse(
    details_dir: Path,
    out: Annotated[Path, typer.Option(help="Output JSONL of benchmark documents")],
) -> None:
    """Parse saved IP India 'PatentDetails' pages into benchmark documents (JSONL)."""
    from app.services.evaluation.ipindia import parse_dir

    docs, skipped = parse_dir(details_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs), encoding="utf-8")
    _echo({"documents": len(docs), "skipped": skipped, "out": str(out)})


@app.command()
def benchmark(
    docs_file: Path,
    api_url: Annotated[str, typer.Option(help="Running classifier API")] = "http://127.0.0.1:8000",
    mode: Annotated[str, typer.Option(help="abstract | abstract+claims")] = "abstract",
    max_results: int = 5,
    limit: Annotated[int | None, typer.Option(help="Only the first N documents")] = None,
    out_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Classify each document through the running API and score against its recorded IPC symbols.

    Measures agreement with the IPC recorded on the document (often applicant-declared, older IPC
    versions), not examiner-verified accuracy. Resumable: finished documents are skipped.
    """
    import time

    import httpx

    from app.services.evaluation import benchmark as bm

    if mode not in ("abstract", "abstract+claims"):
        raise typer.BadParameter("mode must be abstract or abstract+claims")
    docs = [json.loads(x) for x in docs_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    docs = docs[:limit] if limit else docs
    out_dir = out_dir or get_settings().manifests_dir / "benchmarks" / f"{docs_file.stem}__{mode.replace('+', '_')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    done = {}
    if results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["application_number"]] = r

    present = _catalogue_presence(docs)

    client = httpx.Client(base_url=api_url, timeout=300)
    with results_path.open("a", encoding="utf-8") as fh:
        for i, d in enumerate(docs, 1):
            if d["application_number"] in done:
                continue
            body = {"abstract": d["abstract"], "title": d.get("title"), "max_results": max_results,
                    "target_level": "group", "use_llm": False, "include_ranked_debug": True}
            if mode == "abstract+claims":
                body["claims"] = d.get("claims")
            t0 = time.perf_counter()
            try:
                resp = client.post("/v1/classify", json=body)
                payload = resp.json()
            except Exception as exc:  # noqa: BLE001
                resp, payload = None, {"error": "client_error", "message": str(exc)[:300]}
            rec = {"application_number": d["application_number"], "seconds": round(time.perf_counter() - t0, 1),
                   "http": resp.status_code if resp is not None else None}
            if resp is not None and resp.status_code == 200:
                # Full reranked list (before the selection policy) measures candidate generation + ranking.
                debug = payload.get("ranked_debug") or []
                retrieval_debug = payload.get("retrieval_debug") or {}
                ranked = retrieval_debug.get("reranked_candidates") or [c["display_code"] for c in debug]
                rec |= {"status": payload["status"],
                        "supported": [c["display_code"] for c in payload["candidates"]],
                        "ranked": ranked,
                        "ranked_debug": debug,
                        "retrieval_debug": retrieval_debug,
                        "field_label": payload["field_of_invention"]["label"],
                        "field_key": payload["field_of_invention"]["mapping_key"],
                        "pipeline": {
                            "embedding_run_id": payload["provenance"]["embedding_run_id"],
                            "field_mapping_version": payload["provenance"]["field_mapping_version"],
                            "policy": {k: payload["provenance"]["retrieval_settings"].get(k) for k in (
                                "min_reranker_score", "max_reranker_gap", "min_evidence_score",
                                "enforce_lexical_checks", "hierarchical_retrieval", "claims_retrieval",
                                "hierarchical_subclass_limit")}}}
            else:
                rec |= {"status": "error", "error": payload.get("error"), "message": payload.get("message")}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            done[d["application_number"]] = rec
            typer.echo(f"[{i}/{len(docs)}] {d['application_number']} {rec['status']} {rec.get('supported', '')} "
                       f"truth={d['ground_truth'][:3]} {rec['seconds']}s")

    rows, scored_docs, errors, missing, field_pairs = _score_docs(docs, done, present)

    report = {
        "benchmark": "IP India Public Search, published applications",
        "docs_file": str(docs_file), "mode": mode, "api_url": api_url, "max_results": max_results,
        "ground_truth_note": ("Recorded 'Classification (IPC)' of each published application: usually declared at "
                              "filing, may use older IPC versions and may later be revised. Scores measure agreement "
                              "with the recorded IPC, not examiner-verified accuracy."),
        "pipeline": sorted({json.dumps(r.get("pipeline"), sort_keys=True) for r in done.values() if r.get("pipeline")}),
        "documents_total": len(docs), "documents_scored": len(rows), "errors": errors,
        "status_counts": bm.status_counts([done[d["application_number"]] for d in docs if d["application_number"] in done]),
        "recorded_symbols_missing_from_catalogue": sorted(set(missing)),
        "metrics": bm.aggregate(rows),
        "by_section_of_first_recorded_symbol": bm.section_breakdown(scored_docs, rows),
        "field_of_invention_label_agreement_with_ipindia": bm.rate(
            sum(1 for a, b in field_pairs if a and b and a.strip().upper() == b.strip().upper()),
            sum(1 for a, b in field_pairs if a)),
        "field_label_pairs": sorted({f"{a} -> {b}" for a, b in field_pairs}),
        "median_seconds": sorted(r["seconds"] for r in done.values())[len(done) // 2] if done else None,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _echo({k: report[k] for k in ("documents_total", "documents_scored", "status_counts", "metrics")})


@app.command("benchmark-tune")
def benchmark_tune(
    docs_file: Path,
    results_file: Path,
    dev_fraction: float = 0.25,
    seed: int = 13,
    objective_level: Annotated[str, typer.Option(help="subclass | maingroup | exact")] = "maingroup",
) -> None:
    """Tune the pre-calibration selection thresholds on a seeded DEV split; report the untouched TEST split.

    Replays the pipeline's selection policy offline on recorded ranked_debug rows (no model calls).
    Objective on dev: top-1 agreement over ALL documents at --objective-level (abstention counts as a
    miss). The replay is first checked against the live results for the current settings.
    """
    import hashlib
    import itertools

    from app.services.evaluation import benchmark as bm

    s = get_settings()
    docs = [json.loads(x) for x in docs_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    done = {r["application_number"]: r for r in
            (json.loads(x) for x in results_file.read_text(encoding="utf-8").splitlines() if x.strip())}
    docs = [d for d in docs if d["application_number"] in done and done[d["application_number"]]["status"] != "error"]

    def is_dev(app_no: str) -> bool:
        h = int(hashlib.sha256(f"{seed}:{app_no}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        return h < dev_fraction

    dev = [d for d in docs if is_dev(d["application_number"])]
    test = [d for d in docs if not is_dev(d["application_number"])]
    present = _catalogue_presence(docs)

    live = bm.PolicyParams(s.min_reranker_score, s.max_reranker_gap, s.min_evidence_score, 5,
                           enforce_lexical=s.enforce_lexical_checks)
    mismatches = [d["application_number"] for d in docs
                  if bm.simulate_policy(done[d["application_number"]]["ranked_debug"], live,
                                        bm.rule_demoted_codes(done[d["application_number"]]["ranked_debug"]))
                  != done[d["application_number"]]["supported"]]

    def evaluate(subset: list[dict], p: bm.PolicyParams) -> dict:
        override = {d["application_number"]: bm.simulate_policy(done[d["application_number"]]["ranked_debug"], p)
                    for d in subset}
        rows, *_ = _score_docs(subset, done, present, supported_override=override)
        return bm.aggregate(rows)

    def objective(agg: dict) -> tuple:
        lvl = agg.get(objective_level) or {}
        top1 = (lvl.get("top1_all_documents") or {}).get("rate") or 0.0
        f1 = lvl.get("mean_f1_all_documents") or 0.0
        return (round(top1, 6), round(f1, 6))

    grid = [bm.PolicyParams(f, g, e, 5, enforce_lexical=lex) for f, g, e, lex in itertools.product(
        [None, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0], [0.25, 0.5, 1.0, 1.5, 2.0, 3.0],
        [None, -6.0, -5.0, -4.0, -3.0, -2.0], [False, True])]
    scored = sorted(((objective(evaluate(dev, p)), p) for p in grid), key=lambda x: x[0], reverse=True)
    best = scored[0][1]

    def pdict(p: bm.PolicyParams) -> dict:
        return {"min_reranker_score": p.min_reranker_score, "max_reranker_gap": p.max_reranker_gap,
                "min_evidence_score": p.min_evidence_score, "max_results": p.max_results,
                "enforce_lexical_checks": p.enforce_lexical}

    report = {
        "split": {"seed": seed, "dev_fraction": dev_fraction, "dev_documents": len(dev), "test_documents": len(test),
                  "method": "sha256(seed:application_number) < dev_fraction"},
        "replay_check": {"documents": len(docs), "mismatches_vs_live": mismatches},
        "objective": f"{objective_level} top-1 over all dev documents (abstention = miss); tie-break mean set-F1",
        "current_params": pdict(live),
        "tuned_params": pdict(best),
        "top5_dev_configs": [{"params": pdict(p), "dev_objective": o} for o, p in scored[:5]],
        "dev": {"current": evaluate(dev, live), "tuned": evaluate(dev, best)},
        "test": {"current": evaluate(test, live), "tuned": evaluate(test, best)},
    }
    out = results_file.parent / f"tuning_{objective_level}_seed{seed}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _echo({k: report[k] for k in ("split", "replay_check", "current_params", "tuned_params")} | {"out": str(out)})


@app.command("benchmark-compare")
def benchmark_compare(docs_file: Path, results_file: Path) -> None:
    """Compare, on the same documents: the live run, the v1 policy replay, and top-k baselines."""
    from app.services.evaluation import benchmark as bm

    docs = [json.loads(x) for x in docs_file.read_text(encoding="utf-8").splitlines() if x.strip()]
    done = {r["application_number"]: r for r in
            (json.loads(x) for x in results_file.read_text(encoding="utf-8").splitlines() if x.strip())}
    docs = [d for d in docs if d["application_number"] in done and done[d["application_number"]]["status"] != "error"]
    present = _catalogue_presence(docs)
    v1 = bm.PolicyParams(min_reranker_score=-1.0, max_reranker_gap=1.5, min_evidence_score=-3.0, enforce_lexical=True)

    def agg(override: dict | None) -> dict:
        rows, *_ = _score_docs(docs, done, present, supported_override=override)
        return bm.aggregate(rows)

    s = get_settings()
    current = bm.PolicyParams(min_reranker_score=s.min_reranker_score, max_reranker_gap=s.max_reranker_gap,
                              min_evidence_score=s.min_evidence_score, enforce_lexical=s.enforce_lexical_checks)
    with_rules = {a: bm.simulate_policy(done[a]["ranked_debug"], current,
                                        bm.rule_demoted_codes(done[a]["ranked_debug"])) for a in done}
    replay_mismatches = sorted(a for a in done if with_rules[a] != done[a]["supported"])
    variants = {
        "live": agg(None),
        "current_policy_rules_off_replay": agg({a: bm.simulate_policy(done[a]["ranked_debug"], current)
                                                for a in done}),
        "policy_v1_replay": agg({a: bm.simulate_policy(done[a]["ranked_debug"], v1) for a in done}),
        "baseline_top1_always": agg({a: list(dict.fromkeys(done[a]["ranked"]))[:1] for a in done}),
        "baseline_top3_always": agg({a: list(dict.fromkeys(done[a]["ranked"]))[:3] for a in done}),
    }
    field_pairs = [(d.get("field_of_invention"), done[d["application_number"]].get("field_label")) for d in docs]
    rule_effects = {a: sorted(c["display_code"] + " <- " + c["outcome"] for c in done[a]["ranked_debug"]
                              if c["code"] in bm.rule_demoted_codes(done[a]["ranked_debug"])) for a in done}
    out = {"documents": len(docs), "variants": variants,
           "replay_with_recorded_rules_mismatches_vs_live": replay_mismatches,
           "documents_where_wipo_rules_demoted_a_candidate": {a: v for a, v in rule_effects.items() if v},
           "live_field_label_agreement_with_ipindia": bm.rate(
               sum(1 for a, b in field_pairs if a and b and a.strip().upper() == b.strip().upper()),
               sum(1 for a, _ in field_pairs if a)),
           "field_label_confusion": dict(sorted(Counter_pairs(field_pairs).items(), key=lambda kv: -kv[1])[:40])}
    path = results_file.parent / "comparison.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for name, a in variants.items():
        typer.echo(f"{name:22} abstain={a['abstained']['rate']:.3f} " + " ".join(
            f"{lvl[:4]}:top1={a[lvl]['top1_all_documents']['rate']:.3f}{a[lvl]['top1_all_documents']['ci95']}"
            f"/any={a[lvl]['any_supported_all_documents']['rate']:.3f}/f1={a[lvl]['mean_f1_all_documents']:.3f}"
            for lvl in ("subclass", "maingroup", "exact")))
    typer.echo(f"replay mismatches vs live: {len(replay_mismatches)}; documents with a WIPO-rule demotion: "
               f"{len(out['documents_where_wipo_rules_demoted_a_candidate'])}")
    typer.echo(f"field label agreement: {out['live_field_label_agreement_with_ipindia']}  -> {path}")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


@app.command("families")
def families_cmd(
    docs_files: Annotated[list[str], typer.Argument(help="Document JSONL files, each as PATH or PATH=SPLIT "
                                                         "(a split pins every document in that file)")],
    out: Annotated[Path, typer.Option(help="Output JSON: document -> family and split")],
    raw_dir: Annotated[list[Path] | None, typer.Option(help="Saved IP India detail pages, to read priority "
                                                            "claims missing from older JSONL files")] = None,
    ratios: Annotated[str, typer.Option(help="Split ratios for unpinned families")] = "fit=0.7,dev=0.15,test=0.15",
    seed: str = "families-v1",
) -> None:
    """Group documents into patent families and assign family-safe splits; report cross-split families."""
    from app.services.evaluation.families import assign_splits, group_families, split_conflicts
    from app.services.evaluation.ipindia import parse_detail

    priority: dict[str, dict] = {}
    for d in raw_dir or []:
        for f in sorted(d.glob("*.html")):
            p = parse_detail(f.read_text(encoding="utf-8", errors="replace"))
            priority[p["application_number"]] = {k: p[k] for k in ("priority_number", "priority_country")}
    docs, pinned, source = [], {}, {}
    for spec in docs_files:
        path, _, split = spec.partition("=")
        for d in _read_jsonl(Path(path)):
            did = str(d.get("document_id") or d["application_number"])
            for k, v in priority.get(did, {}).items():
                if not d.get(k):
                    d[k] = v
            docs.append(d)
            source.setdefault(did, []).append(path)
            if split:
                pinned[did] = split
    dup_docs = {k: v for k, v in source.items() if len(v) > 1}
    families = group_families(docs)
    split_ratios = {k: float(v) for k, v in (x.split("=") for x in ratios.split(","))}
    assignment = assign_splits(families, split_ratios, seed=seed, pinned=pinned)
    conflicts = split_conflicts(families, {**assignment, **pinned})
    report = {
        "documents": len(source), "families": len(families),
        "multi_member_families": [{"family_id": f.family_id, "members": f.members, "keys": f.keys,
                                   "near_duplicate_links": f.near_duplicate_links}
                                  for f in families if len(f.members) > 1],
        "documents_in_more_than_one_file": dup_docs,
        "pinned_split_conflicts": conflicts,
        "assignment": {m: {"family_id": f.family_id, "split": assignment[m]} for f in families for m in f.members},
        "ratios": split_ratios, "seed": seed,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _echo({k: report[k] for k in ("documents", "families", "pinned_split_conflicts", "documents_in_more_than_one_file")}
          | {"multi_member_families": len(report["multi_member_families"]), "out": str(out)})
    if conflicts or dup_docs:
        raise typer.Exit(1)


@app.command("review-queue")
def review_queue_cmd(
    docs_file: Path,
    results_file: Annotated[Path, typer.Option(help="Benchmark results.jsonl holding each document's ranking")],
    families_file: Annotated[Path, typer.Option(help="Output of `ipc families`")],
    out: Annotated[Path, typer.Option(help="Review queue JSONL (give this to reviewers)")],
    key_out: Annotated[Path, typer.Option(help="Candidate-source key JSONL (do NOT give this to reviewers)")],
    top_k: int = 10,
) -> None:
    """Build blinded expert-review items: the pipeline's top-k plus declared codes, shuffled."""
    from app.db import repository as repo
    from app.db.session import api_session
    from app.services.classification.scope import parent_path
    from app.services.evaluation.review_queue import build_review_item
    from app.services.training.hard_negatives import classification_codes

    fams =json.loads(families_file.read_text(encoding="utf-8"))["assignment"]
    results = {r["application_number"]: r for r in _read_jsonl(results_file)}
    items, keys, missing = [], [], []
    with api_session() as session:
        version = repo.active_release(session).taxonomy_version
        cache: dict[str, str | None] = {}

        def internal(display: str) -> str | None:
            if display not in cache:
                rec = repo.get_record_by_any_code(session, version, display)
                cache[display] = rec.code if rec else None
            return cache[display]

        def describe(codes: list[str]) -> dict[str, dict]:
            keep = classification_codes(session, version, codes)  # indexing codes are never review candidates
            recs = repo.get_records(session, version, [c for c in codes if c in keep])
            recs.update(repo.get_records(session, version,
                                         [a for r in recs.values() for a in r.ancestor_codes if a not in recs]))
            return {c: {"display_code": recs[c].display_code, "title": recs[c].title,
                        "path": [f"{p['display_code']}: {p['title']}" for p in parent_path(recs[c], recs)]}
                    for c in codes if c in recs}

        for d in _read_jsonl(docs_file):
            did = str(d.get("document_id") or d["application_number"])
            r = results.get(did)
            if did not in fams or not r or r.get("http") != 200:
                missing.append(did)
                continue
            ranked = [c for c in (internal(x) for x in r["ranked"][: top_k * 2]) if c]
            truth = d["ground_truth"] if isinstance(d["ground_truth"], list) else json.loads(
                d["ground_truth"].replace("'", '"'))
            declared = [c for c in (internal(x) for x in truth) if c]
            item, key = build_review_item(d, ranked, declared, describe, split=fams[did]["split"],
                                          family_id=fams[did]["family_id"], taxonomy_version=version, top_k=top_k)
            key["declared_missing_from_catalogue"] = [x for x in truth if not internal(x)]
            items.append(item)
            keys.append(key)
    _write_jsonl(out, items)
    _write_jsonl(key_out, keys)
    _echo({"items": len(items), "candidates": sum(len(i["candidates"]) for i in items),
           "documents_without_results_or_family": missing, "out": str(out), "key_out": str(key_out)})


@app.command("review-finalize")
def review_finalize_cmd(
    queue_file: Path,
    out: Annotated[Path, typer.Option(help="Expert-labelled evaluation JSONL")],
    source: Annotated[str, typer.Option(help="Provenance source, e.g. 'IP India 2026-09 review, examiner team A'")],
) -> None:
    """Convert completed review items into expert-labelled JSONL (validated with the calibration loader)."""
    from app.db import repository as repo
    from app.db.session import api_session
    from app.services.confidence.evaluation import load_eval_jsonl
    from app.services.evaluation.review_queue import finalize_reviews

    with api_session() as session:
        version = repo.active_release(session).taxonomy_version
        docs, skipped = finalize_reviews(
            _read_jsonl(queue_file), source=source,
            code_exists=lambda c: bool(repo.get_records(session, version, [c])))
    _write_jsonl(out, docs)
    load_eval_jsonl(out)  # family/split and schema checks
    _echo({"labelled_documents": len(docs), "skipped": len(skipped),
           "complete": sum(d["judgement_completeness"] == "complete" for d in docs), "out": str(out)})


@app.command("mine-negatives")
def mine_negatives_cmd(
    eval_file: Annotated[Path, typer.Argument(help="Expert-labelled evaluation JSONL")],
    out: Annotated[Path, typer.Option(help="Training groups JSONL (positives + typed hard negatives)")],
    results_file: Annotated[Path | None, typer.Option(help="Benchmark results.jsonl for ranked confusables")] = None,
    flag_out: Annotated[Path | None, typer.Option(help="Also write FlagEmbedding reranker rows")] = None,
    orientation: str = "document_first",
    splits: str = "fit",
) -> None:
    """Mine reranker training groups with hard negatives from nearby parts of the scheme."""
    from app.db import repository as repo
    from app.db.session import api_session
    from app.services.confidence.evaluation import load_eval_jsonl
    from app.services.training.hard_negatives import DbCatalogue, mine_group, to_flag_reranker_rows

    s = get_settings()
    wanted = set(splits.split(","))
    docs = [d for d in load_eval_jsonl(eval_file) if d["split"] in wanted]
    results = {r["application_number"]: r for r in _read_jsonl(results_file)} if results_file else {}
    groups, skipped = [], []
    with api_session() as session:
        version = repo.active_release(session).taxonomy_version
        cat = DbCatalogue(session, version, official_context=s.rerank_official_context)
        for d in docs:
            r = results.get(d["document_id"]) or {}
            ranked = [rec.code for x in r.get("ranked", [])[:20]
                      if (rec := repo.get_record_by_any_code(session, version, x))]
            g = mine_group(d, cat, ranked_codes=ranked)
            if g:
                groups.append(g)
            else:
                skipped.append(d["document_id"])
    _write_jsonl(out, groups)
    if flag_out:
        _write_jsonl(flag_out, [row for g in groups for row in to_flag_reranker_rows(g, orientation=orientation)])
    kinds: dict[str, int] = {}
    for g in groups:
        for n in g["negatives"]:
            kinds[n["kind"]] = kinds.get(n["kind"], 0) + 1
    _echo({"groups": len(groups), "skipped_no_accepted_code": skipped, "negatives_by_kind": kinds,
           "splits": sorted(wanted), "scope_formatter": cat.scope_version, "out": str(out)})


def Counter_pairs(pairs: list[tuple]) -> dict[str, int]:  # noqa: N802
    from collections import Counter

    return dict(Counter(f"{a} -> {b}" for a, b in pairs))


if __name__ == "__main__":
    app()
