"""Operator CLI: repeatable import, embedding, indexing, validation, evaluation, calibration.

Run from apps/api:  uv run ipc --help
Select the database with IPC_DB_TARGET=sakthi|local (default sakthi) or --target.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Annotated

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


def _load_pipeline():
    """Load models + pipeline in-process (same code path as the API)."""
    from app.runtime import get_runtime

    rt = get_runtime()
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
