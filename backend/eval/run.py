"""Retrieval evaluation: index a labelled dataset with the production pipeline, then score each search mode.

    python -m eval.run --dataset scifact          # public BEIR benchmark (downloaded on first run)
    python -m eval.run --dataset membrain         # bundled in-domain set (eval/data/membrain.json)

Documents go through the same chunker, embedding model and Postgres indexes as the app, and queries go
through `app.services.search.search`, so the numbers describe what the API actually returns. Everything
runs in a separate database (`membrain_eval` by default), one workspace per dataset; indexing is cached
across runs unless `--reindex` is passed.
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

import asyncpg
from alembic import command
from alembic.config import Config
from sqlalchemy import delete, func, select

from app.config import BACKEND_DIR, Settings
from app.db import make_engine, make_sessionmaker
from app.models import Memory
from app.services.embeddings import FastEmbedEmbedder
from app.services.memories import store_memory
from app.services.search import keyword_candidates, reciprocal_rank_fusion, search, vector_candidates
from eval.datasets import BEIR_DATASETS, EVAL_DIR, Dataset, load, load_beir
from eval.metrics import summarize

MODES = ("vector", "keyword", "hybrid")
RESULTS_DIR = EVAL_DIR / "results"


async def ensure_database(url: str) -> None:
    raw = url.replace("postgresql+asyncpg://", "postgresql://")
    base, name = raw.rsplit("/", 1)
    conn = await asyncpg.connect(f"{base}/postgres")
    try:
        if not await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", name):
            await conn.execute(f'CREATE DATABASE "{name}"')
    finally:
        await conn.close()
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", url)
    await asyncio.to_thread(command.upgrade, cfg, "head")


async def index(sessionmaker, embedder, settings: Settings, ds: Dataset, workspace: str, reindex: bool) -> None:
    async with sessionmaker() as session:
        count = await session.scalar(select(func.count()).select_from(Memory).where(Memory.workspace == workspace))
        if count == len(ds.docs) and not reindex:
            print(f"Using cached index: {count} documents in workspace {workspace!r}")
            return
        await session.execute(delete(Memory).where(Memory.workspace == workspace))
        await session.commit()

        started = time.perf_counter()
        for i, doc in enumerate(ds.docs, start=1):
            await store_memory(
                session,
                embedder,
                workspace,
                title=doc.title[:300],
                body=doc.text,
                type="eval",
                tags=[f"doc:{doc.id}"],  # tags are neither embedded nor full-text indexed
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                commit=False,
            )
            if i % 250 == 0 or i == len(ds.docs):
                await session.commit()
                print(f"  indexed {i}/{len(ds.docs)} ({time.perf_counter() - started:.0f}s)", flush=True)


async def evaluate(sessionmaker, embedder, settings: Settings, ds: Dataset, workspace: str, k: int):
    async with sessionmaker() as session:
        rows = await session.execute(select(Memory.id, Memory.tags).where(Memory.workspace == workspace))
        doc_id = {mid: tags[0].removeprefix("doc:") for mid, tags in rows}

        runs: dict[str, dict[str, list[str]]] = {mode: {} for mode in MODES}
        latency: dict[str, list[float]] = defaultdict(list)
        for qid, query in ds.queries.items():
            t = time.perf_counter()
            qvec = embedder.embed_query(query)
            latency["embed"].append((time.perf_counter() - t) * 1000)
            for mode in MODES:
                t = time.perf_counter()
                hits = await search(
                    session,
                    workspace,
                    query,
                    qvec,
                    mode=mode,
                    k=k,
                    rrf_k=settings.rrf_k,
                    keyword_weight=settings.keyword_weight,
                    bm25_k1=settings.bm25_k1,
                    bm25_b=settings.bm25_b,
                    candidate_pool=settings.candidate_pool,
                    retention_weight=0.0,  # every document is equally fresh; measure retrieval alone
                )
                latency[mode].append((time.perf_counter() - t) * 1000)
                runs[mode][qid] = [doc_id[h.memory.id] for h in hits]
    return runs, latency


SWEEP_WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
SWEEP_RRF_K = (10, 30, 60)


async def sweep(sessionmaker, embedder, settings: Settings, ds: Dataset, workspace: str, k: int) -> None:
    """Score hybrid fusion for a grid of keyword weights, using the same candidates and fusion as search()."""
    async with sessionmaker() as session:
        rows = await session.execute(select(Memory.id, Memory.tags).where(Memory.workspace == workspace))
        doc_id = {mid: tags[0].removeprefix("doc:") for mid, tags in rows}
        candidates = {}
        for qid, query in ds.queries.items():
            vec = await vector_candidates(session, workspace, embedder.embed_query(query), settings.candidate_pool)
            kw = await keyword_candidates(
                session, workspace, query, settings.candidate_pool, k1=settings.bm25_k1, b=settings.bm25_b
            )
            candidates[qid] = ([m for m, _, _ in vec], [m for m, _ in kw])

    print(f"\nFusion sweep on {ds.source} ({len(ds.qrels)} queries)\n")
    print("| RRF k | keyword weight | recall@5 | recall@10 | mrr@10 | ndcg@10 |\n|---:|---:|---:|---:|---:|---:|")
    for rrf_k in SWEEP_RRF_K:
        for weight in SWEEP_WEIGHTS:
            runs = {}
            for qid, (vec, kw) in candidates.items():
                lists = [(r, w) for r, w in ((vec, 1.0), (kw, weight)) if r]
                fused = reciprocal_rank_fusion([r for r, _ in lists], k=rrf_k, weights=[w for _, w in lists])
                runs[qid] = [doc_id[m] for m in sorted(fused, key=fused.get, reverse=True)[:k]]
            m = summarize(runs, ds.qrels)
            print(
                f"| {rrf_k} | {weight} | {m['recall@5']:.3f} | {m['recall@10']:.3f} "
                f"| {m['mrr@10']:.3f} | {m['ndcg@10']:.3f} |"
            )


def pct(values: list[float], q: float) -> float:
    return statistics.quantiles(values, n=100)[int(q) - 1] if len(values) > 1 else values[0]


def report(ds: Dataset, runs, latency, settings: Settings) -> dict:
    metrics = {mode: summarize(runs[mode], ds.qrels) for mode in MODES}
    by_type = {}
    for category in sorted(set(ds.categories.values())):
        qrels = {q: r for q, r in ds.qrels.items() if ds.categories.get(q) == category}
        by_type[category] = {
            "queries": len(qrels),
            **{mode: summarize({q: runs[mode][q] for q in qrels}, qrels) for mode in MODES},
        }
    return {
        "dataset": ds.name,
        "source": ds.source,
        "documents": len(ds.docs),
        "queries": len(ds.qrels),
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "rrf_k": settings.rrf_k,
        "candidate_pool": settings.candidate_pool,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": metrics,
        "by_query_type": by_type,
        "latency_ms": {name: {"p50": round(pct(v, 50), 1), "p95": round(pct(v, 95), 1)} for name, v in latency.items()},
    }


def to_markdown(r: dict) -> str:
    cols = ["recall@1", "recall@5", "recall@10", "mrr@10", "ndcg@10"]
    lines = [
        f"### {r['dataset']}: {r['source']}",
        "",
        f"{r['documents']:,} documents, {r['queries']} queries; `{r['embedding_model']}`, "
        f"chunks {r['chunk_size']}/{r['chunk_overlap']}, RRF k={r['rrf_k']}.",
        "",
        "| Mode | " + " | ".join(c.replace("@", "@") for c in cols) + " | p50 latency |",
        "|---|" + "---:|" * (len(cols) + 1),
    ]
    for mode in MODES:
        m = r["metrics"][mode]
        lines.append(
            f"| {mode} | " + " | ".join(f"{m[c]:.3f}" for c in cols) + f" | {r['latency_ms'][mode]['p50']} ms |"
        )
    vec, hyb = r["metrics"]["vector"], r["metrics"]["hybrid"]
    lines += [
        "",
        f"Hybrid vs vector-only: Recall@5 {vec['recall@5']:.3f} → {hyb['recall@5']:.3f} "
        f"({(hyb['recall@5'] / vec['recall@5'] - 1) * 100:+.1f}% relative, "
        f"{(hyb['recall@5'] - vec['recall@5']) * 100:+.1f} points); "
        f"MRR@10 {vec['mrr@10']:.3f} → {hyb['mrr@10']:.3f} ({(hyb['mrr@10'] / vec['mrr@10'] - 1) * 100:+.1f}%).",
    ]
    if r["by_query_type"]:
        lines += ["", "| Query type | n | vector R@5 | keyword R@5 | hybrid R@5 |", "|---|---:|---:|---:|---:|"]
        for category, t in r["by_query_type"].items():
            lines.append(
                f"| {category} | {t['queries']} | " + " | ".join(f"{t[mode]['recall@5']:.3f}" for mode in MODES) + " |"
            )
    return "\n".join(lines) + "\n"


async def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # results contain arrows; Windows consoles default to cp1252
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="membrain", help="scifact, nfcorpus, or a file in eval/data/")
    parser.add_argument("--database-url", default=None, help="defaults to the app database URL with an _eval suffix")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--split", default="test", help="BEIR split to evaluate (tune on train, report on test)")
    parser.add_argument("--sweep", action="store_true", help="grid-search the keyword weight instead of reporting")
    args = parser.parse_args()

    settings = Settings()
    url = args.database_url or settings.database_url.rsplit("/", 1)[0] + "/membrain_eval"
    ds = load_beir(args.dataset, args.split) if args.dataset in BEIR_DATASETS else load(args.dataset)
    print(f"{ds.source}: {len(ds.docs)} documents, {len(ds.qrels)} queries")

    await ensure_database(url)
    engine = make_engine(url)
    sessionmaker = make_sessionmaker(engine)
    embedder = FastEmbedEmbedder(settings.embedding_model)
    workspace = f"eval-{ds.name}"
    try:
        await index(sessionmaker, embedder, settings, ds, workspace, args.reindex)
        if args.sweep:
            await sweep(sessionmaker, embedder, settings, ds, workspace, args.k)
            return
        runs, latency = await evaluate(sessionmaker, embedder, settings, ds, workspace, args.k)
    finally:
        await engine.dispose()

    result = report(ds, runs, latency, settings)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{ds.name}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md = to_markdown(result)
    (RESULTS_DIR / f"{ds.name}.md").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    asyncio.run(main())
