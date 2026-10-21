import uuid
from dataclasses import dataclass, field
from typing import Literal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, Memory
from app.services.decay import retention

SearchMode = Literal["hybrid", "vector", "keyword"]


@dataclass
class Hit:
    memory: Memory
    score: float  # final score: relevance x retention prior
    relevance: float  # fused score normalised to 0..1 (1.0 = ranked first by every retriever)
    retention: float
    snippet: str
    vector_similarity: float | None = None
    vector_rank: int | None = None
    keyword_rank: int | None = None
    matched_by: list[str] = field(default_factory=list)


def reciprocal_rank_fusion(rankings: list[list[uuid.UUID]], k: int = 60) -> dict[uuid.UUID, float]:
    """Combine ranked lists without needing comparable scores: score(d) = sum 1 / (k + rank_d)."""
    scores: dict[uuid.UUID, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


async def vector_candidates(
    session: AsyncSession, workspace: str, query_vec: list[float], limit: int
) -> list[tuple[uuid.UUID, str, float]]:
    """Nearest chunks by cosine distance, collapsed to the best chunk per memory."""
    distance = Chunk.embedding.cosine_distance(query_vec)
    rows = await session.execute(
        select(Chunk.memory_id, Chunk.content, distance.label("distance"))
        .join(Memory, Memory.id == Chunk.memory_id)
        .where(Memory.workspace == workspace)
        .order_by(distance)
        .limit(limit * 3)  # several chunks may belong to the same memory
    )
    best: dict[uuid.UUID, tuple[str, float]] = {}
    for memory_id, content, dist in rows:
        if memory_id not in best:
            best[memory_id] = (content, 1.0 - float(dist))
    return [(mid, content, sim) for mid, (content, sim) in list(best.items())[:limit]]


async def keyword_candidates(
    session: AsyncSession, workspace: str, query: str, limit: int
) -> list[tuple[uuid.UUID, str]]:
    """Postgres full-text search (web-style syntax: quotes, OR, -exclude), ranked by cover density."""
    tsquery = func.websearch_to_tsquery("english", query)
    rank = func.ts_rank_cd(Memory.search_tsv, tsquery)
    snippet = func.ts_headline("english", Memory.body, tsquery, "StartSel=,StopSel=,MaxWords=40,MinWords=15")
    rows = await session.execute(
        select(Memory.id, snippet)
        .where(Memory.workspace == workspace, Memory.search_tsv.op("@@")(tsquery))
        .order_by(rank.desc())
        .limit(limit)
    )
    return [(mid, snip) for mid, snip in rows]


async def search(
    session: AsyncSession,
    workspace: str,
    query: str,
    query_vec: list[float] | None,
    *,
    mode: SearchMode = "hybrid",
    k: int = 6,
    rrf_k: int = 60,
    candidate_pool: int = 50,
    retention_weight: float = 0.0,
) -> list[Hit]:
    """Hybrid retrieval, then a light retention prior: score = relevance * (1 - w + w * retention).

    With a small w, relevance dominates; between similarly relevant memories, ones in active use win.
    """
    vec = await vector_candidates(session, workspace, query_vec, candidate_pool) if mode != "keyword" else []
    kw = await keyword_candidates(session, workspace, query, candidate_pool) if mode != "vector" else []

    rankings = [r for r in ([m for m, _, _ in vec], [m for m, _ in kw]) if r]
    if not rankings:
        return []
    fused = reciprocal_rank_fusion(rankings, k=rrf_k)
    best_possible = len(rankings) / (rrf_k + 1)
    # Re-rank a wider pool than k so the prior can promote a memory from just outside the cut.
    pool_ids = sorted(fused, key=fused.get, reverse=True)[: max(k * 3, 20)]
    memories = {m.id: m for m in (await session.scalars(select(Memory).where(Memory.id.in_(pool_ids))))}
    relevance = {mid: fused[mid] / best_possible for mid in pool_ids}
    kept = {mid: retention(memories[mid].stability, memories[mid].last_accessed_at) for mid in pool_ids}
    final = {mid: relevance[mid] * (1 - retention_weight + retention_weight * kept[mid]) for mid in pool_ids}
    top_ids = sorted(pool_ids, key=lambda mid: (final[mid], relevance[mid]), reverse=True)[:k]

    vec_info = {mid: (rank, content, sim) for rank, (mid, content, sim) in enumerate(vec, start=1)}
    kw_info = {mid: (rank, snip) for rank, (mid, snip) in enumerate(kw, start=1)}

    hits = []
    for mid in top_ids:
        v, w = vec_info.get(mid), kw_info.get(mid)
        hits.append(
            Hit(
                memory=memories[mid],
                score=round(final[mid], 4),
                relevance=round(relevance[mid], 4),
                retention=round(kept[mid], 4),
                # Prefer the semantically closest chunk; fall back to the keyword highlight.
                snippet=v[1] if v else w[1],
                vector_similarity=round(v[2], 4) if v else None,
                vector_rank=v[0] if v else None,
                keyword_rank=w[0] if w else None,
                matched_by=[name for name, info in (("vector", v), ("keyword", w)) if info],
            )
        )
    return hits
