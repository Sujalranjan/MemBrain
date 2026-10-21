import re
import uuid
from dataclasses import dataclass, field
from typing import Literal

from sqlalchemy import select, text
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


def reciprocal_rank_fusion(
    rankings: list[list[uuid.UUID]], k: int = 60, weights: list[float] | None = None
) -> dict[uuid.UUID, float]:
    """Combine ranked lists without needing comparable scores: score(d) = sum w_i / (k + rank_i(d))."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[uuid.UUID, float] = {}
    for ranking, weight in zip(rankings, weights, strict=True):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
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


SEARCH_SYNTAX = re.compile(r'"|\bOR\b|(?:^|\s)-\w')

# BM25 over the memory_terms postings (see migration 0005). Any query term can match; IDF makes rare terms
# count for more than common ones, which Postgres' built-in ts_rank cannot do. Queries using web-search
# syntax (quotes, OR, -word) are additionally filtered by that expression, so phrases and exclusions hold.
BM25_SQL = text(
    """
    WITH q AS (
        SELECT DISTINCT lexeme FROM unnest(tsvector_to_array(to_tsvector('english', :query))) AS lexeme
    ),
    corpus AS (
        SELECT count(*)::float8 AS n, greatest(avg(term_count), 1)::float8 AS avglen
        FROM memories WHERE workspace = :workspace
    ),
    df AS (
        SELECT t.lexeme, count(*)::float8 AS df
        FROM memory_terms AS t JOIN q USING (lexeme)
        WHERE t.workspace = :workspace
        GROUP BY t.lexeme
    ),
    scored AS (
        SELECT t.memory_id,
               sum(ln(1 + (c.n - df.df + 0.5) / (df.df + 0.5))
                   * t.tf * (CAST(:k1 AS float8) + 1)
                   / (t.tf + CAST(:k1 AS float8)
                      * (1 - CAST(:b AS float8) + CAST(:b AS float8) * m.term_count / c.avglen))) AS score
        FROM memory_terms AS t
        JOIN df USING (lexeme)
        JOIN memories AS m ON m.id = t.memory_id
        CROSS JOIN corpus AS c
        WHERE t.workspace = :workspace
          AND (NOT CAST(:strict AS boolean) OR m.search_tsv @@ websearch_to_tsquery('english', :query))
        GROUP BY t.memory_id
        ORDER BY score DESC
        LIMIT :limit
    ),
    highlight AS (
        SELECT CASE WHEN CAST(:strict AS boolean) THEN websearch_to_tsquery('english', :query)
                    ELSE (SELECT coalesce(string_agg(quote_literal(lexeme), ' | '), '')::tsquery FROM q)
               END AS tq
    )
    SELECT s.memory_id, ts_headline('english', m.body, h.tq, 'StartSel=,StopSel=,MaxWords=40,MinWords=15')
    FROM scored AS s JOIN memories AS m ON m.id = s.memory_id CROSS JOIN highlight AS h
    ORDER BY s.score DESC
    """
)


async def keyword_candidates(
    session: AsyncSession, workspace: str, query: str, limit: int, *, k1: float = 1.2, b: float = 0.75
) -> list[tuple[uuid.UUID, str]]:
    """BM25-ranked keyword matches, with a highlighted snippet for each."""
    rows = await session.execute(
        BM25_SQL,
        {
            "query": query,
            "workspace": workspace,
            "limit": limit,
            "k1": k1,
            "b": b,
            "strict": bool(SEARCH_SYNTAX.search(query)),
        },
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
    keyword_weight: float = 1.0,
    candidate_pool: int = 50,
    bm25_k1: float = 1.2,
    bm25_b: float = 0.75,
    retention_weight: float = 0.0,
) -> list[Hit]:
    """Hybrid retrieval, then a light retention prior: score = relevance * (1 - w + w * retention).

    With a small w, relevance dominates; between similarly relevant memories, ones in active use win.
    """
    vec = await vector_candidates(session, workspace, query_vec, candidate_pool) if mode != "keyword" else []
    kw = (
        await keyword_candidates(session, workspace, query, candidate_pool, k1=bm25_k1, b=bm25_b)
        if mode != "vector"
        else []
    )

    # Weighted RRF: keyword evidence counts for `keyword_weight` of a vector vote (see the eval README).
    lists = [(r, w) for r, w in (([m for m, _, _ in vec], 1.0), ([m for m, _ in kw], keyword_weight)) if r]
    if not lists:
        return []
    rankings, weights = [r for r, _ in lists], [w for _, w in lists]
    fused = reciprocal_rank_fusion(rankings, k=rrf_k, weights=weights)
    best_possible = sum(weights) / (rrf_k + 1)
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
