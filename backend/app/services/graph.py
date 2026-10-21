import uuid
from dataclasses import dataclass

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Memory

# For every memory, its k nearest neighbours by cosine similarity of document embeddings.
# The LATERAL subquery orders by the distance operator so the HNSW index can serve it.
NEIGHBOURS_SQL = text(
    """
    SELECT m.id AS source, n.id AS target, 1 - (m.embedding <=> n.embedding) AS similarity
    FROM memories AS m
    CROSS JOIN LATERAL (
        SELECT o.id, o.embedding
        FROM memories AS o
        WHERE o.workspace = :workspace AND o.id <> m.id AND o.embedding IS NOT NULL
        ORDER BY o.embedding <=> m.embedding
        LIMIT :k
    ) AS n
    WHERE m.workspace = :workspace AND m.embedding IS NOT NULL
    """
)


@dataclass
class Edge:
    source: uuid.UUID
    target: uuid.UUID
    similarity: float
    shared_tags: list[str]


def select_edges(
    pairs: list[tuple[uuid.UUID, uuid.UUID, float]], *, margin: float, min_similarity: float
) -> dict[tuple[uuid.UUID, uuid.UUID], float]:
    """Rank-based edge selection over directed kNN pairs.

    Each source keeps its best neighbour plus any near-ties (within `margin` of the best), all subject to the
    `min_similarity` floor. Directed picks are then merged into undirected edges (A->B and B->A become one).
    """
    by_source: dict[uuid.UUID, list[tuple[float, uuid.UUID]]] = {}
    for source, target, similarity in pairs:
        by_source.setdefault(source, []).append((similarity, target))

    edges: dict[tuple[uuid.UUID, uuid.UUID], float] = {}
    for source, candidates in by_source.items():
        candidates.sort(key=lambda c: c[0], reverse=True)
        best = candidates[0][0]
        for similarity, target in candidates:
            if similarity < min_similarity or similarity < best - margin:
                break
            key = (source, target) if str(source) < str(target) else (target, source)
            edges[key] = max(edges.get(key, 0.0), similarity)
    return edges


async def build_graph(
    session: AsyncSession, workspace: str, *, neighbors: int, margin: float, min_similarity: float
) -> tuple[list[Memory], list[Edge]]:
    memories = list(
        await session.scalars(select(Memory).where(Memory.workspace == workspace).order_by(Memory.created_at.desc()))
    )
    rows = await session.execute(NEIGHBOURS_SQL, {"workspace": workspace, "k": neighbors})
    merged = select_edges(
        [(r.source, r.target, float(r.similarity)) for r in rows], margin=margin, min_similarity=min_similarity
    )

    tags = {m.id: set(m.tags) for m in memories}
    edges = [
        Edge(a, b, round(sim, 4), sorted(tags.get(a, set()) & tags.get(b, set())))
        for (a, b), sim in sorted(merged.items(), key=lambda kv: kv[1], reverse=True)
    ]
    return memories, edges
