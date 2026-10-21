from typing import Annotated

from fastapi import APIRouter, Query

from app.deps import SessionDep, SettingsDep
from app.schemas import GraphEdge, GraphNode, GraphResponse
from app.services.decay import retention
from app.services.graph import build_graph

router = APIRouter(prefix="/api", tags=["graph"])


@router.get("/graph", response_model=GraphResponse)
async def graph(
    session: SessionDep,
    settings: SettingsDep,
    neighbors: Annotated[int | None, Query(ge=1, le=20)] = None,
    margin: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    min_similarity: Annotated[float | None, Query(ge=-1.0, le=1.0)] = None,
):
    """Knowledge graph: memories as nodes, semantic similarity between them as weighted edges."""
    k = neighbors or settings.graph_neighbors
    margin = settings.graph_margin if margin is None else margin
    threshold = settings.graph_min_similarity if min_similarity is None else min_similarity
    memories, edges = await build_graph(
        session, settings.workspace, neighbors=k, margin=margin, min_similarity=threshold
    )
    return GraphResponse(
        nodes=[
            GraphNode(
                id=m.id,
                title=m.title,
                type=m.type,
                tags=m.tags,
                retention=round(retention(m.stability, m.last_accessed_at), 4),
            )
            for m in memories
        ],
        edges=[
            GraphEdge(source=e.source, target=e.target, similarity=e.similarity, shared_tags=e.shared_tags)
            for e in edges
        ],
        neighbors=k,
        margin=margin,
        min_similarity=threshold,
    )
