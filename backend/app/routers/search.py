import logging

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.deps import EmbedderDep, SessionDep, SettingsDep
from app.schemas import SearchRequest, SearchResponse, SearchResult
from app.services import search as svc
from app.services.embeddings import Embedder

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["search"])


async def run_search(
    session: AsyncSession, embedder: Embedder, settings: Settings, query: str, *, mode: svc.SearchMode, k: int
) -> list[svc.Hit]:
    query_vec = await run_in_threadpool(embedder.embed_query, query) if mode != "keyword" else None
    hits = await svc.search(
        session,
        settings.workspace,
        query,
        query_vec,
        mode=mode,
        k=k,
        rrf_k=settings.rrf_k,
        candidate_pool=settings.candidate_pool,
    )
    logger.info("Search mode=%s k=%d hits=%d", mode, k, len(hits))
    return hits


def to_result(h: svc.Hit) -> SearchResult:
    return SearchResult(
        id=h.memory.id,
        title=h.memory.title,
        type=h.memory.type,
        tags=h.memory.tags,
        snippet=h.snippet,
        score=h.score,
        vector_similarity=h.vector_similarity,
        vector_rank=h.vector_rank,
        keyword_rank=h.keyword_rank,
        matched_by=h.matched_by,
    )


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest, session: SessionDep, embedder: EmbedderDep, settings: SettingsDep):
    hits = await run_search(session, embedder, settings, req.query, mode=req.mode, k=req.k)
    return SearchResponse(query=req.query, mode=req.mode, results=[to_result(h) for h in hits])
