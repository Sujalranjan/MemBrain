import logging

from fastapi import APIRouter
from fastapi.concurrency import run_in_threadpool

from app.deps import EmbedderDep, SessionDep, SettingsDep
from app.schemas import SearchRequest, SearchResponse, SearchResult
from app.services import search as svc

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest, session: SessionDep, embedder: EmbedderDep, settings: SettingsDep):
    query_vec = await run_in_threadpool(embedder.embed_query, req.query) if req.mode != "keyword" else None
    hits = await svc.search(
        session,
        settings.workspace,
        req.query,
        query_vec,
        mode=req.mode,
        k=req.k,
        rrf_k=settings.rrf_k,
        candidate_pool=settings.candidate_pool,
    )
    logger.info("Search mode=%s k=%d hits=%d", req.mode, req.k, len(hits))
    return SearchResponse(
        query=req.query,
        mode=req.mode,
        results=[
            SearchResult(
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
            for h in hits
        ],
    )
