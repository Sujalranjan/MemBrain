import logging

from fastapi import APIRouter

from app.deps import AnswererDep, EmbedderDep, SessionDep, SettingsDep, WorkspaceDep, ask_rate_limit
from app.routers.search import run_search, to_result
from app.schemas import AskRequest, AskResponse, CitationOut
from app.services.answers import AnswerUnavailable, Source, source_excerpt
from app.services.memories import record_usage

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["ask"])

NO_MEMORIES = "I couldn't find any memories related to that question. Try capturing the relevant knowledge first."


@router.post("/ask", response_model=AskResponse, dependencies=[ask_rate_limit])
async def ask(
    req: AskRequest,
    session: SessionDep,
    embedder: EmbedderDep,
    answerer: AnswererDep,
    settings: SettingsDep,
    workspace: WorkspaceDep,
):
    """Retrieve the most relevant memories, then answer from them with verifiable citations."""
    hits = await run_search(
        session, embedder, settings, workspace, req.question, mode="hybrid", k=req.k or settings.answer_sources
    )
    results = [to_result(h) for h in hits]
    if not hits:
        return AskResponse(question=req.question, answer=NO_MEMORIES, grounded=False, citations=[], sources=[])

    sources = [
        Source(
            memory_id=h.memory.id,
            title=h.memory.title,
            type=h.memory.type,
            created_at=h.memory.created_at.date().isoformat(),
            text=source_excerpt(h.memory.body, h.snippet, settings.answer_max_source_chars),
        )
        for h in hits
    ]
    try:
        answer = await answerer.answer(req.question, sources)
    except AnswerUnavailable as e:
        logger.warning("Answer unavailable: %s", e)
        return AskResponse(
            question=req.question, answer=None, grounded=False, citations=[], sources=results, notice=str(e)
        )

    # Memories the answer actually cited count as used, which slows their decay.
    cited = {c.source for c in answer.citations if 1 <= c.source <= len(hits)}
    await record_usage(session, [hits[n - 1].memory for n in sorted(cited)])

    return AskResponse(
        question=req.question,
        answer=answer.text,
        grounded=answer.grounded,
        citations=[
            CitationOut(
                source=c.source,
                memory_id=sources[c.source - 1].memory_id,
                title=sources[c.source - 1].title,
                cited_text=c.cited_text,
            )
            for c in answer.citations
            if 1 <= c.source <= len(sources)
        ],
        sources=results,
        model=answer.model,
    )
