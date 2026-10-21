import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Response

from app.deps import EmbedderDep, SessionDep, SettingsDep, WorkspaceDep, writable
from app.schemas import MemoryCreate, MemoryOut
from app.services import memories as svc

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/memories", tags=["memories"])


@router.post("", response_model=MemoryOut, status_code=201, dependencies=[writable])
async def create_memory(
    data: MemoryCreate, session: SessionDep, embedder: EmbedderDep, settings: SettingsDep, workspace: WorkspaceDep
):
    memory = await svc.create_memory(
        session,
        embedder,
        workspace,
        data,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    logger.info("Stored memory %s type=%s chars=%d", memory.id, memory.type, len(memory.body))
    return svc.to_out(memory)


@router.get("", response_model=list[MemoryOut])
async def list_memories(
    session: SessionDep,
    workspace: WorkspaceDep,
    limit: Annotated[int, Query(ge=1, le=1000)] = 200,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    return [svc.to_out(m) for m in await svc.list_memories(session, workspace, limit, offset)]


@router.delete("/{memory_id}", status_code=204, dependencies=[writable])
async def delete_memory(memory_id: uuid.UUID, session: SessionDep, workspace: WorkspaceDep):
    if not await svc.delete_memory(session, workspace, memory_id):
        raise HTTPException(status_code=404, detail="Memory not found")
    return Response(status_code=204)


@router.post("/{memory_id}/reinforce", response_model=MemoryOut, dependencies=[writable])
async def reinforce_memory(memory_id: uuid.UUID, session: SessionDep, workspace: WorkspaceDep):
    memory = await svc.get_memory(session, workspace, memory_id)
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return svc.to_out(await svc.reinforce_memory(session, memory))
