import uuid
from datetime import datetime, timezone

from fastapi.concurrency import run_in_threadpool
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, Memory
from app.schemas import MemoryCreate, MemoryOut
from app.services.chunking import chunk_text
from app.services.decay import current_strength, reinforced_strength
from app.services.embeddings import Embedder


def to_out(memory: Memory) -> MemoryOut:
    return MemoryOut(
        id=memory.id,
        title=memory.title,
        body=memory.body,
        type=memory.type,
        tags=memory.tags,
        created_at=memory.created_at,
        last_accessed_at=memory.last_accessed_at,
        strength=memory.strength,
        current_strength=round(current_strength(memory.strength, memory.last_accessed_at), 4),
    )


async def create_memory(
    session: AsyncSession,
    embedder: Embedder,
    workspace: str,
    data: MemoryCreate,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> Memory:
    chunks = chunk_text(data.body, size=chunk_size, overlap=chunk_overlap)
    # The title is embedded with every chunk so each one carries its document's context.
    vectors = await run_in_threadpool(embedder.embed_documents, [f"{data.title}\n\n{c}" for c in chunks])
    centroid = [sum(dim) / len(vectors) for dim in zip(*vectors, strict=True)]

    now = datetime.now(timezone.utc)
    memory = Memory(
        workspace=workspace,
        title=data.title,
        body=data.body,
        type=data.type,
        tags=data.tags,
        created_at=now,
        last_accessed_at=now,
        strength=1.0,
        embedding=centroid,
        chunks=[Chunk(ordinal=i, content=c, embedding=v) for i, (c, v) in enumerate(zip(chunks, vectors, strict=True))],
    )
    session.add(memory)
    await session.commit()
    await session.refresh(memory)
    return memory


async def list_memories(session: AsyncSession, workspace: str, limit: int, offset: int) -> list[Memory]:
    result = await session.scalars(
        select(Memory)
        .where(Memory.workspace == workspace)
        .order_by(Memory.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(result)


async def get_memory(session: AsyncSession, workspace: str, memory_id: uuid.UUID) -> Memory | None:
    return await session.scalar(select(Memory).where(Memory.id == memory_id, Memory.workspace == workspace))


async def delete_memory(session: AsyncSession, workspace: str, memory_id: uuid.UUID) -> bool:
    result = await session.execute(delete(Memory).where(Memory.id == memory_id, Memory.workspace == workspace))
    await session.commit()
    return result.rowcount > 0


async def reinforce_memory(session: AsyncSession, memory: Memory) -> Memory:
    now = datetime.now(timezone.utc)
    memory.strength = reinforced_strength(memory.strength, memory.last_accessed_at, now)
    memory.last_accessed_at = now
    await session.commit()
    return memory
