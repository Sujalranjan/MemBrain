import uuid
from datetime import datetime, timezone

from fastapi.concurrency import run_in_threadpool
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, Memory
from app.schemas import MemoryCreate, MemoryOut
from app.services.chunking import chunk_text
from app.services.decay import INITIAL_STABILITY_DAYS, half_life_days, retention, reviewed_stability
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
        stability=round(memory.stability, 2),
        access_count=memory.access_count,
        retention=round(retention(memory.stability, memory.last_accessed_at), 4),
        half_life_days=round(half_life_days(memory.stability), 1),
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
        stability=INITIAL_STABILITY_DAYS,
        access_count=0,
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


def review(memory: Memory, now: datetime) -> None:
    """Record a use of the memory: grow its stability per the spacing effect and reset its clock."""
    memory.stability = reviewed_stability(memory.stability, memory.last_accessed_at, now)
    memory.last_accessed_at = now
    memory.access_count += 1


async def reinforce_memory(session: AsyncSession, memory: Memory) -> Memory:
    review(memory, datetime.now(timezone.utc))
    await session.commit()
    return memory


async def record_usage(session: AsyncSession, memories: list[Memory]) -> None:
    now = datetime.now(timezone.utc)
    for memory in memories:
        review(memory, now)
    await session.commit()
