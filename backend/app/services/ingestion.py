"""Background document ingestion: a Postgres-backed job queue and the worker that drains it.

Uploads are stored as `ingestion_jobs` rows and acknowledged immediately; a worker claims them with
`FOR UPDATE SKIP LOCKED`, so any number of workers (in the API process or standalone) can run without
double-processing a job, and no extra broker is needed. A job whose worker died mid-way is reclaimed
after `ingestion_stale_seconds`.
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import undefer

from app.config import Settings
from app.models import IngestionJob
from app.services.embeddings import Embedder
from app.services.extraction import ExtractionError, UnsupportedDocument, extract_text
from app.services.memories import store_memory

logger = logging.getLogger(__name__)

CLAIM_SQL = text(
    """
    UPDATE ingestion_jobs
    SET status = 'processing', attempts = attempts + 1, started_at = now()
    WHERE id = (
        SELECT id FROM ingestion_jobs
        WHERE status = 'queued'
           OR (status = 'processing' AND started_at < now() - make_interval(secs => :stale_seconds))
        ORDER BY created_at
        FOR UPDATE SKIP LOCKED
        LIMIT 1
    )
    RETURNING id
    """
)


async def enqueue(
    session: AsyncSession,
    workspace: str,
    *,
    filename: str,
    content_type: str | None,
    data: bytes,
    title: str,
    type: str,
    tags: list[str],
) -> IngestionJob:
    job = IngestionJob(
        workspace=workspace,
        filename=filename,
        content_type=content_type,
        size_bytes=len(data),
        payload=data,
        title=title,
        type=type,
        tags=tags,
        status="queued",
        attempts=0,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job


async def get_job(session: AsyncSession, workspace: str, job_id: uuid.UUID) -> IngestionJob | None:
    return await session.scalar(
        select(IngestionJob).where(IngestionJob.id == job_id, IngestionJob.workspace == workspace)
    )


async def list_jobs(session: AsyncSession, workspace: str, limit: int) -> list[IngestionJob]:
    result = await session.scalars(
        select(IngestionJob)
        .where(IngestionJob.workspace == workspace)
        .order_by(IngestionJob.created_at.desc())
        .limit(limit)
    )
    return list(result)


async def claim_next(session: AsyncSession, stale_seconds: int) -> uuid.UUID | None:
    job_id = await session.scalar(CLAIM_SQL, {"stale_seconds": stale_seconds})
    await session.commit()
    return job_id


async def _finish(session: AsyncSession, job_id: uuid.UUID, status: str, **values) -> None:
    await session.execute(
        update(IngestionJob)
        .where(IngestionJob.id == job_id)
        .values(status=status, finished_at=datetime.now(timezone.utc), payload=None, **values)
    )
    await session.commit()


async def process_job(
    sessionmaker: async_sessionmaker[AsyncSession], embedder: Embedder, settings: Settings, job_id: uuid.UUID
) -> str:
    """Extract, chunk, embed and store one claimed job. Returns its resulting status."""
    async with sessionmaker() as session:
        job = await session.get(IngestionJob, job_id, options=[undefer(IngestionJob.payload)])
        if job is None:
            return "missing"
        if job.attempts > settings.ingestion_max_attempts:
            await _finish(session, job_id, "failed", error=f"Gave up after {job.attempts - 1} attempts")
            return "failed"
        try:
            body = await run_in_threadpool(extract_text, job.filename, job.payload or b"")
            if len(body) > settings.max_document_chars:
                raise ExtractionError(
                    f"Document has {len(body):,} characters of text; the limit is {settings.max_document_chars:,}"
                )
            # The memory and the job's success are committed together, so a crash never leaves one without the other.
            memory = await store_memory(
                session,
                embedder,
                job.workspace,
                title=job.title,
                body=body,
                type=job.type,
                tags=job.tags,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                commit=False,
            )
            job.status = "succeeded"
            job.memory_id = memory.id
            job.chars = len(body)
            job.error = None
            job.payload = None
            job.finished_at = datetime.now(timezone.utc)
            await session.commit()
            logger.info("Ingested %s as memory %s (%d chars)", job.filename, memory.id, len(body))
            return "succeeded"
        except (UnsupportedDocument, ExtractionError) as e:
            # The file itself is the problem: retrying cannot help.
            await session.rollback()
            await _finish(session, job_id, "failed", error=str(e))
            logger.info("Ingestion of %s failed: %s", job_id, e)
            return "failed"
        except Exception as e:
            await session.rollback()
            attempts = (await session.get(IngestionJob, job_id)).attempts
            logger.exception("Ingestion of %s errored (attempt %d)", job_id, attempts)
            if attempts >= settings.ingestion_max_attempts:
                await _finish(session, job_id, "failed", error=f"Internal error after {attempts} attempts: {e}")
                return "failed"
            await session.execute(
                update(IngestionJob).where(IngestionJob.id == job_id).values(status="queued", error=str(e))
            )
            await session.commit()
            return "queued"


class IngestionWorker:
    """Drains the job queue, then sleeps until notified of a new upload or the poll interval passes."""

    def __init__(self, sessionmaker: async_sessionmaker[AsyncSession], embedder: Embedder, settings: Settings):
        self.sessionmaker = sessionmaker
        self.embedder = embedder
        self.settings = settings
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self.run_forever(), name="ingestion-worker")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def notify(self) -> None:
        self._wake.set()

    async def run_once(self) -> bool:
        """Process the next claimable job, if any. Returns whether one was processed."""
        async with self.sessionmaker() as session:
            job_id = await claim_next(session, self.settings.ingestion_stale_seconds)
        if job_id is None:
            return False
        await process_job(self.sessionmaker, self.embedder, self.settings, job_id)
        return True

    async def run_forever(self) -> None:
        logger.info("Ingestion worker started")
        while True:
            self._wake.clear()  # cleared before draining, so an upload arriving mid-drain is not missed
            try:
                while await self.run_once():
                    pass
            except Exception:
                logger.exception("Ingestion worker failed to claim a job; retrying after the poll interval")
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self.settings.ingestion_poll_seconds)
            except asyncio.TimeoutError:
                pass
