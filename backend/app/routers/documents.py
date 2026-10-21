import logging
import uuid
from pathlib import PurePath
from typing import Annotated

from fastapi import APIRouter, Form, HTTPException, Query, Request, UploadFile

from app.deps import SessionDep, SettingsDep, WorkspaceDep
from app.schemas import IngestionJobOut
from app.services import ingestion as svc
from app.services.extraction import ExtractionError, UnsupportedDocument, detect_format

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("", response_model=IngestionJobOut, status_code=202)
async def upload_document(
    request: Request,
    session: SessionDep,
    settings: SettingsDep,
    workspace: WorkspaceDep,
    file: UploadFile,
    title: Annotated[str | None, Form(max_length=300)] = None,
    memory_type: Annotated[str, Form(alias="type", min_length=1, max_length=50)] = "file",
    tags: Annotated[str, Form(max_length=2_000, description="Comma-separated")] = "",
):
    """Queue a PDF, DOCX or text file for background extraction and indexing; poll the returned job."""
    filename = PurePath(file.filename or "upload").name[:300]
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_bytes // (1024 * 1024)} MB")
    if not data:
        raise HTTPException(status_code=422, detail="File is empty")
    # Reject wrong types up front; the expensive extraction happens in the worker.
    try:
        detect_format(filename, data)
    except UnsupportedDocument as e:
        raise HTTPException(status_code=415, detail=str(e)) from e
    except ExtractionError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    job = await svc.enqueue(
        session,
        workspace,
        filename=filename,
        content_type=file.content_type,
        data=data,
        title=(title or "").strip() or PurePath(filename).stem,
        type=memory_type.strip(),
        tags=list(dict.fromkeys(t.strip()[:100] for t in tags.split(",") if t.strip()))[:50],
    )
    if worker := getattr(request.app.state, "worker", None):
        worker.notify()
    logger.info("Queued %s (%d bytes) as job %s", filename, len(data), job.id)
    return job


@router.get("/jobs", response_model=list[IngestionJobOut])
async def list_jobs(session: SessionDep, workspace: WorkspaceDep, limit: Annotated[int, Query(ge=1, le=100)] = 20):
    return await svc.list_jobs(session, workspace, limit)


@router.get("/jobs/{job_id}", response_model=IngestionJobOut)
async def get_job(job_id: uuid.UUID, session: SessionDep, workspace: WorkspaceDep):
    job = await svc.get_job(session, workspace, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
