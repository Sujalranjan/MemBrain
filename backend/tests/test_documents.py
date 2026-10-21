import asyncio
import time

import asyncpg

from app.config import Settings
from app.db import make_engine, make_sessionmaker
from app.services.ingestion import IngestionWorker
from tests.conftest import FakeEmbedder, _raw_dsn
from tests.documents import make_docx, make_pdf

PDF = make_pdf(
    "Incident review: the payments outage was caused by an expired TLS certificate.",
    "Action item: automate certificate renewal with cert-manager.",
)


def upload(client, filename, data, **form):
    r = client.post("/api/documents", files={"file": (filename, data)}, data=form)
    return r


def wait_for(client, job_id, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/documents/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed"):
            return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} still {job['status']} after {timeout}s")


async def _sql(url, query, *args):
    conn = await asyncpg.connect(_raw_dsn(url))
    try:
        return await conn.fetch(query, *args)
    finally:
        await conn.close()


class FlakyEmbedder(FakeEmbedder):
    """Fails the first `failures` calls, like a model server that is briefly down."""

    def __init__(self, failures):
        self.failures = failures

    def embed_documents(self, texts):
        if self.failures:
            self.failures -= 1
            raise RuntimeError("embedding backend unavailable")
        return super().embed_documents(texts)


async def _run_worker_once(url, embedder=None, times=1):
    engine = make_engine(url)
    try:
        worker = IngestionWorker(make_sessionmaker(engine), embedder or FakeEmbedder(), Settings(database_url=url))
        return [await worker.run_once() for _ in range(times)][-1]
    finally:
        await engine.dispose()


def test_pdf_upload_is_indexed_in_the_background(client):
    r = upload(client, "outage-review.pdf", PDF, type="meeting", tags="incident, payments")
    assert r.status_code == 202, r.text
    assert r.json()["status"] == "queued" and r.json()["title"] == "outage-review"

    job = wait_for(client, r.json()["id"])
    assert job["status"] == "succeeded", job["error"]
    assert job["attempts"] == 1 and job["chars"] > 100

    [memory] = client.get("/api/memories").json()
    assert memory["id"] == job["memory_id"]
    assert memory["type"] == "meeting" and memory["tags"] == ["incident", "payments"]
    assert "cert-manager" in memory["body"]

    hits = client.post("/api/search", json={"query": "certificate renewal"}).json()["results"]
    assert hits[0]["id"] == memory["id"]


def test_docx_upload_uses_given_title(client):
    data = make_docx(["Onboarding SOP", "Request laptop access on day one."])
    job = wait_for(client, upload(client, "sop.docx", data, title="New hire onboarding").json()["id"])
    assert job["status"] == "succeeded"
    assert client.get("/api/memories").json()[0]["title"] == "New hire onboarding"


def test_upload_rejections(make_client):
    client = make_client(max_upload_bytes=1024)
    assert upload(client, "deck.pptx", b"PK\x03\x04").status_code == 415
    assert upload(client, "fake.pdf", b"not a pdf").status_code == 422
    assert upload(client, "empty.txt", b"").status_code == 422
    assert upload(client, "big.txt", b"x" * 2048).status_code == 413
    assert client.get("/api/documents/jobs").json() == []


def test_unreadable_document_fails_without_retry(client, database_url):
    job = wait_for(client, upload(client, "broken.pdf", b"%PDF-1.4\ngarbage").json()["id"])
    assert job["status"] == "failed" and job["attempts"] == 1
    assert "Could not read PDF" in job["error"]
    assert client.get("/api/memories").json() == []
    # The upload is not kept once the job is finished.
    [row] = asyncio.run(_sql(database_url, "SELECT payload FROM ingestion_jobs WHERE id = $1::uuid", job["id"]))
    assert row["payload"] is None


def test_jobs_are_listed_newest_first_and_scoped_to_workspace(make_client):
    a, b = make_client("alpha"), make_client("beta")
    first = upload(a, "one.txt", b"first note").json()["id"]
    second = upload(a, "two.txt", b"second note").json()["id"]
    assert [j["id"] for j in a.get("/api/documents/jobs").json()] == [second, first]
    assert b.get("/api/documents/jobs").json() == []
    assert b.get(f"/api/documents/jobs/{first}").status_code == 404


def test_job_abandoned_by_a_crashed_worker_is_reclaimed(make_client, database_url):
    client = make_client(ingestion_worker=False)
    job_id = upload(client, "notes.txt", b"Standup: deploy freeze starts Friday.").json()["id"]
    assert client.get(f"/api/documents/jobs/{job_id}").json()["status"] == "queued"

    # Simulate a worker that claimed the job an hour ago and then died.
    asyncio.run(
        _sql(
            database_url,
            "UPDATE ingestion_jobs SET status = 'processing', attempts = 1, started_at = now() - interval '1 hour' "
            "WHERE id = $1::uuid",
            job_id,
        )
    )
    assert asyncio.run(_run_worker_once(database_url))
    job = client.get(f"/api/documents/jobs/{job_id}").json()
    assert job["status"] == "succeeded" and job["attempts"] == 2


def test_job_that_keeps_crashing_workers_is_given_up(make_client, database_url):
    client = make_client(ingestion_worker=False)
    job_id = upload(client, "notes.txt", b"poison").json()["id"]
    asyncio.run(
        _sql(
            database_url,
            "UPDATE ingestion_jobs SET status = 'processing', attempts = 3, started_at = now() - interval '1 hour' "
            "WHERE id = $1::uuid",
            job_id,
        )
    )
    assert asyncio.run(_run_worker_once(database_url))
    job = client.get(f"/api/documents/jobs/{job_id}").json()
    assert job["status"] == "failed" and "Gave up after 3 attempts" in job["error"]
    assert client.get("/api/memories").json() == []


def test_fresh_processing_job_is_not_stolen(make_client, database_url):
    client = make_client(ingestion_worker=False)
    job_id = upload(client, "notes.txt", b"in progress elsewhere").json()["id"]
    asyncio.run(
        _sql(
            database_url,
            "UPDATE ingestion_jobs SET status = 'processing', attempts = 1, started_at = now() WHERE id = $1::uuid",
            job_id,
        )
    )
    assert not asyncio.run(_run_worker_once(database_url))


def test_transient_error_is_retried(make_client, database_url):
    client = make_client(ingestion_worker=False)
    job_id = upload(client, "notes.txt", b"Retro: keep the release train weekly.").json()["id"]

    asyncio.run(_run_worker_once(database_url, FlakyEmbedder(failures=1)))
    job = client.get(f"/api/documents/jobs/{job_id}").json()
    assert job["status"] == "queued" and "unavailable" in job["error"]

    asyncio.run(_run_worker_once(database_url))
    job = client.get(f"/api/documents/jobs/{job_id}").json()
    assert job["status"] == "succeeded" and job["attempts"] == 2 and job["error"] is None


def test_persistent_error_fails_after_max_attempts(make_client, database_url):
    client = make_client(ingestion_worker=False)
    job_id = upload(client, "notes.txt", b"Retro notes").json()["id"]
    asyncio.run(_run_worker_once(database_url, FlakyEmbedder(failures=99), times=3))
    job = client.get(f"/api/documents/jobs/{job_id}").json()
    assert job["status"] == "failed" and job["attempts"] == 3 and "after 3 attempts" in job["error"]
