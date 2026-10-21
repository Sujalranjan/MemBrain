import asyncio
import uuid

import asyncpg
import pytest

from app.db import make_engine, make_sessionmaker
from app.services import auth
from tests.conftest import _raw_dsn


async def _create_key(url, workspace, name="test"):
    engine = make_engine(url)
    try:
        async with make_sessionmaker(engine)() as session:
            _, key = await auth.create_key(session, workspace, name)
            return key
    finally:
        await engine.dispose()


async def _revoke(url, key):
    engine = make_engine(url)
    try:
        async with make_sessionmaker(engine)() as session:
            return await auth.revoke_key(session, auth.KEY_PATTERN.match(key).group(1))
    finally:
        await engine.dispose()


@pytest.fixture
def secured(make_client, database_url):
    """An auth-enabled client plus a factory for keys: key('acme') -> headers for a key in workspace acme."""
    client = make_client(auth_enabled=True)

    def key(workspace, name="test"):
        return {"Authorization": f"Bearer {asyncio.run(_create_key(database_url, workspace, name))}"}

    return client, key


def test_requests_without_a_valid_key_are_rejected(secured):
    client, _ = secured
    r = client.get("/api/memories")
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
    assert client.get("/api/memories", headers={"Authorization": "Bearer nope"}).status_code == 401
    forged = "mb_000000000000_" + "x" * 43
    assert client.get("/api/memories", headers={"X-API-Key": forged}).status_code == 401
    # Health and the UI stay public.
    assert client.get("/health").status_code == 200


def test_key_scopes_every_request_to_its_workspace(secured):
    client, key = secured
    acme, globex = key("acme"), key("globex")
    mid = client.post("/api/memories", json={"title": "Acme secret", "body": "Q3 launch plan"}, headers=acme).json()[
        "id"
    ]

    assert client.get("/api/me", headers=globex).json()["workspace"] == "globex"
    assert client.get("/api/memories", headers=globex).json() == []
    assert client.post("/api/search", json={"query": "launch plan"}, headers=globex).json()["results"] == []
    assert client.get("/api/graph", headers=globex).json()["nodes"] == []
    assert client.post(f"/api/memories/{mid}/reinforce", headers=globex).status_code == 404
    assert client.delete(f"/api/memories/{mid}", headers=globex).status_code == 404

    assert [m["id"] for m in client.get("/api/memories", headers=acme).json()] == [mid]


def test_document_jobs_are_isolated(secured):
    client, key = secured
    acme, globex = key("acme"), key("globex")
    job = client.post("/api/documents", files={"file": ("plan.txt", b"Acme roadmap")}, headers=acme).json()
    assert client.get(f"/api/documents/jobs/{job['id']}", headers=globex).status_code == 404
    assert client.get("/api/documents/jobs", headers=globex).json() == []


def test_x_api_key_header_and_revocation(secured, database_url):
    client, _ = secured
    key = asyncio.run(_create_key(database_url, "acme", "laptop"))
    me = client.get("/api/me", headers={"X-API-Key": key}).json()
    assert me == {"workspace": "acme", "key": "laptop", "auth_enabled": True, "read_only": False}

    assert asyncio.run(_revoke(database_url, key))
    assert client.get("/api/me", headers={"X-API-Key": key}).status_code == 401


def test_only_a_hash_of_the_key_is_stored(secured, database_url):
    _, key = secured
    raw = key("acme")["Authorization"].removeprefix("Bearer ")

    async def stored():
        conn = await asyncpg.connect(_raw_dsn(database_url))
        try:
            return await conn.fetchrow("SELECT prefix, key_hash FROM api_keys")
        finally:
            await conn.close()

    row = asyncio.run(stored())
    assert raw not in row.values() and row["key_hash"] == auth.hash_key(raw) and raw.startswith(f"mb_{row['prefix']}_")


def test_invalid_workspace_names_are_refused(database_url):
    with pytest.raises(ValueError):
        asyncio.run(_create_key(database_url, "Not A Slug!"))


def test_rate_limit_returns_429_with_retry_after(make_client):
    client = make_client(rate_limit_enabled=True, rate_limit_per_minute=3)
    statuses = [client.get("/api/memories").status_code for _ in range(4)]
    assert statuses == [200, 200, 200, 429]
    r = client.get("/api/memories")
    assert r.status_code == 429 and 1 <= int(r.headers["retry-after"]) <= 20
    assert r.headers["x-ratelimit-remaining"] == "0"


def test_rate_limit_headers_count_down(make_client):
    client = make_client(rate_limit_enabled=True, rate_limit_per_minute=10)
    remaining = [int(client.get("/api/memories").headers["x-ratelimit-remaining"]) for _ in range(3)]
    assert remaining == [9, 8, 7]


def test_rate_limits_are_per_key(make_client, database_url):
    client = make_client(auth_enabled=True, rate_limit_enabled=True, rate_limit_per_minute=2)
    a = {"X-API-Key": asyncio.run(_create_key(database_url, "acme"))}
    b = {"X-API-Key": asyncio.run(_create_key(database_url, "globex"))}
    assert [client.get("/api/memories", headers=a).status_code for _ in range(3)] == [200, 200, 429]
    assert client.get("/api/memories", headers=b).status_code == 200


def test_ask_has_its_own_tighter_limit(make_client):
    client = make_client(rate_limit_enabled=True, rate_limit_per_minute=100, ask_rate_limit_per_minute=1)
    assert client.post("/api/ask", json={"question": "anything?"}).status_code == 200
    assert client.post("/api/ask", json={"question": "again?"}).status_code == 429
    assert client.get("/api/memories").status_code == 200  # the general bucket is unaffected


def test_concurrent_requests_cannot_overspend_a_bucket(database_url):
    """Many simultaneous requests on one bucket: exactly `capacity` succeed, even across connections."""
    from app.services.ratelimit import take

    bucket = f"burst-{uuid.uuid4()}"

    async def burst():
        engine = make_engine(database_url)
        sessionmaker = make_sessionmaker(engine)

        async def one():
            async with sessionmaker() as session:
                return (await take(session, bucket, 5)).allowed

        try:
            return await asyncio.gather(*(one() for _ in range(20)))
        finally:
            await engine.dispose()

    assert sum(asyncio.run(burst())) == 5


def test_read_only_demo_blocks_writes_but_serves_reads(make_client):
    writer = make_client("demo")
    mid = writer.post("/api/memories", json={"title": "Runbook", "body": "Restart the worker."}).json()["id"]

    demo = make_client("demo", read_only=True)
    assert demo.post("/api/memories", json={"title": "x", "body": "y"}).status_code == 403
    assert demo.delete(f"/api/memories/{mid}").status_code == 403
    assert demo.post(f"/api/memories/{mid}/reinforce").status_code == 403
    assert demo.post("/api/documents", files={"file": ("a.txt", b"hello")}).status_code == 403
    assert demo.post("/api/search", json={"query": "restart worker"}).json()["results"][0]["id"] == mid
    assert demo.post("/api/ask", json={"question": "how to restart?"}).status_code == 200
    assert demo.get("/api/me").json()["read_only"] is True
