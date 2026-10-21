import asyncio
import hashlib
import math
import os
import re

import asyncpg
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient

from app.config import BACKEND_DIR, Settings
from app.main import create_app
from app.services.answers import Answer, AnswerUnavailable, Citation

TEST_DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL", "postgresql+asyncpg://membrain:membrain@localhost:5432/membrain_test"
)
DIM = 384


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder: texts sharing words get similar vectors."""

    dim = DIM

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * DIM
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


class FakeAnswerer:
    """Cites the first sentence of the top source; set `fail` to simulate an unavailable provider."""

    def __init__(self):
        self.calls = []
        self.fail = False

    async def answer(self, question, sources):
        self.calls.append((question, sources))
        if self.fail:
            raise AnswerUnavailable("Anthropic credentials are not configured")
        quote = sources[0].text.split(".")[0]
        return Answer(text=f"{quote}.[1]", citations=[Citation(source=1, cited_text=quote)], model="fake-model")


def _raw_dsn(url: str) -> str:
    return url.replace("postgresql+asyncpg://", "postgresql://")


async def _ensure_database(url: str) -> None:
    base, name = _raw_dsn(url).rsplit("/", 1)
    conn = await asyncpg.connect(f"{base}/postgres", timeout=3)
    try:
        if not await conn.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", name):
            await conn.execute(f'CREATE DATABASE "{name}"')
    finally:
        await conn.close()


async def _truncate(url: str) -> None:
    conn = await asyncpg.connect(_raw_dsn(url))
    try:
        await conn.execute("TRUNCATE memories, ingestion_jobs CASCADE")
    finally:
        await conn.close()


@pytest.fixture(scope="session")
def database_url():
    try:
        asyncio.run(_ensure_database(TEST_DATABASE_URL))
    except (OSError, asyncpg.PostgresError, asyncio.TimeoutError) as e:
        if os.getenv("REQUIRE_TEST_DATABASE"):
            raise
        pytest.skip(f"Postgres not available for integration tests: {e}")
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    cfg.set_main_option("sqlalchemy.url", TEST_DATABASE_URL)
    command.upgrade(cfg, "head")
    return TEST_DATABASE_URL


@pytest.fixture
def answerer():
    return FakeAnswerer()


@pytest.fixture
def make_client(database_url, answerer):
    asyncio.run(_truncate(database_url))
    clients = []

    def _make(workspace: str = "test", **overrides) -> TestClient:
        settings = Settings(database_url=database_url, workspace=workspace, ingestion_poll_seconds=0.2, **overrides)
        app = create_app(settings, embedder=FakeEmbedder(), answerer=answerer)
        c = TestClient(app)
        c.__enter__()
        clients.append(c)
        return c

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture
def client(make_client):
    return make_client()
